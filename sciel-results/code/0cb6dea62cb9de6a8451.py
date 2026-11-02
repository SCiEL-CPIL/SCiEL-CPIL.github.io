import time
import numpy as np
import random
from collections import defaultdict
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls (neighbors)
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    neighbors = np.zeros((total, ball_size), dtype=np.int32)
    
    for w in range(total):
        neighbors[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                neighbors[w, idx] = w + (new_digit - digit) * weight
                idx += 1

    best_size = total
    best_set = None
    
    np.random.seed(42)
    random.seed(42)
    
    def prune_redundant(selected):
        """Remove codewords that are not needed"""
        if not selected:
            return selected
        
        selected_list = list(selected)
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected_list:
            cover_count[neighbors[c]] += 1
        
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            indices = list(range(len(selected_list)))
            random.shuffle(indices)
            for i in indices:
                c = selected_list[i]
                if np.min(cover_count[neighbors[c]]) > 1:
                    cover_count[neighbors[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            if changed:
                selected_list = new_selected
            else:
                break
            
        return selected_list

    def local_search_swap(selected):
        """
        Swap-based local search:
        1. Identify critical words for each codeword.
        2. Find candidates c' that cover ALL critical words of c.
        3. If c' is not in selected, swap c with c'.
        """
        if not selected:
            return selected
        
        selected_set = set(selected)
        max_passes = 50
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
                
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[neighbors[c]] += 1
                
            improved = False
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                c = selected[i]
                crit_words = neighbors[c][cover_count[neighbors[c]] == 1]
                
                if len(crit_words) == 0:
                    continue
                
                candidates = set(neighbors[crit_words[0]].tolist())
                
                for w in crit_words[1:]:
                    candidates.intersection_update(neighbors[w].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                new_candidates = [c_prime for c_prime in candidates if c_prime not in selected_set]
                
                if new_candidates:
                    c_prime = random.choice(new_candidates)
                    selected[i] = c_prime
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    improved = True
                    break
            
            if not improved:
                break
                
            selected = prune_redundant(selected)
            selected_set = set(selected)
            
        return selected

    def add_remove_local_search(selected):
        """
        Add-remove local search (2-for-1 exchange):
        Try to remove 2 codewords and add 1 to maintain coverage.
        """
        if len(selected) < 3:
            return selected
            
        selected_set = set(selected)
        improved = True
        max_passes = 20
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
                
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[neighbors[c]] += 1
                
            n_selected = len(selected)
            pairs_to_try = min(150, n_selected * (n_selected - 1) // 2)
            
            found_swap = False
            for _ in range(pairs_to_try):
                i, j = random.sample(range(n_selected), 2)
                c1, c2 = selected[i], selected[j]
                
                risk_words = set()
                for w in neighbors[c1]:
                    if cover_count[w] <= 2:
                        risk_words.add(w)
                for w in neighbors[c2]:
                    if cover_count[w] <= 2:
                        risk_words.add(w)
                        
                if not risk_words:
                    selected.remove(c1)
                    selected.remove(c2)
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    found_swap = True
                    break
                    
                if len(risk_words) > 6:
                    continue
                    
                risk_list = list(risk_words)
                candidates = set(neighbors[risk_list[0]].tolist())
                for w in risk_list[1:]:
                    candidates.intersection_update(neighbors[w].tolist())
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                valid_candidates = [c_prime for c_prime in candidates if c_prime not in selected_set]
                
                if valid_candidates:
                    c_prime = random.choice(valid_candidates)
                    selected.remove(c1)
                    selected.remove(c2)
                    selected.append(c_prime)
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    selected_set.add(c_prime)
                    found_swap = True
                    break
            
            if not found_swap:
                break
                
            selected = prune_redundant(selected)
            selected_set = set(selected)
            
        return selected

    def add_remove_3_for_2(selected):
        """
        3-for-2 exchange: Remove 3, add 2.
        This is more expensive but can escape deeper local minima.
        """
        if len(selected) < 4:
            return selected
            
        selected_set = set(selected)
        max_passes = 5
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
                
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[neighbors[c]] += 1
                
            n_selected = len(selected)
            triples_to_try = min(20, n_selected) # Limit triples due to cost
            
            found_swap = False
            for _ in range(triples_to_try):
                i, j, k = random.sample(range(n_selected), 3)
                c1, c2, c3 = selected[i], selected[j], selected[k]
                
                # Words that become uncovered if we remove c1, c2, c3
                # They must be covered by at least one of the new 2 codewords.
                # Actually, we need to find 2 codewords that cover all "risk" words.
                # Risk words: w where cover_count[w] <= 3 and w is in N(c1) U N(c2) U N(c3)
                # AND w is not covered by any other selected codeword? No, cover_count includes all.
                # If we remove c1, c2, c3, the new count for w is cover_count[w] - delta.
                # We need new_count >= 1.
                # So we need cover_count[w] - delta >= 1 => delta <= cover_count[w] - 1.
                # delta is the number of {c1, c2, c3} that cover w.
                
                risk_words = set()
                for w in neighbors[c1]:
                    if cover_count[w] <= 3:
                        risk_words.add(w)
                for w in neighbors[c2]:
                    if cover_count[w] <= 3:
                        risk_words.add(w)
                for w in neighbors[c3]:
                    if cover_count[w] <= 3:
                        risk_words.add(w)
                        
                if not risk_words:
                    # Can remove all 3 safely? Then we just remove them.
                    # But we must add 2? No, if we remove 3 and add 0, size drops by 3.
                    # But we need to maintain coverage. If risk_words is empty, it means
                    # all words in the union of balls are covered by at least 4 codewords.
                    # So removing 3 leaves them covered by at least 1.
                    # We can just remove them.
                    selected.remove(c1)
                    selected.remove(c2)
                    selected.remove(c3)
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    selected_set.discard(c3)
                    found_swap = True
                    break
                    
                if len(risk_words) > 8:
                    continue
                    
                # Find 2 codewords that cover all risk_words.
                # This is a set cover problem on risk_words.
                # Heuristic: Pick one candidate that covers most risk words, then find another that covers the rest.
                
                # Get all candidates that cover at least one risk word.
                # Actually, the new codewords must cover ALL risk words.
                # Let's iterate over pairs of candidates? Too slow.
                
                # Alternative:
                # 1. Pick a random risk word w1.
                # 2. Candidates for first new codeword: N(w1).
                # 3. For each candidate c_new1 in N(w1):
                #    remaining_risk = risk_words - N(c_new1)
                #    If remaining_risk is empty, we only need 1 new codeword (so we remove 3 add 1, net -2).
                #    If remaining_risk is not empty, pick a random word w2 in remaining_risk.
                #    Candidates for second new codeword: N(w2) intersect (covers all remaining_risk).
                
                w1 = random.choice(list(risk_words))
                cand1_list = neighbors[w1].tolist()
                
                for c_new1 in cand1_list:
                    if c_new1 in selected_set:
                        continue
                    remaining = risk_words - set(neighbors[c_new1].tolist())
                    if not remaining:
                        # Remove 3, Add 1
                        selected.remove(c1)
                        selected.remove(c2)
                        selected.remove(c3)
                        selected.append(c_new1)
                        selected_set.discard(c1)
                        selected_set.discard(c2)
                        selected_set.discard(c3)
                        selected_set.add(c_new1)
                        found_swap = True
                        break
                    else:
                        w2 = random.choice(list(remaining))
                        cand2_list = neighbors[w2].tolist()
                        for c_new2 in cand2_list:
                            if c_new2 in selected_set or c_new2 == c_new1:
                                continue
                            # Check if c_new2 covers all remaining
                            if remaining.issubset(set(neighbors[c_new2].tolist())):
                                # Remove 3, Add 2
                                selected.remove(c1)
                                selected.remove(c2)
                                selected.remove(c3)
                                selected.append(c_new1)
                                selected.append(c_new2)
                                selected_set.discard(c1)
                                selected_set.discard(c2)
                                selected_set.discard(c3)
                                selected_set.add(c_new1)
                                selected_set.add(c_new2)
                                found_swap = True
                                break
                        if found_swap:
                            break
                if found_swap:
                    break
            
            if not found_swap:
                break
                
            selected = prune_redundant(selected)
            selected_set = set(selected)
            
        return selected

    def greedy_construct(randomize=False):
        """Randomized greedy set cover construction"""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size
        
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            
            if max_count <= 0:
                break
            
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            
            if randomize and len(candidates) > 1:
                # Pick from top 10% or random among top
                top_k = min(10, len(candidates))
                best_indices = np.random.choice(len(candidates), top_k, replace=False)
                best_idx = candidates[best_indices[np.random.randint(top_k)]]
            else:
                best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            active[best_idx] = False
            
            for w in neighbors[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    for c in neighbors[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
            
            if time.time() > deadline - 10.0:
                break
        
        return selected, uncovered_count

    def random_init():
        """Dense random initialization followed by pruning"""
        # Start with a large random sample
        # 6561 / 17 * 1.5 ~ 580
        size = 700
        selected = random.sample(range(total), size)
        selected = prune_redundant(selected)
        
        # Check validity
        cover_count = np.zeros(total, dtype=np.int32)
        for c in selected:
            cover_count[neighbors[c]] += 1
        if np.min(cover_count) == 0:
            # If not valid, add missing covers greedily
            missing = np.where(cover_count == 0)[0]
            for w in missing:
                # Find a center covering w
                c = neighbors[w][np.random.randint(ball_size)]
                if c not in selected:
                    selected.append(c)
                    cover_count[neighbors[c]] += 1
        return selected

    # Main loop
    while time.time() < deadline - 5.0:
        # Alternate strategies
        r = random.random()
        if r < 0.5:
            selected, uncovered_count = greedy_construct(randomize=False)
        elif r < 0.8:
            selected, uncovered_count = greedy_construct(randomize=True)
        else:
            selected = random_init()
            
        if uncovered_count > 0 and r < 0.8:
            # Fix missing covers
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[neighbors[c]] += 1
            missing = np.where(cover_count == 0)[0]
            for w in missing:
                c = neighbors[w][np.random.randint(ball_size)]
                if c not in selected:
                    selected.append(c)
        
        # Optimization pipeline
        selected = prune_redundant(selected)
        selected = local_search_swap(selected)
        selected = add_remove_local_search(selected)
        selected = local_search_swap(selected) # Another pass of swaps
        selected = add_remove_3_for_2(selected)
        selected = prune_redundant(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            if best_size <= 486 and time.time() > deadline - 15.0:
                break
        if time.time() > deadline - 8.0:
            break
    
    if best_set is None:
        best_set = list(range(total))
    
    result = []
    for w in best_set:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
        
    return result
