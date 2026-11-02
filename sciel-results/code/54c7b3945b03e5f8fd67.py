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
    # neighbors[w] contains w and all words at distance 1 from w
    # Due to symmetry, neighbors[w] is also the list of centers that cover w
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
            # Iterate in random order to break symmetry
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
        1. Identify critical words for each codeword (words covered ONLY by this codeword).
        2. Find candidates c' that cover ALL critical words of c.
        3. If c' is not in selected, swap c with c'.
        4. Repeat until no swaps possible.
        5. Prune redundants.
        """
        if not selected:
            return selected
        
        selected_set = set(selected)
        
        # We iterate in rounds. In each round, we try to improve the set.
        # To avoid infinite loops or getting stuck, we limit the number of passes.
        max_passes = 50
        for _ in range(max_passes):
            if time.time() > deadline - 5.0:
                break
                
            # Recompute cover counts
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[neighbors[c]] += 1
                
            improved = False
            # Shuffle to break symmetry
            indices = list(range(len(selected)))
            random.shuffle(indices)
            
            for i in indices:
                c = selected[i]
                # Find critical words: words in neighbors[c] with cover_count == 1
                crit_words = neighbors[c][cover_count[neighbors[c]] == 1]
                
                if len(crit_words) == 0:
                    # This codeword is redundant, remove it
                    # Note: prune_redundant will handle this, but we can do it here for speed
                    # However, removing changes cover_count, so we should just let prune handle it
                    # Or we can mark it for removal. For simplicity, let prune handle it.
                    continue
                    
                # Find candidates c' that cover all crit_words
                # Start with neighbors of the first critical word
                # neighbors[crit_words[0]] gives all centers covering crit_words[0]
                candidates = set(neighbors[crit_words[0]].tolist())
                
                for w in crit_words[1:]:
                    # Intersect with neighbors of next critical word
                    candidates.intersection_update(neighbors[w].tolist())
                    if not candidates:
                        break
                
                if not candidates:
                    continue
                
                # Check if any candidate is NOT in selected
                # We prefer candidates that are not already selected
                # If all candidates are already selected, we can't swap to a new one
                # (Swapping to an existing one would just duplicate, which is invalid)
                new_candidates = [c_prime for c_prime in candidates if c_prime not in selected_set]
                
                if new_candidates:
                    # Pick one randomly
                    c_prime = random.choice(new_candidates)
                    
                    # Perform swap
                    selected[i] = c_prime
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    improved = True
                    # Break to re-evaluate cover counts, as they have changed
                    break
            
            if not improved:
                break
                
            # After swaps, prune redundants
            selected = prune_redundant(selected)
            selected_set = set(selected)
            
        return selected

    def add_remove_local_search(selected):
        """
        Add-remove local search:
        Try to remove 2 codewords and add 1 to maintain coverage.
        This explores a larger neighborhood than single swap.
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
            # Try a limited number of random pairs to save time
            pairs_to_try = min(100, n_selected * (n_selected - 1) // 2)
            
            found_swap = False
            for _ in range(pairs_to_try):
                i, j = random.sample(range(n_selected), 2)
                c1, c2 = selected[i], selected[j]
                
                # Words that would become uncovered if we remove c1 and c2
                # These are words w where cover_count[w] == 1 and w is in neighbors[c1] or neighbors[c2]
                # Actually, if cover_count[w] == 1, it's covered by exactly one of c1, c2 (or another).
                # If we remove c1 and c2, a word w becomes uncovered if:
                # 1. w is in neighbors[c1] or neighbors[c2]
                # 2. cover_count[w] <= 2 (if it was 2, removing 1 makes it 1, removing 2 makes it -1? No, counts are per codeword)
                # Let's be precise:
                # new_cover_count[w] = cover_count[w] - (1 if w in N(c1) else 0) - (1 if w in N(c2) else 0)
                # We need new_cover_count[w] >= 1 for all w.
                # So we need cover_count[w] - delta >= 1.
                # delta is 0, 1, or 2.
                
                # Identify words that are "at risk"
                # Words in N(c1) or N(c2) with low cover count.
                risk_words = set()
                for w in neighbors[c1]:
                    if cover_count[w] <= 2:
                        risk_words.add(w)
                for w in neighbors[c2]:
                    if cover_count[w] <= 2:
                        risk_words.add(w)
                        
                if not risk_words:
                    # Both can be removed safely
                    selected.remove(c1)
                    selected.remove(c2)
                    selected_set.discard(c1)
                    selected_set.discard(c2)
                    found_swap = True
                    break
                    
                # Find a candidate c_prime that covers ALL risk_words
                # and is not c1 or c2
                # c_prime must be in the intersection of neighbors[w] for all w in risk_words
                
                # Optimization: If risk_words is large, intersection is likely empty.
                # Limit to small sets of risk words.
                if len(risk_words) > 5:
                    continue
                    
                risk_list = list(risk_words)
                candidates = set(neighbors[risk_list[0]].tolist())
                for w in risk_list[1:]:
                    candidates.intersection_update(neighbors[w].tolist())
                    if not candidates:
                        break
                        
                if not candidates:
                    continue
                    
                # Filter out c1 and c2 and existing selected
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

    def greedy_construct():
        """Randomized greedy set cover construction"""
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size
        
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            # Find the best candidate among active centers
            # We want the one covering the most uncovered words
            # gain[c] = sum(uncovered[w] for w in neighbors[c])
            # To speed this up, we can just compute it on the fly or maintain cand_count.
            # cand_count is maintained.
            
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            
            if max_count <= 0:
                break
            
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            active[best_idx] = False
            
            # Update cover counts and candidate counts
            for w in neighbors[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    # Decrement cand_count for all centers that cover w
                    # centers covering w are neighbors[w]
                    for c in neighbors[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
            
            if time.time() > deadline - 10.0:
                break
        
        return selected, uncovered_count

    # Main loop: try multiple strategies with random restarts
    while time.time() < deadline - 5.0:
        # Alternate between greedy and random initialization
        if random.random() < 0.7:
            # Greedy construction
            selected, uncovered_count = greedy_construct()
            if uncovered_count > 0:
                continue
        else:
            # Random initialization: pick 600 random words, prune
            # This provides a different starting landscape
            selected = random.sample(range(total), 600)
            selected = prune_redundant(selected)
            # Check validity
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[neighbors[c]] += 1
            if np.min(cover_count) == 0:
                continue
        
        # Local Search: Swap-based optimization
        selected = local_search_swap(selected)
        
        # Add-remove local search
        selected = add_remove_local_search(selected)
        
        # Final pruning
        selected = prune_redundant(selected)
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
            
            # Early stop if we've found a very good solution and time is running low
            if best_size <= 486 and time.time() > deadline - 15.0:
                break
        if time.time() > deadline - 8.0:
            break
    
    # Convert to list of digit lists
    if best_set is None:
        # Fallback: return a large valid set
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
