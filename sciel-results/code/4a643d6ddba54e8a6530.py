import time
import numpy as np
import random
from collections import defaultdict
from itertools import combinations

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1(deadline)
    
    # Fallback for other cases
    import itertools
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    return [list(c) for c in itertools.product(range(q), repeat=n)]

def _construct_ternary_8_1(deadline):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Precompute balls efficiently
    weights = [3 ** (n - 1 - i) for i in range(n)]
    
    # Build balls array: balls[w] = [w, neighbors...]
    balls = np.zeros((total, ball_size), dtype=np.int32)
    
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                balls[w, idx] = w + (new_digit - digit) * weight
                idx += 1
    
    # Build inverse mapping: for each word w, which ball centers cover it
    # inv[w] = list of center indices c such that w is in balls[c]
    inv = [defaultdict(int) for _ in range(total)]
    for c in range(total):
        for w in balls[c]:
            inv[w][c] += 1
            
    # Convert to lists for faster iteration
    inv_lists = [list(inv[w].keys()) for w in range(total)]
    
    best_size = total
    best_set = None
    
    # Use multiple random restarts with greedy + pruning + local search
    np.random.seed(42)
    random.seed(42)
    
    # Precompute conflict graph: two centers conflict if their balls overlap
    # We'll use this for MIS-based optimization
    # But first, let's focus on improving the local search with better swap heuristics
    
    while time.time() < deadline - 5.0:
        # Randomized greedy construction with incremental updates
        selected = []
        cover_count = np.zeros(total, dtype=np.int32)
        uncovered = np.ones(total, dtype=bool)
        uncovered_count = total
        
        # For each candidate center, track how many uncovered words it covers
        # cand_count[c] = number of uncovered words in balls[c]
        cand_count = np.zeros(total, dtype=np.int32)
        for c in range(total):
            cand_count[c] = ball_size  # Initially all words are uncovered
        
        # Track which centers are still "active" (not yet selected)
        active = np.ones(total, dtype=bool)
        
        while uncovered_count > 0:
            # Find the best candidate among active centers
            # Use a simple approach: find max cand_count among active
            active_counts = np.where(active, cand_count, -1)
            max_count = np.max(active_counts)
            
            if max_count <= 0:
                break
            
            # Pick randomly among max-count active candidates for diversity
            candidates = np.where((active_counts == max_count) & active)[0]
            if len(candidates) == 0:
                break
            best_idx = candidates[np.random.randint(len(candidates))]
            
            selected.append(best_idx)
            active[best_idx] = False
            
            # Update cover counts and candidate counts
            for w in balls[best_idx]:
                if cover_count[w] == 0:
                    uncovered_count -= 1
                    # Decrement cand_count for all centers that cover w
                    for c in inv_lists[w]:
                        if active[c]:
                            cand_count[c] -= 1
                cover_count[w] += 1
            
            if time.time() > deadline - 10.0:
                break
        
        # Check if valid
        if uncovered_count > 0:
            continue
        
        # Pruning: iteratively remove codewords that are not needed
        # A codeword c is removable if for all w in balls[c], cover_count[w] > 1
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
            
        # Local Search: Enhanced swap-based optimization with MIS-inspired heuristics
        improved = True
        while improved and time.time() < deadline - 2.0:
            improved = False
            # Recompute cover counts from scratch to be safe after swaps/removals
            cover_count = np.zeros(total, dtype=np.int32)
            for c in selected:
                cover_count[balls[c]] += 1
                
            selected_set = set(selected)
            
            # Identify critical codewords and their unique words
            critical_candidates = []
            for c in selected:
                unique_words = [w for w in balls[c] if cover_count[w] == 1]
                if unique_words:
                    critical_candidates.append((c, unique_words))
            
            # Sort critical codewords by number of unique words (hardest to replace first)
            critical_candidates.sort(key=lambda x: len(x[1]), reverse=True)
            
            # Try to find a swap for any critical codeword
            for c, unique_words in critical_candidates:
                if not unique_words:
                    continue
                    
                # Find intersection of inv_lists for all unique words
                # These are the candidates c' that cover ALL unique words
                common_candidates = set(inv_lists[unique_words[0]])
                for w in unique_words[1:]:
                    common_candidates.intersection_update(inv_lists[w])
                    
                # Filter out already selected candidates
                available_candidates = [c_prime for c_prime in common_candidates if c_prime not in selected_set]
                
                if not available_candidates:
                    continue
                
                # Score candidates: prefer those that cover more words overall (higher degree in conflict graph)
                # This is an MIS-inspired heuristic: choose vertices with higher potential to reduce future conflicts
                def candidate_score(c_prime):
                    # Count how many words in balls[c_prime] are covered by exactly 2 codewords
                    # (swapping to this would make them covered by exactly 1, potentially creating new criticals)
                    # vs words covered by 3+ (swapping keeps them safe)
                    score = 0
                    for w in balls[c_prime]:
                        if cover_count[w] == 2:
                            score -= 1  # Bad: creates new critical
                        elif cover_count[w] >= 3:
                            score += 1  # Good: maintains safety margin
                    # Also prefer candidates that don't conflict with other selected codewords
                    # (i.e., their ball doesn't overlap much with other selected balls)
                    for c_other in selected:
                        if c_other == c_prime:
                            continue
                        # Count overlap between balls[c_prime] and balls[c_other]
                        overlap = np.sum(cover_count[balls[c_prime]] >= 1)  # approximate
                        score -= overlap * 0.1
                    return score
                
                # Sort candidates by score (highest first)
                available_candidates.sort(key=candidate_score, reverse=True)
                
                # Try the best candidate
                c_prime = available_candidates[0]
                
                # Verify the swap is valid: after removing c and adding c_prime, all words still covered
                # Check that c_prime covers all unique words of c
                valid_swap = True
                for w in unique_words:
                    if w not in balls[c_prime]:
                        valid_swap = False
                        break
                
                if valid_swap:
                    # Perform swap
                    selected.remove(c)
                    selected.append(c_prime)
                    selected_set.remove(c)
                    selected_set.add(c_prime)
                    improved = True
                    break # Restart the loop to re-evaluate criticals
            
            # After swaps, try to remove redundant codewords again
            if time.time() < deadline - 2.0:
                cover_count = np.zeros(total, dtype=np.int32)
                for c in selected:
                    cover_count[balls[c]] += 1
                
                new_selected = []
                for c in selected:
                    if np.min(cover_count[balls[c]]) > 1:
                        cover_count[balls[c]] -= 1
                    else:
                        new_selected.append(c)
                
                if len(new_selected) < len(selected):
                    selected = new_selected
                    improved = True
            
            # Additional optimization: try to remove pairs of codewords if possible
            # (this is more expensive but can find better solutions)
            if time.time() < deadline - 2.0 and len(selected) > 100:
                cover_count = np.zeros(total, dtype=np.int32)
                for c in selected:
                    cover_count[balls[c]] += 1
                
                # Try removing pairs of non-critical codewords
                non_critical = [c for c in selected if np.min(cover_count[balls[c]]) > 1]
                
                if len(non_critical) >= 2:
                    # Try a few random pairs
                    for _ in range(min(10, len(non_critical) * (len(non_critical) - 1) // 2)):
                        if time.time() > deadline - 2.0:
                            break
                        i, j = random.sample(range(len(non_critical)), 2)
                        c1, c2 = non_critical[i], non_critical[j]
                        
                        # Check if removing both is safe
                        # A word is safe if it's covered by at least 2 other codewords
                        # or by one of c1/c2 and at least one other
                        safe = True
                        for w in np.concatenate([balls[c1], balls[c2]]):
                            # Count covers excluding c1 and c2
                            count_without = 0
                            for c_other in selected:
                                if c_other == c1 or c_other == c2:
                                    continue
                                if w in balls[c_other]:
                                    count_without += 1
                                    if count_without >= 1:
                                        break
                            if count_without == 0:
                                safe = False
                                break
                        
                        if safe:
                            selected.remove(c1)
                            selected.remove(c2)
                            improved = True
                            break
        
        if len(selected) < best_size:
            best_size = len(selected)
            best_set = selected.copy()
        
        # If we've found a solution close to the known best (486), do a more thorough search
        if best_size <= 500 and time.time() < deadline - 10.0:
            # Try more restarts with different random seeds
            for extra_restart in range(3):
                if time.time() > deadline - 10.0:
                    break
                
                np.random.seed(42 + extra_restart * 1000)
                random.seed(42 + extra_restart * 1000)
                
                selected = []
                cover_count = np.zeros(total, dtype=np.int32)
                uncovered = np.ones(total, dtype=bool)
                uncovered_count = total
                
                cand_count = np.zeros(total, dtype=np.int32)
                for c in range(total):
                    cand_count[c] = ball_size
                
                active = np.ones(total, dtype=bool)
                
                while uncovered_count > 0 and time.time() < deadline - 10.0:
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
                    
                    for w in balls[best_idx]:
                        if cover_count[w] == 0:
                            uncovered_count -= 1
                            for c in inv_lists[w]:
                                if active[c]:
                                    cand_count[c] -= 1
                        cover_count[w] += 1
                
                if uncovered_count > 0:
                    continue
                
                # Pruning
                changed = True
                while changed and time.time() < deadline - 10.0:
                    changed = False
                    new_selected = []
                    for c in selected:
                        if np.min(cover_count[balls[c]]) > 1:
                            cover_count[balls[c]] -= 1
                            changed = True
                        else:
                            new_selected.append(c)
                    selected = new_selected
                
                if len(selected) < best_size:
                    best_size = len(selected)
                    best_set = selected.copy()
        
        # Early stop if we've found a very good solution and time is running low
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
