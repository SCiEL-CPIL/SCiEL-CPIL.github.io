import time
import numpy as np
import random
import itertools

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0

    # Handle general case fallback
    def get_fallback():
        if n == 4 and q == 3 and radius == 1:
            # Tetracode
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n > 4 and q == 3 and radius == 1:
            # Direct sum with tetracode
            tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        # General greedy for other params
        return _general_greedy(q, n, radius, deadline)

    if q != 3 or n != 8 or radius != 1:
        return get_fallback()

    total_words = 3 ** 8
    num_neighbors = 1 + n * (q - 1) # 17

    # Precompute neighbors
    # int_to_digits
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    best_code = None
    best_size = total_words

    # Run multiple randomized greedy attempts
    seed = 42
    while time.time() < deadline - 3.0:
        random.seed(seed)
        seed += 1
        
        uncovered = np.ones(total_words, dtype=np.uint8)
        selected_indices = []
        
        # Greedy Phase with randomization
        while uncovered.sum() > 0:
            if time.time() > deadline - 8.0:
                break
            # Calculate gains
            gains = uncovered[neighbors].sum(axis=1)
            
            # Find max gain
            max_gain = np.max(gains)
            if max_gain == 0:
                break
            
            # Random pick from top candidates (diversification)
            candidates = np.where(gains == max_gain)[0]
            if len(candidates) > 1:
                # Pick randomly from top-k to diversify
                k = min(10, len(candidates))
                best_idx = candidates[random.randint(0, len(candidates) - 1)]
            else:
                best_idx = candidates[0]
                
            selected_indices.append(best_idx)
            uncovered[neighbors[best_idx]] = 0
            
            # Early exit if we already exceeded best_size
            if len(selected_indices) >= best_size:
                break

        if len(selected_indices) >= best_size:
            continue

        # Redundancy Removal Phase
        covered_counts = np.zeros(total_words, dtype=np.uint8)
        for idx in selected_indices:
            covered_counts[neighbors[idx]] += 1
            
        # Iteratively remove redundant codewords
        changed = True
        while changed and time.time() < deadline - 3.0:
            changed = False
            indices_to_check = list(range(len(selected_indices)))
            random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if time.time() > deadline - 3.0:
                    break
                
                idx = selected_indices[i]
                if np.all(covered_counts[neighbors[idx]] > 1):
                    covered_counts[neighbors[idx]] -= 1
                    selected_indices.pop(i)
                    changed = True
                    break

        # Local Search: Single swap + 2-exchange
        improved = True
        while improved and time.time() < deadline - 3.0:
            improved = False
            indices_to_check = list(range(len(selected_indices)))
            random.shuffle(indices_to_check)
            
            for i in indices_to_check:
                if time.time() > deadline - 3.0:
                    break
                idx = selected_indices[i]
                
                # Find critical words
                crit_mask = (covered_counts[neighbors[idx]] == 1)
                crit_words = neighbors[idx][crit_mask]
                
                if len(crit_words) == 0:
                    covered_counts[neighbors[idx]] -= 1
                    selected_indices.pop(i)
                    improved = True
                    break

                # Find candidates that cover all critical words using numpy
                # Build a boolean mask for each critical word
                # candidate covers all crit_words if for all v in crit_words, v is in neighbors[candidate]
                
                # Efficient approach: for each candidate position in neighbors, check if it covers all crit_words
                # We can use the fact that neighbors is precomputed
                # For a candidate c, it covers word v iff v is in neighbors[c]
                # Equivalently, c is in neighbors[v]
                
                # Use the first critical word to get initial candidates
                initial_candidates = neighbors[crit_words[0]]
                
                # Filter: candidate must be in neighbors[v] for all v in crit_words
                # And candidate != idx
                valid_mask = np.ones(total_words, dtype=bool)
                for v in crit_words:
                    # v is covered by candidate c iff c is in neighbors[v]
                    # neighbors[v] gives all c such that dist(c,v) <= 1
                    # We need c in neighbors[v] for all v
                    # So: c must be in intersection of neighbors[v] for all v in crit_words
                    # But we only have neighbors as (word -> its neighbors), not (word -> words that cover it)
                    # However, Hamming distance is symmetric, so neighbors[v] = words within dist 1 of v
                    # So candidate c covers v iff c is in neighbors[v]
                    
                    # Build mask for this v
                    v_cover_mask = np.zeros(total_words, dtype=bool)
                    v_cover_mask[neighbors[v]] = True
                    valid_mask &= v_cover_mask
                
                valid_mask[idx] = False  # Exclude current codeword
                
                valid_candidates = np.where(valid_mask)[0]
                
                if len(valid_candidates) == 0:
                    continue
                
                # Pick random candidate
                c_prime = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                
                # Perform the swap
                covered_counts[neighbors[idx]] -= 1
                covered_counts[neighbors[c_prime]] += 1
                selected_indices[i] = c_prime
                improved = True
                break

            # After swaps, do another removal pass
            if improved:
                changed = True
                while changed and time.time() < deadline - 3.0:
                    changed = False
                    indices_to_check = list(range(len(selected_indices)))
                    random.shuffle(indices_to_check)
                    
                    for i in indices_to_check:
                        if time.time() > deadline - 3.0:
                            break
                        
                        idx = selected_indices[i]
                        if np.all(covered_counts[neighbors[idx]] > 1):
                            covered_counts[neighbors[idx]] -= 1
                            selected_indices.pop(i)
                            changed = True
                            break

                # Try 2-exchange: remove 2, add 2
                if time.time() < deadline - 3.0 and len(selected_indices) >= 2:
                    # Find pairs of codewords that are both "weakly covered"
                    # A codeword is weakly covered if it has few critical words
                    # Try removing two codewords and see if we can replace them with two others
                    for _ in range(5):  # Try a few 2-exchanges
                        if time.time() > deadline - 3.0:
                            break
                        i1 = random.randint(0, len(selected_indices) - 1)
                        i2 = random.randint(0, len(selected_indices) - 1)
                        if i1 == i2:
                            continue
                        
                        idx1 = selected_indices[i1]
                        idx2 = selected_indices[i2]
                        
                        # Critical words for both
                        crit1 = neighbors[idx1][covered_counts[neighbors[idx1]] == 1]
                        crit2 = neighbors[idx2][covered_counts[neighbors[idx2]] == 1]
                        
                        if len(crit1) == 0 and len(crit2) == 0:
                            # Both redundant
                            covered_counts[neighbors[idx1]] -= 1
                            covered_counts[neighbors[idx2]] -= 1
                            # Remove higher index first to preserve indices
                            if i1 > i2:
                                i1, i2 = i2, i1
                            selected_indices.pop(i2)
                            selected_indices.pop(i1)
                            improved = True
                            break
                        elif len(crit1) == 0:
                            covered_counts[neighbors[idx1]] -= 1
                            selected_indices.pop(i1)
                            improved = True
                            break
                        elif len(crit2) == 0:
                            covered_counts[neighbors[idx2]] -= 1
                            selected_indices.pop(i2)
                            improved = True
                            break
                        else:
                            # Try to find two replacements that cover all critical words
                            all_crit = np.concatenate([crit1, crit2])
                            # Find candidates that cover all critical words
                            valid_mask = np.ones(total_words, dtype=bool)
                            for v in all_crit:
                                v_cover_mask = np.zeros(total_words, dtype=bool)
                                v_cover_mask[neighbors[v]] = True
                                valid_mask &= v_cover_mask
                            valid_mask[idx1] = False
                            valid_mask[idx2] = False
                            
                            valid_candidates = np.where(valid_mask)[0]
                            if len(valid_candidates) >= 2:
                                # Pick two different candidates
                                c1 = valid_candidates[random.randint(0, len(valid_candidates) - 1)]
                                remaining = np.delete(valid_candidates, np.where(valid_candidates == c1)[0][0])
                                c2 = remaining[random.randint(0, len(remaining) - 1)]
                                if c1 != c2:
                                    covered_counts[neighbors[idx1]] -= 1
                                    covered_counts[neighbors[idx2]] -= 1
                                    covered_counts[neighbors[c1]] += 1
                                    covered_counts[neighbors[c2]] += 1
                                    selected_indices[i1] = c1
                                    selected_indices[i2] = c2
                                    improved = True
                                    break

        if len(selected_indices) < best_size:
            best_size = len(selected_indices)
            best_code = [int_to_digits(idx, n, q) for idx in selected_indices]
            
    if best_code is None:
        return get_fallback()
        
    return best_code

def _general_greedy(q, n, radius, deadline):
    # Simple greedy for other parameters
    total_words = q ** n
    if total_words > 10000:
        return [list(t) for t in __import__('itertools').product(range(q), repeat=n)]
        
    def int_to_digits(num, n, q):
        res = [0] * n
        for i in range(n):
            res[i] = num % q
            num //= q
        return res

    num_neighbors = 1 + n * (q - 1)
    neighbors = np.zeros((total_words, num_neighbors), dtype=np.int32)
    for w in range(total_words):
        digits = int_to_digits(w, n, q)
        idx = 0
        neighbors[w, idx] = w
        idx += 1
        for pos in range(n):
            orig_val = digits[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                neighbors[w, idx] = w + (new_val - orig_val) * (q ** pos)
                idx += 1

    uncovered = np.ones(total_words, dtype=np.uint8)
    selected_indices = []
    
    while uncovered.sum() > 0:
        if time.time() > deadline:
            break
        gains = uncovered[neighbors].sum(axis=1)
        best_idx = np.argmax(gains)
        if gains[best_idx] == 0:
            break
        selected_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = 0
        
    return [int_to_digits(idx, n, q) for idx in selected_indices]
