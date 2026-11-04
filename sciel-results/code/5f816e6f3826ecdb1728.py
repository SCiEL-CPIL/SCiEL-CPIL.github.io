"""Rewritten program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit

    # For the specific case q=3, n=6, radius=1, use an improved greedy set cover
    # with better heuristics and local search
    if q == 3 and n == 6 and radius == 1:
        total = 3 ** 6
        powers = [3 ** p for p in range(6)]

        # Precompute for each word, the 13 words within Hamming distance 1 (including itself)
        neighbors = []
        for i in range(total):
            d = [(i // powers[p]) % 3 for p in range(6)]
            covered = [i]
            for p in range(6):
                for alt in range(3):
                    if alt != d[p]:
                        new_i = i + (alt - d[p]) * powers[p]
                        covered.append(new_i)
            neighbors.append(covered)

        # Convert neighbors to numpy arrays for faster operations
        neighbors_arr = np.array(neighbors, dtype=np.int32)

        def greedy_cover(uncovered_mask):
            """Greedy set cover: repeatedly pick the candidate covering most uncovered words."""
            uncovered = uncovered_mask.copy()
            codewords = []
            
            while uncovered.any():
                if time.time() > deadline:
                    break
                
                # Find candidate that covers most uncovered words
                # Only consider candidates that are themselves uncovered for speed
                uncovered_indices = np.where(uncovered)[0]
                
                if len(uncovered_indices) == 0:
                    break
                
                best_count = -1
                best_candidate = -1
                
                # Check all uncovered candidates
                for cand in uncovered_indices:
                    count = np.sum(uncovered[neighbors_arr[cand]])
                    if count > best_count:
                        best_count = count
                        best_candidate = cand
                
                if best_candidate == -1:
                    break
                
                codewords.append(best_candidate)
                
                # Mark covered
                uncovered[neighbors_arr[best_candidate]] = False
            
            return codewords

        # Run greedy with different orderings to find a good solution
        best_codewords = None
        best_len = float('inf')
        
        # Standard greedy
        uncovered_mask = np.ones(total, dtype=bool)
        result = greedy_cover(uncovered_mask)
        if len(result) < best_len:
            best_len = len(result)
            best_codewords = result[:]
        
        # Try greedy with random perturbations
        np.random.seed(42)
        for trial in range(20):
            if time.time() > deadline - 10:
                break
            
            # Randomize the order in which candidates are considered
            # by randomly permuting the initial uncovered set slightly
            uncovered_mask = np.ones(total, dtype=bool)
            
            # Add a small random bias: randomly cover a few words first
            # to break symmetry
            if trial > 0:
                num_perturb = min(trial * 2, 20)
                perturb_indices = np.random.choice(total, size=num_perturb, replace=False)
                # Don't actually cover them, just use as tie-breakers
                # Instead, let's try a different approach: randomize neighbor order
                
            # Run greedy but with randomized tie-breaking
            uncovered = uncovered_mask.copy()
            codewords = []
            
            while uncovered.any():
                if time.time() > deadline - 5:
                    break
                
                uncovered_indices = np.where(uncovered)[0]
                if len(uncovered_indices) == 0:
                    break
                
                best_count = -1
                best_candidate = -1
                best_candidates = []
                
                for cand in uncovered_indices:
                    count = np.sum(uncovered[neighbors_arr[cand]])
                    if count > best_count:
                        best_count = count
                        best_candidate = cand
                        best_candidates = [cand]
                    elif count == best_count:
                        best_candidates.append(cand)
                
                if best_candidate == -1:
                    break
                
                # Randomly pick from best candidates
                chosen = best_candidates[np.random.randint(len(best_candidates))]
                codewords.append(chosen)
                uncovered[neighbors_arr[chosen]] = False
            
            if len(codewords) < best_len:
                best_len = len(codewords)
                best_codewords = codewords[:]
        
        # Try local search: swap codewords to see if we can reduce the count
        if best_codewords is not None and len(best_codewords) > 70:
            if time.time() < deadline - 15:
                # Try removing each codeword and seeing if coverage is still maintained
                # or if we can replace it with a better one
                current = best_codewords[:]
                
                # Compute coverage map
                covered = np.zeros(total, dtype=np.int32)
                for cw in current:
                    covered[neighbors_arr[cw]] += 1
                
                improved = True
                while improved and time.time() < deadline - 10:
                    improved = False
                    
                    # Try removing each codeword
                    for i in range(len(current)):
                        if time.time() > deadline - 10:
                            break
                        
                        # Check if removing current[i] leaves any word uncovered
                        if np.all(covered[neighbors_arr[current[i]]] > 1):
                            # Safe to remove
                            current.pop(i)
                            covered[neighbors_arr[current[i]]] -= 1
                            improved = True
                            break
                    
                    # Try swapping: remove one, add another that covers more
                    for i in range(len(current)):
                        if time.time() > deadline - 10:
                            break
                        
                        # Temporarily remove
                        covered[neighbors_arr[current[i]]] -= 1
                        
                        # Find best candidate to add
                        best_gain = 0
                        best_add = -1
                        
                        # Only check uncovered words
                        uncovered_mask = covered == 0
                        if uncovered_mask.any():
                            uncovered_indices = np.where(uncovered_mask)[0]
                            for cand in uncovered_indices:
                                if cand in current:
                                    continue
                                gain = np.sum(uncovered_mask[neighbors_arr[cand]])
                                if gain > best_gain:
                                    best_gain = gain
                                    best_add = cand
                        
                        if best_add != -1 and best_gain > 0:
                            # Do the swap
                            covered[neighbors_arr[best_add]] += 1
                            current[i] = best_add
                            improved = True
                        else:
                            # Undo removal
                            covered[neighbors_arr[current[i]]] += 1
                
                if len(current) < best_len:
                    best_len = len(current)
                    best_codewords = current[:]

        # Convert indices to codeword lists
        if best_codewords is not None:
            result = []
            for idx in best_codewords:
                word = [(idx // powers[p]) % 3 for p in range(6)]
                result.append(word)
            
            # Verify coverage
            covered = np.zeros(total, dtype=bool)
            for cw in best_codewords:
                covered[neighbors_arr[cw]] = True
            
            if covered.all():
                return result
        
        # Fallback: direct product construction
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]

    # General case: fallback to direct sum
    if n >= 4 and q == 3:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        return [h + list(tail) for h in tetracode for tail in itertools.product(range(q), repeat=n - 4)]

    # Very general fallback
    return [[0] * n]
# EVOLVE-BLOCK-END
