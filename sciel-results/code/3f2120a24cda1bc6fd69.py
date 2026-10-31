"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    best_score = -1
    best_set = [0, 1, 3]  # seed fallback
    
    def compute_score(U):
        """Compute score using set operations for small sets, FFT for large."""
        U = sorted(set(U))
        if not U or 0 not in U:
            return -1
        max_u = max(U)
        if max_u == 0:
            return 0
        n = len(U)
        
        # Use set operations for small n (fast and exact)
        if n < 15:
            sumset = set()
            diffset = set()
            for a in U:
                for b in U:
                    sumset.add(a + b)
                    diffset.add(a - b)
            n_sum = len(sumset)
            n_diff = len(diffset)
        else:
            # Use FFT for larger n
            size_sum = 2 * max_u + 1
            fft_size = 1
            while fft_size < 2 * size_sum:
                fft_size *= 2
            
            ind = np.zeros(fft_size)
            for x in U:
                ind[x] = 1.0
            
            f = np.fft.rfft(ind)
            conv = np.fft.irfft(f * f, n=fft_size)
            n_sum = int(np.sum(conv[:size_sum] > 0.5))
            
            rev_ind = np.zeros(fft_size)
            for x in U:
                rev_ind[max_u - x] = 1.0
            
            f_rev = np.fft.rfft(rev_ind)
            conv_diff = np.fft.irfft(f * f_rev, n=fft_size)
            n_diff = int(np.sum(conv_diff[:2 * max_u + 1] > 0.5))
        
        if n_sum == 0:
            return -1
        
        base = 2 * max_u + 1
        if base <= 1:
            return 0
        ratio = n_diff / n_sum
        if ratio <= 1:
            return 1.0
        score = 1 + math.log(ratio) / math.log(base)
        return score

    def try_set(cand):
        nonlocal best_score, best_set
        cand = sorted(set(cand))
        if not cand or 0 not in cand:
            return
        if max(cand) == 0:
            return
        s = compute_score(cand)
        if s > best_score:
            best_score = s
            best_set = list(cand)

    # Initialize with a few good small sets
    initial_sets = [
        [0, 1, 3],
        [0, 1, 4, 6],
        [0, 1, 2, 5, 10],
        [0, 1, 3, 4, 9, 10],
        [0, 1, 2, 4, 8, 16]
    ]
    for s in initial_sets:
        try_set(s)
    
    # Greedy Construction Phase
    # Start from the best initial set and greedily add elements
    current = list(best_set)
    max_val = 200  # Limit search space for additions
    
    while time.time() < deadline - 20:
        improved = False
        # Generate candidates: neighbors of existing elements and new large elements
        candidates = set()
        for x in current:
            for dx in range(-5, 6):
                if x + dx > 0:
                    candidates.add(x + dx)
        
        # Add some random larger elements to explore new territory
        for _ in range(10):
            candidates.add(random.randint(max(current) + 1, max_val))
        
        # Also consider extending the max
        candidates.update(range(max(current) + 1, max(current) + 10))
        
        # Filter candidates: must be non-negative, not in current
        candidates = {x for x in candidates if x > 0 and x not in current}
        
        # Evaluate all candidates and pick the best
        best_cand = None
        best_cand_score = -1
        
        # Sort candidates by heuristic (e.g., larger values might increase base more, 
        # but we need to check the ratio). Just evaluate all if count is small.
        if len(candidates) > 100:
            # Sample if too many
            candidates = list(random.sample(list(candidates), 100))
        
        for c in candidates:
            cand_set = sorted(current + [c])
            s = compute_score(cand_set)
            if s > best_cand_score:
                best_cand_score = s
                best_cand = c
        
        if best_cand is not None and best_cand_score > best_score:
            current.append(best_cand)
            current.sort()
            best_score = best_cand_score
            best_set = list(current)
            improved = True
            max_val = max(max_val, current[-1] + 50)
            
            if time.time() > deadline - 20:
                break
        else:
            # No improvement, try local search (swap)
            if len(current) > 2:
                swap_improved = False
                # Try swapping one element with a new one
                for i in range(len(current)):
                    if current[i] == 0:
                        continue
                    original = current[i]
                    # Try replacing with neighbors
                    for new_val in range(max(1, original - 10), original + 11):
                        if new_val in current:
                            continue
                        cand_set = current.copy()
                        cand_set[i] = new_val
                        cand_set.sort()
                        s = compute_score(cand_set)
                        if s > best_score:
                            best_score = s
                            best_set = cand_set
                            current = cand_set
                            swap_improved = True
                            break
                    if swap_improved:
                        break
                
                if swap_improved:
                    continue
                else:
                    # Try removing an element
                    remove_improved = False
                    for i in range(len(current)):
                        if current[i] == 0:
                            continue
                        cand_set = current.copy()
                        cand_set.pop(i)
                        s = compute_score(cand_set)
                        if s > best_score:
                            best_score = s
                            best_set = cand_set
                            current = cand_set
                            remove_improved = True
                            break
                    
                    if remove_improved:
                        continue
                    
            # If no improvement in add/swap/remove, break
            break

    return list(best_set)
# EVOLVE-BLOCK-END
