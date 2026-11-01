import numpy as np
import time
import itertools

def construct(q, n, radius):
    # Specific case for K_3(7,1)
    if q == 3 and n == 7 and radius == 1:
        return _construct_k3_7_1()
    
    # Generic fallback
    if q == 3:
        # Direct sum with tetracode
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            # n < 4, just return all words or simple set
            # For n=1,2,3 perfect codes exist or small sets
            pass
            
    # Absolute fallback: all words
    return [list(t) for t in itertools.product(range(q), repeat=n)]

def _construct_k3_7_1():
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    space_size = 2187
    # Precompute spheres
    spheres = np.zeros((space_size, 22), dtype=np.int32)
    powers = [1, 3, 9, 27, 81, 243, 729]
    
    for i in range(space_size):
        digits = [0]*7
        temp = i
        for j in range(7):
            digits[j] = temp % 3
            temp //= 3
            
        spheres[i, 0] = i
        idx = 1
        for pos in range(7):
            orig_val = digits[pos]
            p = powers[pos]
            for new_val in range(3):
                if new_val == orig_val:
                    continue
                spheres[i, idx] = i + (new_val - orig_val) * p
                idx += 1
                
    def idx_to_word(idx):
        w = [0]*7
        for i in range(7):
            w[i] = idx % 3
            idx //= 3
        return w
        
    best_code = None
    best_size = space_size
    
    # Run multiple restarts
    for restart in range(500):
        if time.time() > deadline:
            break
            
        # Greedy
        covered = np.zeros(space_size, dtype=bool)
        chosen = []
        num_covered = 0
        
        while num_covered < space_size:
            # Vectorized calculation of uncovered count per sphere
            # ~covered is boolean array
            # covered[spheres] is boolean matrix
            uncovered_in_spheres = np.sum(~covered[spheres], axis=1)
            max_uncov = np.max(uncovered_in_spheres)
            
            if max_uncov == 0:
                break
                
            candidates = np.where(uncovered_in_spheres == max_uncov)[0]
            best_idx = np.random.choice(candidates)
            
            chosen.append(best_idx)
            
            # Update coverage
            # Count new covers
            new_mask = ~covered[spheres[best_idx]]
            num_covered += np.sum(new_mask)
            covered[spheres[best_idx]] = True
            
        # Cleanup
        chosen_list = chosen[:]
        
        # Compute cover counts
        cover_counts = np.zeros(space_size, dtype=np.int32)
        for c in chosen_list:
            cover_counts[spheres[c]] += 1
            
        # Iterative removal
        changed = True
        while changed:
            changed = False
            for i in range(len(chosen_list)):
                c = chosen_list[i]
                # Check if any word in sphere is covered ONLY by c
                # i.e., cover_counts[spheres[c]] has any 1s
                if np.any(cover_counts[spheres[c]] == 1):
                    continue
                else:
                    # Remove
                    cover_counts[spheres[c]] -= 1
                    chosen_list.pop(i)
                    changed = True
                    break
                    
        current_size = len(chosen_list)
        
        if current_size < best_size:
            best_size = current_size
            best_code = [idx_to_word(c) for c in chosen_list]
            
            if best_size == 186:
                break

    if best_code is None:
        # Fallback
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        best_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=3)]

    return best_code
