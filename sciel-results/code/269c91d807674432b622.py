import math
import random

def calc_score(U):
    if not U:
        return 0
    U = list(set(U))
    if 0 not in U:
        return 0
    max_u = max(U)
    if max_u == 0:
        return 0
    sums = set()
    diffs = set()
    for a in U:
        for b in U:
            sums.add(a + b)
            diffs.add(a - b)
    num = len(diffs) / len(sums)
    den = math.log(2 * max_u + 1)
    if den == 0:
        return 0
    return 1 + math.log(num) / den

def construct():
    best = [0, 1, 3]
    best_score = calc_score(best)
    
    # Search over base-B binary expansion sets with local search on subsets
    
    for B in [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]:
        for n in range(2, 11):
            # Generate all base-B binary numbers
            all_vals = []
            for mask in range(1 << n):
                val = 0
                for j in range(n):
                    if mask & (1 << j):
                        val += B ** j
                all_vals.append(val)
            
            # Check full set first
            U_full = all_vals
            s_full = calc_score(U_full)
            if s_full > best_score:
                best_score = s_full
                best = U_full

            # Local search (Hill Climbing)
            # We represent U as a bitmask of indices in all_vals
            # Start with a few different initial masks
            
            # Limit iterations based on set size to stay within time limits
            # Score calc is O(N^2). N = 2^n.
            N = len(all_vals)
            if N <= 64:
                max_iters = 200
            elif N <= 256:
                max_iters = 50
            else:
                max_iters = 10

            initial_masks = [
                (1 << N) - 1,  # Full set
                1,  # Just 0
            ]
            
            # Add some random masks
            for _ in range(5):
                size = random.randint(2, N)
                indices = random.sample(range(N), size)
                mask = 0
                for i in indices:
                    mask |= (1 << i)
                initial_masks.append(mask)
            
            for mask in initial_masks:
                current_mask = mask
                # Ensure 0 is in set
                if not (current_mask & 1):
                    current_mask |= 1
                    
                current_U = [all_vals[i] for i in range(N) if (current_mask >> i) & 1]
                current_score = calc_score(current_U)
                
                no_improve_count = 0
                for _ in range(max_iters):
                    if no_improve_count > 10:
                        break
                    
                    # Try adding or removing a random element
                    idx = random.randint(0, N - 1)
                    
                    if (current_mask >> idx) & 1:
                        # Remove
                        if idx == 0:
                            continue # Keep 0
                        new_mask = current_mask & ~(1 << idx)
                    else:
                        # Add
                        new_mask = current_mask | (1 << idx)
                    
                    new_U = [all_vals[i] for i in range(N) if (new_mask >> i) & 1]
                    new_score = calc_score(new_U)
                    
                    if new_score > current_score:
                        current_mask = new_mask
                        current_score = new_score
                        no_improve_count = 0
                        
                        if current_score > best_score:
                            best_score = current_score
                            best = new_U
                    else:
                        no_improve_count += 1

    return best
