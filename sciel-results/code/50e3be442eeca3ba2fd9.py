import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    N = 1000
    max_sum = 2 * N
    
    # Helper to get shifted mask for convolution update
    # Returns an array of length max_sum+1
    # shifted[i] = mask[i-x] if 0 <= i-x <= N else 0
    def get_shifted(mask, x):
        shifted = np.zeros(max_sum + 1, dtype=int)
        # mask is length N+1
        # We want shifted[x : x + (N+1)] = mask[0 : N+1]
        # But we must respect bounds.
        # i ranges from x to x+N.
        # i-x ranges from 0 to N.
        # So shifted[x : x+N+1] = mask
        # However, shifted has length 2N+1.
        # x+N+1 might exceed 2N+1 if x > N. But x <= N.
        # So x+N <= 2N.
        # So slice is valid.
        shifted[x : x + N + 1] = mask
        return shifted

    # Greedy Initialization
    mask = np.zeros(N + 1, dtype=bool)
    C = np.zeros(max_sum + 1, dtype=int)
    D = np.zeros(max_sum + 1, dtype=int)
    size = 0
    
    # We need to check validity efficiently.
    # R[s] = (C[s] + D[s]) / 2 <= 2  => C[s] + D[s] <= 4.
    
    for x in range(1, N + 1):
        # Check if adding x is valid
        # New C = C + shifted_mask (where shifted_mask uses current mask)
        # New D = D with D[2x] = 1
        
        # Affected sums:
        # 1. s = 2x.
        #    New R[2x] = (C[2x] + 1 + D[2x] + 1) / 2 = (C[2x] + D[2x]) / 2 + 1.
        #    We need (C[2x] + D[2x]) / 2 + 1 <= 2  => C[2x] + D[2x] <= 2.
        
        if C[2*x] + D[2*x] > 2:
            continue
            
        # 2. s = x + y for y in S.
        #    New R[s] = (C[s] + 1 + D[s]) / 2.
        #    We need (C[s] + 1 + D[s]) <= 4 => C[s] + D[s] <= 3.
        
        ys = np.where(mask)[0]
        if len(ys) > 0:
            sums = x + ys
            # Exclude s=2x?
            # If y=x, s=2x. But x is not in S yet, so y != x.
            # So s != 2x.
            vals = C[sums] + D[sums]
            if np.any(vals > 3):
                continue
        
        # Valid, add x
        # Update C
        shifted = get_shifted(mask, x)
        C += shifted
        # Update mask
        mask[x] = True
        # Update D
        D[2*x] = 1
        size += 1
        
    best_mask = mask.copy()
    best_size = size
    
    # Local Search with Simulated Annealing
    T = 1.0
    T_min = 0.01
    cooling = 0.999
    
    current_mask = mask.copy()
    current_C = C.copy()
    current_D = D.copy()
    current_size = size
    current_violations = np.sum((current_C + current_D) > 4)
    
    while time.time() < deadline:
        # Choose move
        if random.random() < 0.6:
            # Try Insert
            candidates = np.where(~current_mask)[0]
            candidates = candidates[candidates > 0] # Exclude 0
            if len(candidates) == 0:
                continue
            x = random.choice(candidates)
            
            # Check validity of adding x
            # New C = current_C + shifted
            # New D = current_D with D[2x]=1
            
            # Check s=2x
            if current_C[2*x] + current_D[2*x] > 2:
                # Invalid
                # Calculate penalty
                # We might still accept if it helps?
                # For now, only accept valid moves in insert branch to keep it simple and safe
                # Or use SA logic
                pass
            else:
                # Check s=x+y
                ys = np.where(current_mask)[0]
                if len(ys) > 0:
                    sums = x + ys
                    vals = current_C[sums] + current_D[sums]
                    if np.any(vals > 3):
                        pass
                    else:
                        # Valid
                        shifted = get_shifted(current_mask, x)
                        new_C = current_C + shifted
                        new_D = current_D.copy()
                        new_D[2*x] = 1
                        new_size = current_size + 1
                        new_violations = np.sum((new_C + new_D) > 4)
                        
                        # Accept
                        current_C = new_C
                        current_D = new_D
                        current_mask[x] = True
                        current_size = new_size
                        current_violations = new_violations
                        
                        if current_violations == 0 and current_size > best_size:
                            best_size = current_size
                            best_mask = current_mask.copy()
                else:
                    # No ys, only s=2x to check
                    shifted = get_shifted(current_mask, x)
                    new_C = current_C + shifted
                    new_D = current_D.copy()
                    new_D[2*x] = 1
                    new_size = current_size + 1
                    new_violations = np.sum((new_C + new_D) > 4)
                    
                    current_C = new_C
                    current_D = new_D
                    current_mask[x] = True
                    current_size = new_size
                    current_violations = new_violations
                    
                    if current_violations == 0 and current_size > best_size:
                        best_size = current_size
                        best_mask = current_mask.copy()
                        
        else:
            # Try Swap
            # Pick x in S, y not in S
            in_set = np.where(current_mask)[0]
            in_set = in_set[in_set > 0]
            if len(in_set) == 0:
                continue
            x = random.choice(in_set)
            
            out_set = np.where(~current_mask)[0]
            out_set = out_set[out_set > 0]
            if len(out_set) == 0:
                continue
            y = random.choice(out_set)
            
            # Remove x
            shifted_x = get_shifted(current_mask, x)
            temp_C = current_C - shifted_x
            temp_D = current_D.copy()
            temp_D[2*x] = 0
            temp_mask = current_mask.copy()
            temp_mask[x] = False
            temp_size = current_size - 1
            
            # Add y to temp state
            # Check validity
            if temp_C[2*y] + temp_D[2*y] > 2:
                continue
            ys = np.where(temp_mask)[0]
            if len(ys) > 0:
                sums = y + ys
                vals = temp_C[sums] + temp_D[sums]
                if np.any(vals > 3):
                    continue
            else:
                pass # Only 2y check
            
            # Valid swap
            shifted_y = get_shifted(temp_mask, y)
            new_C = temp_C + shifted_y
            new_D = temp_D.copy()
            new_D[2*y] = 1
            new_size = temp_size + 1
            new_violations = np.sum((new_C + new_D) > 4)
            
            # Accept
            current_C = new_C
            current_D = new_D
            current_mask = temp_mask
            current_mask[y] = True
            current_size = new_size
            current_violations = new_violations
            
            if current_violations == 0 and current_size > best_size:
                best_size = current_size
                best_mask = current_mask.copy()
                
        T *= cooling
        if T < T_min:
            T = T_min

    # Return best set
    best_elements = np.where(best_mask)[0]
    best_elements = best_elements[best_elements > 0]
    return list(best_elements)

# Run construct
# if __name__ == "__main__":
#     res = construct()
#     print(len(res))
