import time
import numpy as np

def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n_max = 4096
    
    # Use a bitset for O(1) collision checks
    # Max triple sum is 3*4096 = 12288
    max_sum = 3 * n_max + 1
    bitset = np.zeros(max_sum + 1, dtype=np.uint8)
    
    selected = []
    
    for x in range(1, n_max + 1):
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
            
        if not selected:
            selected.append(x)
            bitset[3 * x] = 1
            continue
            
        sel = np.array(selected, dtype=np.int32)
        n = len(sel)
        
        # Generate all new triple sums involving x
        # Types:
        # 1. 3x
        # 2. 2x + s for s in sel
        # 3. x + 2s for s in sel
        # 4. x + s_i + s_j for i < j (strictly less to avoid overlap with type 3)
        
        total_size = 1 + n + n + n * (n - 1) // 2
        if total_size == 0:
            continue
            
        new_sums = np.empty(total_size, dtype=np.int32)
        
        idx = 0
        new_sums[idx] = 3 * x
        idx += 1
        
        new_sums[idx:idx + n] = 2 * x + sel
        idx += n
        
        new_sums[idx:idx + n] = x + 2 * sel
        idx += n
        
        # Pairs i < j
        if n >= 2:
            ii, jj = np.triu_indices(n, k=1)  # k=1 means i < j
            new_sums[idx:] = x + sel[ii] + sel[jj]
        
        # Check for internal duplicates (two new triples with same sum)
        # Use sorting + comparison for speed
        sorted_sums = np.sort(new_sums)
        if np.any(sorted_sums[1:] == sorted_sums[:-1]):
            continue
            
        # Check against existing sums using bitset
        if np.any(bitset[new_sums]):
            continue
            
        # Valid - add x
        selected.append(x)
        bitset[new_sums] = 1
        
    return selected
