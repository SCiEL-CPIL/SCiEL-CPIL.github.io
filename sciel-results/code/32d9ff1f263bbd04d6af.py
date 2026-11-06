import time
import numpy as np

def construct():
    deadline = time.time() + 85.0  # Safety margin for 90s limit
    n_max = 4096
    
    # Use a bitset for O(1) collision checks
    # Max triple sum is 3*4096 = 12288
    max_sum = 3 * n_max + 1
    bitset = np.zeros(max_sum + 1, dtype=np.uint8)
    
    # Start with a guaranteed-valid skeleton: powers of 4
    # For powers of 4, any triple sum has unique base-4 representation
    # (max digit is 3, base is 4, so no carries)
    # 4^0=1, 4^1=4, 4^2=16, 4^3=64, 4^4=256, 4^5=1024, 4^6=4096
    skeleton = [1, 4, 16, 64, 256, 1024]
    # 4^6 = 4096 is the boundary, include it
    skeleton.append(4096)
    
    selected = []
    for x in skeleton:
        if x > n_max:
            continue
        selected.append(x)
        # Mark all triple sums involving only skeleton elements
        sel_arr = np.array(selected, dtype=np.int32)
        n = len(sel_arr)
        # Generate all triples from the current set
        ii, jj, kk = np.meshgrid(
            np.arange(n), np.arange(n), np.arange(n), indexing='ij'
        )
        # Only keep i <= j <= k to get unordered triples with repetition
        mask = (ii <= jj) & (jj <= kk)
        ii, jj, kk = ii[mask], jj[mask], kk[mask]
        sums = sel_arr[ii] + sel_arr[jj] + sel_arr[kk]
        bitset[sums] = 1
    
    # Now greedily try to insert remaining integers
    # Try in increasing order, skipping those already in skeleton
    skeleton_set = set(skeleton)
    
    for x in range(1, n_max + 1):
        if time.time() > deadline:
            break
        if len(selected) >= 80:
            break
        if x in skeleton_set:
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
