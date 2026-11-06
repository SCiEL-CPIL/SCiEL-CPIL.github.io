"""Optimized greedy construction for B_3[1] sets using NumPy vectorization."""
import time
import numpy as np

def construct():
    deadline = time.time() + 85.0
    
    selected = []
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        if not selected:
            selected.append(x)
            existing_sums.add(3 * x)
            continue
        
        sel = np.array(selected, dtype=np.int32)
        n = len(sel)
        
        # Compute all new sums involving x:
        # Type 1: 3*x
        # Type 2: 2*x + y for each y in sel
        # Type 3: x + 2*y for each y in sel
        # Type 4: x + y + z for y <= z in sel
        
        new_sums = np.empty(1 + 2 * n + n * (n + 1) // 2, dtype=np.int32)
        idx = 0
        
        new_sums[idx] = 3 * x
        idx += 1
        
        new_sums[idx:idx + n] = 2 * x + sel
        idx += n
        
        new_sums[idx:idx + n] = x + 2 * sel
        idx += n
        
        # x + y + z for y <= z
        # Create upper triangular indices
        ii, jj = np.triu_indices(n)
        new_sums[idx:] = x + sel[ii] + sel[jj]
        
        # Check for internal duplicates: count unique values
        unique_sums = np.unique(new_sums)
        if len(unique_sums) < len(new_sums):
            continue
        
        # Check against existing sums
        # Convert to set for intersection check
        new_sums_set = set(unique_sums.tolist())
        if existing_sums & new_sums_set:
            continue
        
        # Valid - add x and update existing_sums
        selected.append(x)
        existing_sums |= new_sums_set
    
    return selected
