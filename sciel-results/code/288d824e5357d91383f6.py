"""Optimized B_3[1] construction using vectorized greedy with incremental sum checks."""
import time
import numpy as np

def construct():
    deadline = time.time() + 80.0
    
    selected = []
    existing_sums = set()
    
    for x in range(1, 4097):
        if time.time() > deadline:
            break
        
        if not selected:
            selected.append(x)
            existing_sums.add(3 * x)
            continue
        
        n = len(selected)
        sel = np.array(selected, dtype=np.int32)
        
        # Generate all new sums involving x:
        # Type 1: 3x
        # Type 2: 2x + y for each y in sel
        # Type 3: x + 2y for each y in sel
        # Type 4: x + y + z for each pair y <= z in sel
        
        # Count expected distinct triples
        num_triples = 1 + 2 * n + n * (n + 1) // 2
        
        # Build all new sums as a numpy array for fast collision checking
        # Part 1: 3x
        parts = [np.array([3 * x], dtype=np.int32)]
        
        # Part 2: 2x + y
        parts.append(2 * x + sel)
        
        # Part 3: x + 2y
        parts.append(x + 2 * sel)
        
        # Part 4: x + y + z for y <= z
        # Use upper triangle of outer sum
        ii, jj = np.triu_indices(n)
        parts.append(x + sel[ii] + sel[jj])
        
        new_sums_arr = np.concatenate(parts)
        
        # Check internal uniqueness: if any duplicate in new_sums_arr, reject
        if len(np.unique(new_sums_arr)) != num_triples:
            continue
        
        # Check against existing sums
        new_sums_set = set(new_sums_arr.tolist())
        if existing_sums & new_sums_set:
            continue
        
        # Valid - add x
        selected.append(x)
        existing_sums |= new_sums_set
    
    return selected
