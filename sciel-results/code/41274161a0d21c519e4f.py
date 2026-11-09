import numpy as np


def heuristics(demands, capacity):
    n = len(demands)
    # Stable descending sort by size
    order = np.argsort(-demands, kind='mergesort')
    
    # FFD packing
    bin_loads = []
    bin_members = []
    for idx in order:
        s = demands[idx]
        placed = False
        for b in range(len(bin_loads)):
            if s <= bin_loads[b]:
                bin_loads[b] -= s
                bin_members[b].append(idx)
                placed = True
                break
        if not placed:
            bin_loads.append(capacity - s)
            bin_members.append([idx])
    
    # Build adjacency-based heuristic: only consecutive items in each FFD bin get high weight
    H = np.full((n, n), 1e-9, dtype=np.float64)
    for members in bin_members:
        m = len(members)
        for k in range(m - 1):
            a = members[k]
            b = members[k + 1]
            H[a, b] = 1.0
            H[b, a] = 1.0
    
    return H
