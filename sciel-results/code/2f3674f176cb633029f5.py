"""
Optimal solution for packing edge-disjoint Hamiltonian cycles in the 21-vertex circulant graph C_21(1,2,5).
The graph is 6-regular with 63 edges. The maximum number of edge-disjoint Hamiltonian cycles is 3.
We construct them directly using steps 1, 2, and 5 modulo 21.
"""
import time

# EVOLVE-BLOCK-START
P = {'n': 21, 'edges': [[0, 1], [0, 2], [0, 5], [0, 16], [0, 19], [0, 20], [1, 2], [1, 3], [1, 6], [1, 17], [1, 20], [2, 3], [2, 4], [2, 7], [2, 18], [3, 4], [3, 5], [3, 8], [3, 19], [4, 5], [4, 6], [4, 9], [4, 20], [5, 6], [5, 7], [5, 10], [6, 7], [6, 8], [6, 11], [7, 8], [7, 9], [7, 12], [8, 9], [8, 10], [8, 13], [9, 10], [9, 11], [9, 14], [10, 11], [10, 12], [10, 15], [11, 12], [11, 13], [11, 16], [12, 13], [12, 14], [12, 17], [13, 14], [13, 15], [13, 18], [14, 15], [14, 16], [14, 19], [15, 16], [15, 17], [15, 20], [16, 17], [16, 18], [17, 18], [17, 19], [18, 19], [18, 20], [19, 20]]}

def construct():
    n = P['n']
    
    # The graph is a circulant graph C_21(1, 2, 5).
    # We can decompose it into 3 edge-disjoint Hamiltonian cycles
    # using steps 1, 2, and 5.
    steps = [1, 2, 5]
    
    cycles = []
    for k in steps:
        cycle = [0]
        current = 0
        for _ in range(n - 1):
            current = (current + k) % n
            cycle.append(current)
        cycles.append(cycle)
    
    # Verification (optional but safe)
    # 1. Check if all edges in cycles exist in P['edges']
    host_edges = set()
    for u, v in P['edges']:
        host_edges.add((min(u, v), max(u, v)))
        
    used_edges = set()
    valid = True
    for cyc in cycles:
        # Check vertex set size and distinctness
        if len(set(cyc)) != n:
            valid = False
            break
        for i in range(n):
            u = cyc[i]
            v = cyc[(i + 1) % n]
            e = (min(u, v), max(u, v))
            if e not in host_edges:
                valid = False
                break
            if e in used_edges:
                valid = False
                break
            used_edges.add(e)
        if not valid:
            break
            
    if valid:
        # Ensure types are strictly Python int in lists
        return [[int(v) for v in c] for c in cycles]
    else:
        # Fallback: return empty list if verification fails (should not happen)
        return []

# EVOLVE-BLOCK-END
