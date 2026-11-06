import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Seed solution n=9
    seed_n = 9
    seed_colors = [min(j - i, seed_n - j + i) - 1 for i in range(seed_n) for j in range(i + 1, seed_n)]
    best = {'n': seed_n, 'colors': seed_colors}

    # Try to find higher n
    # n=13 is 78 edges, n=12 is 66, n=11 is 55, n=10 is 45.
    # We will try 13, then 12, then 11, then 10.

    for n in [13, 12, 11, 10]:
        if time.time() > deadline - 2.0:
            break

        # Allocate time based on difficulty
        budget = 28.0 if n == 13 else (22.0 if n == 12 else (18.0 if n == 11 else 8.0))
        end_time = time.time() + budget

        res = _find_coloring(n, end_time)
        if res:
            return res

    return best

def _find_coloring(n, deadline):
    # Generate all edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    num_edges = len(all_edges)
    
    # Order edges by "degree" heuristic: edges connecting high-degree vertices first
    # In complete graph, all vertices have degree n-1, so use a different heuristic
    # Try ordering by sum of vertex indices to break symmetry
    edge_order = sorted(range(num_edges), key=lambda idx: (all_edges[idx][0] + all_edges[idx][1], all_edges[idx][0]))
    edges = [all_edges[i] for i in edge_order]
    edge_to_orig = [edge_order[i] for i in range(num_edges)]
    
    # Precompute for each edge position, which original edge index it corresponds to
    # For the final answer, we need to map back to lexicographic order
    
    for attempt in range(10000):
        if time.time() > deadline:
            break
        
        colors_ordered = [-1] * num_edges
        adj = [[0] * n for _ in range(4)]
        
        def check_c6(u, v, c):
            nu = adj[c][u]
            nv = adj[c][v]
            
            # Iterate x1 in nu
            x1_mask = nu
            while x1_mask:
                lsb = x1_mask & -x1_mask
                x1 = lsb.bit_length() - 1
                x1_mask ^= lsb
                if x1 == v: continue

                # Iterate x4 in nv
                x4_mask = nv
                while x4_mask:
                    lsb4 = x4_mask & -x4_mask
                    x4 = lsb4.bit_length() - 1
                    x4_mask ^= lsb4
                    if x4 == u or x4 == x1: continue

                    # N(x1) and N(x4) excluding forbidden {u, v, x1, x4}
                    forbidden = (1 << u) | (1 << v) | (1 << x1) | (1 << x4)

                    nx1 = adj[c][x1] & ~forbidden
                    nx4 = adj[c][x4] & ~forbidden

                    # Check if union of N(x2) for x2 in nx1 intersects nx4
                    # x3 must be in N(x2) ∩ N(x4) and x3 ∉ {u,v,x1,x4}
                    if not nx1: continue
                    if not nx4: continue

                    union = 0
                    vx2 = nx1
                    while vx2:
                        lsb2 = vx2 & -vx2
                        x2 = lsb2.bit_length() - 1
                        vx2 ^= lsb2
                        union |= adj[c][x2] & ~forbidden

                    if union & nx4:
                        return True
            return False

        def backtrack(idx):
            if time.time() > deadline:
                return False
            if idx == num_edges:
                return True

            u, v = edges[idx]

            # Try colors in a deterministic but varied order
            # Use attempt number to vary the starting color
            start_c = attempt % 4
            c_vals = [(start_c + k) % 4 for k in range(4)]
            
            for c in c_vals:
                if check_c6(u, v, c):
                    continue

                colors_ordered[idx] = c
                adj[c][u] |= (1 << v)
                adj[c][v] |= (1 << u)

                if backtrack(idx + 1):
                    return True

                adj[c][u] &= ~(1 << v)
                adj[c][v] &= ~(1 << u)
                colors_ordered[idx] = -1

            return False

        if backtrack(0):
            # Map back to lexicographic order
            colors_lex = [-1] * num_edges
            for i in range(num_edges):
                orig_idx = edge_to_orig[i]
                colors_lex[orig_idx] = colors_ordered[i]
            return {'n': n, 'colors': colors_lex}

    return None
