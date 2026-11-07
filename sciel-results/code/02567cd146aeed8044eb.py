"""Triangle-free diameter-four augmentation via bipartite ILP."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import linprog

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
    n = P['n']
    base_edges = P['edges']
    
    # Build base adjacency
    base_adj = [[False] * n for _ in range(n)]
    for u, v in base_edges:
        base_adj[u][v] = True
        base_adj[v][u] = True
    
    # Parity sets for triangle-freeness guarantee
    evens = [i for i in range(n) if i % 2 == 0]
    odds = [i for i in range(n) if i % 2 == 1]
    
    # Candidate cross-parity edges (not in base)
    candidates = []
    for u in evens:
        for v in odds:
            if not base_adj[u][v]:
                candidates.append((u, v))
    
    m = len(candidates)
    
    # Precompute: for each candidate edge, which pairs (i,j) does it help?
    # A candidate edge (u,v) helps pair (i,j) if there exist a in N(i) and b in N(j)
    # such that a is adjacent to u or v (via base or other candidates), and b is adjacent to u or v.
    # For LP relaxation, we use a sufficient condition:
    # Edge (u,v) helps (i,j) if:
    #   (i is adjacent to u or v) AND (j is adjacent to u or v), OR
    #   (i is adjacent to some a where a is adjacent to u or v) AND (j is adjacent to some b where b is adjacent to u or v)
    # Simplified: edge (u,v) helps (i,j) if dist_base(i,u)<=1 or dist_base(i,v)<=1 AND dist_base(j,u)<=1 or dist_base(j,v)<=1
    # Actually, with the shortcut, dist(i,j) <= 1 + dist(i, {u,v}) + 1 + dist(j, {u,v}) if both can reach the edge.
    # More precisely: if i can reach u or v in <=1 step, and j can reach u or v in <=1 step, then dist(i,j) <= 1+1+1+1 = 4.
    
    # For each pair (i,j) with base distance > 4, we need coverage.
    # Let's compute base distances first.
    # BFS from each node for base distances
    base_dist = [[-1] * n for _ in range(n)]
    for s in range(n):
        # BFS
        dist = [-1] * n
        dist[s] = 0
        queue = [s]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in range(n):
                if base_adj[u][v] and dist[v] == -1:
                    dist[v] = dist[u] + 1
                    queue.append(v)
        base_dist[s] = dist
    
    # Pairs that need help (base distance > 4)
    need_pairs = []
    for i in range(n):
        for j in range(i+1, n):
            if base_dist[i][j] > 4:
                need_pairs.append((i, j))
    
    k = len(need_pairs)
    
    # For each candidate edge e_idx and each need_pair p_idx, determine if edge covers the pair
    # Edge (u,v) covers pair (i,j) if:
    #   min(dist_base(i,u), dist_base(i,v)) <= 1 AND min(dist_base(j,u), dist_base(j,v)) <= 1
    # This gives a path of length at most 1 + 1 + 1 + 1 = 4.
    
    cover = np.zeros((k, m), dtype=int)
    for pi, (i, j) in enumerate(need_pairs):
        for ei, (u, v) in enumerate(candidates):
            di = min(base_dist[i][u], base_dist[i][v])
            dj = min(base_dist[j][u], base_dist[j][v])
            if di <= 1 and dj <= 1:
                cover[pi][ei] = 1
    
    # Set up ILP: minimize sum(x_e) subject to sum_e cover[p,e] * x_e >= 1 for all p, x_e in {0,1}
    # Use LP relaxation with linprog, then round up.
    
    # Objective: minimize sum x_e
    c = np.ones(m, dtype=float)
    
    # Constraints: A_ub @ x <= b_ub (linprog uses <= form)
    # We want: cover @ x >= 1  =>  -cover @ x <= -1
    A_ub = -cover.astype(float)
    b_ub = -np.ones(k, dtype=float)
    
    # Bounds: 0 <= x_e <= 1
    bounds = [(0, 1)] * m
    
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
    
    if res.success:
        x = res.x
        # Round: take edges where x > 0.5
        selected = [ei for ei in range(m) if x[ei] > 0.5]
    else:
        # Fallback: greedy set cover
        selected = []
        remaining = set(range(k))
        while remaining:
            best_ei = None
            best_count = -1
            for ei in range(m):
                if ei in selected:
                    continue
                count = 0
                for pi in remaining:
                    if cover[pi][ei] == 1:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_ei = ei
            if best_ei is None or best_count <= 0:
                break
            selected.append(best_ei)
            remaining = {pi for pi in remaining if cover[pi][best_ei] == 0}
    
    # Verify and if needed, add more edges greedily
    added_edges = [list(candidates[ei]) for ei in selected]
    
    # Verify diameter constraint
    def check_diameter(edges_added):
        adj = [[False] * n for _ in range(n)]
        for u, v in base_edges:
            adj[u][v] = True
            adj[v][u] = True
        for u, v in edges_added:
            adj[u][v] = True
            adj[v][u] = True
        
        # Check triangle-free
        for i in range(n):
            for j in range(i+1, n):
                if adj[i][j]:
                    for k in range(j+1, n):
                        if adj[i][k] and adj[j][k]:
                            return False
        
        # Check connected
        visited = [False] * n
        visited[0] = True
        queue = [0]
        qi = 0
        while qi < len(queue):
            u = queue[qi]
            qi += 1
            for v in range(n):
                if adj[u][v] and not visited[v]:
                    visited[v] = True
                    queue.append(v)
        if not all(visited):
            return False
        
        # Check diameter
        for s in range(n):
            dist = [-1] * n
            dist[s] = 0
            queue = [s]
            qi = 0
            while qi < len(queue):
                u = queue[qi]
                qi += 1
                for v in range(n):
                    if adj[u][v] and dist[v] == -1:
                        dist[v] = dist[u] + 1
                        queue.append(v)
            if max(dist) > 4:
                return False
        return True
    
    if not check_diameter(added_edges):
        # Greedy fix: add more edges until valid
        remaining_candidates = [ei for ei in range(m) if ei not in selected]
        while remaining_candidates:
            best_ei = None
            best_improvement = -1
            for ei in remaining_candidates:
                test_edges = added_edges + [list(candidates[ei])]
                # Quick check: does this edge help?
                u, v = candidates[ei]
                help_count = 0
                for pi, (i, j) in enumerate(need_pairs):
                    if cover[pi][ei] == 1:
                        help_count += 1
                if help_count > best_improvement:
                    best_improvement = help_count
                    best_ei = ei
            if best_ei is None or best_improvement <= 0:
                break
            added_edges.append(list(candidates[best_ei]))
            remaining_candidates.remove(best_ei)
            if check_diameter(added_edges):
                break
    
    # Final verification
    if not check_diameter(added_edges):
        # Last resort: use the known-good single hub solution
        hub = 11
        added_edges = []
        for e in evens:
            if e != hub - 1 and e != hub + 1:
                if not base_adj[hub][e]:
                    added_edges.append([hub, e])
    
    return added_edges
# EVOLVE-BLOCK-END
