import random
import time
import sys

sys.setrecursionlimit(10000)

def _time_up(deadline):
    return time.time() > deadline

def _check_triangle_free(adj, n):
    for i in range(n):
        for j in adj[i]:
            if j < i:
                continue
            if adj[i] & adj[j]:
                return False
    return True

def _exact_alpha(adj, n, deadline):
    """
    Returns the size of the maximum independent set.
    Uses branch and bound with bitmasks (n <= 64).
    """
    masks = [0] * n
    for i in range(n):
        m = 0
        for j in adj[i]:
            m |= (1 << j)
        masks[i] = m
        
    best = [0]
    
    # Pre-compute popcount for small integers is fast in Python 3.
    # We use bin(x).count('1') which is optimized in C.
    
    def dfs(candidates, current_size):
        if _time_up(deadline):
            return
        
        if not candidates:
            if current_size > best[0]:
                best[0] = current_size
            return
            
        # Upper bound: current_size + popcount(candidates)
        # If this bound <= best[0], prune
        if current_size + bin(candidates).count('1') <= best[0]:
            return
            
        # Choose a vertex to branch on
        # Heuristic: pick vertex with highest degree in the induced subgraph
        # to maximize pruning in the "exclude" branch (which removes fewer candidates? No, exclude removes v, include removes v+neighbors).
        # Actually, standard branch and bound for Independent Set:
        # Branch on v.
        # Include v: candidates -> candidates \ (N(v) U {v})
        # Exclude v: candidates -> candidates \ {v}
        # To prune effectively, we want to find a large set quickly (to raise best) and prune deep.
        # Picking high degree vertices for inclusion is good because it removes many candidates, leading to a smaller subtree for the "include" branch,
        # but the "exclude" branch is large.
        # A common heuristic is to pick a vertex with high degree in the remaining graph.
        
        best_v = -1
        max_deg = -1
        temp_cands = candidates
        while temp_cands:
            lsb = temp_cands & -temp_cands
            v = lsb.bit_length() - 1
            deg = bin(candidates & masks[v]).count('1')
            if deg > max_deg:
                max_deg = deg
                best_v = v
            temp_cands ^= lsb
            
        v = best_v
        if v == -1:
            return
            
        bit_v = 1 << v
        
        # Option 1: Include v
        new_cands_1 = candidates & ~masks[v] & ~bit_v
        dfs(new_cands_1, current_size + 1)
        
        # Option 2: Exclude v
        new_cands_2 = candidates & ~bit_v
        dfs(new_cands_2, current_size)
        
    all_mask = (1 << n) - 1
    dfs(all_mask, 0)
    
    return best[0]

def construct():
    deadline = time.time() + 110.0 # Safety margin
    
    best_n = 0
    best_edges = []
    
    # Strategy:
    # 1. Start with a known small valid graph (C5 blowup or similar) to have a baseline.
    # 2. Try to grow it by adding vertices (local search / hill climbing).
    # 3. Also try random restarts with triangle-free enforcement.
    
    # Helper to check if graph is valid
    def is_valid(adj, n):
        if not _check_triangle_free(adj, n):
            return False
        alpha = _exact_alpha(adj, n, deadline)
        return alpha < 10

    # Initial best: C5 (n=5, alpha=2)
    # Edges: 0-1, 1-2, 2-3, 3-4, 4-0
    adj_5 = [set() for _ in range(5)]
    edges_5 = [[0,1], [1,2], [2,3], [3,4], [4,0]]
    for u, v in edges_5:
        adj_5[u].add(v)
        adj_5[v].add(u)
    if _check_triangle_free(adj_5, 5) and _exact_alpha(adj_5, 5, deadline) < 10:
        best_n = 5
        best_edges = edges_5

    # Try to find larger graphs
    # We will attempt to build graphs of increasing size n.
    # For each n, we do several restarts.
    
    for target_n in range(6, 41):
        if _time_up(deadline):
            break
            
        found = False
        attempts = 0
        max_attempts = 200 # Adjust based on time
        
        while not found and not _time_up(deadline) and attempts < max_attempts:
            attempts += 1
            n = target_n
            adj = [set() for _ in range(n)]
            
            # Initialize with random sparse graph
            p = 0.3
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i].add(j)
                        adj[j].add(i)
                        
            # Enforce triangle-free
            # Remove edges that form triangles
            changed = True
            while changed:
                changed = False
                for i in range(n):
                    if _time_up(deadline):
                        break
                    for j in list(adj[i]):
                        if j < i: continue
                        common = adj[i] & adj[j]
                        if common:
                            # Remove edge (i,j)
                            adj[i].remove(j)
                            adj[j].remove(i)
                            changed = True
                if _time_up(deadline):
                    break
            
            if not _check_triangle_free(adj, n):
                continue

            # Check alpha
            # Quick heuristic rejection
            # Greedy independent set
            best_heur = 0
            for _ in range(10):
                vertices = list(range(n))
                random.shuffle(vertices)
                ind_set = set()
                for v in vertices:
                    if not (ind_set & adj[v]):
                        ind_set.add(v)
                if len(ind_set) > best_heur:
                    best_heur = len(ind_set)
            
            if best_heur >= 10:
                continue

            # Exact check
            alpha = _exact_alpha(adj, n, deadline)
            if alpha < 10:
                edges = []
                for i in range(n):
                    for j in adj[i]:
                        if i < j:
                            edges.append([i, j])
                if n > best_n:
                    best_n = n
                    best_edges = edges
                found = True
                break
                
        # If we found a valid graph for this n, continue to try n+1
        # If not, we can still continue, maybe a random restart works
        # But if we fail for a long time, we might want to stop trying smaller n and focus on larger?
        # The loop structure tries 6, then 7, etc.
        # If 6 fails, we try 7. This is inefficient if 6 is easy.
        # Better: if we find valid for n, keep it. If we fail for n, move to n+1.
        # This ensures we don't waste time on small n once we have a good one.
        
        # Optimization: If we failed to find a valid graph for n, 
        # and n is significantly larger than current best, maybe try harder?
        # For now, simple increment.

    if best_n > 0:
        return {'n': best_n, 'edges': best_edges}
    else:
        # Fallback to C5 if nothing worked (shouldn't happen)
        return {'n': 5, 'edges': [[0,1],[1,2],[2,3],[3,4],[4,0]]}
