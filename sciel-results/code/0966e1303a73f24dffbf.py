import time
import random
import numpy as np

def construct():
    start_time = time.time()
    DEADLINE = start_time + 110.0

    def gen_graph(n, p, seed):
        rng = random.Random(seed)
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.append((u, v))
        return edges

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def popcount(x):
        return x.bit_count()

    def max_clique_in_induced(n, adj, mask, rng):
        """
        Find max clique in G[mask].
        adj is the global adjacency list.
        mask is a bitmask of vertices to consider.
        """
        best_clique = []
        
        def expand(R, P):
            nonlocal best_clique
            if not P:
                if len(R) > len(best_clique):
                    best_clique = R[:]
                return
            
            pc = popcount(P)
            if len(R) + pc <= len(best_clique):
                return
            
            # Pivot
            max_deg = -1
            candidates = []
            P_copy = P
            while P_copy:
                v_bit = P_copy & (-P_copy)
                v = v_bit.bit_length() - 1
                P_copy ^= v_bit
                deg = popcount(P & adj[v])
                if deg > max_deg:
                    max_deg = deg
                    candidates = [v]
                elif deg == max_deg:
                    candidates.append(v)
                    
            if not candidates:
                return
            
            u = rng.choice(candidates)
            Q = P & ~adj[u]
            
            while Q:
                # Pick v from Q
                # Heuristic: pick v with max degree in P
                max_v_deg = -1
                v_candidates = []
                Q_copy = Q
                while Q_copy:
                    v_bit = Q_copy & (-Q_copy)
                    i = v_bit.bit_length() - 1
                    Q_copy ^= v_bit
                    deg = popcount(P & adj[i])
                    if deg > max_v_deg:
                        max_v_deg = deg
                        v_candidates = [i]
                    elif deg == max_v_deg:
                        v_candidates.append(i)
                        
                if not v_candidates:
                    break
                
                v = rng.choice(v_candidates)
                
                new_R = R + [v]
                new_P = P & adj[v]
                expand(new_R, new_P)
                
                P &= ~(1 << v)
                Q = Q & P
                
        expand([], mask)
        return best_clique

    def solve_instance(n, p, seed, time_limit):
        edges = gen_graph(n, p, seed)
        adj = build_adj(n, edges)
        
        # Baseline
        baseline = [[u, v] for u, v in edges]
        best_cliques = baseline
        best_count = len(baseline)
        
        start = time.time()
        attempt = 0
        while time.time() < start + time_limit:
            rng = random.Random(seed * 1000 + attempt)
            
            rem_adj = adj[:]
            cliques = []
            
            # Precompute edge list and weights?
            # We need to dynamically remove edges.
            # Using adj bitmasks is efficient for removal.
            
            # Strategy: Hard Edge First
            # We need to find the edge with min common neighbors in the REMAINING graph.
            # This requires scanning all remaining edges.
            # Number of remaining edges decreases.
            
            # Optimization:
            # Instead of scanning all edges every time, we can just scan.
            # For n=64, E~1100. Scanning is cheap.
            
            has_edges = any(rem_adj[i] != 0 for i in range(n))
            
            while has_edges:
                # Find edge (u,v) with min popcount(rem_adj[u] & rem_adj[v])
                min_common = n + 1
                best_u, best_v = -1, -1
                
                for u in range(n):
                    if not rem_adj[u]:
                        continue
                    # Iterate neighbors v > u
                    v_mask = rem_adj[u] & (~((1 << (u+1)) - 1)) # v > u
                    while v_mask:
                        v_bit = v_mask & (-v_mask)
                        v = v_bit.bit_length() - 1
                        v_mask ^= v_bit
                        
                        common = popcount(rem_adj[u] & rem_adj[v])
                        if common < min_common:
                            min_common = common
                            best_u, best_v = u, v
                            # If min_common is 0, we can't do better (edge is isolated in terms of triangles)
                            # But we still need to cover it.
                            # Actually if common is 0, the clique is just {u,v}.
                            # We can break early if we find 0? 
                            # Yes, 0 is the minimum possible.
                            if min_common == 0:
                                break
                    if min_common == 0:
                        break
                
                if best_u == -1:
                    break # No edges found? Should not happen if has_edges is true.
                
                # Find max clique containing best_u, best_v
                # Common neighbors of best_u and best_v in rem_adj
                common_mask = rem_adj[best_u] & rem_adj[best_v]
                
                # Find max clique in G[common_mask]
                clique_core = max_clique_in_induced(n, rem_adj, common_mask, rng)
                
                # The clique is core + {best_u, best_v}
                clique = clique_core + [best_u, best_v]
                
                # Remove edges of this clique
                for i in range(len(clique)):
                    for j in range(i + 1, len(clique)):
                        u, v = clique[i], clique[j]
                        rem_adj[u] &= ~(1 << v)
                        rem_adj[v] &= ~(1 << u)
                
                cliques.append(clique)
                
                has_edges = any(rem_adj[i] != 0 for i in range(n))
            
            count = len(cliques)
            if count < best_count:
                best_count = count
                best_cliques = cliques
            
            attempt += 1
            
        return best_cliques

    # Solve instances
    # Time budget allocation
    # n=32 is much faster than n=64.
    # Give 20% to n=32, 80% to n=64?
    # Or just split evenly?
    # Let's give 15s to first, 90s to second.
    
    t1 = 15.0
    t2 = 90.0
    
    current = time.time()
    remaining = DEADLINE - current
    if t1 + t2 > remaining:
        scale = remaining / (t1 + t2)
        t1 *= scale
        t2 *= scale

    result1 = solve_instance(32, 0.6, 601, t1)
    
    if time.time() > DEADLINE:
        edges2 = gen_graph(64, 0.55, 602)
        result2 = [[u, v] for u, v in edges2]
    else:
        result2 = solve_instance(64, 0.55, 602, t2)
    
    return [result1, result2]
