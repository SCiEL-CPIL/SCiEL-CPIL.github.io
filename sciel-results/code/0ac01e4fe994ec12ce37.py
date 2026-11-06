import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110  # safety margin under 120s
    
    n = 40
    # Precompute indices for vectorized C6 check
    # u < v < w < x < y < z
    u_idx, v_idx, w_idx, x_idx, y_idx, z_idx = np.meshgrid(
        np.arange(n), np.arange(n), np.arange(n), np.arange(n), np.arange(n), np.arange(n),
        indexing='ij'
    )
    mask = (u_idx < v_idx) & (v_idx < w_idx) & (w_idx < x_idx) & (x_idx < y_idx) & (y_idx < z_idx)
    u_f = u_idx[mask]
    v_f = v_idx[mask]
    w_f = w_idx[mask]
    x_f = x_idx[mask]
    y_f = y_idx[mask]
    z_f = z_idx[mask]
    
    def has_c6(adj):
        """
        Check if graph with adjacency matrix adj (0/1) contains a C6.
        Uses vectorized NumPy operations.
        """
        # Edges: (u,v), (v,w), (w,x), (x,y), (y,z), (z,u)
        e1 = adj[u_f, v_f]
        e2 = adj[v_f, w_f]
        e3 = adj[w_f, x_f]
        e4 = adj[x_f, y_f]
        e5 = adj[y_f, z_f]
        e6 = adj[z_f, u_f]
        
        # Early exit if any edge type is missing
        if not e1.any() or not e2.any() or not e3.any() or not e4.any() or not e5.any() or not e6.any():
            return False
            
        # Combine all edge conditions
        all_edges = e1 & e2 & e3 & e4 & e5 & e6
        return all_edges.any()

    def count_edges(adj):
        return np.sum(adj) // 2

    best_adj = None
    best_cnt = 0
    
    # Strategy: Random sampling of graphs with varying edge densities.
    # We know ex(40; C6) is around 105.
    # Total possible edges: 40*39/2 = 780.
    # Density for 105 edges: 105/780 ~ 0.135.
    # We will sample p in a range around this, slightly higher to find dense valid graphs.
    
    # We will also try to "complete" or "refine" if we find a valid graph, 
    # but the primary driver is sampling.
    
    # To increase the chance of finding valid graphs, we can bias the sampling
    # or use a repair strategy. However, pure random sampling at p=0.13-0.15
    # might yield very few C6-free graphs.
    # A better approach: Sample from a slightly lower density, then greedily add edges?
    # No, the directive asks for a DIFFERENT strategy.
    # Let's stick to "Random Sampling + Verification" but with a twist:
    # We will sample many graphs. If we find a valid one, we record it.
    # We will also try "structured" random graphs: e.g., random regular graphs?
    # Or simply random graphs with p tuned.
    
    # Actually, random graphs at p=0.135 have a huge number of C6s.
    # The probability that a specific 6-tuple is a C6 is p^6 ~ 10^-5.
    # Number of 6-tuples is ~ 40^6 ~ 4*10^9.
    # Expected number of C6s is ~ 4*10^4.
    # So random graphs are almost certainly NOT C6-free.
    
    # Therefore, pure random sampling will fail.
    # We need a strategy that generates C6-free graphs more reliably.
    # The "Spectral Construction" hint suggests using structures with controlled eigenvalues.
    # One such class is **Random Graphs with Forbidden Subgraphs** via **Gibbs Sampling** or **Local Search from Random Start**.
    
    # Let's pivot the "Different Strategy" to: **Greedy Removal from a Dense Random Graph**.
    # 1. Generate a random graph with p=0.2 (very dense, ~156 edges).
    # 2. While it has a C6, remove a random edge from a found C6.
    # 3. This guarantees a C6-free graph.
    # 4. Repeat with many restarts and different p values.
    # 5. Keep the one with the most edges.
    
    # This is "Top-Down" (start dense, remove) vs "Bottom-Up" (start empty, add).
    # It is materially different from the current "Bottom-Up" greedy/local search.
    
    def find_c6_edge(adj):
        """
        Find one edge that is part of a C6. Returns (u, v) or None.
        Uses the precomputed arrays.
        """
        e1 = adj[u_f, v_f]
        e2 = adj[v_f, w_f]
        e3 = adj[w_f, x_f]
        e4 = adj[x_f, y_f]
        e5 = adj[y_f, z_f]
        e6 = adj[z_f, u_f]
        
        all_edges = e1 & e2 & e3 & e4 & e5 & e6
        if not all_edges.any():
            return None
        
        # Pick a random index from the valid C6s
        idx = np.random.choice(np.where(all_edges)[0])
        # Return the edge (u, v) of this C6. 
        # Removing any edge in the cycle breaks it.
        # We return (u, v) for simplicity.
        return int(u_f[idx]), int(v_f[idx])

    best_adj = None
    best_cnt = 0
    
    # Parameters for top-down search
    # We want to end up with ~105 edges.
    # Start with p such that expected edges > 105.
    # p=0.15 -> 117 edges.
    # p=0.20 -> 156 edges.
    
    # We will run several rounds of "Random Dense Start + Greedy Removal".
    
    # To make it faster, we can check for C6 and remove in a loop.
    # The C6 check is O(N^6) effectively but vectorized. 
    # With n=40, N^6 is large, but the boolean array is fixed size.
    # Let's benchmark the cost of has_c6.
    
    # We will try different starting densities.
    densities = [0.14, 0.15, 0.16, 0.18, 0.20, 0.25]
    
    for p in densities:
        # Try a few restarts for each density
        for _ in range(5):
            if time.time() > deadline - 5:
                break
                
            # Generate random graph
            # adj is symmetric
            # Generate upper triangle
            tri_u, tri_v = np.triu_indices(n, k=1)
            r = np.random.random(len(tri_u))
            adj = np.zeros((n, n), dtype=np.uint8)
            mask_p = r < p
            adj[tri_u[mask_p], tri_v[mask_p]] = 1
            adj[tri_v[mask_p], tri_u[mask_p]] = 1
            
            # Greedy Removal
            # We want to remove as FEW edges as possible.
            # Random removal might be suboptimal.
            # We can prioritize removing edges that are in MANY C6s?
            # That's expensive. Let's stick to random removal for simplicity first.
            
            # Optimization: Instead of checking the whole graph for a C6 each time,
            # we can just check and remove.
            
            while True:
                if time.time() > deadline - 2:
                    break
                    
                c6_edge = find_c6_edge(adj)
                if c6_edge is None:
                    # No C6 found
                    break
                else:
                    u, v = c6_edge
                    adj[u, v] = 0
                    adj[v, u] = 0
            
            if time.time() > deadline - 2:
                continue
                
            cnt = count_edges(adj)
            if cnt > best_cnt:
                best_cnt = cnt
                best_adj = adj.copy()
                
                # If we found a very good graph, we might want to try to add edges back?
                # But the task says "Maximize edges".
                # The "Different Strategy" is the Top-Down construction.
                # We can add a small "Refinement" phase: try to add random edges if they don't create C6.
                # This combines Top-Down (to get a valid base) with Bottom-Up (to maximize).
                # But let's keep the core strategy distinct.
                
                # Actually, the current program DOES local search.
                # If I just do Top-Down, I might get stuck at 100 edges.
                # Let's add a lightweight "Add-Edge" pass on the result of Top-Down.
                
                # Add-Edge Pass
                # Try to add random non-edges.
                non_edges = np.where((adj == 0) & (np.arange(n)[:, None] < np.arange(n)[None, :]))
                if len(non_edges[0]) > 0:
                    # Shuffle non-edges
                    perm = np.random.permutation(len(non_edges[0]))
                    for i in perm:
                        if time.time() > deadline - 2:
                            break
                        u, v = non_edges[0][i], non_edges[1][i]
                        if adj[u, v] == 0:
                            # Try adding
                            adj[u, v] = 1
                            adj[v, u] = 1
                            if has_c6(adj):
                                # Revert
                                adj[u, v] = 0
                                adj[v, u] = 0
                            else:
                                # Keep
                                pass

                # Update best
                cnt = count_edges(adj)
                if cnt > best_cnt:
                    best_cnt = cnt
                    best_adj = adj.copy()

    # If no graph was found (unlikely), fallback to a simple valid graph
    if best_adj is None:
        # 8 disjoint K5s
        best_adj = np.zeros((n, n), dtype=np.uint8)
        for k in range(8):
            base = k * 5
            for i in range(5):
                for j in range(i + 1, 5):
                    u = base + i
                    v = base + j
                    best_adj[u, v] = 1
                    best_adj[v, u] = 1
        best_cnt = count_edges(best_adj)

    # Convert to list of edges
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if best_adj[u, v]:
                edges.append([u, v])

    return edges
