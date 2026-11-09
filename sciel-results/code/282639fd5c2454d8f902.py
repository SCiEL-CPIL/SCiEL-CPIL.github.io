import time
import random
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Leave 10s margin for 120s limit

    # Helper: Check if graph is C4-free using bitmasks
    def is_c4_free(adj, n):
        for u in range(n):
            for v in range(u + 1, n):
                if (adj[u] >> v) & 1:
                    # If u and v are adjacent, they can't be opposite in C4, 
                    # but the definition "no two vertices have two common neighbours" 
                    # applies to ALL pairs? 
                    # Wait. A C4 is u-v-w-x-u. 
                    # u and w are opposite. v and x are opposite.
                    # u and v are adjacent.
                    # The condition "No two vertices may have two common neighbours" 
                    # is the standard definition of C4-free.
                    # This applies to ANY pair of vertices, adjacent or not.
                    # If u and v are adjacent, can they have 2 common neighbors?
                    # If u-v is an edge, and u-w-v is a path? No, simple graph.
                    # If u and v are adjacent, and they share neighbors w and x,
                    # then u-w-v-x-u is a C4? No, u-w-v is a triangle if w-v is edge?
                    # If u and v are adjacent, and share w, then u-v-w-u is a triangle.
                    # If they share w and x, we have u-v, u-w, v-w (triangle) and u-x, v-x (triangle).
                    # Does this form a C4?
                    # A C4 requires 4 distinct vertices.
                    # If u and v are adjacent, and share w, the cycle u-v-w-u is length 3.
                    # To have a C4 involving u and v as opposite vertices, they must NOT be adjacent.
                    # To have a C4 involving u and v as adjacent vertices, e.g., u-v-w-x-u,
                    # u and v are adjacent. Common neighbors of u and v?
                    # Neighbors of u: v, w, ...
                    # Neighbors of v: u, w, ...
                    # Common neighbor w.
                    # If they have another common neighbor x, then u-x-v is a path.
                    # Cycle u-v-w-x-u? 
                    # Edges: u-v, v-w, w-x, x-u.
                    # u and v share w. Do they share x?
                    # x is neighbor of u and v.
                    # So u and v have common neighbors w and x.
                    # Yes, the condition "no two vertices have >1 common neighbor" covers all C4s.
                    pass
                
                common = adj[u] & adj[v]
                if common.bit_count() > 1:
                    return False
        return True

    # Helper: Check if alpha < 11
    def has_independent_set_11(adj, n):
        full_mask = (1 << n) - 1
        non_neighbors = [0] * n
        for i in range(n):
            # Non-neighbors of i are vertices not in adj[i] and not i itself
            non_neighbors[i] = full_mask ^ adj[i] ^ (1 << i)
            
        # Use a recursive backtracking with pruning
        # candidates is a list of vertices that are mutually non-adjacent to each other
        # and non-adjacent to all previously chosen vertices
        def backtrack(candidates, size):
            if size == 11:
                return True
            if len(candidates) < 11 - size:
                return False
            
            # Heuristic: pick vertex with fewest connections to remaining candidates?
            # Or just first?
            # Let's pick the one that restricts the most (lowest degree in induced subgraph of candidates)
            # This is expensive. Let's stick to first for speed, or random.
            # Actually, picking the vertex with the smallest index is fine if we shuffle candidates initially?
            # No, let's just pick candidates[0].
            
            v = candidates[0]
            rest = candidates[1:]
            
            # New candidates must be non-neighbors of v
            new_candidates = [u for u in rest if (non_neighbors[v] >> u) & 1]
            
            if backtrack(new_candidates, size + 1):
                return True
            
            if backtrack(rest, size):
                return True
                
            return False
            
        # Initial candidates: all vertices
        return backtrack(list(range(n)), 0)

    # Fallback: Seed program (n=30)
    def get_seed_graph():
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}

    # Polarity graph of PG(2, q)
    def get_polarity_graph_q5():
        q = 5
        points = []
        for x in range(q):
            for y in range(q):
                points.append((x, y, 1))
        for m in range(q):
            points.append((1, m, 0))
        points.append((0, 1, 0))
        
        n = len(points)
        adj = [0] * n
        edges = []
        
        for i in range(n):
            xi, yi, zi = points[i]
            for j in range(i + 1, n):
                xj, yj, zj = points[j]
                if (xi * xj + yi * yj + zi * zj) % q == 0:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
                    edges.append([i, j])
        
        # Verify
        if is_c4_free(adj, n) and not has_independent_set_11(adj, n):
            return {'n': n, 'edges': edges}
        return None

    # Try polarity graph first (n=31)
    result = get_polarity_graph_q5()
    if result is not None:
        # We have a valid n=31 graph.
        # Try to improve to n=32..39
        best_result = result
        best_n = result['n']
        
        for n in range(32, 40):
            if time.time() > deadline:
                break
                
            # Try random search for this n
            found = False
            for trial in range(200): # Adjust trials based on time
                if time.time() > deadline:
                    break
                    
                # Generate random graph
                # Density for C4-free is low.
                p = 0.15
                adj = [0] * n
                edges = []
                
                for u in range(n):
                    for v in range(u + 1, n):
                        if random.random() < p:
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                            edges.append([u, v])
                            
                # Remove C4s
                # Iterate until no C4
                changed = True
                while changed:
                    changed = False
                    for u in range(n):
                        for v in range(u + 1, n):
                            common = adj[u] & adj[v]
                            if common.bit_count() > 1:
                                # Remove one edge from the C4
                                # Find two common neighbors w, x
                                # Cycle u-w-v-x-u
                                # Remove (u, w)
                                mask = common
                                lsb = mask & -mask
                                w = lsb.bit_length() - 1
                                adj[u] &= ~(1 << w)
                                adj[w] &= ~(1 << u)
                                changed = True
                                # Rebuild edges list is expensive, do it later
                                break
                        if changed:
                            break
                    if not changed:
                        break
                
                # Rebuild edges
                edges = []
                for u in range(n):
                    mask = adj[u]
                    while mask:
                        lsb = mask & -mask
                        v = lsb.bit_length() - 1
                        if u < v:
                            edges.append([u, v])
                        mask ^= lsb
                
                if not is_c4_free(adj, n):
                    continue
                    
                if not has_independent_set_11(adj, n):
                    return {'n': n, 'edges': edges}
                    
        return best_result

    # If polarity graph failed (shouldn't), try seed
    return get_seed_graph()
