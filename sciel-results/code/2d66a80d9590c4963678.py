import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin

    def get_cycle(n, adj_bits):
        """Returns a 5-cycle if found, else None."""
        for u in range(n):
            # Get non-neighbors of u
            # We only care about v > u to avoid duplicates? 
            # Actually, just checking all v != u is fine, but v > u is faster.
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    continue # u and v are adjacent, no C5 can have u-v as chord? 
                             # Wait, C5 is a cycle. If u-v is an edge, 
                             # can it be part of a C5? 
                             # Yes, u-v is an edge in the cycle.
                             # But my logic above was for u, v being non-adjacent 
                             # in the cycle? No, in a C5, all consecutive pairs are adjacent.
                             # The non-consecutive pairs are non-adjacent (chords).
                             # My previous logic: "Path u-a-b-c-v". 
                             # Here u and v are NOT adjacent in the path, 
                             # but they ARE adjacent in the cycle (v-u).
                             # So u and v must be adjacent in the graph?
                             # No, in the cycle u-a-b-c-v-u, u and v are adjacent.
                             # So we should iterate over EDGES (u, v).
                             # And look for a path of length 4 between u and v 
                             # that does not use the edge (u, v).
                             # Path: u-a-b-c-v.
                             # a != v, c != u.
                             # a, b, c distinct from u, v.
                             # a != c, a != b, c != b.
                             
                             # Let's restart the C5 check logic.
                             # For each edge (u, v):
                             #   Find a, c such that a in N(u), c in N(v).
                             #   a != v, c != u.
                             #   Find b in N(a) & N(c).
                             #   b not in {u, v, a, c}.
                             #   Also need to ensure a, b, c are distinct.
                             #   And the path is simple.
                             
                             # This is more direct.
            # Reset loop to iterate over edges
            pass
        
        # Corrected C5 Check
        for u in range(n):
            for v in range(u + 1, n):
                if not ((adj_bits[u] >> v) & 1):
                    continue # Not an edge, so u-v is not part of the cycle edge set in this iteration
                
                # We are looking for a cycle u-v-c-b-a-u? 
                # Or u-a-b-c-v-u.
                # Edge (u, v) is in the cycle.
                # Path u-a-b-c-v of length 4.
                
                nu = adj_bits[u] & ~(1 << v) # Neighbors of u excluding v
                nv = adj_bits[v] & ~(1 << u) # Neighbors of v excluding u
                
                # a in nu, c in nv
                # b in N(a) & N(c)
                # b not in {u, v, a, c}
                
                a_mask = nu
                while a_mask:
                    lsb_a = a_mask & -a_mask
                    a = lsb_a.bit_length() - 1
                    a_mask ^= lsb_a
                    
                    # a is neighbor of u
                    # c must be neighbor of v
                    c_mask = nv
                    while c_mask:
                        lsb_c = c_mask & -c_mask
                        c = lsb_c.bit_length() - 1
                        c_mask ^= lsb_c
                        
                        if c == a:
                            continue
                        
                        # b in N(a) & N(c)
                        common = adj_bits[a] & adj_bits[c]
                        forbidden = (1 << u) | (1 << v) | (1 << a) | (1 << c)
                        common &= ~forbidden
                        
                        if common:
                            # Found a cycle
                            # u - a - b - c - v - u
                            # We need b's index
                            lsb_b = common & -common
                            b = lsb_b.bit_length() - 1
                            return [u, a, b, c, v]
        return None

    def get_independent_set_9(n, adj_bits):
        """Returns an independent set of size 9 if found, else None."""
        # Find K9 in complement
        comp_adj = [0] * n
        mask_all = (1 << n) - 1
        for u in range(n):
            comp_adj[u] = mask_all ^ adj_bits[u] ^ (1 << u)
            
        # Backtracking
        def search(candidates, k, current_set):
            if k == 0:
                return current_set
            if len(candidates) < k:
                return None
            if k == 1:
                return current_set + [candidates[0]]
                
            # Heuristic: pick vertex with highest degree in subgraph
            # For simplicity, just pick first
            # Better: iterate and try to prune
            
            # Try each vertex as the next in the clique
            # To avoid duplicates, we maintain order
            for i in range(len(candidates)):
                v = candidates[i]
                # New candidates are neighbors of v in comp_adj, from candidates[i+1:]
                new_cands = []
                for j in range(i + 1, len(candidates)):
                    u = candidates[j]
                    if (comp_adj[v] >> u) & 1:
                        new_cands.append(u)
                        
                res = search(new_cands, k - 1, current_set + [v])
                if res is not None:
                    return res
            return None

        return search(list(range(n)), 9, [])

    def is_valid(n, adj_bits):
        if get_cycle(n, adj_bits) is not None:
            return False
        if get_independent_set_9(n, adj_bits) is not None:
            return False
        return True

    def adj_to_edges(n, adj_bits):
        edges = []
        for u in range(n):
            for v in range(u + 1, n):
                if (adj_bits[u] >> v) & 1:
                    edges.append([u, v])
        return edges

    # Base solution: 8 K4s (n=32)
    base_edges = []
    for i in range(8):
        start = i * 4
        for u in range(4):
            for v in range(u + 1, 4):
                base_edges.append([start + u, start + v])
                
    best_n = 32
    best_edges = base_edges
    
    # Try to find better solutions for n = 35, 34, 33
    for target_n in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        found = False
        for restart in range(100): # Reduced restarts to allow more steps per restart
            if time.time() > deadline:
                break
                
            p = 0.5
            adj_bits = [0] * target_n
            for u in range(target_n):
                for v in range(u + 1, target_n):
                    if random.random() < p:
                        adj_bits[u] |= (1 << v)
                        adj_bits[v] |= (1 << u)
                        
            steps = 0
            max_steps = 500
            while steps < max_steps and time.time() < deadline:
                steps += 1
                
                cycle = get_cycle(target_n, adj_bits)
                if cycle is not None:
                    # Remove a random edge from the cycle
                    # Cycle: u, a, b, c, v
                    # Edges: (u,a), (a,b), (b,c), (c,v), (v,u)
                    edges_in_cycle = [(cycle[0], cycle[1]), (cycle[1], cycle[2]), 
                                      (cycle[2], cycle[3]), (cycle[3], cycle[4]), 
                                      (cycle[4], cycle[0])]
                    e_u, e_v = random.choice(edges_in_cycle)
                    adj_bits[e_u] &= ~(1 << e_v)
                    adj_bits[e_v] &= ~(1 << e_u)
                    continue
                
                ind_set = get_independent_set_9(target_n, adj_bits)
                if ind_set is not None:
                    # Add an edge between two vertices in the independent set
                    # This breaks the independent set
                    u, v = random.sample(ind_set, 2)
                    adj_bits[u] |= (1 << v)
                    adj_bits[v] |= (1 << u)
                    continue
                
                # Valid
                found = True
                best_n = target_n
                best_edges = adj_to_edges(target_n, adj_bits)
                break
                
        if found:
            break

    return {'n': best_n, 'edges': best_edges}
