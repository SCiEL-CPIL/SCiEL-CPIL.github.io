"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import itertools
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin
    
    n = 16
    
    # Precompute Q3 structure
    q3_edges = set()
    for i in range(8):
        for j in range(i+1, 8):
            diff = bin(i ^ j).count('1')
            if diff == 1:
                q3_edges.add((i, j))
    
    # Build Q3 adjacency lists
    q3_adj = [set() for _ in range(8)]
    for (a, b) in q3_edges:
        q3_adj[a].add(b)
        q3_adj[b].add(a)
    
    def check_q3_free_fast(adj_sets, new_edge):
        """Fast check: only verify if adding new_edge creates a Q3 using structural properties."""
        u, v = new_edge
        
        # Get neighbors of u and v (excluding each other)
        nu = [x for x in adj_sets[u] if x != v]
        nv = [x for x in adj_sets[v] if x != u]
        
        # Q3 structure relative to edge (0,1):
        # 0 connected to: 1, 2, 4
        # 1 connected to: 0, 3, 5
        # 2 connected to: 0, 6
        # 3 connected to: 1, 7
        # 4 connected to: 0, 6
        # 5 connected to: 1, 7
        # 6 connected to: 2, 4, 7
        # 7 connected to: 3, 5, 6
        
        # We need:
        # x, y in N(u) \ {v} (images of 2, 4)
        # z, w in N(v) \ {u} (images of 3, 5)
        # p in N(x) ∩ N(y) (image of 6), p distinct from {u,v,x,y,z,w}
        # q in N(z) ∩ N(w) (image of 7), q distinct from {u,v,x,y,z,w,p}
        # edge (p,q) must exist
        
        for i in range(len(nu)):
            x = nu[i]
            for j in range(i+1, len(nu)):
                y = nu[j]
                for k in range(len(nv)):
                    z = nv[k]
                    if z == x or z == y:
                        continue
                    for l in range(k+1, len(nv)):
                        w = nv[l]
                        if w == x or w == y or w == z:
                            continue
                        
                        forbidden = {u, v, x, y, z, w}
                        p_candidates = adj_sets[x] & adj_sets[y]
                        p_candidates -= forbidden
                        if not p_candidates:
                            continue
                        
                        common_zw = adj_sets[z] & adj_sets[w]
                        for p in p_candidates:
                            q_candidates = common_zw - {u, v, x, y, z, w, p}
                            for q in q_candidates:
                                if q in adj_sets[p]:
                                    return False
        return True
    
    def greedy_build(part_a, part_b, time_limit):
        """Greedy edge addition starting from K_{|A|,|B|}."""
        adj = [[0]*n for _ in range(n)]
        adj_sets = [set() for _ in range(n)]
        edges = []
        
        # Add all cross edges
        for a in part_a:
            for b in part_b:
                adj[a][b] = 1
                adj[b][a] = 1
                adj_sets[a].add(b)
                adj_sets[b].add(a)
                edges.append([a, b])
        
        # Collect remaining possible edges
        all_possible = []
        for i in range(n):
            for j in range(i+1, n):
                if not adj[i][j]:
                    all_possible.append((i, j))
        
        # Priority: within-part-a first, then within-part-b
        # Within part-a edges are very safe (small set)
        a_set = set(part_a)
        b_set = set(part_b)
        
        def edge_priority(edge):
            i, j = edge
            i_in_a = i in a_set
            j_in_a = j in a_set
            if i_in_a and j_in_a:
                return 0
            elif not i_in_a and not j_in_a:
                return 1
            else:
                return 2  # shouldn't happen
        
        all_possible.sort(key=edge_priority)
        
        best_edges = list(edges)
        best_count = len(edges)
        
        for edge in all_possible:
            if time.time() > time_limit:
                break
            
            u, v = edge
            # Try adding this edge
            adj[u][v] = 1
            adj[v][u] = 1
            adj_sets[u].add(v)
            adj_sets[v].add(u)
            edges.append([u, v])
            
            if check_q3_free_fast(adj_sets, (u, v)):
                best_edges = list(edges)
                best_count = len(edges)
            else:
                # Revert
                adj[u][v] = 0
                adj[v][u] = 0
                adj_sets[u].remove(v)
                adj_sets[v].remove(u)
                edges.pop()
        
        return best_edges, best_count
    
    # Try multiple starting configurations
    configurations = [
        ([0, 1, 2], list(range(3, 16))),       # K_{3,13}
        ([0, 1, 2, 3], list(range(4, 16))),    # K_{4,12}
        ([0, 1], list(range(2, 16))),          # K_{2,14}
        ([0, 1, 2, 3, 4], list(range(5, 16))), # K_{5,11}
    ]
    
    # Allocate time: give more time to promising configs
    total_time_budget = deadline - start_time - 5  # 5s safety margin
    # We'll try each config with a fraction of the time
    # Start with K_{3,13} as it's most likely to work well
    
    best_overall_edges = None
    best_overall_count = 0
    
    # Time allocation: first config gets 50%, others split remaining
    if len(configurations) > 0:
        t1 = start_time + total_time_budget * 0.5
        edges1, count1 = greedy_build(configurations[0][0], configurations[0][1], t1)
        best_overall_edges = edges1
        best_overall_count = count1
    
    remaining_configs = configurations[1:]
    if remaining_configs:
        remaining_time = deadline - start_time - 5
        per_config_time = remaining_time / len(remaining_configs)
        for part_a, part_b in remaining_configs:
            t_end = start_time + total_time_budget - 5  # leave safety margin
            if time.time() > t_end:
                break
            edges_c, count_c = greedy_build(part_a, part_b, t_end)
            if count_c > best_overall_count:
                best_overall_edges = edges_c
                best_overall_count = count_c
    
    if best_overall_edges is None:
        # Fallback: just K_{3,13}
        return [[a, b] for a in range(3) for b in range(3, 16)]
    
    return best_overall_edges
# EVOLVE-BLOCK-END
