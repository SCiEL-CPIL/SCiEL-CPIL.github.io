import time
import random
import numpy as np

def construct():
    deadline = time.time() + 135  # safety margin before 150s kill
    n = 20
    m = 10
    K = 20  # number of available colours
    
    left = list(range(m))
    right = list(range(m, 2*m))
    
    # Precompute all 4-cycles in K_{m,m}
    cycles = []
    for l1 in range(m):
        for l2 in range(l1+1, m):
            for r1 in range(m):
                for r2 in range(r1+1, m):
                    cycles.append((l1, l2, r1, r2))
    
    def check_colouring(colour_matrix):
        """Check if all 4-cycles are rainbow. colour_matrix[l][r] = color."""
        for l1, l2, r1, r2 in cycles:
            c1 = colour_matrix[l1][r1]
            c2 = colour_matrix[l2][r1]
            c3 = colour_matrix[l2][r2]
            c4 = colour_matrix[l1][r2]
            if len({c1, c2, c3, c4}) < 4:
                return False
        return True

    best_edges = None
    best_count = 0
    
    # Strategy 1: Try to find valid K_{10,10} coloring via structured + random search
    found_valid = False
    
    # Try linear forms C[l][r] = (a*l + b*r) % K
    for a in range(1, K):
        for b in range(1, K):
            if time.time() > deadline - 20:
                break
            colour_matrix = [[(a*l + b*r) % K for r in range(m)] for l in range(m)]
            if check_colouring(colour_matrix):
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
        if found_valid:
            break
    
    if not found_valid:
        # Try C[l][r] = (l * r + c) % K
        for c in range(K):
            if time.time() > deadline - 20:
                break
            colour_matrix = [[(l * r + c) % K for r in range(m)] for l in range(m)]
            if check_colouring(colour_matrix):
                found_valid = True
                best_count = m * m
                best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                             for l in range(m) for r in range(m)]
                break
        if found_valid:
            pass
    
    if not found_valid:
        # Random search with repair on K_{10,10}
        for attempt in range(5000):
            if time.time() > deadline - 10:
                break
            
            # Random initialization
            colour_matrix = [[random.randint(0, K-1) for _ in range(m)] for _ in range(m)]
            
            # Repair loop
            for repair_iter in range(100):
                if time.time() > deadline - 10:
                    break
                # Find a violated cycle
                violated = None
                for l1, l2, r1, r2 in cycles:
                    c1 = colour_matrix[l1][r1]
                    c2 = colour_matrix[l2][r1]
                    c3 = colour_matrix[l2][r2]
                    c4 = colour_matrix[l1][r2]
                    if len({c1, c2, c3, c4}) < 4:
                        violated = (l1, l2, r1, r2)
                        break
                
                if violated is None:
                    # Valid!
                    found_valid = True
                    best_count = m * m
                    best_edges = [[left[l], right[r], colour_matrix[l][r]] 
                                 for l in range(m) for r in range(m)]
                    break
                
                l1, l2, r1, r2 = violated
                # Change one of the four colors randomly
                edge_choice = random.randint(0, 3)
                if edge_choice == 0:
                    colour_matrix[l1][r1] = random.randint(0, K-1)
                elif edge_choice == 1:
                    colour_matrix[l2][r1] = random.randint(0, K-1)
                elif edge_choice == 2:
                    colour_matrix[l2][r2] = random.randint(0, K-1)
                else:
                    colour_matrix[l1][r2] = random.randint(0, K-1)
            
            if found_valid:
                break
    
    # Strategy 2: If K_{10,10} coloring not found, use C4-free greedy bipartite
    if best_edges is None:
        # Build C4-free bipartite graph greedily
        adjL = [set() for _ in range(m)]
        adjR = [set() for _ in range(m)]
        
        edges = []
        all_edges = [(i, j) for i in range(m) for j in range(m)]
        random.shuffle(all_edges)
        
        for i, j in all_edges:
            if time.time() > deadline - 2:
                break
            
            # Check if adding (i, j) creates a C4
            creates_c4 = False
            for i_prime in adjR[j]:
                if adjL[i] & adjL[i_prime]:
                    creates_c4 = True
                    break
            
            if not creates_c4:
                adjL[i].add(j)
                adjR[j].add(i)
                c = random.randint(0, K - 1)
                edges.append([left[i], right[j], c])
        
        best_edges = edges
        best_count = len(edges)
    
    # Strategy 3: General graph greedy with efficient validity checking
    if best_count < 100:
        n_gen = 20
        all_edges_gen = []
        for i in range(n_gen):
            for j in range(i+1, n_gen):
                all_edges_gen.append((i, j))
        
        best_gen_edges = [[0, v, 0] for v in range(1, n_gen)]
        best_gen_count = 19
        
        def get_valid_colors(u, v, adj, color_mat, neighbors):
            """
            Returns the set of colors that can be assigned to edge (u, v)
            without violating the rainbow 4-cycle property.
            Assumes (u, v) is NOT currently in the graph.
            """
            forbidden = set()
            # Check all 3-paths v-w-x-u
            # A 4-cycle is formed if we add (u,v) and there exists a path u-x-w-v of length 3.
            # The edges are (u,v), (v,w), (w,x), (x,u).
            # Colors must be distinct: c_uv != c_vw != c_wx != c_xu != c_uv.
            # Since c_uv is new, we just need to ensure c_uv is not in {c_vw, c_wx, c_xu}
            # AND that {c_vw, c_wx, c_xu} are distinct among themselves.
            
            for w in neighbors[v]:
                if w == u:
                    continue
                c_vw = color_mat[v][w]
                for x in neighbors[u]:
                    if x == v or x == w:
                        continue
                    if not adj[w][x]:
                        continue
                    c_wx = color_mat[w][x]
                    c_xu = color_mat[x][u]
                    
                    # Check if existing path colors are distinct
                    if c_vw == c_wx or c_vw == c_xu or c_wx == c_xu:
                        # This path is already "bad" for any new edge? 
                        # Wait, if the existing 3-path has duplicate colors, 
                        # adding (u,v) creates a 4-cycle with duplicate colors regardless of c_uv?
                        # Yes, if c_vw == c_wx, the cycle has duplicate colors.
                        # So if such a path exists, we CANNOT add (u,v) at all?
                        # No, the condition is "Every 4-cycle ... must receive four distinct colours".
                        # If there is a path u-x-w-v with c_wx == c_vw, and we add (u,v),
                        # the cycle u-v-w-x-u has colors c_uv, c_vw, c_wx, c_xu.
                        # Since c_vw == c_wx, the cycle is NOT rainbow.
                        # Thus, if ANY such path exists, we cannot add (u,v).
                        return set()
                    
                    forbidden.add(c_vw)
                    forbidden.add(c_wx)
                    forbidden.add(c_xu)
            
            return set(range(n_gen)) - forbidden
        
        # Run multiple greedy passes with different seeds and edge orderings
        for trial in range(100):
            if time.time() > deadline - 5:
                break
            
            seed = trial
            random.seed(seed)
            
            adj = {i: set() for i in range(n_gen)}
            color_mat = {}
            neighbors = [set() for _ in range(n_gen)]
            current_edges = []
            
            edge_order = all_edges_gen[:]
            random.shuffle(edge_order)
            
            for u, v in edge_order:
                if time.time() > deadline - 2:
                    break
                
                valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                if valid:
                    # Prefer a structured color choice
                    pref = (u + v) % n_gen
                    if pref in valid:
                        c = pref
                    else:
                        c = random.choice(list(valid))
                    
                    adj[u].add(v)
                    adj[v].add(u)
                    color_mat[(u, v)] = c
                    color_mat[(v, u)] = c
                    neighbors[u].add(v)
                    neighbors[v].add(u)
                    current_edges.append([u, v, c])
            
            if len(current_edges) > best_gen_count:
                best_gen_count = len(current_edges)
                best_gen_edges = current_edges
        
        if best_gen_count > best_count:
            best_edges = best_gen_edges
            best_count = best_gen_count
    
    # Strategy 4: Local search to add more edges via recoloring
    if best_edges is not None and best_count < 190:
        if time.time() < deadline - 10:
            # Build adjacency from current best
            adj = {i: set() for i in range(n)}
            color_mat = {}
            neighbors = [set() for _ in range(n)]
            
            for u, v, c in best_edges:
                adj[u].add(v)
                adj[v].add(u)
                color_mat[(u, v)] = c
                color_mat[(v, u)] = c
                neighbors[u].add(v)
                neighbors[v].add(u)
            
            cur_edges = [e[:] for e in best_edges]
            cur_count = best_count
            edge_list = [(u, v) for u, v, c in best_edges]
            
            for iteration in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Try adding missing edges
                missing = [(i, j) for i in range(n) for j in range(i+1, n) if j not in adj[i]]
                if not missing:
                    break
                random.shuffle(missing)
                
                added = False
                for u, v in missing:
                    valid = get_valid_colors(u, v, adj, color_mat, neighbors)
                    if valid:
                        pref = (u + v) % n
                        if pref in valid:
                            c = pref
                        else:
                            c = random.choice(list(valid))
                        
                        adj[u].add(v)
                        adj[v].add(u)
                        color_mat[(u, v)] = c
                        color_mat[(v, u)] = c
                        neighbors[u].add(v)
                        neighbors[v].add(u)
                        cur_edges.append([u, v, c])
                        cur_count += 1
                        edge_list.append((u, v))
                        added = True
                        break
                
                if not added:
                    # Try recoloring existing edges to free up colors
                    random.shuffle(edge_list)
                    improved = False
                    for eu, ev in edge_list[:50]:
                        old_c = color_mat[(eu, ev)]
                        other_colors = [c for c in range(n) if c != old_c]
                        random.shuffle(other_colors)
                        
                        for new_c in other_colors[:10]:
                            # Check if recoloring (eu, ev) to new_c is valid
                            valid_recolor = True
                            for w in neighbors[ev]:
                                if w == eu:
                                    continue
                                c_ev_w = color_mat[(ev, w)]
                                for x in neighbors[eu]:
                                    if x == ev or x == w:
                                        continue
                                    if not adj[w].__contains__(x):
                                        continue
                                    c_wx = color_mat[(w, x)]
                                    c_xu = color_mat[(x, eu)]
                                    
                                    if c_ev_w == c_wx or c_ev_w == c_xu or c_wx == c_xu:
                                        continue
                                    
                                    if new_c in (c_ev_w, c_wx, c_xu):
                                        valid_recolor = False
                                        break
                                if not valid_recolor:
                                    break
                            if valid_recolor:
                                color_mat[(eu, ev)] = new_c
                                color_mat[(ev, eu)] = new_c
                                for i, e in enumerate(cur_edges):
                                    if e[0] == eu and e[1] == ev:
                                        cur_edges[i][2] = new_c
                                        break
                                improved = True
                                break
                        if improved:
                            break
                
                if cur_count > best_count:
                    best_count = cur_count
                    best_edges = cur_edges
    
    return best_edges
