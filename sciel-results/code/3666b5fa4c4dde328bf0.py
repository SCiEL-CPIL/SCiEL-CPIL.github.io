import time
import random
import numpy as np

def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n = 20
    m = 10
    K = 20  # number of available colours
    
    left = list(range(m))
    right = list(range(m, 2*m))
    
    # Precompute all 4-cycles in K_{m,m}
    # A 4-cycle is (l1, l2, r1, r2) with l1!=l2, r1!=r2
    # Cycle: l1-r1-l2-r2-l1
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
    
    # Strategy 3: Also try non-bipartite greedy with the current program's approach
    # as a fallback if we haven't found something good
    if best_count < 50:
        # Try the original greedy approach on general graphs
        n_gen = 20
        all_edges_gen = []
        for i in range(n_gen):
            for j in range(i+1, n_gen):
                all_edges_gen.append((i, j))
        
        best_gen_edges = [[0, v, 0] for v in range(1, n_gen)]
        best_gen_count = 19
        
        def is_valid_gen(adj, colors, u, v, c):
            for a in range(n_gen):
                if a == u or a == v:
                    continue
                if a not in adj[u]:
                    continue
                ca = colors.get((u, a)) or colors.get((a, u))
                if ca is None:
                    continue
                
                for b in range(n_gen):
                    if b == u or b == v or b == a:
                        continue
                    if b not in adj[a]:
                        continue
                    if v not in adj[b]:
                        continue
                    
                    cab = colors.get((a, b)) or colors.get((b, a))
                    cbv = colors.get((b, v)) or colors.get((v, b))
                    
                    if cab is None or cbv is None:
                        continue
                    
                    cols = [c, ca, cab, cbv]
                    if len(set(cols)) != 4:
                        return False
            return True
        
        for _ in range(500):
            if time.time() > deadline - 5:
                break
                
            random.shuffle(all_edges_gen)
            adj = {i: set() for i in range(n_gen)}
            colors = {}
            current_edges = []
            
            for u, v in all_edges_gen:
                found = False
                color_order = list(range(n_gen))
                random.shuffle(color_order)
                
                for c in color_order:
                    if is_valid_gen(adj, colors, u, v, c):
                        adj[u].add(v)
                        adj[v].add(u)
                        colors[(u, v)] = c
                        colors[(v, u)] = c
                        current_edges.append([u, v, c])
                        found = True
                        break
                
                if not found:
                    pass
            
            if len(current_edges) > best_gen_count:
                best_gen_count = len(current_edges)
                best_gen_edges = current_edges
        
        if best_gen_count > best_count:
            best_edges = best_gen_edges
            best_count = best_gen_count
    
    return best_edges
