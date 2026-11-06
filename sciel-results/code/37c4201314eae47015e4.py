import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Target n=46, min_deg=7
    # Using Polarity Graph of PG(2,7)
    # v = 57, d = 8
    
    # 1. Generate PG(2,7) vertices
    # Points are non-zero vectors in GF(7)^3 up to scalar multiplication.
    # We can represent each point by its canonical form: first non-zero entry is 1.
    points = []
    for x0 in range(7):
        for x1 in range(7):
            for x2 in range(7):
                if x0 == 0 and x1 == 0 and x2 == 0:
                    continue
                # Canonicalize
                # Find first non-zero
                if x0 != 0:
                    inv = pow(x0, 6, 7) # 7 is prime, x^(p-2) = x^-1
                    p = (1, (x1 * inv) % 7, (x2 * inv) % 7)
                elif x1 != 0:
                    inv = pow(x1, 6, 7)
                    p = (0, 1, (x2 * inv) % 7)
                else: # x2 != 0
                    p = (0, 0, 1)
                
                if p not in points:
                    points.append(p)
    
    n_total = len(points) # Should be 57
    # Map points to indices
    point_to_idx = {p: i for i, p in enumerate(points)}
    
    # 2. Build adjacency list for Polarity Graph
    # Polarity: x0*y2 + x1*y1 + x2*y0 = 0
    # Note: In characteristic 7, this is a symmetric bilinear form.
    # Two points u, v are adjacent if u . v = 0.
    # Self-orthogonality: u . u = x0*x2 + x1*x1 + x2*x0 = 2*x0*x2 + x1^2.
    # If u . u = 0, u is adjacent to itself? No, simple graph, no self-loops.
    # We only add edge (u,v) if u != v and u . v = 0.
    
    adj = [[0] * n_total for _ in range(n_total)]
    for i in range(n_total):
        u = points[i]
        for j in range(i + 1, n_total):
            v = points[j]
            # Dot product mod 7
            dot = (u[0] * v[2] + u[1] * v[1] + u[2] * v[0]) % 7
            if dot == 0:
                adj[i][j] = 1
                adj[j][i] = 1
                
    # Verify degree is 8
    # deg = sum(adj[i])
    
    # 3. Find subset T of size 11 to remove such that min degree of S (size 46) >= 7
    # Equivalent to e(T) >= 21
    
    def count_edges_in_subset(T):
        # T is a set of indices
        e = 0
        T_list = list(T)
        for i in range(len(T_list)):
            u = T_list[i]
            for j in range(i + 1, len(T_list)):
                v = T_list[j]
                if adj[u][v]:
                    e += 1
        return e
    
    best_T = None
    best_e = -1
    
    # Local search to maximize e(T)
    # Initial T: random 11 vertices
    all_vertices = set(range(n_total))
    
    # Try a few random starts
    for _ in range(500):
        if time.time() > deadline - 5:
            break
            
        T = set(random.sample(list(all_vertices), 11))
        e = count_edges_in_subset(T)
        
        improved = True
        while improved:
            if time.time() > deadline - 5:
                break
            improved = False
            # Try all swaps
            T_list = list(T)
            S_list = list(all_vertices - T)
            
            for i in range(len(T_list)):
                if time.time() > deadline - 5:
                    break
                u = T_list[i]
                for v in S_list:
                    # Calculate delta e
                    # Remove u, add v
                    # Edges lost: edges from u to T\{u}
                    # Edges gained: edges from v to T\{u}
                    
                    # Optimize: precompute degrees within T?
                    # For simplicity, just recalculate if potential gain is high?
                    # Or just swap and check.
                    
                    # Quick check: 
                    # deg_u_in_T = sum(1 for w in T if w != u and adj[u][w])
                    # deg_v_in_T = sum(1 for w in T if w != u and adj[v][w])
                    # delta = deg_v_in_T - deg_u_in_T
                    # If delta > 0, swap.
                    
                    deg_u_in_T = 0
                    deg_v_in_T = 0
                    T_minus_u = T - {u}
                    for w in T_minus_u:
                        if adj[u][w]:
                            deg_u_in_T += 1
                        if adj[v][w]:
                            deg_v_in_T += 1
                    
                    if deg_v_in_T > deg_u_in_T:
                        # Swap
                        T.remove(u)
                        T.add(v)
                        e += deg_v_in_T - deg_u_in_T
                        improved = True
                        if e > best_e:
                            best_e = e
                            best_T = T.copy()
                        break # Restart loop after swap
            if e > best_e:
                best_e = e
                best_T = T.copy()
                
    # Check if best_T is valid
    if best_T is not None and best_e >= 21:
        T = best_T
        S = list(all_vertices - T)
        
        # Verify min degree in S
        min_deg = 99
        for u in S:
            deg = 0
            for v in S:
                if u != v and adj[u][v]:
                    deg += 1
            if deg < min_deg:
                min_deg = deg
        
        if min_deg >= 7:
            # Construct edges for S
            edges = []
            # Map old indices to new 0..len(S)-1
            idx_map = {old: new for new, old in enumerate(S)}
            n_s = len(S)
            for i in range(n_s):
                for j in range(i + 1, n_s):
                    u = S[i]
                    v = S[j]
                    if adj[u][v]:
                        edges.append([idx_map[u], idx_map[v]])
            
            # Safety check: ensure all vertices in edges are < n_s
            # and n_s is within limits
            if all(0 <= e[0] < n_s and 0 <= e[1] < n_s for e in edges):
                return {'n': n_s, 'edges': edges}
            
    # Fallback: n=45
    # Remove any 12 vertices
    T_fallback = set(random.sample(list(all_vertices), 12))
    S = list(all_vertices - T_fallback)
    
    # Verify min degree in S (should be >= 6)
    min_deg = 99
    for u in S:
        deg = 0
        for v in S:
            if u != v and adj[u][v]:
                deg += 1
        if deg < min_deg:
            min_deg = deg
            
    if min_deg >= 6:
        edges = []
        idx_map = {old: new for new, old in enumerate(S)}
        n_s = len(S)
        for i in range(n_s):
            for j in range(i + 1, n_s):
                u = S[i]
                v = S[j]
                if adj[u][v]:
                    edges.append([idx_map[u], idx_map[v]])
        if all(0 <= e[0] < n_s and 0 <= e[1] < n_s for e in edges):
            return {'n': n_s, 'edges': edges}
        
    # Fallback n=39
    return {'n': 39, 'edges': []}
