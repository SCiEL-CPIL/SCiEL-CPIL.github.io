import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # 115s safety margin
    
    # GF(4) arithmetic
    mult_table = [
        [0, 1, 2, 3],
        [1, 1, 2, 3],
        [2, 2, 3, 1],
        [3, 3, 1, 2]
    ]
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_mul(a, b):
        return mult_table[a][b]
    
    # Generate points of PG(2,4)
    points = []
    for x in range(4):
        for y in range(4):
            for z in range(4):
                if x == 0 and y == 0 and z == 0:
                    continue
                vec = (x, y, z)
                is_new = True
                for p in points:
                    for k in [1, 2, 3]:
                        if (gf4_mul(k, p[0]) == vec[0] and 
                            gf4_mul(k, p[1]) == vec[1] and 
                            gf4_mul(k, p[2]) == vec[2]):
                            is_new = False
                            break
                    if not is_new:
                        break
                if is_new:
                    points.append(vec)
    
    n4 = len(points)
    
    # Build polarity graph: u ~ v if dot(u,v) = 0
    adj4 = [[False]*n4 for _ in range(n4)]
    edges4 = []
    for i in range(n4):
        for j in range(i+1, n4):
            u = points[i]
            v = points[j]
            dot = gf4_add(gf4_add(gf4_mul(u[0], v[2]), gf4_mul(u[1], v[1])), gf4_mul(u[2], v[0]))
            if dot == 0:
                adj4[i][j] = True
                adj4[j][i] = True
                edges4.append([i, j])
    
    # Verify C4-free
    def check_c4_free(adj, n):
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
        for u in range(n):
            for v in range(u+1, n):
                common = adj_mask[u] & adj_mask[v]
                if bin(common).count('1') >= 2:
                    return False
        return True
    
    if not check_c4_free(adj4, n4):
        # Fallback: 10 disjoint triangles
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # The 21-vertex polarity graph is C4-free
    # Now try to extend it by adding vertices
    # A new vertex v can be added if:
    # 1. For any two existing neighbors of v, they don't share another common neighbor (C4-free)
    # 2. The independence number stays < 11
    
    # Greedy extension: try to add vertices one by one
    # Start with the 21-vertex graph
    best_n = n4
    best_edges = edges4[:]
    best_adj = [row[:] for row in adj4]
    
    # Try adding up to 22 more vertices (total up to 43)
    max_vertices = 43
    for new_v in range(n4, max_vertices):
        if time.time() > deadline - 10:
            break
        
        # Try different adjacency patterns for new_v
        # The new vertex must connect to a subset S of existing vertices
        # such that no two vertices in S have a common neighbor (other than new_v)
        
        # For C4-free: for any u, w in S, u and w must not have a common neighbor
        # (since new_v would be a common neighbor, and if they have another common neighbor, C4)
        
        # So S must be a set of vertices where no two share a common neighbor
        # This means S is an independent set in the "common neighbor" graph
        
        # Build common neighbor relation
        # u and w are "conflicting" if they have a common neighbor
        
        # For efficiency, let's try a few random subsets
        found = False
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            # Random subset of existing vertices
            S = set(random.sample(range(new_v), random.randint(0, min(8, new_v))))
            
            # Check C4 constraint: no two in S have a common neighbor
            valid = True
            S_list = list(S)
            for i in range(len(S_list)):
                for j in range(i+1, len(S_list)):
                    u, w = S_list[i], S_list[j]
                    # Check if u and w have a common neighbor
                    common = 0
                    for k in range(new_v):
                        if best_adj[u][k] and best_adj[w][k]:
                            common += 1
                            if common >= 1:
                                valid = False
                                break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Check independence number constraint
            # Add new_v with adjacency to S
            test_adj = [row[:] for row in best_adj]
            test_adj.append([False]*max_vertices)
            for i in range(new_v):
                test_adj[i].append(False)
            test_adj.append([False]*max_vertices)
            
            for u in S:
                test_adj[new_v][u] = True
                test_adj[u][new_v] = True
            
            # Quick check: is there an independent set of size 11?
            # Use a fast heuristic
            # Greedy independent set
            used = [False] * (new_v + 1)
            indep = []
            for v in range(new_v + 1):
                if not used[v]:
                    indep.append(v)
                    for u in range(new_v + 1):
                        if test_adj[v][u]:
                            used[u] = True
            
            if len(indep) >= 11:
                continue
            
            # Looks good, accept this vertex
            for u in S:
                best_adj[new_v][u] = True
                best_adj[u][new_v] = True
                best_edges.append([new_v, u])
            
            best_n = new_v + 1
            found = True
            break
        
        if not found:
            # Try with no edges (isolated vertex) - but this increases alpha
            # Only if alpha allows
            # Skip for now
            pass
    
    # Final verification
    final_n = best_n
    final_adj = [row[:final_n] for row in best_adj[:final_n]]
    
    if not check_c4_free(final_adj, final_n):
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    # Compute final independence number
    def max_independent_set(adj, n):
        adj_mask = [0] * n
        for i in range(n):
            for j in range(n):
                if adj[i][j]:
                    adj_mask[i] |= (1 << j)
        
        best = [0]
        
        def dfs(candidates, size):
            if size > best[0]:
                best[0] = size
            if candidates == 0:
                return
            if size + bin(candidates).count('1') <= best[0]:
                return
            max_deg = -1
            chosen = -1
            c = candidates
            while c:
                lsb = c & (-c)
                v = lsb.bit_length() - 1
                c ^= lsb
                deg = bin(adj_mask[v] & candidates).count('1')
                if deg > max_deg:
                    max_deg = deg
                    chosen = v
            if chosen == -1:
                return
            new_candidates = candidates & ~(adj_mask[chosen] | (1 << chosen))
            dfs(new_candidates, size + 1)
            new_candidates = candidates & ~(1 << chosen)
            dfs(new_candidates, size)
        
        all_candidates = (1 << n) - 1
        dfs(all_candidates, 0)
        return best[0]
    
    alpha = max_independent_set(final_adj, final_n)
    
    if alpha >= 11:
        # Fallback
        edges = []
        for t in range(10):
            a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
            edges += [[a, b], [a, c], [b, c]]
        return {'n': 30, 'edges': edges}
    
    return {'n': final_n, 'edges': best_edges}
