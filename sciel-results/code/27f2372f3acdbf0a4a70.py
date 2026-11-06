    def run_sa(A, time_limit_remaining):
        edges = get_edges(A)
        current_score = score(A)
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        
        while time.time() < time_limit_remaining:
            if not edges: break
            
            u, v = random.choice(edges)
            
            # Find non-edge
            while True:
                u2 = random.randint(0, n-1)
                v2 = random.randint(0, n-1)
                if u2 == v2:
                    continue
                if A[u2, v2] == 0:
                    if (u2 == u and v2 == v) or (u2 == v and v2 == u):
                        continue
                    break
            
            # Apply
            A[u, v] = 0
            A[v, u] = 0
            A[u2, v2] = 1
            A[v2, u2] = 1
            
            # Update edge list
            # Remove (u,v)
            # Find index
            idx = edges.index((u,v)) if (u,v) in edges else edges.index((v,u))
            # Actually, we stored them as (min, max) or just (u,v)?
            # get_edges returns (i,j) with i<j.
            # So (u,v) might be (v,u) in list if u>v.
            # Let's normalize edges in list to (min, max)
            
            # This is getting complicated. 
            # Given n=16, just recalculating edges is fast enough?
            # 16*15/2 = 120 checks.
            # If we do 10000 steps, 1.2M checks. Fast.
            # So I will stick to recalculating edges to avoid bugs.
            
            new_score = score(A)
            
            if new_score < current_score:
                current_score = new_score
                yield A, new_score
            else:
                delta = new_score - current_score
                if delta > 0:
                    p = np.exp(-delta / T)
                    if random.random() < p:
                        current_score = new_score
                    else:
                        A[u, v] = 1
                        A[v, u] = 1
                        A[u2, v2] = 0
                        A[v2, u2] = 0
                else:
                    current_score = new_score
                    yield A, new_score
                    
            T = max(T * cooling, T_min)
