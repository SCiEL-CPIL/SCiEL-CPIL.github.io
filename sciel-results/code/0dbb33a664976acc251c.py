     def sa_search(n, deadline):
         m = n * (n - 1) // 2
         colors = [random.randint(0, 4) for _ in range(m)]
         
         # Precompute edge indices
         edge_idx = [[-1]*n for _ in range(n)]
         idx = 0
         for i in range(n):
             for j in range(i+1, n):
                 edge_idx[i][j] = idx
                 edge_idx[j][i] = idx
                 idx += 1
                 
         adj = [[0]*n for _ in range(5)]
         total_violations = 0
         
         # Initialize
         for i in range(n):
             for j in range(i+1, n):
                 c = colors[edge_idx[i][j]]
                 adj[c][i] |= (1 << j)
                 adj[c][j] |= (1 << i)
                 
         # Calculate initial total violations
         # For each color, for each pair u<v, count common neighbors
         for c in range(5):
             for u in range(n):
                 for v in range(u+1, n):
                     common = (adj[c][u] & adj[c][v]).bit_count()
                     if common > 1:
                         total_violations += common * (common - 1) // 2
                         
         if total_violations == 0:
             return colors
             
         T = 1.0
         T_min = 0.01
         step = 0
         max_steps = 1000000 # Adjust based on time
         
         while time.time() < deadline:
             # Pick random edge
             u = random.randint(0, n-1)
             v = random.randint(0, n-1)
             if u == v: continue
             if u > v: u, v = v, u
             
             c_old = colors[edge_idx[u][v]]
             c_new = random.randint(0, 4)
             if c_new == c_old: continue
             
             # Calculate delta
             delta = 0
             
             # Remove from c_old
             # For each w != u, v
             # Check if v in adj[c_old][w] -> affects K(u, w)
             # Check if u in adj[c_old][w] -> affects K(v, w)
             
             for w in range(n):
                 if w == u or w == v: continue
                 
                 # Pair (u, w)
                 if (adj[c_old][w] >> v) & 1:
                     # v is common neighbor of u and w
                     K_old = (adj[c_old][u] & adj[c_old][w]).bit_count()
                     # Contribution change: C(K_old, 2) - C(K_old-1, 2) = K_old - 1
                     delta -= (K_old - 1)
                     
                 # Pair (v, w)
                 if (adj[c_old][w] >> u) & 1:
                     K_old = (adj[c_old][v] & adj[c_old][w]).bit_count()
                     delta -= (K_old - 1)
                     
             # Add to c_new
             for w in range(n):
                 if w == u or w == v: continue
                 
                 # Pair (u, w)
                 if (adj[c_new][w] >> v) & 1:
                     K_old = (adj[c_new][u] & adj[c_new][w]).bit_count()
                     # Contribution change: C(K_old+1, 2) - C(K_old, 2) = K_old
                     delta += K_old
                     
                 # Pair (v, w)
                 if (adj[c_new][w] >> u) & 1:
                     K_old = (adj[c_new][v] & adj[c_new][w]).bit_count()
                     delta += K_old
                     
             # Accept?
             if delta < 0 or random.random() < pow(2.71828, -delta / T):
                 # Update
                 colors[edge_idx[u][v]] = c_new
                 total_violations += delta
                 
                 # Update adj
                 adj[c_old][u] ^= (1 << v)
                 adj[c_old][v] ^= (1 << u)
                 adj[c_new][u] ^= (1 << v)
                 adj[c_new][v] ^= (1 << u)
                 
                 if total_violations == 0:
                     return colors
                 
             # Cool down
             if step % 100 == 0:
                 T *= 0.99
                 if T < T_min: T = T_min
             step += 1
             
             # Restart if stuck?
             if step % 10000 == 0 and T == T_min:
                 # Reheat
                 T = 1.0
                 
         return None
     