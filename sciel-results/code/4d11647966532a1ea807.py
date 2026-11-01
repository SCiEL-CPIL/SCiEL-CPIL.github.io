  def search_local_320(k, g, start_time, deadline):
      if k != 4:
          return None, None
      n = 320
      max_time = deadline - 5
      current_time = time.time()
      
      def generate_random_4reg(n):
          while True:
              stubs = []
              for i in range(n):
                  for _ in range(k):
                      stubs.append(i)
              random.shuffle(stubs)
              edges = set()
              valid = True
              for i in range(0, len(stubs), 2):
                  u = stubs[i]
                  v = stubs[i+1]
                  if u == v:
                      valid = False
                      break
                  e = (min(u,v), max(u,v))
                  if e in edges:
                      valid = False
                      break
                  edges.add(e)
              if valid:
                  return list(edges)
      
      def build_adj(n, edges):
          adj = [[] for _ in range(n)]
          for u, v in edges:
              adj[u].append(v)
              adj[v].append(u)
          return adj
      
      def find_short_cycle(adj, n, g):
          # Returns (length, edges_in_cycle) or None
          for start in range(n):
              dist = [-1] * n
              parent = [-1] * n
              dist[start] = 0
              q = deque([start])
              while q:
                  u = q.popleft()
                  if dist[u] >= g // 2: # We only care about cycles < g
                      continue
                  for v in adj[u]:
                      if dist[v] == -1:
                          dist[v] = dist[u] + 1
                          parent[v] = u
                          q.append(v)
                      elif v != parent[u]:
                          # Found a back edge
                          # Cycle length
                          # We need to ensure this is a simple cycle
                          # The path from start to u and start to v are simple.
                          # The union forms a cycle.
                          # Length = dist[u] + dist[v] + 1
                          # But wait, if v is an ancestor of u, it's a tree edge? No, v != parent[u] handles direct parent.
                          # If v is a deeper ancestor, it's a back edge.
                          # If v is in a different branch, it's a cross edge.
                          # In undirected graph BFS, cross edges don't exist? 
                          # Actually, in undirected graphs, non-tree edges are back edges to ancestors.
                          # So v is an ancestor of u.
                          # Cycle length = dist[u] - dist[v] + 1?
                          # No.
                          # Path start->v->...->u->v.
                          # Length = (dist[u] - dist[v]) + 1.
                          # Let's verify.
                          # If v is parent of u, dist[u]=dist[v]+1. Length = 1+1=2? No, simple graph no multi-edges, so no 2-cycles.
                          # If v is grandparent, dist[u]=dist[v]+2. Path u->parent->v->u. Length 3.
                          # Formula: dist[u] - dist[v] + 1.
                          
                          # Wait, the BFS tree path from start to u is unique.
                          # The edge (u,v) closes the cycle.
                          # The cycle is start...v...u-v-start? No.
                          # The cycle is the path from v to u in the tree, plus the edge (u,v).
                          # Length = dist[u] - dist[v] + 1.
                          
                          # Is it possible that dist[u] + dist[v] + 1 is the cycle?
                          # That would be if v and u are in different branches.
                          # In undirected BFS, if v is visited and not parent, v must be an ancestor of u.
                          # Because if v was in a different branch, it would have been visited earlier, and u would have been visited from v?
                          # No.
                          # Standard result: In undirected graph, every non-tree edge connects a node to an ancestor.
                          
                          cycle_len = dist[u] - dist[v] + 1
                          if cycle_len < g:
                              # Reconstruct edges
                              edges_in_cycle = []
                              curr = u
                              while curr != v:
                                  p = parent[curr]
                                  edges_in_cycle.append((min(curr, p), max(curr, p)))
                                  curr = p
                              edges_in_cycle.append((min(u, v), max(u, v)))
                              return cycle_len, edges_in_cycle
          return None, None

      def get_girth(adj, n, g):
          # Just returns the length of the shortest cycle found
          res, _ = find_short_cycle(adj, n, g)
          return res if res is not None else g # Or larger

      edges = generate_random_4reg(n)
      adj = build_adj(n, edges)
      
      while time.time() < max_time:
          cycle_len, cycle_edges = find_short_cycle(adj, n, g)
          if cycle_len is None:
              return n, edges
              
          # Pick two random edges from the cycle
          if len(cycle_edges) < 2:
              # Should not happen for cycle_len >= 3
              break
              
          e1, e2 = random.sample(cycle_edges, 2)
          
          # Try to switch
          # e1 = (u1, v1), e2 = (u2, v2)
          # Remove e1, e2
          # Add (u1, u2), (v1, v2) or (u1, v2), (v1, u2)
          
          u1, v1 = e1
          u2, v2 = e2
          
          # Option 1: (u1, u2) and (v1, v2)
          new_e1 = (min(u1, u2), max(u1, u2))
          new_e2 = (min(v1, v2), max(v1, v2))
          
          valid = True
          if new_e1 == new_e2: valid = False
          if new_e1 in edges or new_e2 in edges: valid = False
          if u1 == u2 or v1 == v2: valid = False # Loop check (implicitly handled by min/max if distinct, but u1==u2 means loop)
          
          if valid:
              # Apply
              edges.remove(e1)
              edges.remove(e2)
              edges.add(new_e1)
              edges.add(new_e2)
              
              # Rebuild adj? Or update?
              # For simplicity, rebuild. n=320 is small.
              adj = build_adj(n, edges)
              
              # Check new girth
              new_cycle_len, _ = find_short_cycle(adj, n, g)
              new_girth = new_cycle_len if new_cycle_len is not None else g
              
              # Accept if better or random
              if new_girth > cycle_len:
                  pass # Accepted
              elif new_girth == cycle_len:
                  if random.random() > 0.5:
                      # Reject
                      edges.remove(new_e1)
                      edges.remove(new_e2)
                      edges.add(e1)
                      edges.add(e2)
                      adj = build_adj(n, edges)
              else:
                  # Reject
                  edges.remove(new_e1)
                  edges.remove(new_e2)
                  edges.add(e1)
                  edges.add(e2)
                  adj = build_adj(n, edges)
                  
      return None, None
  