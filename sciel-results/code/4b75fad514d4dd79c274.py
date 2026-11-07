   def is_valid(edges):
       n = 15
       adj = [set() for _ in range(n)]
       for u, v in edges:
           adj[u].add(v)
           adj[v].add(u)
       
       deg = [len(adj[i]) for i in range(n)]
       
       # Check all paths of length 3
       for u in range(n):
           for v in adj[u]:
               for t in adj[v]:
                   if t == u: continue
                   # Path s-u-v-t
                   # s must be in adj[u] and s != v
                   for s in adj[u]:
                       if s == v: continue
                       if deg[s] == deg[t]:
                           return False
       return True
   