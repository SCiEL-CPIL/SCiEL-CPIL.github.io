   while time < deadline:
       # Perturb
       ...
       # Local Opt
       changed = True
       while changed:
           changed = False
           for v in range(n):
               if not in_set[v]:
                   if np.sum(in_set[adj[v]]) == 0:
                       in_set[v] = True
                       changed = True
   