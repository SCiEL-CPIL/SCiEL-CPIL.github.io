     def construct():
         n = P['n']
         base_edges = set()
         for e in P['edges']:
             u, v = e
             if u > v: u, v = v, u
             base_edges.add((u, v))
         
         E = [i for i in range(n) if i % 2 == 0]
         O = [i for i in range(n) if i % 2 == 1]
         
         added = []
         
         # Hub 0 (in E) connects to all O except 1
         for o in O:
             if o != 1:
                 u, v = 0, o
                 if u > v: u, v = v, u
                 if (u, v) not in base_edges:
                     added.append([u, v])
                     
         # Hub 1 (in O) connects to all E except 0 and 2
         for e in E:
             if e not in [0, 2]:
                 u, v = 1, e
                 if u > v: u, v = v, u
                 if (u, v) not in base_edges:
                     added.append([u, v])
             
         return added
     