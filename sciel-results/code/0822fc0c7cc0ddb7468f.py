   def construct():
       n = 35
       edges = []
       # 7 disjoint K4s
       for b in range(7):
           base = 4 * b
           for i in range(4):
               for j in range(i + 1, 4):
                   edges.append([base + i, base + j])
       # 1 K7
       base = 28
       for i in range(7):
           for j in range(i + 1, 7):
               edges.append([base + i, base + j])
       return {'n': n, 'edges': edges}
   