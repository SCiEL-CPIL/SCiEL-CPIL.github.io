def construct():
      edges = []
      # Component 1: K3,3 on 0..5
      # A = {0, 1, 2}, B = {3, 4, 5}
      for u in [0, 1, 2]:
          for v in [3, 4, 5]:
              edges.append([u, v])
      
      # Component 2: K3,3 on 6..11
      # A = {6, 7, 8}, B = {9, 10, 11}
      for u in [6, 7, 8]:
          for v in [9, 10, 11]:
              edges.append([u, v])
              
      return edges
