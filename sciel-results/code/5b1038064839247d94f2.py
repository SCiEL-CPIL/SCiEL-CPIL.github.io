def construct():
      n = 16
      edges = []
      # Construct complete bipartite graph K_{8,8}
      # This is the maximum edge count for a triangle-free graph on 16 nodes
      # which is required for edge-critical diameter 2 property.
      for u in range(8):
          for v in range(8, 16):
              edges.append([u, v])
      return edges
