def construct():
      import time
      import random
      
      deadline = time.time() + 110
      best_size = 0
      best_set = set()
      
      # Precomputation
      # edges: list of (u, v)
      # face_conflicts[e_idx]: list of tuples of 3 edge indices
      # Or better: for each edge e, list of (j, edge_a, edge_b, edge_c) where j is the face index
      # Actually, just list the 3 edges for each of the 6 faces.
      
      d = 7
      nverts = 1 << d
      all_edges = []
      edge_map = {} # (u,v) -> index
      
      for u in range(nverts):
          for i in range(d):
              v = u ^ (1 << i)
              if u < v:
                  all_edges.append((u, v))
                  edge_map[(u, v)] = len(all_edges) - 1
                  
      # For each edge, find the 3 other edges in each of the 6 faces
      # Edge (u,v) differs in bit i.
      # Face defined by bits i and j.
      # Vertices: u, v, u^1<<j, v^1<<j.
      # Edges: (u,v), (u, u^1<<j), (v, v^1<<j), (u^1<<j, v^1<<j).
      
      conflicts = [[] for _ in range(len(all_edges))]
      for idx, (u, v) in enumerate(all_edges):
          # Find bit i
          diff = u ^ v
          i = diff.bit_length() - 1
          for j in range(d):
              if j == i: continue
              # Other 3 edges
              u2 = u ^ (1 << j)
              v2 = v ^ (1 << j)
              e1 = (min(u, u2), max(u, u2))
              e2 = (min(v, v2), max(v, v2))
              e3 = (min(u2, v2), max(u2, v2))
              # Get indices
              idx1 = edge_map[e1]
              idx2 = edge_map[e2]
              idx3 = edge_map[e3]
              conflicts[idx].append((idx1, idx2, idx3))
      
      # Greedy Loop
      edges_indices = list(range(len(all_edges)))
      
      while time.time() < deadline:
          random.shuffle(edges_indices)
          chosen = [False] * len(all_edges)
          count = 0
          
          for idx in edges_indices:
              # Check if adding idx creates a C4
              ok = True
              for c in conflicts[idx]:
                  # If all 3 are chosen, we have a C4
                  if chosen[c[0]] and chosen[c[1]] and chosen[c[2]]:
                      ok = False
                      break
              
              if ok:
                  chosen[idx] = True
                  count += 1
          
          if count > best_size:
              best_size = count
              best_set = set(i for i, val in enumerate(chosen) if val)
              
              # Early exit if we hit upper bound?
              if best_size >= 308:
                  break
                  
      # Convert best_set to list of edges
      result = [list(all_edges[i]) for i in best_set]
      return result
