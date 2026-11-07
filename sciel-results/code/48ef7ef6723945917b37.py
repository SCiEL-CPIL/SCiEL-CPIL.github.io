import numpy as np
from collections import deque

P = {'n': 24, 'edges': [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [11, 12], [12, 13], [13, 14], [14, 15], [15, 16], [16, 17], [17, 18], [18, 19], [19, 20], [20, 21], [21, 22], [22, 23]], 'diameter': 4}

def construct():
       n = P['n']
       base_edges = P['edges']
       
       # State
       adj = [[] for _ in range(n)]
       edge_set = set()
       for u, v in base_edges:
           adj[u].append(v)
           adj[v].append(u)
           edge_set.add((min(u, v), max(u, v)))
           
       added_edges = []

       def get_dist_matrix():
           # Returns numpy array of distances
           dist = np.full((n, n), 999)
           for i in range(n):
               dist[i][i] = 0
               q = deque([i])
               while q:
                   node = q.popleft()
                   for nei in adj[node]:
                       if dist[i][nei] > dist[i][node] + 1:
                           dist[i][nei] = dist[i][node] + 1
                           q.append(nei)
           return dist

       def get_diameter(dist_mat):
           return np.max(dist_mat)

       def would_create_triangle(u, v):
           # Check if adding (u,v) creates a triangle
           # Common neighbor?
           # Since n is small, set intersection is fast
           set_u = set(adj[u])
           set_v = set(adj[v])
           if set_u & set_v:
               return True
           return False

       def try_add_edge(u, v):
           if (u, v) in edge_set or (v, u) in edge_set:
               return False
           if u == v:
               return False
           if would_create_triangle(u, v):
               return False
           
           adj[u].append(v)
           adj[v].append(u)
           edge_set.add((min(u, v), max(u, v)))
           added_edges.append([u, v])
           return True

       def try_remove_edge(u, v):
           # Remove edge
           adj[u].remove(v)
           adj[v].remove(u)
           edge_set.remove((min(u, v), max(u, v)))
           added_edges.remove([u, v])
           
           # Check diameter
           dist_mat = get_dist_matrix()
           d = np.max(dist_mat)
           if d > 4:
               # Restore
               adj[u].append(v)
               adj[v].append(u)
               edge_set.add((min(u, v), max(u, v)))
               added_edges.append([u, v])
               return False
               
           # Triangle-free is guaranteed if it was before and we only removed an edge
           return True

       # Phase 1: Greedy Add
       while True:
           dist_mat = get_dist_matrix()
           d = np.max(dist_mat)
           if d <= 4:
               break
               
           best_score = -1
           best_edge = None
           
           # Vectorized scoring
           # For each candidate edge (u, v), calculate new distances
           # new_dist[i, j] = min(dist[i, j], dist[i, u] + 1 + dist[v, j], dist[i, v] + 1 + dist[u, j])
           
           for u in range(n):
               for v in range(u + 1, n):
                   if (u, v) in edge_set:
                       continue
                   if would_create_triangle(u, v):
                       continue
                   
                   # Calculate new distance matrix efficiently
                   # dist[i, u] is column u of dist_mat
                   # dist[v, j] is row v of dist_mat
                   # path via u->v: dist[i, u] + 1 + dist[v, j]
                   # path via v->u: dist[i, v] + 1 + dist[u, j]
                   
                   # We want to count how many pairs have new_dist > 4
                   # Or simply, does it reduce the diameter?
                   
                   # Let's compute the max new distance
                   # new_dist = np.minimum(dist_mat, dist_mat[:, u:u+1] + 1 + dist_mat[v:v+1, :])
                   # new_dist = np.minimum(new_dist, dist_mat[:, v:v+1] + 1 + dist_mat[u:u+1, :])
                   
                   # This is O(n^2) per candidate. 276 * 576 = 158,000 ops.
                   # NumPy will handle this fast.
                   
                   cand_dist = np.minimum(dist_mat, dist_mat[:, u:u+1] + 1 + dist_mat[v:v+1, :])
                   cand_dist = np.minimum(cand_dist, dist_mat[:, v:v+1] + 1 + dist_mat[u:u+1, :])
                   
                   # Score: minimize max distance, then minimize number of pairs > 4
                   new_max = np.max(cand_dist)
                   new_bad = np.sum(cand_dist > 4)
                   
                   # We want to minimize new_max, then new_bad
                   # Since we break when max <= 4, if new_max <= 4, it's a great candidate
                   # We can prioritize candidates that bring max down to 4
                   
                   if new_max < d: # Strict improvement in diameter
                       # Tie breaker: fewer bad pairs
                       if new_max < best_score or (new_max == best_score and new_bad < best_bad):
                           best_score = new_max
                           best_bad = new_bad
                           best_edge = (u, v)
                   elif new_max == d:
                       # Try to reduce bad pairs if diameter doesn't change?
                       # Usually we want to reduce diameter first.
                       # But if we can't reduce diameter, maybe reduce bad pairs?
                       # The loop breaks when d <= 4. So if d > 4, we MUST reduce d eventually.
                       # If no edge reduces d, we are stuck?
                       # Actually, if d > 4, there must be an edge that reduces the diameter 
                       # unless the graph is already "maximally" connected for triangle-free?
                       # But we can always add more edges.
                       pass
           
           if best_edge is None:
               # If no edge reduces diameter, try to reduce "bad" pairs?
               # Or just break to avoid infinite loop
               # But we should find an edge.
               # Let's fallback to any edge that reduces bad pairs
               best_score = np.sum(dist_mat > 4)
               best_edge = None
               for u in range(n):
                   for v in range(u + 1, n):
                       if (u, v) in edge_set:
                           continue
                       if would_create_triangle(u, v):
                           continue
                       cand_dist = np.minimum(dist_mat, dist_mat[:, u:u+1] + 1 + dist_mat[v:v+1, :])
                       cand_dist = np.minimum(cand_dist, dist_mat[:, v:v+1] + 1 + dist_mat[u:u+1, :])
                       new_bad = np.sum(cand_dist > 4)
                       if new_bad < best_score:
                           best_score = new_bad
                           best_edge = (u, v)
                           
               if best_edge is None:
                   break
           
           u, v = best_edge
           try_add_edge(u, v)

       # Phase 2: Greedy Remove
       changed = True
       while changed:
           changed = False
           for edge in list(added_edges):
               u, v = edge
               if try_remove_edge(u, v):
                   changed = True

       # Phase 3: Local Search
       for _ in range(3):
           # Add
           while True:
               dist_mat = get_dist_matrix()
               d = np.max(dist_mat)
               if d <= 4:
                   break
               best_score = -1
               best_bad = 999
               best_edge = None
               
               for u in range(n):
                   for v in range(u + 1, n):
                       if (u, v) in edge_set:
                           continue
                       if would_create_triangle(u, v):
                           continue
                       
                       cand_dist = np.minimum(dist_mat, dist_mat[:, u:u+1] + 1 + dist_mat[v:v+1, :])
                       cand_dist = np.minimum(cand_dist, dist_mat[:, v:v+1] + 1 + dist_mat[u:u+1, :])
                       
                       new_max = np.max(cand_dist)
                       new_bad = np.sum(cand_dist > 4)
                       
                       if new_max < d:
                           if new_max < best_score or (new_max == best_score and new_bad < best_bad):
                               best_score = new_max
                               best_bad = new_bad
                               best_edge = (u, v)
                               
               if best_edge is None:
                   break
               
               u, v = best_edge
               try_add_edge(u, v)
               
           # Remove
           changed = True
           while changed:
               changed = False
               for edge in list(added_edges):
                   u, v = edge
                   if try_remove_edge(u, v):
                       changed = True

       return added_edges
