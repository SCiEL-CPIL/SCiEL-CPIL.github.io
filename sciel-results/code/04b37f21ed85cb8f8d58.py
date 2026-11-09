import random
import time

def construct():
       n = 15
       # Target degrees: 14, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
       # Sum = 118, Edges = 59
       
       # Try to construct a graph with these degrees
       # Use Havel-Hakimi or similar, or just random generation + repair
       
       # We will try a specific construction or a quick search
       
       # Start with complete graph
       adj = [[0]*n for _ in range(n)]
       for i in range(n):
           for j in range(i+1, n):
               adj[i][j] = 1
               adj[j][i] = 1
               
       # Degrees are all 14
       # We need to reduce degrees to targets
       targets = [14, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
       
       # Sort vertices by target degree descending?
       # v0:14, v1:13, v2:13, v3:12, ...
       
       # We need to remove edges.
       # v0: remove 0
       # v1: remove 1
       # v2: remove 1
       # v3: remove 2
       # ...
       
       # Let's try to remove edges specifically to avoid v1-v2 path
       # v1, v2 are the degree 13 vertices.
       
       # If we remove (v1, v2), then v1, v2 lose 1 degree each.
       # v1 needs 1 removal, v2 needs 1 removal.
       # So remove (v1, v2).
       adj[1][2] = 0
       adj[2][1] = 0
       
       # Now v1, v2 have degree 13.
       # v0 has degree 14.
       # v3 needs degree 12 (remove 2 from 14).
       # v4 needs degree 11 (remove 3).
       # ...
       
       # We need to remove more edges.
       # To be safe, let's just use a local search to find ANY valid graph with 59 edges.
       
       def get_edges(adj):
           edges = []
           for i in range(n):
               for j in range(i+1, n):
                   if adj[i][j]:
                       edges.append([i, j])
           return edges
           
       def is_valid(edges):
           deg = [0]*n
           for u, v in edges:
               deg[u] += 1
               deg[v] += 1
               
           # Check all paths s-u-v-t
           # For each edge (u,v), check all s in N(u)\{v}, t in N(v)\{u}
           # If s != t and deg[s] == deg[t], return False
           
           # Build adjacency list for speed
           adj_list = [[] for _ in range(n)]
           for u, v in edges:
               adj_list[u].append(v)
               adj_list[v].append(u)
               
           for u, v in edges:
               for s in adj_list[u]:
                   if s == v: continue
                   for t in adj_list[v]:
                       if t == u: continue
                       if s == t: continue
                       if deg[s] == deg[t]:
                           return False
           return True
       
       # Random restarts
       start_time = time.time()
       best_edges = None
       
       # Initial guess: K7,8 is valid with 56 edges.
       # Let's try to build the 59 edge graph.
       
       # We'll use a simple hill climbing on the edge set.
       # Start with a random graph of 59 edges?
       # Or start with K15 and remove 46 edges randomly, then repair degrees?
       
       # Better: Start with the degree sequence construction.
       
       # Let's try to construct the specific degree sequence graph.
       # v0..v14
       # Targets: 14, 13, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1
       
       # We'll use a greedy approach to remove edges from K15.
       # Removals needed:
       # v0: 0
       # v1: 1
       # v2: 1
       # v3: 2
       # v4: 3
       # v5: 4
       # v6: 5
       # v7: 6
       # v8: 7
       # v9: 8
       # v10: 9
       # v11: 10
       # v12: 11
       # v13: 12
       # v14: 13
       
       # Total removals: 92 (46 edges)
       
       # We want to avoid path v1-x-y-v2.
       # If we remove (v1, v2), we satisfy the "1 removal" for both.
       # Then v1, v2 are not connected.
       # Common neighbors: all others.
       # We need to remove edges among others to break paths.
       
       # This is getting complex. Let's just run a local search.
       
       def random_graph_with_degree_sequence(targets):
           # Simple random generation might be slow.
           # We'll just return None and use hill climbing from K7,8.
           pass
           
       # Hill climbing from K7,8
       edges = []
       for u in range(7):
           for v in range(7, 15):
               edges.append([u, v])
               
       # Try to add edges
       # Current 56. Target 59.
       # Add 3 random intra-partition edges and check validity.
       # If valid, we have 59.
       
       # But we know intra-partition edges in K7,8 cause issues if degrees are uniform.
       # However, if we ALSO modify the degrees of the other partition, it might work.
       
       # Let's just try to find a valid graph with 59 edges via random search.
       
       import itertools
       
       # This might be too slow if we do it from scratch.
       # Let's stick to the K7,8 base and see if we can find a valid 59-edge graph.
       
       # Actually, the inspiration suggests distinct degrees.
       # Let's try to construct the distinct degree graph explicitly.
       
       # We'll use a known algorithm for degree sequence realization.
       
       # For now, return K7,8 (56) if we can't find better quickly.
       # But we should try to improve.
       
       # Let's try a few random perturbations of K7,8.
       
       candidates = []
       for _ in range(1000):
           e = edges[:]
           # Add 3 random edges from A-A or B-B
           # A: 0-6, B: 7-14
           # Possible intra edges:
           intra_a = [(i, j) for i in range(7) for j in range(i+1, 7)]
           intra_b = [(i, j) for i in range(7, 15) for j in range(i+1, 15)]
           
           # Pick 3 random
           added = set()
           while len(added) < 3:
               if random.random() < 0.5:
                   e_try = random.choice(intra_a)
               else:
                   e_try = random.choice(intra_b)
               if e_try not in added and e_try not in [(u,v) for u,v in e] and (e_try[1], e_try[0]) not in [(u,v) for u,v in e]:
                   added.add(e_try)
                   
           if len(added) < 3: continue
           
           for u, v in added:
               e.append([u, v])
               
           if is_valid(e):
               return e
               
       return edges
