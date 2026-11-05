import time
import random

def construct():
       start_time = time.time()
       deadline = start_time + 110 # Safety margin
       
       n = 40
       best_edges = []
       best_count = 0
       
       # Helper to check C6 creation
       def creates_c6(u, v, adj, neighbors):
           # DFS from v to find path of length 4 to a neighbor of u
           # Path: v -> ... -> w -> u
           # Depth 4 means 4 edges from v to w.
           # So v(0) - x(1) - y(2) - z(3) - w(4)
           # Check if u in N(w)
           
           def dfs(node, depth, visited):
               if depth == 4:
                   return (1 << u) & adj[node]
               for nxt in neighbors[node]:
                   if nxt in visited or nxt == u:
                       continue
                   if dfs(nxt, depth + 1, visited | {nxt}):
                       return True
               return False
           
           return dfs(v, 0, {v})
       
       def count_edges(adj):
           return sum(bin(adj[i]).count('1') for i in range(n)) // 2
           
       def get_edges(adj):
           edges = []
           for i in range(n):
               for j in range(i+1, n):
                   if (1 << j) & adj[i]:
                       edges.append([i, j])
           return edges
       
       for restart in range(1000):
           if time.time() > deadline:
               break
           
           # Initialize random graph
           # p = 0.1 to 0.15
           p = 0.12
           adj = [0] * n
           neighbors = [set() for _ in range(n)]
           
           for i in range(n):
               for j in range(i+1, n):
                   if random.random() < p:
                       adj[i] |= (1 << j)
                       adj[j] |= (1 << i)
                       neighbors[i].add(j)
                       neighbors[j].add(i)
           
           # Clean phase: Remove edges that are part of C6
           # We need to find existing C6s.
           # This is harder than checking if an *added* edge creates a C6.
           # If we start with a random graph, it likely has many C6s.
           # We can iteratively find a C6 and remove an edge.
           # How to find a C6?
           # Pick a random edge (u,v). Check if it's part of a C6.
           # A C6 containing (u,v) is u-v-x-y-z-w-u.
           # So we need a path of length 4 from v to u? No, from v to u avoiding (u,v)?
           # The rest of the cycle is a path of length 4 from v to u in G \ {(u,v)}.
           # So we can reuse the logic.
           
           def has_c6_in_graph(adj, neighbors):
               # Try random edges
               edges_list = []
               for i in range(n):
                   for j in range(i+1, n):
                       if (1 << j) & adj[i]:
                           edges_list.append((i,j))
               
               random.shuffle(edges_list)
               for u, v in edges_list:
                   # Check if (u,v) is in a C6
                   # Temporarily remove (u,v)?
                   # Or just check if there is a path of length 4 from v to u in G \ {(u,v)}
                   # The DFS above checks path in G.
                   # If the path uses (u,v), it's not a simple path in G\{(u,v)}.
                   # But the DFS forbids revisiting nodes and u is only at end.
                   # The path is v-x-y-z-w-u.
                   # Does this path use the edge (u,v)?
                   # The edge (u,v) connects u and v.
                   # The path starts at v and ends at u.
                   # It does not use the edge (u,v) internally.
                   # So if `creates_c6(u, v, adj, neighbors)` returns True,
                   # it means there is a path v-x-y-z-w-u.
                   # Together with edge (u,v), this forms a C6.
                   # So we can just use the same function!
                   
                   if creates_c6(u, v, adj, neighbors):
                       return (u, v)
               return None
           
           # Clean loop
           changed = True
           while changed:
               if time.time() > deadline:
                   break
               edge_in_c6 = has_c6_in_graph(adj, neighbors)
               if edge_in_c6 is None:
                   changed = False
               else:
                   u, v = edge_in_c6
                   # Remove a random edge from the C6?
                   # We only know (u,v) is in a C6.
                   # We can just remove (u,v).
                   adj[u] &= ~(1 << v)
                   adj[v] &= ~(1 << u)
                   neighbors[u].remove(v)
                   neighbors[v].remove(u)
           
           # Add phase: Greedily add edges
           # Iterate through all missing edges
           for i in range(n):
               for j in range(i+1, n):
                   if not ((1 << j) & adj[i]):
                       if not creates_c6(i, j, adj, neighbors):
                           # Add edge
                           adj[i] |= (1 << j)
                           adj[j] |= (1 << i)
                           neighbors[i].add(j)
                           neighbors[j].add(i)
           
           # Update best
           current_edges = get_edges(adj)
           if len(current_edges) > best_count:
               best_count = len(current_edges)
               best_edges = current_edges
               
       return best_edges
