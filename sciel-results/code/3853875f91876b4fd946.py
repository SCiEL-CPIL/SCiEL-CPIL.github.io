    class Graph:
        def __init__(self, n):
            self.n = n
            self.adj = [set() for _ in range(n)]
            self.deg = [0] * n
            self.edge_set = set()
            self.edges = []
            self.deg_set = [set() for _ in range(n)]

        def add_edge(self, u, v):
            if u == v or (u, v) in self.edge_set or (v, u) in self.edge_set:
                return False
            
            # Check validity
            if self.deg_set[u] & self.deg_set[v]:
                return False
            
            old_deg_u = self.deg[u]
            old_deg_v = self.deg[v]
            
            # Update adjacency
            self.adj[u].add(v)
            self.adj[v].add(u)
            
            # Update degrees
            self.deg[u] += 1
            self.deg[v] += 1
            new_deg_u = self.deg[u]
            new_deg_v = self.deg[v]
            
            # Update deg_sets
            # 1. u and v are new neighbors of each other
            self.deg_set[u].add(new_deg_v)
            self.deg_set[v].add(new_deg_u)
            
            # 2. Neighbors of u (excluding v) see u's degree change
            for x in self.adj[u]:
                if x == v: continue
                self.deg_set[x].discard(old_deg_u)
                self.deg_set[x].add(new_deg_u)
                
            # 3. Neighbors of v (excluding u) see v's degree change
            for y in self.adj[v]:
                if y == u: continue
                self.deg_set[y].discard(old_deg_v)
                self.deg_set[y].add(new_deg_v)
                
            self.edge_set.add((u, v))
            self.edges.append([u, v])
            return True

        def remove_edge(self, u, v):
            if (u, v) not in self.edge_set and (v, u) not in self.edge_set:
                return False
            
            # Ensure u, v order doesn't matter for removal logic, but let's normalize
            # Actually, edge_set contains (min, max) or just (u,v)? 
            # In add_edge, we add (u,v). If we call remove_edge(v,u), 
            # (v,u) might not be in edge_set if we only added (u,v).
            # Let's make edge_set store sorted tuples or check both.
            # Current code: self.edge_set.add((u, v)).
            # If add_edge(1, 2) is called, (1, 2) is in set.
            # If remove_edge(2, 1) is called, (2, 1) is NOT in set.
            # Fix: check both or normalize.
            
            if (u, v) in self.edge_set:
                pass
            elif (v, u) in self.edge_set:
                u, v = v, u
            else:
                return False

            old_deg_u = self.deg[u]
            old_deg_v = self.deg[v]
            
            # Update adjacency
            self.adj[u].remove(v)
            self.adj[v].remove(u)
            
            # Update degrees
            self.deg[u] -= 1
            self.deg[v] -= 1
            new_deg_u = self.deg[u]
            new_deg_v = self.deg[v]
            
            # Update deg_sets
            # 1. u and v are no longer neighbors
            self.deg_set[u].discard(old_deg_v)
            self.deg_set[v].discard(old_deg_u)
            
            # 2. Neighbors of u (now excluding v) see u's degree revert
            for x in self.adj[u]:
                self.deg_set[x].discard(old_deg_u)
                self.deg_set[x].add(new_deg_u)
                
            # 3. Neighbors of v (now excluding u) see v's degree revert
            for y in self.adj[v]:
                self.deg_set[y].discard(old_deg_v)
                self.deg_set[y].add(new_deg_v)
                
            self.edge_set.discard((u, v))
            if (v, u) in self.edge_set:
                self.edge_set.discard((v, u))
            
            # Remove from edges list
            # This is O(E), but E is small.
            for i in range(len(self.edges)):
                if self.edges[i] == [u, v] or self.edges[i] == [v, u]:
                    self.edges.pop(i)
                    break
                    
            return True

        def can_add(self, u, v):
            if u == v or (u, v) in self.edge_set or (v, u) in self.edge_set:
                return False
            return not (self.deg_set[u] & self.deg_set[v])
