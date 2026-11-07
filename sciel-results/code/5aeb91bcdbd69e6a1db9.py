"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def construct():
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
            
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
            
        def copy(self):
            new_dsu = DSU(n)
            new_dsu.parent = self.parent[:]
            new_dsu.rank = self.rank[:]
            return new_dsu

    def solve_with_k_colors(k):
        coloring = [-1] * n
        # dsus will store DSU for each pair of colors (c1, c2) with c1 < c2
        # Key: (c1, c2)
        dsus = {}
        
        # Initialize DSUs for all pairs
        for i in range(k):
            for j in range(i+1, k):
                dsus[(i, j)] = DSU(n)
        
        def backtrack(idx):
            if idx == n:
                return True
            
            # Order vertices by degree? Static order is fine for now, 
            # but dynamic might be better. Let's stick to static for simplicity 
            # and rely on fast pruning.
            
            for c in range(k):
                # Check proper coloring
                valid = True
                for neighbor in adj[idx]:
                    if coloring[neighbor] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check bicolored cycles
                # We need to check for each color c' != c
                # If we add idx with color c, it connects to neighbors with color c'
                # This adds edges (idx, neighbor) to the subgraph of (c, c')
                # A cycle is formed if idx connects to two neighbors u1, u2 of color c'
                # that are already connected in the subgraph of (c, c')
                
                conflict = False
                # List of (neighbor, c') to union later
                unions_to_do = []
                
                for c_prime in range(k):
                    if c_prime == c:
                        continue
                    
                    key = (min(c, c_prime), max(c, c_prime))
                    dsu = dsus[key]
                    
                    # Find neighbors of idx with color c_prime
                    neighbors_of_c_prime = []
                    for neighbor in adj[idx]:
                        if coloring[neighbor] == c_prime:
                            neighbors_of_c_prime.append(neighbor)
                    
                    if len(neighbors_of_c_prime) >= 2:
                        # Check if any two are connected in dsu
                        # Note: dsu contains all previously assigned vertices of colors c and c_prime
                        # We need to check connectivity among the neighbors
                        first = neighbors_of_c_prime[0]
                        for other in neighbors_of_c_prime[1:]:
                            if dsu.find(first) == dsu.find(other):
                                conflict = True
                                break
                    elif len(neighbors_of_c_prime) == 1:
                        # No cycle formed yet, but we will union
                        unions_to_do.append((idx, neighbors_of_c_prime[0], key))
                    
                    if conflict:
                        break
                
                if conflict:
                    continue
                
                # Assign color
                coloring[idx] = c
                
                # Save DSU states for rollback
                dsu_states = {}
                for key in dsus:
                    dsu_states[key] = dsus[key].copy()
                
                # Perform unions
                for u, v, key in unions_to_do:
                    dsus[key].union(u, v)
                
                # Also, we must union idx with neighbors of color c? 
                # No, neighbors of color c are forbidden.
                # What about neighbors of other colors? 
                # We handled c_prime.
                # Are there edges to uncolored vertices? Yes, but they don't affect current DSU state 
                # until they are colored.
                
                if backtrack(idx + 1):
                    return True
                
                # Rollback
                coloring[idx] = -1
                for key in dsus:
                    dsus[key] = dsu_states[key]
                    
            return False
        
        return backtrack(0), coloring

    # Try 3 colors
    found, coloring = solve_with_k_colors(3)
    if found:
        return [int(c) for c in coloring]
    
    # Try 4 colors
    found, coloring = solve_with_k_colors(4)
    if found:
        return [int(c) for c in coloring]
        
    # Fallback
    return [int(i) for i in range(n)]
# EVOLVE-BLOCK-END
