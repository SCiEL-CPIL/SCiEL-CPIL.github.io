"""Few colors without any bicolored cycle - Backtracking solver."""
import random
import time
import sys

# Increase recursion limit just in case, though depth is 50
sys.setrecursionlimit(10000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']
ADJ = [[] for _ in range(N)]
for a, b in EDGES:
    ADJ[a].append(b)
    ADJ[b].append(a)

class UF:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def copy(self):
        new_uf = UF.__new__(UF)
        new_uf.parent = self.parent[:]
        new_uf.rank = self.rank[:]
        return new_uf

def get_pair_key(c1, c2):
    if c1 > c2:
        c1, c2 = c2, c1
    return (c1, c2)

def construct():
    deadline = time.time() + 50.0  # Safety margin
    rng = random.Random(42)
    
    # Try k=3 first, then k=4, etc.
    for k in [3, 4, 5]:
        if time.time() > deadline:
            break
        
        # Multiple restarts to avoid getting stuck in bad orderings
        for attempt in range(1000):
            if time.time() > deadline:
                break
            
            # Randomize vertex order to break symmetry?
            # MRV is dynamic, so initial order matters less, but let's shuffle colors
            color_order = list(range(k))
            rng.shuffle(color_order)
            
            # State:
            # coloring: list of -1 (uncolored) or color index
            # ufs: dict (c1, c2) -> UF object
            # We only create UFs for pairs that have at least one edge? 
            # Or just create all possible pairs for simplicity? 
            # Max pairs for k=5 is 10. 10 * 50 ints is small.
            ufs = {}
            for i in range(k):
                for j in range(i+1, k):
                    ufs[(i, j)] = UF(N)
            
            coloring = [-1] * N
            
            # Recursive Backtracking
            def solve(idx, colored_count):
                if time.time() > deadline:
                    return False
                
                if colored_count == N:
                    return True
                
                # MRV: Find uncolored vertex with fewest valid colors
                best_v = -1
                best_colors = None
                best_count = k + 1
                
                for v in range(N):
                    if coloring[v] != -1:
                        continue
                    
                    valid_colors = []
                    for c in color_order:
                        if is_valid_color(v, c):
                            valid_colors.append(c)
                    
                    if len(valid_colors) < best_count:
                        best_count = len(valid_colors)
                        best_v = v
                        best_colors = valid_colors
                        if best_count == 0:
                            return False
                        if best_count == 1:
                            break # Can't get better than 1
                    
                if best_v == -1:
                    return False # No uncolored vertices? Should be handled by colored_count
                
                for c in best_colors:
                    # Save state
                    old_color = coloring[best_v]
                    old_ufs = {key: uf.copy() for key, uf in ufs.items()}
                    
                    # Assign
                    coloring[best_v] = c
                    for u in ADJ[best_v]:
                        if coloring[u] != -1:
                            d = coloring[u]
                            if d == c:
                                continue
                            key = get_pair_key(c, d)
                            ufs[key].union(best_v, u)
                    
                    if solve(0, colored_count + 1):
                        return True
                    
                    # Restore state
                    coloring[best_v] = old_color
                    ufs.clear()
                    ufs.update(old_ufs)
                
                return False

            def is_valid_color(v, c):
                for u in ADJ[v]:
                    d = coloring[u]
                    if d == -1:
                        continue
                    if d == c:
                        return False
                    key = get_pair_key(c, d)
                    uf = ufs[key]
                    if uf.find(v) == uf.find(u):
                        return False
                return True

            if solve(0, 0):
                # Valid coloring found
                # Ensure types are Python int
                return [int(x) for x in coloring]
            
    # Fallback: Return a trivial valid coloring with many colors
    # This guarantees no bicolored cycles (each pair has at most 1 edge? No, 
    # if we use distinct colors, each pair has 0 edges, which is a forest).
    # But we want to minimize colors. 
    # If search fails, we return a valid one with N colors.
    return [int(i) for i in range(N)]

if __name__ == "__main__":
    res = construct()
    print("colors used:", len(set(res)))
