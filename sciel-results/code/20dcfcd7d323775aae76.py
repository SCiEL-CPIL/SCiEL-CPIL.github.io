import time
import sys
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_valid(coloring):
    """Check if coloring is valid: adjacent differ, and no bicolored cycles."""
    n = P['n']
    edges = P['edges']
    
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    colors_used = set(coloring)
    
    for c1 in colors_used:
        for c2 in colors_used:
            if c1 >= c2:
                continue
            
            parent = list(range(n))
            
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            
            def union(x, y):
                rx, ry = find(x), find(y)
                if rx == ry:
                    return True
                parent[rx] = ry
                return False
            
            has_cycle = False
            for u, v in edges:
                cu, cv = coloring[u], coloring[v]
                if (cu == c1 or cu == c2) and (cv == c1 or cv == c2):
                    if union(u, v):
                        has_cycle = True
                        break
            
            if has_cycle:
                return False
    
    return True

def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _find_coloring_fast(max_colors, deadline):
    n = P['n']
    adj = _build_adj()
    
    # Ordering: sort by degree descending (High degree first constrains more)
    order = list(range(n))
    order.sort(key=lambda x: len(adj[x]), reverse=True)
    
    coloring = [-1] * n
    used_colors = set()
    
    # Union-Find for each pair of colors
    # uf[c1][c2] stores parent array for subgraph G[c1 U c2]
    # We only need to maintain these for pairs that are active.
    # Since max_colors is small (3 or 4), we can pre-allocate.
    # But we need rollback capability.
    
    # Let's use a simpler approach: 
    # For each pair (c1, c2), maintain a list of "active vertices" and a DSU.
    # Actually, since we backtrack, we can just store the state of the DSU.
    
    # Optimization: 
    # We will maintain a dictionary `dsus` mapping (c1, c2) to a DSU object.
    # DSU object will support undo.
    
    class DSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []
            
        def find(self, x):
            while self.parent[x] != x:
                self.parent[x] = self.parent[self.parent[x]]
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False # Cycle
            # Union by rank
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            # Record change for undo
            self.history.append((ry, rx, self.rank[rx]))
            return True
        
        def undo(self, count):
            for _ in range(count):
                ry, rx, rank_rx = self.history.pop()
                self.parent[ry] = ry
                self.rank[rx] = rank_rx
        
        def connected(self, x, y):
            return self.find(x) == self.find(y)

    # Initialize DSUs for all pairs up to max_colors
    # But we only create them when needed? 
    # For simplicity and speed, pre-allocate for all pairs in 0..max_colors-1
    dsus = {}
    for c1 in range(max_colors):
        for c2 in range(max_colors):
            if c1 < c2:
                dsus[(c1, c2)] = DSU(n)

    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n:
            return True
        
        v = order[idx]
        
        for color in range(max_colors):
            # Check proper coloring
            valid = True
            for nb in adj[v]:
                if coloring[nb] == color:
                    valid = False
                    break
            if not valid:
                continue
            
            # Check bicolored cycles incrementally
            # We need to check pairs (color, c') for all c' in used_colors, c' != color
            # And also if color is new, we don't have pairs with it yet?
            # Wait, if we assign 'color' to v, we are adding v to the set of vertices with 'color'.
            # This affects subgraphs G[color U c'] for all existing c'.
            
            # We need to store the state of DSUs to rollback
            # Number of DSUs affected: len(used_colors)
            # For each affected DSU, we might perform multiple unions.
            
            affected_dsus = []
            unions_per_dsu = []
            
            ok = True
            for c_prime in used_colors:
                if c_prime == color:
                    continue
                
                key = (min(color, c_prime), max(color, c_prime))
                dsu = dsus[key]
                
                # Find neighbors of v with color c_prime
                neighbors_cp = [nb for nb in adj[v] if coloring[nb] == c_prime]
                
                # Check if any two neighbors are connected in current DSU
                # If yes, adding v creates a cycle
                # Note: v is not in the DSU yet.
                # If neighbors_cp has size >= 2, check connectivity between them.
                if len(neighbors_cp) >= 2:
                    # Check if any pair is connected
                    # Since we are adding v, which connects to all of them,
                    # if any two are connected, a cycle is formed.
                    # Actually, if they are all in the same component, cycle.
                    # If they are in different components, no cycle yet?
                    # Wait. If u1 and u2 are connected, path u1...u2.
                    # Add v. Edges v-u1, v-u2.
                    # Cycle v-u1-...-u2-v. Yes.
                    # So we just need to check if the neighbors are in the same component.
                    # It's enough to check if they are all in the same component?
                    # No, if u1, u2 connected, cycle.
                    # So check if any two are connected.
                    # This is equivalent to checking if the number of distinct components
                    # among neighbors_cp is 1? No.
                    # If neighbors are in components C1, C2.
                    # If |neighbors in C1| >= 2, cycle.
                    # So we just need to check if any component has >= 2 neighbors.
                    
                    comp_count = {}
                    for nb in neighbors_cp:
                        root = dsu.find(nb)
                        comp_count[root] = comp_count.get(root, 0) + 1
                    
                    if any(count >= 2 for count in comp_count.values()):
                        ok = False
                        break
                
                # If valid so far, we will perform unions
                # We need to record how many unions we do to undo later
                # We union v with each neighbor in neighbors_cp
                # But we haven't added v to the DSU parent array?
                # The DSU parent array is for vertices 0..n-1.
                # v is a vertex index.
                # So we can just union v with neighbors.
                
                affected_dsus.append(dsu)
                unions_per_dsu.append(len(neighbors_cp))
            
            if not ok:
                continue
            
            # Perform the unions
            # We must do this for all affected DSUs
            # We need to be careful: if we fail later, we must undo.
            
            # Assign color
            coloring[v] = color
            used_colors.add(color)
            
            # Perform unions
            for i, c_prime in enumerate([c for c in used_colors if c != color]):
                # Note: used_colors changed, so we need to map back to the original list
                # This is inefficient. Let's re-iterate the original used_colors list
                pass
            
            # Let's redo the union logic properly
            # We stored affected_dsus and unions_per_dsu in the check loop.
            # But the check loop iterated over `used_colors` (before adding `color`? No, `used_colors` is the set of colors assigned to *other* vertices).
            # `color` might be new or existing.
            # If `color` is existing, `used_colors` already contains it.
            # The loop `for c_prime in used_colors: if c_prime == color: continue` handles this.
            
            # So we just need to execute the unions that we "planned" in the check.
            # But we didn't store the neighbors.
            # Let's just re-iterate and do the unions.
            
            for c_prime in list(used_colors): # Use list to avoid modification issues if we were adding, but we aren't
                if c_prime == color:
                    continue
                key = (min(color, c_prime), max(color, c_prime))
                dsu = dsus[key]
                neighbors_cp = [nb for nb in adj[v] if coloring[nb] == c_prime]
                for nb in neighbors_cp:
                    if not dsu.union(v, nb):
                        # This should not happen if check passed, but just in case
                        # Rollback all unions done so far in this step
                        # This logic is getting complicated.
                        pass
                    # Record that we did a union
                    # We need to undo these later.
            
            # To make undo easy, let's just store the count of unions per DSU
            # and call undo(count).
            
            # Let's refine the "do it" part
            unions_done = []
            for c_prime in list(used_colors):
                if c_prime == color:
                    continue
                key = (min(color, c_prime), max(color, c_prime))
                dsu = dsus[key]
                neighbors_cp = [nb for nb in adj[v] if coloring[nb] == c_prime]
                for nb in neighbors_cp:
                    if not dsu.union(v, nb):
                        # Rollback
                        for d, cnt in reversed(unions_done):
                            d.undo(cnt)
                        coloring[v] = -1
                        used_colors.discard(color)
                        ok = False
                        break
                    unions_done.append((dsu, 1)) # Wait, undo takes count.
                if not ok: break
            
            if not ok:
                continue
            
            if backtrack(idx + 1):
                return True
            
            # Undo
            # We need to undo the unions in reverse order?
            # DSU undo is stack-based.
            # We just need to undo the total number of unions for each DSU.
            # But we interleaved unions for different DSUs?
            # No, we iterated c_prime, then neighbors.
            # So for a specific DSU, we did K unions.
            # We can just call undo(K) for each DSU.
            
            for c_prime in list(used_colors):
                if c_prime == color:
                    continue
                key = (min(color, c_prime), max(color, c_prime))
                dsu = dsus[key]
                neighbors_cp = [nb for nb in adj[v] if coloring[nb] == c_prime]
                dsu.undo(len(neighbors_cp))
            
            coloring[v] = -1
            used_colors.discard(color)
            
        return False

    if backtrack(0):
        return coloring
    return None

def construct():
    deadline = time.time() + 80
    
    # Try increasing numbers of colors
    for num_colors in range(3, 15):
        if time.time() > deadline - 5:
            break
        
        coloring = _find_coloring_fast(num_colors, deadline)
        if coloring is not None:
            if _check_valid(coloring):
                return [int(c) for c in coloring]
    
    # Fallback: identity coloring
    return [int(i) for i in range(P['n'])]

if __name__ == '__main__':
    result = construct()
    print(result)
    print(len(set(result)))
