import time
import sys
from collections import defaultdict

# Increase recursion limit just in case
sys.setrecursionlimit(2000)

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _check_coloring(coloring, edges):
    """Verify the coloring is valid."""
    n = len(coloring)
    for u, v in edges:
        if coloring[u] == coloring[v]:
            return False
    
    # Check bicolored cycles using Union-Find for each pair
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = coloring[u], coloring[v]
        if cu != cv:
            pair = (min(cu, cv), max(cu, cv))
            pair_edges[pair].append((u, v))
    
    for pair, elist in pair_edges.items():
        parent = list(range(n))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False
            parent[rx] = ry
            return True
        for u, v in elist:
            if not union(u, v):
                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s limit, 5s safety margin
    
    n = P['n']
    edges = P['edges']
    
    # Build adjacency list
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Order vertices by degree (descending) for better pruning
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    
    # Precompute neighbor lists in the new order? No, just use original indices.
    # But we need to check neighbors' colors.
    
    # We will try k = 3, 4, 5, ...
    # Why start at 3? k=2 is impossible for graphs with cycles (which this has).
    
    for k in range(3, 15):
        if time.time() > deadline - 5:
            break
            
        # State for backtracking
        coloring = [-1] * n
        
        # Union-Find for each pair of colors
        # There are k*(k-1)/2 pairs.
        # We store parent arrays for each pair.
        # To allow undo, we maintain a global undo stack.
        # Each entry in undo stack: (pair_idx, u, v, parent_before, rank_before)
        # Actually, simpler: store the changes to parent and rank.
        
        num_pairs = k * (k - 1) // 2
        pair_to_idx = {}
        idx = 0
        for c1 in range(k):
            for c2 in range(c1 + 1, k):
                pair_to_idx[(c1, c2)] = idx
                idx += 1
        
        # Initialize UFs
        # parents[pair_idx] is a list of n ints
        # ranks[pair_idx] is a list of n ints
        parents = [list(range(n)) for _ in range(num_pairs)]
        ranks = [[0] * n for _ in range(num_pairs)]
        
        def find(pi, x):
            p = parents[pi]
            while p[x] != x:
                p[x] = p[p[x]]
                x = p[x]
            return x
        
        def union(pi, u, v):
            p = parents[pi]
            r = ranks[pi]
            ru = find(pi, u)
            rv = find(pi, v)
            if ru == rv:
                return False
            # Union by rank
            if r[ru] < r[rv]:
                ru, rv = rv, ru
            # Record undo info: (pi, rv, ru, r[ru])
            # We need to restore p[rv] and r[ru]
            # p[rv] was ru (after union). Before it was rv (or something else if path compressed? No, we don't path compress in undoable UF usually, or we do but it's messy. Let's avoid path compression in the undoable part to keep undo simple, or store full state? No, too big.)
            # Standard reversible UF:
            # We don't use path compression. We use union by rank.
            # find() does not compress path.
            def find_no_comp(pi, x):
                p = parents[pi]
                while p[x] != x:
                    x = p[x]
                return x
            
            ru = find_no_comp(pi, u)
            rv = find_no_comp(pi, v)
            if ru == rv:
                return False
            
            if r[ru] < r[rv]:
                ru, rv = rv, ru
            
            # Perform union
            p[rv] = ru
            if r[ru] == r[rv]:
                r[ru] += 1
            
            # Push to undo stack
            undo_stack.append((pi, rv, ru, r[ru] - (1 if r[ru] == r[rv] else 0))) 
            # Wait, we need to restore r[ru] to its old value.
            # Let's store the old value.
            # Re-do:
            undo_stack.pop() # remove wrong push
            
            old_rank_ru = r[ru]
            # If we incremented rank, old_rank_ru is the current - 1?
            # No, we increment AFTER checking.
            # So:
            # 1. Check ranks.
            # 2. Swap if needed.
            # 3. Store old_rank_ru (current value).
            # 4. p[rv] = ru.
            # 5. If r[ru] == r[rv], r[ru] += 1.
            
            # Let's rewrite union properly with undo stack.
            
            return True
            
        # Let's implement a cleaner Reversible UF class inside the loop to avoid global state mess.
        
        class RevUF:
            def __init__(self, n):
                self.parent = list(range(n))
                self.rank = [0] * n
                self.stack = []
                
            def find(self, x):
                # No path compression for easy undo
                while self.parent[x] != x:
                    x = self.parent[x]
                return x
            
            def union(self, u, v):
                ru = self.find(u)
                rv = self.find(v)
                if ru == rv:
                    return False
                if self.rank[ru] < self.rank[rv]:
                    ru, rv = rv, ru
                # Record state
                self.stack.append((rv, ru, self.rank[ru]))
                self.parent[rv] = ru
                if self.rank[ru] == self.rank[rv]:
                    self.rank[ru] += 1
                return True
            
            def undo(self):
                if not self.stack:
                    return
                rv, ru, old_rank_ru = self.stack.pop()
                self.parent[rv] = rv
                self.rank[ru] = old_rank_ru

        # Create UFs for all pairs
        ufs = [RevUF(n) for _ in range(num_pairs)]
        
        def backtrack(pos):
            if time.time() > deadline - 2:
                return False
            if pos == n:
                return True
            v = order[pos]
            
            # Try all colors
            for c in range(k):
                # Check adjacency
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Assign color
                coloring[v] = c
                
                # Check bicolored cycles
                snapshots = []
                failed = False
                for nb in adj[v]:
                    d = coloring[nb]
                    if d == -1:
                        continue
                    if d == c:
                        continue
                    # Get UF for pair (c, d)
                    c_min, c_max = (c, d) if c < d else (d, c)
                    pi = pair_to_idx[(c_min, c_max)]
                    uf = ufs[pi]
                    snap = len(uf.stack)
                    if not uf.union(v, nb):
                        # Cycle found
                        failed = True
                        # Undo previous unions in this step
                        for uf_prev, snap_prev in snapshots:
                            while len(uf_prev.stack) > snap_prev:
                                uf_prev.undo()
                        break
                    snapshots.append((uf, snap))
                
                if not failed:
                    if backtrack(pos + 1):
                        return True
                
                # Undo
                coloring[v] = -1
                for uf_prev, snap_prev in snapshots:
                    while len(uf_prev.stack) > snap_prev:
                        uf_prev.undo()
            
            return False
        
        if backtrack(0):
            # Verify
            if _check_coloring(coloring, edges):
                return [int(c) for c in coloring]
        
    # Fallback: Greedy
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    coloring = [0] * n
    for v in order:
        used = set(coloring[nb] for nb in adj[v])
        for c in range(50):
            if c not in used:
                coloring[v] = c
                break
    return [int(c) for c in coloring]
