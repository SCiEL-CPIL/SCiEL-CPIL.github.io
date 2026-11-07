import time
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

class ReversibleUnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.undo_stack = []
    
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False
        self.undo_stack.append((rx, ry, self.parent[ry], self.rank[rx], self.rank[ry]))
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1
        return True
    
    def undo(self):
        if not self.undo_stack:
            return
        rx, ry, old_parent_ry, old_rank_rx, old_rank_ry = self.undo_stack.pop()
        self.parent[ry] = old_parent_ry
        self.rank[rx] = old_rank_rx
        self.rank[ry] = old_rank_ry
    
    def snapshot(self):
        return len(self.undo_stack)
    
    def rollback(self, snap):
        while len(self.undo_stack) > snap:
            self.undo()

def construct():
    start_time = time.time()
    deadline = start_time + 80
    n = P['n']
    edges = P['edges']
    
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
        
    # Try k=4 first, then 5, etc.
    for k in range(4, 11):
        if time.time() > deadline - 5:
            break
            
        # Ordering: High degree first (all are 4, so arbitrary or by index)
        # Let's use a static order 0..49
        order = list(range(n))
        
        coloring = [-1] * n
        pair_ufs = {}
        
        def get_uf(c1, c2):
            pair = (min(c1, c2), max(c1, c2))
            if pair not in pair_ufs:
                pair_ufs[pair] = ReversibleUnionFind(n)
            return pair_ufs[pair]
            
        def try_assign(v, c):
            # Check adjacency
            for nb in adj[v]:
                if coloring[nb] == c:
                    return None
                    
            coloring[v] = c
            snapshots = []
            
            for nb in adj[v]:
                d = coloring[nb]
                if d == -1:
                    continue
                if d == c:
                    continue
                uf = get_uf(c, d)
                snap = uf.snapshot()
                if not uf.union(v, nb):
                    coloring[v] = -1
                    for ufk, sn in snapshots:
                        ufk.rollback(sn)
                    return None
                snapshots.append((uf, snap))
            return snapshots
            
        def unassign(v, c, snapshots):
            coloring[v] = -1
            for uf, snap in snapshots:
                uf.rollback(snap)
                
        def backtrack(pos):
            if time.time() > deadline:
                return False
            if pos == n:
                return True
            v = order[pos]
            
            # Symmetry breaking:
            if v == 0:
                colors_to_try = [0]
            elif v == 1:
                # 1 is neighbor of 0
                colors_to_try = [1, 2, 3] # If k>=3
                if k == 2: colors_to_try = [1]
            else:
                colors_to_try = list(range(k))
                
            for c in colors_to_try:
                # Quick check: if neighbor has color c, skip
                valid = True
                for nb in adj[v]:
                    if coloring[nb] == c:
                        valid = False
                        break
                if not valid:
                    continue
                    
                snapshots = try_assign(v, c)
                if snapshots is not None:
                    if backtrack(pos + 1):
                        return True
                    unassign(v, c, snapshots)
            return False
            
        if backtrack(0):
            # Verify
            # Check adjacent differ
            ok = True
            for u, v in edges:
                if coloring[u] == coloring[v]:
                    ok = False
                    break
            if ok:
                # Check forests
                pair_edges = defaultdict(list)
                for u, v in edges:
                    cu, cv = coloring[u], coloring[v]
                    if cu != cv:
                        pair = (min(cu, cv), max(cu, cv))
                        pair_edges[pair].append((u, v))
                for pair, elist in pair_edges.items():
                    uf = ReversibleUnionFind(n)
                    for u, v in elist:
                        if not uf.union(u, v):
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    return [int(c) for c in coloring]
                    
    # Fallback: return a valid coloring with more colors if needed
    # Greedy with many colors
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    order = sorted(range(n), key=lambda x: -len(adj[x]))
    coloring = [0] * n
    for i, v in enumerate(order):
        used = set()
        for nb in adj[v]:
            used.add(coloring[nb])
        for c in range(50):
            if c in used:
                continue
            coloring[v] = c
            # Check if this creates a cycle in any pair
            # Simplified check: just accept if no adjacent same color
            # This might fail the forest condition, but it's a fallback
            break
        else:
            coloring[v] = i % 50
            
    # Final verification and fallback to distinct colors if invalid
    def is_valid(col):
        for u, v in edges:
            if col[u] == col[v]:
                return False
        pair_edges = defaultdict(list)
        for u, v in edges:
            cu, cv = col[u], col[v]
            if cu != cv:
                pair = (min(cu, cv), max(cu, cv))
                pair_edges[pair].append((u, v))
        for pair, elist in pair_edges.items():
            uf = ReversibleUnionFind(n)
            for u, v in elist:
                if not uf.union(u, v):
                    return False
        return True

    if is_valid(coloring):
        return [int(c) for c in coloring]
        
    return list(range(n))
