"""Backtracking search minimizing distinct colors with
incremental bicolored-cycle detection via DSU with rollback."""
# EVOLVE-BLOCK-START
import time
import sys
import random
from collections import defaultdict

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

N = P['n']
EDGES = P['edges']

# Build adjacency
adj = [[] for _ in range(N)]
for u, v in EDGES:
    adj[u].append(v)
    adj[v].append(u)

degree = [len(adj[i]) for i in range(N)]
# Order by degree descending, then by index for determinism
order = sorted(range(N), key=lambda x: (-degree[x], x))
pos_of = [0] * N
for i, v in enumerate(order):
    pos_of[v] = i

# Precompute for each vertex, which neighbors come before it in the order
# This helps with forward checking
prior_neighbors = [[] for _ in range(N)]
for idx, v in enumerate(order):
    for u in adj[v]:
        if pos_of[u] < idx:
            prior_neighbors[v].append(u)

class RollbackDSU:
    """DSU with rollback support for backtracking."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.log = []  # stack of (child, parent, size_parent_before)
    
    def find(self, x):
        # Path compression not used for rollback; use union by size
        while self.parent[x] != x:
            x = self.parent[x]
        return x
    
    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # cycle
        if self.size[rx] > self.size[ry]:
            rx, ry = ry, rx
        self.log.append((ry, rx, self.size[rx]))
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True
    
    def snapshot(self):
        return len(self.log)
    
    def rollback(self, snap):
        while len(self.log) > snap:
            child, parent, size_parent_before = self.log.pop()
            self.parent[child] = child
            self.size[parent] = size_parent_before

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 85s deadline, 5s margin
    
    # Trivial valid solution: all different colors
    trivial = list(range(N))
    best_result = trivial[:]
    best_colors = N
    
    # Try to find solutions with fewer colors
    for max_colors in range(2, N + 1):
        if time.time() > deadline:
            break
        
        # Backtracking search with DSU rollback
        color = [-1] * N
        # DSU for each color pair: key = (min(c1,c2), max(c1,c2))
        dsus = {}
        
        def get_dsu(c1, c2):
            if c1 > c2:
                c1, c2 = c2, c1
            key = (c1, c2)
            if key not in dsus:
                dsus[key] = RollbackDSU(N)
            return dsus[key], key
        
        def check_and_assign(v, c):
            """Check if assigning color c to vertex v is valid.
            Returns (valid, snapshots) where snapshots is dict of dsu_key -> snapshot."""
            # Check adjacency: no neighbor has color c
            for u in adj[v]:
                if color[u] == c:
                    return False, None
            
            # Check bicolored cycles: for each neighbor u with color cu != c,
            # check if edge (v,u) would create a cycle in subgraph {c, cu}
            snapshots = {}
            valid = True
            for u in adj[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                dsu, key = get_dsu(c, cu)
                snap = dsu.snapshot()
                # Check if v and u are already connected
                if dsu.find(v) == dsu.find(u):
                    valid = False
                    # Rollback all previous snapshots in this check
                    for k, s in snapshots.items():
                        dsus[k].rollback(s)
                    return False, None
                # Tentatively union
                dsu.union(v, u)
                snapshots[key] = snap
            
            return True, snapshots
        
        def unassign(v, c, snapshots):
            """Rollback DSU changes made by check_and_assign."""
            if snapshots:
                for key, snap in snapshots.items():
                    dsus[key].rollback(snap)
            color[v] = -1
        
        def backtrack(idx):
            nonlocal best_result, best_colors
            if time.time() > deadline:
                return False
            if idx == N:
                distinct = len(set(color))
                if distinct < best_colors:
                    best_colors = distinct
                    best_result = color[:]
                return True
            
            v = order[idx]
            # Try colors 0..max_colors-1
            for c in range(max_colors):
                valid, snapshots = check_and_assign(v, c)
                if not valid:
                    continue
                
                color[v] = c
                if backtrack(idx + 1):
                    return True
                unassign(v, c, snapshots)
            
            return False
        
        found = backtrack(0)
        if found:
            # We found a valid coloring with at most max_colors colors
            # The actual number of distinct colors used is best_colors
            if best_colors <= max_colors:
                break
        else:
            continue
    
    # Ensure we return a valid list of Python ints
    if best_result is None:
        best_result = trivial
    
    result = [int(c) for c in best_result]
    return result
# EVOLVE-BLOCK-END
