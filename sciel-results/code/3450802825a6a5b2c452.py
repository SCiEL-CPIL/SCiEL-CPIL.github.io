"""Minimize distinct colors for degree-4 graph with bicolored-forest constraint.
Uses optimized backtracking with rollback Union-Find and symmetry breaking."""
import time
import random
from collections import deque
from itertools import combinations

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

_N = P['n']
_EDGES = P['edges']
_ADJ = [[] for _ in range(_N)]
for _a, _b in _EDGES:
    _ADJ[_a].append(_b)
    _ADJ[_b].append(_a)

# Precompute edges incident to each vertex
_EDGES_OF = [[] for _ in range(_N)]
for i, (a, b) in enumerate(_EDGES):
    _EDGES_OF[a].append(i)
    _EDGES_OF[b].append(i)


def _is_forest_edge_set(edge_indices):
    """Check if the subgraph formed by given edge indices is a forest (acyclic)."""
    parent = list(range(_N))
    rank = [0] * _N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for ei in edge_indices:
        a, b = _EDGES[ei]
        ra, rb = find(a), find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if rank[ra] == rank[rb]:
            rank[ra] += 1
    return True


def _check_bicforest(coloring):
    """Check bicolored-forest constraint for a complete coloring."""
    from collections import defaultdict
    pair_edges = defaultdict(list)
    for ei, (a, b) in enumerate(_EDGES):
        ca, cb = coloring[a], coloring[b]
        if ca == cb:
            return False
        cmin, cmax = (ca, cb) if ca < cb else (cb, ca)
        pair_edges[(cmin, cmax)].append(ei)

    for pair, eis in pair_edges.items():
        if not _is_forest_edge_set(eis):
            return False
    return True


class _UF_Rollback:
    """Union-Find with rollback capability for backtracking."""
    def __init__(self, n):
        self.n = n
        self.parent = list(range(n))
        self.size = [1] * n
        self.history = []

    def find(self, x):
        # No path compression for rollback, just walk up
        while self.parent[x] != x:
            x = self.parent[x]
        return x

    def union(self, x, y):
        rx = self.find(x)
        ry = self.find(y)
        if rx == ry:
            self.history.append((None, None)) # Marker for no change
            return False
        
        if self.size[rx] < self.size[ry]:
            rx, ry = ry, rx
            
        self.history.append((ry, rx))
        self.parent[ry] = rx
        self.size[rx] += self.size[ry]
        return True

    def rollback(self, state_size):
        while len(self.history) > state_size:
            ry, rx = self.history.pop()
            if ry is not None:
                self.parent[ry] = ry
                self.size[rx] -= self.size[ry]
                # size[ry] is already correct from previous state? 
                # Actually, size[ry] was not changed in union, so it's fine.
                # Wait, size[rx] was increased. We subtract size[ry].
                # But size[ry] itself might have been changed by other unions?
                # No, ry was a root. Its size only changes when it merges into another root.
                # Since it's a root here, its size is the size of its component.
                # When we merge ry into rx, size[rx] += size[ry]. size[ry] remains the same.
                # So rollback: size[rx] -= size[ry]. Correct.


def _solve_k_coloring(k, deadline):
    """
    Solve for k-coloring with bicolored-forest constraint using backtracking.
    """
    color = [-1] * _N
    
    # Symmetry breaking: Fix vertex 0 to color 0, vertex 1 to color 1 (if adjacent)
    # Vertex 0 is adjacent to 1.
    color[0] = 0
    color[1] = 1
    
    # Initialize Union-Finds for all pairs
    n_pairs = k * (k - 1) // 2
    pair_ids = {}
    idx = 0
    for i in range(k):
        for j in range(i+1, k):
            pair_ids[(i, j)] = idx
            idx += 1
            
    ufs = [_UF_Rollback(_N) for _ in range(n_pairs)]
    
    # Initialize with edges from pre-colored vertices
    # Edges (0,1) is pair (0,1)
    # Edges (0,7), (0,43), (0,49) - 7,43,49 uncolored
    # Edges (1,2), (1,8), (1,44) - 2,8,44 uncolored
    
    # We need to add edges where both ends are colored.
    # Only (0,1) is fully colored initially.
    pi_01 = pair_ids[(0, 1)]
    ufs[pi_01].union(0, 1)
    
    # Store initial state sizes
    initial_sizes = [len(uf.history) for uf in ufs]
    
    # Vertex order: Start with high degree, uncolored
    # 0 and 1 are fixed.
    # Sort remaining by degree descending
    remaining = [v for v in range(_N) if color[v] == -1]
    remaining.sort(key=lambda v: (-len(_ADJ[v]), v))
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == len(remaining):
            return True
            
        v = remaining[idx]
        
        # Determine available colors
        used_colors = set(color[u] for u in _ADJ[v] if color[u] != -1)
        available = [c for c in range(k) if c not in used_colors]
        
        # Heuristic: Try colors that are "least constraining" or just random?
        # Let's try a fixed order first, then random if fails.
        
        for c in available:
            color[v] = c
            
            # Check bicolored forest constraint
            # For each neighbor u with color cu != -1 and cu != c
            # The edge (v, u) belongs to pair (c, cu)
            # We need to check if adding this edge creates a cycle in that pair's subgraph.
            
            ok = True
            changes = []
            
            for u in _ADJ[v]:
                cu = color[u]
                if cu == -1 or cu == c:
                    continue
                ci, cj = (c, cu) if c < cu else (cu, c)
                pi = pair_ids[(ci, cj)]
                
                # Record state before union
                state_size = len(ufs[pi].history)
                
                if not ufs[pi].union(v, u):
                    # Cycle detected
                    ok = False
                    # Rollback this union (which did nothing to structure but added marker)
                    ufs[pi].rollback(state_size)
                    break
                else:
                    changes.append((pi, state_size))
            
            if ok:
                if backtrack(idx + 1):
                    return True
            
            # Rollback changes
            for pi, state_size in reversed(changes):
                ufs[pi].rollback(state_size)
            
            color[v] = -1
            
        return False

    if backtrack(0):
        return color[:]
    return None

def construct():
    t_start = time.time()
    deadline = t_start + 85.0
    
    # Try k=3, 4, 5, 6
    for k in range(3, 7):
        if time.time() > deadline - 10.0:
            break
        result = _solve_k_coloring(k, deadline - 5.0)
        if result is not None:
            return [int(x) for x in result]
            
    # Fallback to greedy with more colors if backtracking fails
    # Greedy usually works for k >= 5 or 6 on 4-regular graphs
    for k in range(6, 10):
        if time.time() > deadline - 5.0:
            break
        order = sorted(range(_N), key=lambda v: (-len(_ADJ[v]), v))
        color = [-1] * _N
        valid = True
        for v in order:
            used = set(color[nb] for nb in _ADJ[v] if color[nb] != -1)
            for c in range(k):
                if c not in used:
                    color[v] = c
                    break
            if color[v] == -1:
                valid = False
                break
        if valid and _check_bicforest(color):
            return [int(x) for x in color]
            
    # Last resort: 50 colors
    return [int(i) for i in range(_N)]
