"""Few colors without any bicolored cycle — backtracking with forward checking."""
import time
from collections import deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

# Precompute adjacency
_n = P['n']
_adj = [[] for _ in range(_n)]
for u, v in P['edges']:
    _adj[u].append(v)
    _adj[v].append(u)

# Vertex ordering: degree-first, then by index for determinism
_order = sorted(range(_n), key=lambda v: (-len(_adj[v]), v))
_pos = [0] * _n
for i, v in enumerate(_order):
    _pos[v] = i


def _check_bicolor_forest(color, k):
    """Check that the subgraph induced by colors {0,1} (within the k-color assignment) is a forest.
    We check all pairs of colors among the k used."""
    # For each pair of colors (a, b), check the induced subgraph is a forest
    for a in range(k):
        for b in range(a + 1, k):
            # Collect vertices with color a or b
            vs = [v for v in range(_n) if color[v] == a or color[v] == b]
            if len(vs) < 3:
                continue
            # Count edges in induced subgraph
            vs_set = set(vs)
            edge_count = 0
            for v in vs:
                for u in _adj[v]:
                    if u in vs_set and u > v:
                        edge_count += 1
            # A forest on |V| vertices has at most |V| - 1 edges
            if edge_count >= len(vs):
                return False
    return True


def _check_bicolor_forest_partial(color, k, assigned_count):
    """Partial check: only check pairs where both colors have been assigned to at least one vertex.
    This is a necessary condition (not sufficient for unassigned vertices, but useful for pruning)."""
    for a in range(k):
        for b in range(a + 1, k):
            # Count vertices and edges among assigned vertices with color a or b
            vs = []
            for v in range(_n):
                if color[v] >= 0 and (color[v] == a or color[v] == b):
                    vs.append(v)
            if len(vs) < 3:
                continue
            vs_set = set(vs)
            edge_count = 0
            for v in vs:
                for u in _adj[v]:
                    if u in vs_set and u > v and color[u] >= 0:
                        edge_count += 1
            if edge_count >= len(vs):
                return False
    return True


def _solve_with_k_colors(k, deadline):
    """Backtracking search for a valid k-coloring."""
    color = [-1] * _n
    # Use union-find per color-pair for efficient cycle detection
    # For each pair (a,b), maintain a DSU over vertices colored a or b
    
    def make_dsu(n_nodes):
        parent = list(range(n_nodes))
        rank = [0] * n_nodes
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            rx, ry = find(x), find(y)
            if rx == ry:
                return False  # cycle detected
            if rank[rx] < rank[ry]:
                rx, ry = ry, rx
            parent[ry] = rx
            if rank[rx] == rank[ry]:
                rank[rx] += 1
            return True
        return find, union

    # We'll track for each color-pair a DSU. But since k is small (3-6), we can just
    # check acyclicity incrementally.
    
    # Better approach: maintain for each pair (a,b) a set of edges and do incremental cycle check
    # Actually, let's just do the full check at each step for simplicity with k<=6 and n=50.
    
    # Even better: maintain DSU per pair. When we assign a vertex to color c, for each other color d,
    # we add edges from this vertex to neighbors of color d.
    
    # Let's use a simpler approach: at each assignment, check if any pair (a,b) now has a cycle.
    # We'll maintain edge counts per pair and use the fact that a connected component with
    # #edges >= #vertices has a cycle.
    
    # Actually, the simplest correct approach for n=50: just check _check_bicolor_forest_partial
    # after each assignment. With k<=5 and 50 vertices, this should be fast enough.
    
    # Let's try a more efficient approach: maintain for each pair (a,b) a DSU.
    # DSU[a][b] maps vertex -> parent in the induced subgraph of colors a,b.
    # But vertices can be in multiple pairs... this gets complex.
    
    # Simplest: just do the full check. n=50, k<=6, so at most 15 pairs, each with <=50 vertices.
    # The check is O(n*deg) per pair, so O(15*50*4) = O(3000) per step. With 50 steps, that's 150k ops.
    # Backtracking might explore many nodes though.
    
    # Let's use a smarter approach: maintain per-pair edge sets and check acyclicity via
    # counting. A graph is a forest iff for every connected component, edges = vertices - 1.
    # We can maintain this incrementally.
    
    # For now, let's just do the straightforward backtracking with the partial check.
    
    def backtrack(idx):
        if time.time() > deadline:
            return None
        if idx == _n:
            # Final check
            if _check_bicolor_forest(color, k):
                return color[:]
            return None
        
        v = _order[idx]
        
        # Try each color
        for c in range(k):
            # Check adjacency constraint
            conflict = False
            for u in _adj[v]:
                if color[u] == c:
                    conflict = True
                    break
            if conflict:
                continue
            
            color[v] = c
            
            # Forward checking: verify no bicolored cycle is formed
            if _check_bicolor_forest_partial(color, k, idx + 1):
                result = backtrack(idx + 1)
                if result is not None:
                    return result
            
            color[v] = -1
        
        return None
    
    return backtrack(0)


def construct():
    deadline = time.time() + 50  # Safety margin before 60s limit
    
    # Try increasing number of colors
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = _solve_with_k_colors(k, deadline)
        if result is not None:
            # Verify the result is valid
            assert all(isinstance(x, int) and 0 <= x < _n for x in result)
            assert len(result) == _n
            # Check adjacency
            for u, v in P['edges']:
                assert result[u] != result[v], f"Adjacent vertices {u},{v} have same color {result[u]}"
            # Check bicolored forest
            assert _check_bicolor_forest(result, k), "Bicolored cycle detected"
            # Ensure pure Python ints
            return [int(x) for x in result]
    
    # Fallback: trivial coloring (shouldn't happen but safety)
    return list(range(_n))
