"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict, deque

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}

def _build_adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj

def _verify_coloring(colors, n, edges):
    """Verify that the coloring is valid."""
    for u, v in edges:
        if colors[u] == colors[v]:
            return False
    # Check bicolored forest constraint
    pair_edges = defaultdict(list)
    for u, v in edges:
        cu, cv = colors[u], colors[v]
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
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

def _row_is_valid_ring(pattern, k):
    """Check if a row pattern is valid: no adjacent same colors on the ring (including wrap)."""
    for i in range(len(pattern)):
        if pattern[i] == pattern[(i + 1) % len(pattern)]:
            return False
    return True

def _generate_valid_rows(k, row_len=10):
    """Generate all valid row patterns for k colors on a ring of length row_len."""
    valid_rows = []
    
    def backtrack(pos, pattern, used_colors):
        if time.time() > _deadline:
            return
        if pos == row_len:
            # Check wrap-around
            if pattern[0] != pattern[row_len - 1]:
                valid_rows.append(tuple(pattern))
            return
        for c in range(k):
            if pos > 0 and pattern[pos - 1] == c:
                continue
            pattern[pos] = c
            backtrack(pos + 1, pattern, used_colors | {c})
            pattern[pos] = -1
    
    pattern = [-1] * row_len
    backtrack(0, pattern, set())
    return valid_rows

def _check_bicolored_forest_for_rows(row_patterns, n=50):
    """
    Check if the given 5 row patterns form a valid coloring with bicolored forest property.
    row_patterns is a list of 5 tuples, each of length 10.
    """
    # Build the full coloring
    colors = [0] * n
    for r in range(5):
        for c in range(10):
            colors[r * 10 + c] = row_patterns[r][c]
    
    return _verify_coloring(colors, n, P['edges'])

def _check_pair_forest_incremental(row1, row2, k):
    """
    Check if two adjacent rows create any bicolored cycles when combined with 
    the horizontal edges in each row.
    Returns True if no bicolored cycle is formed by the combination of these two rows.
    """
    # The edges between row1 and row2 are vertical: (r1*c, r2*c) for c in 0..9
    # The edges within row1 are horizontal: (r1*c, r1*(c+1)%10)
    # The edges within row2 are horizontal: (r2*c, r2*(c+1)%10)
    
    # For each pair of colors (a, b), check if the subgraph induced by vertices 
    # with colors a or b in these two rows forms a forest.
    
    # Collect all edges in these two rows
    edges_in_rows = []
    # Horizontal edges in row1 (row index 0 in this local view)
    for c in range(10):
        c2 = (c + 1) % 10
        edges_in_rows.append((c, c2))  # both in row1
    # Horizontal edges in row2 (row index 1)
    for c in range(10):
        c2 = (c + 1) % 10
        edges_in_rows.append((10 + c, 10 + c2))  # both in row2
    # Vertical edges
    for c in range(10):
        edges_in_rows.append((c, 10 + c))
    
    # Build coloring for these 20 vertices
    local_colors = list(row1) + list(row2)
    
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in edges_in_rows:
        cu, cv = local_colors[u], local_colors[v]
        if cu == cv:
            return False  # Adjacent same color
        key = (min(cu, cv), max(cu, cv))
        pair_edges[key].append((u, v))
    
    # Check each pair for cycles
    for pair, elist in pair_edges.items():
        parent = list(range(20))
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

def _solve_cylindrical(k, deadline):
    """
    Solve using the cylindrical structure: enumerate valid row patterns,
    then use DP/brute-force to find compatible sequences of 5 rows.
    """
    global _deadline
    _deadline = deadline
    
    start = time.time()
    
    # Generate all valid row patterns for k colors
    valid_rows = _generate_valid_rows(k, 10)
    
    if not valid_rows:
        return None
    
    # Limit the number of patterns to keep search tractable
    # For k=3, there are 3^10 = 59049 possible, but ring constraint reduces it
    # For k=4, 4^10 = 1M, still manageable with pruning
    
    if len(valid_rows) > 50000:
        # Sample a subset if too many
        random.seed(42)
        valid_rows = random.sample(valid_rows, 50000)
    
    # Precompute compatibility between row pairs
    # compat[i][j] = True if row i and row j can be adjacent
    n_rows = len(valid_rows)
    
    # Build compatibility matrix
    # For efficiency, only check pairs that might be needed
    # We need to find a sequence of 5 rows where consecutive rows are compatible
    # and the full 5-row configuration is valid
    
    # First, build adjacency list of compatible row pairs
    compat = [set() for _ in range(n_rows)]
    
    for i in range(n_rows):
        if time.time() > deadline:
            break
        for j in range(n_rows):
            if i == j:
                continue
            if _check_pair_forest_incremental(valid_rows[i], valid_rows[j], k):
                compat[i].add(j)
    
    # Now do DP: find a sequence of 5 rows where consecutive are compatible
    # dp[r][i] = True if we can have a valid prefix of r+1 rows ending with row i
    # We also need to track the actual sequence for reconstruction
    
    # For 5 rows, we can just do BFS/DFS on the compatibility graph
    # State: (row_index, current_row_pattern_index)
    
    # Use BFS to find any valid sequence of 5 rows
    # State: (depth, last_row_idx)
    # We want depth=5
    
    from collections import defaultdict
    
    # dp[depth][last_idx] = (prev_idx, ...) for backtracking
    # Simpler: just do DFS with memoization
    
    # For 5 rows, the search space is at most n_rows^4 in worst case, but
    # with compatibility pruning it should be much less
    
    # Let's do iterative deepening / BFS
    # Actually for 5 rows, let's just enumerate all possibilities with pruning
    
    # Better approach: for each pair of first two rows, extend greedily
    
    # Let me try a different approach: enumerate all valid 5-row sequences
    # using backtracking with the compatibility graph
    
    found = None
    
    # To limit search, try each starting row and do DFS
    # But this could be slow. Let's use a smarter approach.
    
    # Actually, let me just try all combinations with pruning
    # For 5 rows: for each row0, for each compatible row1, for each compatible row2, ...
    
    # This is O(n_rows^4) in worst case which is too slow for n_rows=50000
    
    # Better: use the fact that we only need ONE valid solution
    
    # Let's try: for each starting row, greedily extend
    for start_idx in range(n_rows):
        if time.time() > deadline:
            break
        
        # Try to extend from this starting row
        def extend(current_depth, current_row_idx, path):
            if time.time() > deadline:
                return None
            if current_depth == 5:
                return path[:]
            
            for next_idx in compat[current_row_idx]:
                path.append(next_idx)
                result = extend(current_depth + 1, next_idx, path)
                if result is not None:
                    return result
                path.pop()
            
            return None
        
        result = extend(1, start_idx, [start_idx])
        if result is not None:
            found = result
            break
    
    if found is None:
        return None
    
    # Build the full coloring
    colors = [0] * 50
    for r in range(5):
        row_pattern = valid_rows[found[r]]
        for c in range(10):
            colors[r * 10 + c] = row_pattern[c]
    
    # Verify the full coloring
    if _verify_coloring(colors, 50, P['edges']):
        return colors
    
    return None

def _solve_with_k_colors(n, edges, k, deadline):
    """
    Solve using cylindrical structure if possible, otherwise fall back to backtracking.
    """
    # Try cylindrical approach first
    result = _solve_cylindrical(k, deadline)
    if result is not None:
        return result
    
    # Fallback: generic backtracking
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    
    num_pairs = k * (k - 1) // 2
    pair_to_idx = {}
    pair_idx = 0
    for i in range(k):
        for j in range(i + 1, k):
            pair_to_idx[(i, j)] = pair_idx
            pair_idx += 1
    
    class RollbackDSU:
        def __init__(self, n):
            self.parent = list(range(n))
            self.rank = [0] * n
            self.history = []
        
        def find(self, x):
            while self.parent[x] != x:
                x = self.parent[x]
            return x
        
        def union(self, x, y):
            rx, ry = self.find(x), self.find(y)
            if rx == ry:
                return False
            if self.rank[rx] < self.rank[ry]:
                rx, ry = ry, rx
            self.history.append((ry, rx, self.rank[rx]))
            self.parent[ry] = rx
            if self.rank[rx] == self.rank[ry]:
                self.rank[rx] += 1
            return True
        
        def rollback(self, count):
            for _ in range(count):
                child, parent, rank_parent_before = self.history.pop()
                self.parent[child] = child
                self.rank[parent] = rank_parent_before
    
    dsus = [RollbackDSU(n) for _ in range(num_pairs)]
    
    def get_pair_idx(c1, c2):
        if c1 > c2:
            c1, c2 = c2, c1
        return pair_to_idx[(c1, c2)]
    
    def try_assign(v, color):
        assignments = []
        for nb in adj[v]:
            if colors[nb] == -1:
                continue
            if colors[nb] == color:
                return None
            pi = get_pair_idx(color, colors[nb])
            if dsus[pi].union(v, nb):
                assignments.append((pi, v, nb))
            else:
                for (pi2, u2, v2) in assignments:
                    dsus[pi2].rollback(1)
                return None
        return assignments
    
    def undo_assign(v, color, assignments):
        for (pi, u, v_edge) in assignments:
            dsus[pi].rollback(1)
    
    max_color_used = -1
    colored_count = [0]
    
    def get_candidates(v):
        forbidden = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                forbidden.add(colors[nb])
        candidates = []
        for c in range(k):
            if c in forbidden:
                continue
            if max_color_used >= 0:
                if c <= max_color_used or c == max_color_used + 1:
                    candidates.append(c)
            else:
                if c == 0:
                    candidates.append(c)
        return candidates
    
    def backtrack():
        nonlocal max_color_used
        if time.time() > deadline:
            return False
        
        if colored_count[0] == n:
            return True
        
        best_v = -1
        best_cands = None
        best_count = k + 1
        
        for v in range(n):
            if colors[v] != -1:
                continue
            cands = get_candidates(v)
            cnt = len(cands)
            if cnt < best_count:
                best_count = cnt
                best_v = v
                best_cands = cands
                if cnt <= 1:
                    break
        
        if best_v == -1:
            return True
        
        if best_count == 0:
            return False
        
        best_cands.sort()
        
        for color in best_cands:
            assignments = try_assign(best_v, color)
            if assignments is None:
                continue
            
            colors[best_v] = color
            colored_count[0] += 1
            old_max = max_color_used
            if color > max_color_used:
                max_color_used = color
            
            if backtrack():
                return True
            
            colors[best_v] = -1
            colored_count[0] -= 1
            undo_assign(best_v, color, assignments)
            max_color_used = old_max
        
        return False
    
    if backtrack():
        return colors
    return None

def construct():
    n = P['n']
    edges = P['edges']
    start_time = time.time()
    deadline = start_time + 85
    
    # Try increasing number of colors
    for k in range(3, 10):
        if time.time() > deadline:
            break
        result = _solve_with_k_colors(n, edges, k, deadline)
        if result is not None:
            if _verify_coloring(result, n, edges):
                return [int(c) for c in result]
    
    # Fallback: greedy with many colors
    adj = _build_adjacency(n, edges)
    colors = [-1] * n
    next_color = 0
    for v in range(n):
        used = set()
        for nb in adj[v]:
            if colors[nb] != -1:
                used.add(colors[nb])
        while next_color in used:
            next_color += 1
        colors[v] = next_color
        next_color += 1
    
    return [int(c) for c in colors]
# EVOLVE-BLOCK-END

_deadline = 0
