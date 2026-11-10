"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from functools import lru_cache

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj(n, edges):
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _is_clique_mask(mask, adj):
    """Check if a bitmask represents a clique."""
    if mask == 0:
        return True
    vertices = [i for i in range(len(adj)) if mask & (1 << i)]
    for i, u in enumerate(vertices):
        for v in vertices[i+1:]:
            if v not in adj[u]:
                return False
    return True

def _is_independent_mask(mask, adj):
    """Check if a bitmask represents an independent set."""
    if mask == 0:
        return True
    vertices = [i for i in range(len(adj)) if mask & (1 << i)]
    for i, u in enumerate(vertices):
        for v in vertices[i+1:]:
            if v in adj[u]:
                return False
    return True

def _bron_kerbosch_cliques(adj):
    """Find all maximal cliques using Bron-Kerbosch algorithm."""
    n = len(adj)
    cliques = []
    
    def expand(R, P, X):
        if not P and not X:
            if R:
                cliques.append(R[:])
            return
        
        # Pivot selection: choose vertex in P∪X with max neighbors in P
        pivot = None
        max_count = -1
        for v in P | X:
            count = len(P & adj[v])
            if count > max_count:
                max_count = count
                pivot = v
        
        # Only consider vertices in P not adjacent to pivot
        candidates = list(P - adj[pivot]) if pivot is not None else list(P)
        
        for v in candidates:
            expand(R + [v], P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    expand([], set(range(n)), set())
    return cliques

def _bron_kerbosch_independents(adj):
    """Find all maximal independent sets by finding maximal cliques in complement."""
    n = len(adj)
    # Build complement graph
    comp_adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and v not in adj[u]:
                comp_adj[u].add(v)
    
    # Find maximal cliques in complement = maximal independent sets in original
    cliques = []
    
    def expand(R, P, X):
        if not P and not X:
            if R:
                cliques.append(R[:])
            return
        
        # Pivot selection
        pivot = None
        max_count = -1
        for v in P | X:
            count = len(P & comp_adj[v])
            if count > max_count:
                max_count = count
                pivot = v
        
        candidates = list(P - comp_adj[pivot]) if pivot is not None else list(P)
        
        for v in candidates:
            expand(R + [v], P & comp_adj[v], X & comp_adj[v])
            P.remove(v)
            X.add(v)
    
    expand([], set(range(n)), set())
    return cliques

def _precompute_valid_sets(n, adj):
    """Precompute all maximal cliques and independent sets as bitmasks."""
    # Find all maximal cliques
    max_cliques = _bron_kerbosch_cliques(adj)
    max_independents = _bron_kerbosch_independents(adj)
    
    # Convert to bitmasks and add all subsets (non-maximal cliques/independents are also valid)
    valid_sets = set()
    
    # Add all maximal cliques and their subsets
    for clique in max_cliques:
        mask = 0
        for v in clique:
            mask |= (1 << v)
        # Add this maximal clique and all its non-empty subsets
        sub = mask
        while sub:
            valid_sets.add(sub)
            sub = (sub - 1) & mask
    
    # Add all maximal independents and their subsets
    for indep in max_independents:
        mask = 0
        for v in indep:
            mask |= (1 << v)
        sub = mask
        while sub:
            valid_sets.add(sub)
            sub = (sub - 1) & mask
    
    return list(valid_sets)

def _build_exact_cover_table(n, valid_sets):
    """Build exact cover table for Algorithm X.
    
    Universe: vertices 0..n-1
    Sets: valid_sets (each is a bitmask)
    We want to select a minimum number of sets that partition the universe.
    """
    # For minimum set cover (not exact cover), we need to be more careful.
    # Actually, for cochromatic partition, we need a partition, so each vertex
    # appears in exactly one part. This is an exact cover problem.
    
    # Rows: each valid set
    # Columns: each vertex (0 to n-1)
    # A row covers a column if the vertex is in the set.
    
    # But wait - in a partition, each vertex must be in exactly one part.
    # So we need to select sets such that each vertex is covered exactly once.
    # This is indeed an exact cover problem!
    
    rows = []
    for s in valid_sets:
        row = []
        for v in range(n):
            if s & (1 << v):
                row.append(v)
        if row:  # Only non-empty sets
            rows.append(row)
    
    return rows

def _algorithm_x(rows, n, deadline, start_time):
    """Solve exact cover problem using Algorithm X with Dancing Links.
    
    Returns the minimum number of rows needed, or None if no solution or timeout.
    """
    if time.time() - start_time > deadline:
        return None, None
    
    # Build column header counts
    col_counts = [0] * n
    for row in rows:
        for col in row:
            col_counts[col] += 1
    
    # Dancing Links data structure
    # Each node: (left, right, up, down, col, row_idx)
    # We'll use arrays for efficiency
    
    # First, count total nodes
    total_nodes = sum(len(row) for row in rows) + n + 1  # +1 for header
    
    # Build the matrix
    # Columns are 0..n-1, rows are 0..len(rows)-1
    # We'll create a 2D structure
    
    # For simplicity and speed with n=14, let's use a simpler approach:
    # Backtracking with good pruning
    
    best_solution = None
    best_count = n + 1  # Upper bound
    
    # Sort rows by size (smaller first for better pruning)
    indexed_rows = [(len(row), idx, row) for idx, row in enumerate(rows)]
    indexed_rows.sort()
    
    used_cols = 0
    solution = []
    
    def backtrack(remaining_cols, start_row_idx, current_count):
        nonlocal best_solution, best_count
        
        if time.time() - start_time > deadline:
            return
        
        if not remaining_cols:
            if current_count < best_count:
                best_count = current_count
                best_solution = solution[:]
            return
        
        # Pruning: if current_count + lower_bound >= best_count, prune
        # Lower bound: at least ceil(remaining_cols / max_set_size)
        max_size = max(len(row) for row in rows) if rows else 1
        if current_count + (len(remaining_cols) + max_size - 1) // max_size >= best_count:
            return
        
        # Choose the column with fewest available rows
        min_col = None
        min_count = float('inf')
        for col in remaining_cols:
            count = 0
            for i in range(start_row_idx, len(rows)):
                if col in rows[i]:
                    count += 1
            if count < min_count:
                min_count = count
                min_col = col
            if count == 0:
                return  # No row covers this column
        
        if min_count == 0:
            return
        
        # Try each row that covers min_col
        for i in range(start_row_idx, len(rows)):
            if min_col in rows[i]:
                # Check if row is compatible (no overlap with already chosen)
                compatible = True
                for col in rows[i]:
                    if col not in remaining_cols:
                        compatible = False
                        break
                if not compatible:
                    continue
                
                # Choose this row
                solution.append(i)
                new_remaining = set(remaining_cols)
                for col in rows[i]:
                    new_remaining.discard(col)
                
                backtrack(new_remaining, i + 1, current_count + 1)
                
                solution.pop()
    
    all_cols = set(range(n))
    backtrack(all_cols, 0, 0)
    
    if best_solution is not None:
        return best_count, best_solution
    return None, None

def _greedy_set_cover(n, rows):
    """Greedy set cover for initial solution."""
    if not rows:
        return n, list(range(n))
    
    remaining = set(range(n))
    selected = []
    
    while remaining:
        # Pick the row that covers the most remaining elements
        best_row = None
        best_cover = 0
        for i, row in enumerate(rows):
            cover = len(set(row) & remaining)
            if cover > best_cover:
                best_cover = cover
                best_row = i
        
        if best_row is None or best_cover == 0:
            break
        
        selected.append(best_row)
        for col in rows[best_row]:
            remaining.discard(col)
    
    return len(selected), selected

def _validate_partition(n, parts):
    """Validate that parts form a valid cochromatic partition."""
    # Check each vertex appears exactly once
    seen = [False] * n
    for part in parts:
        for v in part:
            if seen[v]:
                return False
            seen[v] = True
    if not all(seen):
        return False
    
    return True

def construct():
    n = P['n']
    adj = _build_adj(n, P['edges'])
    
    start_time = time.time()
    deadline = 80.0  # Safety margin from 90s limit
    
    # Precompute valid sets (all cliques and independent sets)
    valid_sets = _precompute_valid_sets(n, adj)
    
    # Build rows for exact cover
    rows = _build_exact_cover_table(n, valid_sets)
    
    # Get initial solution from greedy
    greedy_count, greedy_solution = _greedy_set_cover(n, rows)
    best_count = greedy_count
    best_solution = greedy_solution
    
    # Try to find better solution with Algorithm X
    ax_count, ax_solution = _algorithm_x(rows, n, deadline, start_time)
    
    if ax_count is not None and ax_count < best_count:
        best_count = ax_count
        best_solution = ax_solution
    
    # Convert solution to partition format
    if best_solution is None:
        # Fallback: single vertex parts
        final_result = [[v] for v in range(n)]
    else:
        final_result = []
        for row_idx in best_solution:
            part = [v for v in range(n) if rows[row_idx].__contains__(v)]
            # rows[row_idx] is already a list of vertices
            part = rows[row_idx][:]
            if part:
                final_result.append([int(v) for v in part])
    
    # Validate
    if not _validate_partition(n, final_result):
        # Fallback to valid partition
        final_result = [[v] for v in range(n)]
    
    return final_result
# EVOLVE-BLOCK-END
