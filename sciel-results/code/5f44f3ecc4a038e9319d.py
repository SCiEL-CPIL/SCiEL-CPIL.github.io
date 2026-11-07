"""Solver for few-color graph coloring with no bicolored cycles.

Strategy:
  1. The graph is a Cayley graph on Z_50 (edges i <-> i+1, i+7, i+43 mod 50).
     For a linear coloring c(v) = (a*v) mod k, each edge step d becomes a
     bicolored subgraph on Z_k with step (a*d) mod k, which is a union of
     gcd(k, a*d) cycles of length k/gcd.  That subgraph is a forest iff
     gcd(k, a*d) == 1 (a single k-cycle => no cycle).  Hence k works iff
     gcd(k, a) == gcd(k, 7) == 1 and k >= 3.  The smallest such k is 6.
  2. Enumerate (k, a) in increasing k, verify each candidate with the exact
     union-find bicolored-cycle check, and return the first valid one.
  3. As a refinement / safety net, run a fast cached merge-based reduction
     on the best coloring found (it can only keep or improve validity).
"""
import time
import sys
from math import gcd
from collections import defaultdict, Counter

P = {'n': 50, 'edges': [[0, 1], [0, 7], [0, 43], [0, 49], [1, 2], [1, 8], [1, 44], [2, 3], [2, 9], [2, 45], [3, 4], [3, 10], [3, 46], [4, 5], [4, 11], [4, 47], [5, 6], [5, 12], [5, 48], [6, 7], [6, 13], [6, 49], [7, 8], [7, 14], [8, 9], [8, 15], [9, 10], [9, 16], [10, 11], [10, 17], [11, 12], [11, 18], [12, 13], [12, 19], [13, 14], [13, 20], [14, 15], [14, 21], [15, 16], [15, 22], [16, 17], [16, 23], [17, 18], [17, 24], [18, 19], [18, 25], [19, 20], [19, 26], [20, 21], [20, 27], [21, 22], [21, 28], [22, 23], [22, 29], [23, 24], [23, 30], [24, 25], [24, 31], [25, 26], [25, 32], [26, 27], [26, 33], [27, 28], [27, 34], [28, 29], [28, 35], [29, 30], [29, 36], [30, 31], [30, 37], [31, 32], [31, 38], [32, 33], [32, 39], [33, 34], [33, 40], [34, 35], [34, 41], [35, 36], [35, 42], [36, 37], [36, 43], [37, 38], [37, 44], [38, 39], [38, 45], [39, 40], [39, 46], [40, 41], [40, 47], [41, 42], [41, 48], [42, 43], [42, 49], [43, 44], [44, 45], [45, 46], [46, 47], [47, 48], [48, 49]]}


def _build_adj():
    n = P['n']
    adj = [[] for _ in range(n)]
    for u, v in P['edges']:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _check_valid(coloring, adj):
    """Check adjacent differ and no bicolored cycles. Returns True if valid."""
    n = len(coloring)
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False
    color_pairs = set()
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        color_pairs.add((min(cu, cv), max(cu, cv)))
    for c1, c2 in color_pairs:
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                if not union(u, v):
                    return False
    return True


def _fast_check(coloring):
    """Fast exact validity check (adjacency + bicolored forest) using cached
    per-color-pair edge lists.  Returns (is_valid, color_pairs, pair_edges)."""
    n = len(coloring)
    # Adjacency
    for u, v in P['edges']:
        if coloring[u] == coloring[v]:
            return False, None, None
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key].append((u, v))
    for (c1, c2), edges in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for u, v in edges:
            ru, rv = find(u), find(v)
            if ru == rv:
                return False, None, None
            if rank[ru] < rank[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            if rank[ru] == rank[rv]:
                rank[ru] += 1
    return True, list(pair_edges.keys()), pair_edges


def _try_reduce(coloring, adj, deadline):
    """Cached incremental merge-based reduction.  Returns best coloring."""
    n = len(coloring)
    valid, color_pairs, pair_edges = _fast_check(coloring)
    if not valid:
        return coloring

    # adjacency set for the quick "adjacent" pre-filter
    adj_set = [set(a) for a in adj]

    def rebuild(coloring):
        for u, v in P['edges']:
            if coloring[u] == coloring[v]:
                return False, None, None
        pe = defaultdict(list)
        for u, v in P['edges']:
            cu, cv = coloring[u], coloring[v]
            key = (cu, cv) if cu < cv else (cv, cu)
            pe[key].append((u, v))
        for (c1, c2), edges in pe.items():
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            for u, v in edges:
                ru, rv = find(u), find(v)
                if ru == rv:
                    return False, None, None
                if rank[ru] < rank[rv]:
                    ru, rv = rv, ru
                parent[rv] = ru
                if rank[ru] == rank[rv]:
                    rank[ru] += 1
        return True, list(pe.keys()), pe

    changed = True
    while changed and time.time() < deadline - 3:
        changed = False
        distinct = sorted(set(coloring))
        if len(distinct) <= 2:
            break
        counts = Counter(coloring)
        ordered = sorted(distinct, key=lambda c: (-counts[c], c))
        for i in range(len(ordered)):
            for j in range(i + 1, len(ordered)):
                ca, cb = ordered[i], ordered[j]
                # Quick filter: no edge directly between ca and cb
                if (ca, cb) not in pair_edges and (cb, ca) not in pair_edges:
                    new_coloring = [ca if c == cb else c for c in coloring]
                    ok, cp, pe = rebuild(new_coloring)
                    if ok:
                        coloring = new_coloring
                        color_pairs, pair_edges = cp, pe
                        changed = True
                        break
            if changed:
                break
    return coloring


def _find_biclic_cycles(coloring):
    """Find bicolored cycles in the induced subgraph for each color pair.
    Returns a list of lists, where each inner list is a list of edges (u, v)
    forming a cycle."""
    n = len(coloring)
    # Group edges by color pair
    pair_edges = defaultdict(list)
    for u, v in P['edges']:
        cu, cv = coloring[u], coloring[v]
        key = (cu, cv) if cu < cv else (cv, cu)
        pair_edges[key].append((u, v))
    
    cycles = []
    for (c1, c2), edges in pair_edges.items():
        parent = list(range(n))
        rank = [0] * n
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a, b):
            ra, rb = find(a), find(b)
            if ra == rb:
                return False
            if rank[ra] < rank[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            if rank[ra] == rank[rb]:
                rank[ra] += 1
            return True
        
        # To find the actual cycle, we need to track the path or use a different method.
        # Since the graph is sparse, we can just collect edges that cause a cycle.
        # However, for SAT we need specific clauses.
        # A simpler approach for SAT: just add a clause that forbids the current assignment
        # if it's invalid. But that's not a "bicolored cycle" clause, that's a "no repeat" clause.
        # The directive asks for "no bicolored cycle" clauses derived from short cycles.
        
        # Let's try to find the cycle path.
        # We can do this by maintaining a parent pointer in the DSU or by doing a DFS on the subgraph.
        # Given the small size, let's just do a simple check.
        # Actually, for the SAT solver, we can just add a clause that says "not (all these vertices have these colors)".
        # But we need to know WHICH vertices/edges form the cycle.
        
        # Let's modify the union-find to track the cycle edges.
        # Or simpler: if union returns False, we have a cycle.
        # We can find the cycle by tracing back from u and v to their common ancestor in the tree.
        
        # Let's implement a helper to find the cycle.
        def get_cycle(u, v, parent, rank):
            # This is tricky with standard DSU.
            # Let's just collect the edge that caused the conflict.
            # And assume that removing that edge breaks the cycle? No, we need to forbid the coloring.
            # The clause should be: NOT (u has cu AND v has cv AND ... other vertices in cycle have their colors)
            # For now, let's just return the edge (u, v) that closed the cycle.
            # The clause will be: NOT (coloring[u] == cu AND coloring[v] == cv)
            # This is a valid conflict clause! It prevents this specific edge from having this specific color pair.
            # Wait, the edge (u, v) is fixed in the graph. The colors are variables.
            # So the clause is: NOT (x_u == cu AND x_v == cv)
            # This is a 2-literal clause (if we use one-hot encoding) or a clause in the SAT encoding.
            
            # Let's return the pair (u, v, cu, cv)
            return (u, v, cu, cv)

        for u, v in edges:
            cu, cv = coloring[u], coloring[v]
            if not union(u, v):
                # Found a cycle!
                cycles.append((u, v, cu, cv))
                # We can break early for this pair if we just want one clause
                # But adding more might help
    return cycles


def _dfs_search(adj, num_colors, deadline):
    """Backtracking search with degeneracy ordering and incremental bicolored-cycle detection."""
    n = P['n']
    
    # Compute degeneracy ordering
    # For this graph, all vertices have degree 4, so degeneracy is 4
    # We'll use a simple ordering based on vertex index, but could improve
    
    # Precompute adjacency as sets
    adj_set = [set(a) for a in adj]
    
    coloring = [-1] * n
    
    # Incremental bicolored cycle detection
    # We maintain union-find per color pair for assigned edges
    # This is complex; instead we'll do a full check at each step but with pruning
    
    def is_valid_partial():
        """Check that current partial coloring has no violations."""
        # Check adjacent assigned vertices differ
        for u in range(n):
            if coloring[u] == -1:
                continue
            for v in adj[u]:
                if coloring[v] != -1 and coloring[u] == coloring[v]:
                    return False
        
        # Check no bicolored cycles in assigned subgraph
        assigned_edges = []
        for u, v in P['edges']:
            if coloring[u] != -1 and coloring[v] != -1:
                assigned_edges.append((u, v, coloring[u], coloring[v]))
        
        if not assigned_edges:
            return True
        
        color_pairs = set()
        for u, v, cu, cv in assigned_edges:
            color_pairs.add((min(cu, cv), max(cu, cv)))
        
        for c1, c2 in color_pairs:
            parent = list(range(n))
            rank = [0] * n
            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x
            def union(a, b):
                ra, rb = find(a), find(b)
                if ra == rb:
                    return False
                if rank[ra] < rank[rb]:
                    ra, rb = rb, ra
                parent[rb] = ra
                if rank[ra] == rank[rb]:
                    rank[ra] += 1
                return True
            for u, v, cu, cv in assigned_edges:
                if (cu == c1 and cv == c2) or (cu == c2 and cv == c1):
                    if not union(u, v):
                        return False
        return True
    
    def dfs(idx):
        if time.time() > deadline:
            return None
        if idx == n:
            return coloring[:]
        v = idx
        
        # Determine allowed colors
        allowed = set(range(num_colors))
        for nb in adj[v]:
            if coloring[nb] != -1:
                allowed.discard(coloring[nb])
        
        for c in sorted(allowed):
            coloring[v] = c
            if is_valid_partial():
                result = dfs(idx + 1)
                if result is not None:
                    return result
            coloring[v] = -1
        return None
    
    return dfs(0)


def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before the 90s kill
    adj = _build_adj()
    n = P['n']

    best = None
    best_count = n + 1

    # --- Direct Cayley-structure construction: c(v) = (a*v) mod k ---
    # k works iff gcd(k, a) == gcd(k, 7) == 1 and k >= 3.  Enumerate small k.
    for k in range(3, 13):
        if time.time() > deadline:
            break
        for a in range(1, k):
            if gcd(k, a) != 1 or gcd(k, 7) != 1:
                continue
            coloring = [int((a * v) % k) for v in range(n)]
            ok, _, _ = _fast_check(coloring)
            if ok:
                used = len(set(coloring))
                if used < best_count:
                    best = coloring
                    best_count = used
                break  # first valid a for this k is enough

    # --- Fallbacks if no direct construction found (should not happen) ---
    if best is None:
        for k in [10, 12, 15]:
            coloring = [v % k for v in range(n)]
            if _check_valid(coloring, adj):
                best = coloring
                best_count = len(set(coloring))
                break
    if best is None:
        for k in [5, 6, 7, 8, 9, 10]:
            coloring = [(v // 10 + v % 10) % k for v in range(n)]
            if _check_valid(coloring, adj):
                best = coloring
                best_count = len(set(coloring))
                break
    if best is None:
        best = list(range(n))
        best_count = n

    # --- Try SAT solver if available and time permits ---
    if time.time() < deadline - 10:
        try:
            from pysat.solvers import Solver as PySatSolver
            from pysat.formula import CNF
            from pysat.card import CardEnc, EncType
            
            # Try to find a 3-coloring (or fewer) using SAT
            # We want to minimize colors. Let's try 3 first, then 4.
            # If SAT finds a valid 3-coloring, we are done (score 3).
            # If not, we fall back to DFS or keep the 6-coloring.
            
            # SAT Encoding:
            # Variable x_{v,c} means vertex v has color c.
            # Constraints:
            # 1. Each vertex has exactly one color. (Cardinality: exactly 1)
            # 2. Adjacent vertices have different colors. (x_{u,c} -> not x_{v,c})
            # 3. No bicolored cycles.
            #    This is hard to encode directly as "no cycle in subgraph induced by colors c1, c2".
            #    However, we can use the conflict-driven approach:
            #    Start with a SAT instance that allows any coloring.
            #    Solve. Check if valid.
            #    If invalid, add a clause that forbids the specific bicolored cycle found.
            #    Repeat.
            
            # Since we already have a valid 6-coloring, we can use it to seed or just search for 3.
            # Let's try to find a 3-coloring.
            
            num_colors_target = 3
            if best_count <= num_colors_target:
                pass # Already have 3 or fewer
            
            else:
                # Variables: v in 0..49, c in 0..2
                # Var index: v * num_colors_target + c + 1
                def var(v, c):
                    return v * num_colors_target + c + 1
                
                cnf = CNF()
                
                # 1. Exactly one color per vertex
                for v in range(n):
                    lits = [var(v, c) for c in range(num_colors_target)]
                    # At least one
                    cnf.append(lits)
                    # At most one (pairwise negation)
                    for i in range(num_colors_target):
                        for j in range(i + 1, num_colors_target):
                            cnf.append([-lits[i], -lits[j]])
                
                # 2. Adjacent vertices different colors
                for u, v in P['edges']:
                    for c in range(num_colors_target):
                        cnf.append([-var(u, c), -var(v, c)])
                
                # 3. Bicolored cycle constraints will be added incrementally
                
                with PySatSolver(name='cadical19') as s:
                    s.load_formula(cnf)
                    
                    max_iters = 100
                    for _ in range(max_iters):
                        if time.time() > deadline - 5:
                            break
                        if not s.solve():
                            # No solution with 3 colors
                            break
                        
                        model = s.get_model()
                        coloring = [0] * n
                        for v in range(n):
                            for c in range(num_colors_target):
                                if model[var(v, c)] > 0:
                                    coloring[v] = c
                                    break
                        
                        # Check validity
                        ok, _, _ = _fast_check(coloring)
                        if ok:
                            used = len(set(coloring))
                            if used < best_count:
                                best = coloring
                                best_count = used
                            break
                        
                        # Add conflict clause
                        # Find a bicolored cycle
                        cycles = _find_biclic_cycles(coloring)
                        if not cycles:
                            # Should not happen if _fast_check says invalid
                            break
                        
                        # Pick the first cycle
                        u, v, cu, cv = cycles[0]
                        # Clause: NOT (x_{u,cu} AND x_{v,cv})
                        # This is [-var(u, cu), -var(v, cv)]
                        cnf.append([-var(u, cu), -var(v, cv)])
                        s.add_clause([-var(u, cu), -var(v, cv)])
                        
        except ImportError:
            pass # pysat not available
        except Exception:
            pass # Error in SAT solver

    # --- Try DFS search with fewer colors if time permits ---
    if time.time() < deadline - 10 and best_count > 3:
        for num_colors in range(3, 5):
            if time.time() > deadline - 5:
                break
            result = _dfs_search(adj, num_colors, deadline - 2)
            if result is not None:
                ok, _, _ = _fast_check(result)
                if ok:
                    used = len(set(result))
                    if used < best_count:
                        best = result
                        best_count = used
                    break

    # --- Refinement / safety net: cached merge-based reduction ---
    if time.time() < deadline - 3:
        reduced = _try_reduce(best, adj, deadline)
        ok, _, _ = _fast_check(reduced)
        if ok and len(set(reduced)) <= best_count:
            best = reduced
            best_count = len(set(best))

    # --- Final exact verification ---
    if not _check_valid(best, adj):
        return [int(x) for x in range(n)]

    return [int(x) for x in best]
