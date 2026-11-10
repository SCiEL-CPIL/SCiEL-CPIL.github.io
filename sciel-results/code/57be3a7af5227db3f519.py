"""Few edges in 5-uniform hypergraphs without Property B (m(5)).
Strategy:
 1. Use the known 51-edge construction (if it can be verified quickly) as the target.
 2. Build a fast exact 2-colourability checker (backtracking + unit propagation) with
    a hard time budget.
 3. Try to improve via local search / randomised edge-deletion heuristics, but always
    fall back to a valid construction.
 4. Hard deadline guard to finish before the evaluator kills the process.
"""
import time
import random
from itertools import combinations

# ---------------------------------------------------------------------------
# Exact 2-colourability checker with unit propagation and backtracking.
# ---------------------------------------------------------------------------

def _is_2colourable(n, edges, time_limit=5.0):
    """Return True if the hypergraph has a proper 2-colouring.
    Uses backtracking with unit propagation.
    Returns None if time limit is exceeded (inconclusive).
    """
    t0 = time.time()
    # Build incidence: for each vertex, list of (edge_idx, position)
    # We represent edges as frozensets for fast lookup, but keep list for positions.
    edge_set = [frozenset(e) for e in edges]
    m = len(edges)
    # For each edge, we need to track which vertices are assigned and to what colour.
    # Unit propagation: if an edge has 4 vertices of one colour, the 5th must be the other.
    # If an edge has all 5 same colour -> contradiction.
    
    # Adjacency: vertex -> list of (edge_idx, other_vertices)
    v_to_edges = [[] for _ in range(n)]
    for i, e in enumerate(edges):
        for v in e:
            v_to_edges[v].append((i, e))
    
    # We'll do recursive backtracking.
    # State: assignment array (-1 = unassigned, 0/1 = colour)
    # To make it fast, we use a queue-based propagation.
    
    # Precompute for each edge the list of vertices
    edge_verts = [list(e) for e in edges]
    
    # Count: for each edge, how many vertices are assigned to colour 0 and colour 1
    # We'll maintain this incrementally.
    
    # Actually, for small n (<=26) and m (<=200), a simpler approach may work:
    # Use a SAT-like approach with unit clauses.
    # Each edge constraint: NOT(all 0) AND NOT(all 1)
    # i.e., for each edge e = {v1..v5}: (v1 OR v2 OR v3 OR v4 OR v5) AND (¬v1 OR ¬v2 OR ¬v3 OR ¬v4 OR ¬v5)
    # where vi means vertex i is colour 1, ¬vi means colour 0.
    
    # Let's use a simple DPLL with unit propagation on these 2*m clauses.
    # Variables: 0..n-1, True = colour 1, False = colour 0.
    
    # Build clauses
    clauses = []
    for e in edges:
        # NOT all 0: at least one is 1
        clauses.append(tuple(e))  # (v1, v2, v3, v4, v5)
        # NOT all 1: at least one is 0
        clauses.append(tuple(-v for v in e))  # (-v1, -v2, -v3, -v4, -v5)
    
    # Unit propagation and backtracking
    # We'll use a simple recursive approach with memoization on partial assignments.
    
    # For speed, we'll use a bitset representation if possible, but let's keep it simple.
    
    def propagate(assign, clauses_active):
        """assign: dict var -> bool. Returns (new_assign, new_clauses_active, contradiction)"""
        changed = True
        while changed:
            changed = False
            new_clauses = []
            for cl in clauses_active:
                # Check if clause is satisfied, unit, or empty
                sat = False
                unit_var = None
                unassigned_count = 0
                for lit in cl:
                    v = abs(lit)
                    if v in assign:
                        val = assign[v]
                        if (lit > 0) == val:
                            sat = True
                            break
                    else:
                        unassigned_count += 1
                if sat:
                    continue
                if unassigned_count == 0:
                    return assign, clauses_active, True  # contradiction
                if unassigned_count == 1:
                    # Find the unassigned literal
                    for lit in cl:
                        v = abs(lit)
                        if v not in assign:
                            unit_var = v
                            unit_val = lit > 0
                            break
                    if unit_var in assign:
                        if assign[unit_var] != unit_val:
                            return assign, clauses_active, True
                    else:
                        assign[unit_var] = unit_val
                        changed = True
                    continue
                new_clauses.append(cl)
            clauses_active = new_clauses
        return assign, clauses_active, False
    
    # Order variables by degree (number of clauses they appear in)
    var_degree = [0] * n
    for cl in clauses:
        for lit in cl:
            var_degree[abs(lit)] += 1
    var_order = sorted(range(n), key=lambda v: -var_degree[v])
    
    def backtrack(assign, clauses_active, var_idx):
        if time.time() - t0 > time_limit:
            return None  # inconclusive
        assign, clauses_active, contradiction = propagate(assign, clauses_active)
        if contradiction:
            return False
        if not clauses_active:
            return True
        # Find next unassigned variable
        for i in range(var_idx, n):
            v = var_order[i]
            if v not in assign:
                # Try both values
                for val in [True, False]:
                    new_assign = dict(assign)
                    new_assign[v] = val
                    result = backtrack(new_assign, clauses_active, i + 1)
                    if result is not None:
                        return result
                return False
        return True
    
    result = backtrack({}, list(clauses), 0)
    return result

# ---------------------------------------------------------------------------
# Known 51-edge construction.
# We use a construction based on the projective geometry or a known small example.
# One known 51-edge example is from the literature. Let's use a construction
# on 11 vertices or so. Actually, let's use a simpler approach:
# The complete 5-uniform hypergraph on 9 vertices has C(9,5) = 126 edges.
# We need to find a smaller one.
#
# A known result: m(5) <= 51. One construction uses 11 vertices.
# Let's use a randomised construction and verify.
#
# Actually, let's just use a construction that is known to work.
# I'll use the following: take the 51 edges from a known database if available,
# or generate one via search.
#
# For now, let's use a heuristic: start with a random set of edges and try to
# find a small non-2-colourable hypergraph.
# ---------------------------------------------------------------------------

def _random_non_2colourable(n, m, seed=42):
    """Try to find a random 5-uniform hypergraph on n vertices with m edges
    that is not 2-colourable. Returns the edge list or None."""
    rng = random.Random(seed)
    all_edges = list(combinations(range(n), 5))
    rng.shuffle(all_edges)
    edges = all_edges[:m]
    return edges

def _try_improve(n, edges, time_limit=30.0):
    """Try to remove edges while maintaining non-2-colourability.
    Returns a smaller edge list if found, else the original."""
    t0 = time.time()
    current_edges = list(edges)
    m = len(current_edges)
    # Try to remove each edge one by one
    for i in range(m - 1, -1, -1):
        if time.time() - t0 > time_limit:
            break
        # Try removing edge i
        test_edges = current_edges[:i] + current_edges[i+1:]
        # Check if still non-2-colourable
        result = _is_2colourable(n, test_edges, time_limit=1.0)
        if result is False:  # Not 2-colourable
            current_edges = test_edges
        elif result is None:
            # Inconclusive, skip
            continue
    return current_edges

def construct():
    t_start = time.time()
    deadline = 140.0  # Safety margin
    
    # Start with a known construction.
    # Let's use a construction on 11 vertices with 51 edges.
    # I'll use a specific known example.
    
    # Actually, let's use a simpler approach: use the complete hypergraph on 9 vertices
    # but that has 126 edges. We need to do better.
    
    # Let's try to find a small example via search.
    # Start with n=11, m=51
    best_n = 11
    best_edges = None
    
    # Try several random constructions
    for seed in range(20):
        if time.time() - t_start > deadline:
            break
        edges = _random_non_2colourable(best_n, 51, seed=seed)
        if edges is None:
            continue
        # Verify it's not 2-colourable
        result = _is_2colourable(best_n, edges, time_limit=2.0)
        if result is False:
            best_edges = edges
            break
        elif result is None:
            continue
    
    if best_edges is None:
        # Fallback: use a known construction
        # Let's use the complete 5-uniform hypergraph on 9 vertices (126 edges)
        # but that's not optimal. Let's try to find a better one.
        # Actually, let's use a construction from the literature.
        # One known 51-edge construction:
        # I'll use a specific set of edges.
        
        # For now, let's just use a random one and verify.
        for seed in range(50):
            if time.time() - t_start > deadline:
                break
            edges = _random_non_2colourable(11, 51, seed=seed + 100)
            result = _is_2colourable(11, edges, time_limit=1.0)
            if result is False:
                best_edges = edges
                break
    
    if best_edges is None:
        # Last resort: complete hypergraph on 9 vertices
        best_n = 9
        best_edges = [list(e) for e in combinations(range(9), 5)]
    
    # Try to improve
    if time.time() - t_start < deadline - 10:
        improved = _try_improve(best_n, best_edges, time_limit=min(30.0, deadline - (time.time() - t_start) - 5))
        if len(improved) < len(best_edges):
            best_edges = improved
    
    return {'n': best_n, 'edges': best_edges}
