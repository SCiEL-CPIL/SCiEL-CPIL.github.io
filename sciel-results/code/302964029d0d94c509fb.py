"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from random import Random
from collections import defaultdict
from scipy.optimize import linprog

def _random_graph(n, p, seed):
    """Generate a random graph using the same RNG sequence as the seed program."""
    rng = Random(seed)
    edges = []
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                edges.append((u, v))
    return edges

def _bron_kerbosch_maximal_cliques(n, adj):
    """
    Enumerate all maximal cliques using Bron-Kerbosch with pivot.
    adj: list of sets, adj[i] is the set of neighbors of i.
    Returns list of sets (each a clique).
    """
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(set(R))
            return
        
        # Choose pivot u from P ∪ X to minimize branching
        if P:
            u = max(P, key=lambda v: len(adj[v] & P))
            candidates = P - adj[u]
        elif X:
            u = max(X, key=lambda v: len(adj[v] & X))
            candidates = P  # P is empty, so this loop won't run
            # Actually if P is empty, we already returned above
            return
        else:
            return
        
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.remove(v)
            X.add(v)
    
    initial_P = set(range(n))
    bk(set(), initial_P, set())
    return cliques

def _solve_set_cover_lp_guided(cliques, n):
    """
    Solve the set cover problem using LP relaxation duals to guide a greedy selection,
    followed by local search.
    """
    if not cliques:
        return set()
    
    num_cliques = len(cliques)
    
    # Build incidence matrix A (num_cliques x n)
    # A[ci, v] = 1 if v is in cliques[ci]
    A = np.zeros((num_cliques, n))
    for ci, clique in enumerate(cliques):
        for v in clique:
            A[ci, v] = 1.0
    
    # LP Relaxation: min sum(x_v) subject to A x >= 1, 0 <= x_v <= 1
    c = np.ones(n)
    b = np.ones(num_cliques)
    
    # linprog expects constraints in form A_ub x <= b_ub
    # So we use -A x <= -1
    res = linprog(c, A_ub=-A, b_ub=-b, bounds=[(0, 1)] * n, method='highs')
    
    if not res.success:
        # Fallback to simple greedy if LP fails
        return _simple_greedy(cliques, n)

    x_lp = res.x
    # Dual variables (shadow prices) for the constraints A x >= 1
    # In linprog, res.ineqlin.marginals gives the duals for A_ub x <= b_ub
    # Since we used -A x <= -1, the duals correspond to the original constraints.
    # Note: The sign might be tricky. Let's use the marginal values directly.
    # The dual variable lambda_i indicates the cost increase if the RHS of constraint i is increased.
    # Here, increasing RHS of -A x <= -1 is decreasing the lower bound, which is not standard.
    # Let's stick to a robust heuristic: use x_lp values to prioritize vertices.
    
    # Heuristic 1: Pick vertices with high x_lp values.
    # Heuristic 2: Use duals to weight the marginal gain.
    
    # Let's compute duals properly.
    # The dual of min c^T x s.t. A x >= b, x >= 0 is max b^T y s.t. A^T y <= c, y >= 0.
    # linprog gives us the primal solution.
    # We can extract the dual solution if available, but `marginals` is for the bounds/constraints in the form used.
    # For `A_ub x <= b_ub`, the marginals are the dual variables.
    # Our constraint was `-A x <= -1`. So `marginals` are duals for `-A x <= -1`.
    # Let y be the dual for `A x >= 1`. Then dual for `-A x <= -1` is `-y`.
    # So `y = -res.ineqlin.marginals`.
    
    try:
        duals = -res.ineqlin.marginals
        # Ensure non-negative (duals for >= constraints in min problem should be >= 0)
        duals = np.maximum(duals, 0)
    except:
        duals = np.ones(num_cliques)
    
    # Greedy selection guided by LP solution and duals
    selected = set()
    uncovered = set(range(num_cliques))
    
    # Sort vertices by x_lp descending, but we will do a dynamic greedy
    # Score for vertex v: sum of duals of uncovered cliques it covers / (1 + cost)
    # Cost is 1 for all.
    
    # We will iterate and pick the vertex with the highest "dual-weighted coverage"
    # among those not yet selected.
    
    while uncovered:
        best_v = None
        best_score = -1
        best_ties = []
        
        for v in range(n):
            if v in selected:
                continue
            
            # Calculate score: sum of duals of cliques covered by v that are still uncovered
            score = 0.0
            # Also incorporate LP value as a tie-breaker or base weight
            lp_val = x_lp[v]
            
            for ci in range(num_cliques):
                if A[ci, v] > 0 and ci in uncovered:
                    score += duals[ci]
            
            # Combine score with LP value to break ties or guide if duals are similar
            # If duals are all 1, this is standard greedy.
            # We can add a small epsilon * lp_val to favor vertices that LP thinks are important
            total_score = score + 0.1 * lp_val
            
            if total_score > best_score:
                best_score = total_score
                best_v = v
                best_ties = [v]
            elif total_score == best_score and total_score > 0:
                best_ties.append(v)
        
        if best_v is None or best_score <= 0:
            # Fallback: pick any vertex that covers an uncovered clique
            for v in range(n):
                if v in selected:
                    continue
                covers_any = False
                for ci in range(num_cliques):
                    if A[ci, v] > 0 and ci in uncovered:
                        covers_any = True
                        break
                if covers_any:
                    best_v = v
                    break
            
            if best_v is None:
                break # Should not happen if feasible
        
        # Pick from ties randomly if we want diversity, but deterministic is safer for reproducibility
        # Let's pick the one with highest x_lp among ties
        if len(best_ties) > 1:
            best_v = max(best_ties, key=lambda v: x_lp[v])
            
        selected.add(best_v)
        for ci in range(num_cliques):
            if A[ci, best_v] > 0 and ci in uncovered:
                uncovered.remove(ci)
        
        if not uncovered:
            break

    # Local Search: Try to remove vertices
    # Check if each selected vertex is necessary
    for v in list(selected):
        # Temporarily remove v
        selected.remove(v)
        # Check coverage
        all_covered = True
        for ci in range(num_cliques):
            covered = False
            for sv in selected:
                if A[ci, sv] > 0:
                    covered = True
                    break
            if not covered:
                all_covered = False
                break
        
        if all_covered:
            # v is redundant, keep it removed
            pass
        else:
            # v is needed, put it back
            selected.add(v)
    
    # Second pass: Try swaps?
    # For simplicity and speed, stick to removal.
    
    return selected

def _simple_greedy(cliques, n):
    """Fallback greedy solver."""
    if not cliques:
        return set()
    
    num_cliques = len(cliques)
    vertex_to_cliques = defaultdict(list)
    for ci, clique in enumerate(cliques):
        for v in clique:
            vertex_to_cliques[v].append(ci)
    
    selected = set()
    uncovered = set(range(num_cliques))
    
    while uncovered:
        best_v = None
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = sum(1 for ci in vertex_to_cliques[v] if ci in uncovered)
            if count > best_count:
                best_count = count
                best_v = v
        
        if best_v is None or best_count == 0:
            break
        
        selected.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            if ci in uncovered:
                uncovered.remove(ci)
        
        if not uncovered:
            break
    
    return selected

def construct():
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        edges = _random_graph(n, p, seed)
        
        # Build adjacency
        adj = [set() for _ in range(n)]
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Find all maximal cliques of size >= 2
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        cliques = [c for c in cliques if len(c) >= 2]
        
        # Solve set cover
        if not cliques:
            out.append([])
        else:
            selected = _solve_set_cover_lp_guided(cliques, n)
            out.append(sorted(selected))
    
    return out
# EVOLVE-BLOCK-END
