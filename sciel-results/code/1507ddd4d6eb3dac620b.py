"""Solve the transversal problem: minimum set of vertices hitting every maximal clique (size>=2).

Key insight: A hitting set for all maximal cliques of size >= 2 is equivalent to a
vertex cover. By König's theorem duality (complement), vertex cover = V \\ MIS.
So we find the Maximum Independent Set (MIS), and the answer is V \\ MIS.

Strategy:
1. Generate each random graph.
2. Formulate MIS as ILP: maximize sum(x_v) s.t. x_u + x_v <= 1 for each edge.
3. Solve with scipy.optimize.milp (HiGHS) with time budget.
4. Polish with aggressive local search (redundancy removal, 1-swap, 2-for-1, 2-for-2).
5. Fallback to greedy if needed.
"""

import time
from random import Random
import numpy as np

def _random_graph_edges(n, p, seed):
    """Generate edges of a random graph using the same procedure as the seed program."""
    rng = Random(seed)
    return [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]

def _build_adjacency(n, edges):
    """Build adjacency sets from edge list."""
    adj = [set() for _ in range(n)]
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj

def _bron_kerbosch_maximal_cliques(n, adj):
    """Enumerate all maximal cliques using Bron-Kerbosch with pivot."""
    cliques = []
    
    def bronk(R, P, X):
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        PX = P | X
        if not PX:
            return
        best_pivot = None
        best_count = -1
        for u in PX:
            count = len(P & adj[u])
            if count > best_count:
                best_count = count
                best_pivot = u
        candidates = list(P - adj[best_pivot])
        for v in candidates:
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(frozenset(), set(range(n)), set())
    return cliques

def _verify_hitting_set(selected, cliques):
    """Verify that selected vertices hit every clique."""
    selected_set = set(selected)
    for clique in cliques:
        if not (clique & selected_set):
            return False
    return True

def _greedy_hitting_set(n, cliques):
    """Greedy heuristic for minimum hitting set."""
    if not cliques:
        return []
    remaining_cliques = [set(c) for c in cliques]
    selected = set()
    vertex_cliques = {v: [] for v in range(n)}
    for i, clique in enumerate(remaining_cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    uncovered = [True] * len(remaining_cliques)
    uncovered_count = len(remaining_cliques)
    while uncovered_count > 0:
        best_v = -1
        best_score = -1
        for v in range(n):
            if v in selected:
                continue
            cov_count = 0
            for ci in vertex_cliques[v]:
                if uncovered[ci]:
                    cov_count += 1
            if cov_count > best_score:
                best_score = cov_count
                best_v = v
        if best_v == -1:
            break
        selected.add(best_v)
        for ci in vertex_cliques[best_v]:
            if uncovered[ci]:
                uncovered[ci] = False
                uncovered_count -= 1
    return sorted(selected)

def _solve_mis_ilp(n, edges, time_limit=30.0):
    """Solve Maximum Independent Set via ILP. Returns the MIS as a set of vertices."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        m = len(edges)
        if m == 0:
            return set(range(n))
        
        # Build constraint matrix: for each edge (u,v), x_u + x_v <= 1
        A = np.zeros((m, n))
        for i, (u, v) in enumerate(edges):
            A[i, u] = 1.0
            A[i, v] = 1.0
        
        # Maximize sum(x_v) => minimize -sum(x_v)
        c = -np.ones(n)
        constraint = LinearConstraint(A, lb=-np.inf, ub=1.0)
        integrality = np.ones(n)
        bounds = Bounds(0, 1)
        
        result = milp(c=c, constraints=constraint, integrality=integrality, 
                      bounds=bounds, options={"time_limit": time_limit})
        
        if result.x is not None:
            mis = set(i for i in range(n) if result.x[i] > 0.5)
            # Verify it's truly independent
            valid = True
            for u, v in edges:
                if u in mis and v in mis:
                    valid = False
                    break
            if valid:
                return mis
        return None
    except Exception:
        return None

def _greedy_mis(n, adj):
    """Greedy heuristic for Maximum Independent Set."""
    # Sort vertices by degree (ascending) - low degree vertices first
    order = sorted(range(n), key=lambda v: len(adj[v]))
    mis = set()
    for v in order:
        if all(u not in mis for u in adj[v]):
            mis.add(v)
    return mis

def _local_search_polish(selected, cliques, n, max_iters=500):
    """Aggressive local search to minimize the hitting set size."""
    selected_set = set(selected)
    clique_sets = [set(c) for c in cliques]
    
    def is_valid(s):
        for c in clique_sets:
            if not (c & s):
                return False
        return True
    
    # Phase 1: Remove redundant vertices (repeat until stable)
    changed = True
    while changed:
        changed = False
        for v in list(selected_set):
            test_set = selected_set - {v}
            if is_valid(test_set):
                selected_set = test_set
                changed = True
    
    # Phase 2: 1-swap (remove one, add one)
    for _ in range(max_iters):
        improved = False
        for v in list(selected_set):
            test_set = selected_set - {v}
            uncovered = [c for c in clique_sets if not (c & test_set)]
            if not uncovered:
                selected_set = test_set
                improved = True
                break
            if uncovered:
                common = set(uncovered[0])
                for c in uncovered[1:]:
                    common &= c
                for w in common:
                    if w not in selected_set:
                        new_set = test_set | {w}
                        if len(new_set) < len(selected_set):
                            selected_set = new_set
                            improved = True
                            break
                if improved:
                    break
        if not improved:
            break
    
    # Phase 3: 2-for-1 (remove two, add one)
    for _ in range(max_iters // 2):
        improved = False
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            if improved:
                break
            for j in range(i + 1, len(sel_list)):
                v1, v2 = sel_list[i], sel_list[j]
                test_set = selected_set - {v1, v2}
                uncovered = [c for c in clique_sets if not (c & test_set)]
                if not uncovered:
                    selected_set = test_set
                    improved = True
                    break
                if uncovered:
                    common = set(uncovered[0])
                    for c in uncovered[1:]:
                        common &= c
                    for w in common:
                        if w not in selected_set:
                            new_set = test_set | {w}
                            if len(new_set) < len(selected_set):
                                selected_set = new_set
                                improved = True
                                break
                    if improved:
                        break
        if not improved:
            break
    
    # Phase 4: 2-for-2 (remove two, add two)
    for _ in range(max_iters // 3):
        improved = False
        sel_list = list(selected_set)
        for i in range(len(sel_list)):
            if improved:
                break
            for j in range(i + 1, len(sel_list)):
                if improved:
                    break
                v1, v2 = sel_list[i], sel_list[j]
                test_set = selected_set - {v1, v2}
                uncovered = [c for c in clique_sets if not (c & test_set)]
                if not uncovered:
                    selected_set = test_set
                    improved = True
                    break
                if uncovered:
                    # Find two vertices that together cover all uncovered cliques
                    common1 = set(uncovered[0])
                    for c in uncovered[1:]:
                        common1 &= c
                    # Try pairs from common1
                    common1_list = sorted(common1 - selected_set)
                    for a_idx in range(len(common1_list)):
                        if improved:
                            break
                        a = common1_list[a_idx]
                        # Check which uncovered cliques are still not hit by a
                        still_uncovered = [c for c in uncovered if a not in c]
                        if not still_uncovered:
                            new_set = test_set | {a}
                            if len(new_set) < len(selected_set):
                                selected_set = new_set
                                improved = True
                                break
                        # Need a second vertex
                        common2 = set(still_uncovered[0]) if still_uncovered else set()
                        for c in still_uncovered[1:]:
                            common2 &= c
                        common2_list = sorted(common2 - selected_set)
                        for b in common2_list:
                            new_set = test_set | {a, b}
                            if len(new_set) < len(selected_set) and is_valid(new_set):
                                selected_set = new_set
                                improved = True
                                break
                        if improved:
                            break
        if not improved:
            break
    
    # Final redundancy removal
    changed = True
    while changed:
        changed = False
        for v in list(selected_set):
            test_set = selected_set - {v}
            if is_valid(test_set):
                selected_set = test_set
                changed = True
    
    return sorted(selected_set)

def construct():
    """Main entry point. Returns three vertex lists, one per instance."""
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin before 150s kill
    
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() > deadline:
            edges = _random_graph_edges(n, p, seed)
            touched = {x for e in edges for x in e}
            out.append(sorted(touched))
            continue
        
        edges = _random_graph_edges(n, p, seed)
        adj = _build_adjacency(n, edges)
        cliques = _bron_kerbosch_maximal_cliques(n, adj)
        
        if not cliques:
            out.append([])
            continue
        
        # Time allocation for ILP
        elapsed = time.time() - start_time
        remaining = deadline - elapsed - 10.0  # reserve 10s for polish + other instances
        ilp_time = max(5.0, min(remaining / 3.0, 40.0))
        
        # Solve MIS via ILP
        mis = _solve_mis_ilp(n, edges, time_limit=ilp_time)
        
        if mis is None:
            # Fallback: greedy MIS
            mis = _greedy_mis(n, adj)
        
        # Transversal = V \ MIS
        selected = sorted(set(range(n)) - mis)
        
        # Verify and polish
        if _verify_hitting_set(selected, cliques):
            if time.time() < deadline - 5.0:
                selected = _local_search_polish(selected, cliques, n, max_iters=300)
        else:
            # ILP gave invalid solution, fall back to greedy hitting set
            selected = _greedy_hitting_set(n, cliques)
            if time.time() < deadline - 5.0:
                selected = _local_search_polish(selected, cliques, n, max_iters=300)
        
        # Final verification
        if not _verify_hitting_set(selected, cliques):
            touched = {x for e in edges for x in e}
            selected = sorted(touched)
        
        out.append(sorted(selected))
    
    return out
