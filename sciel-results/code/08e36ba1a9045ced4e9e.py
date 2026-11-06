"""Bootstrap direct approach: meet every maximal clique (size>=2) with a minimal-ish transversal."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

_DEADLINE = 150.0
_START_TIME = time.time()

def _remaining():
    return _DEADLINE - time.time()

def construct():
    start = time.time()
    out = []
    for n, p, seed in ((30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)):
        if time.time() - start > _DEADLINE - 5:
            rng = random.Random(seed)
            edges = [(u, v) for u in range(n) for v in range(u + 1, n) if rng.random() < p]
            touched = sorted({x for e in edges for x in e})
            out.append(touched)
            continue
        sol = _solve_instance(n, p, seed)
        out.append(sol)
    return out

def _solve_instance(n, p, seed):
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    
    cliques = []
    def bronk(R, P, X):
        if time.time() - _START_TIME > _DEADLINE - 10:
            return
        if not P and not X:
            if len(R) >= 2:
                cliques.append(frozenset(R))
            return
        best_pivot = None
        max_count = -1
        for v in P:
            c = len(adj[v] & (P | X))
            if c > max_count:
                max_count = c
                best_pivot = v
        if best_pivot is not None:
            candidates = P - adj[best_pivot]
        else:
            candidates = set(P)
        for v in list(candidates):
            bronk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bronk(set(), set(range(n)), set())
    cliques = [c for c in cliques if len(c) >= 2]
    
    if not cliques:
        return []
    
    milp_sol = _solve_milp(n, cliques)
    if milp_sol is not None and time.time() - _START_TIME < _DEADLINE - 15:
        verified = _local_search_fast(n, cliques, milp_sol)
        return verified
    
    greedy_sol = _greedy_set_cover(n, cliques)
    return _local_search_fast(n, cliques, greedy_sol)

def _solve_milp(n, cliques):
    """Solve the minimum clique transversal as a 0-1 ILP using scipy's MILP solver."""
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        import numpy as np
    except ImportError:
        return None
    
    if not cliques:
        return set()
    
    # Variables: x[v] = 1 if vertex v is selected
    # Minimize: sum(x[v])
    # Subject to: for each clique C, sum_{v in C} x[v] >= 1
    #             x[v] in {0, 1}
    
    m = len(cliques)
    
    # Objective: minimize sum of x
    c = np.ones(n)
    
    # Constraints: A @ x >= 1 for each clique
    # Build constraint matrix
    A_rows = []
    for clique in cliques:
        row = np.zeros(n)
        for v in clique:
            row[v] = 1.0
        A_rows.append(row)
    
    if not A_rows:
        return set()
    
    A = np.array(A_rows)
    lb = np.ones(m)  # Each clique must be hit at least once
    ub = np.full(m, np.inf)
    
    # Variable bounds: 0 <= x[v] <= 1, integer
    var_lb = np.zeros(n)
    var_ub = np.ones(n)
    integrality = np.ones(n)  # All variables are binary
    
    try:
        res = milp(
            c=c,
            constraints=LinearConstraint(A, lb, ub),
            integrality=integrality,
            bounds=Bounds(var_lb, var_ub)
        )
        
        if res.status == 0:  # Success
            selected = set()
            for v in range(n):
                if res.x[v] > 0.5:
                    selected.add(v)
            return selected
        else:
            return None
    except Exception:
        return None

def _local_search_fast(n, cliques, initial_sol):
    """Fast local search with proper bookkeeping for clique coverage."""
    selected = set(initial_sol)
    
    # Build incidence: for each vertex, list of clique indices it covers
    vertex_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_cliques[v].append(i)
    
    # Track coverage count for each clique
    m = len(cliques)
    cover_count = [0] * m
    for v in selected:
        for ci in vertex_cliques[v]:
            cover_count[ci] += 1
    
    # Remove redundant vertices
    changed = True
    while changed and time.time() - _START_TIME < _DEADLINE - 10:
        changed = False
        for v in list(selected):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            redundant = True
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    redundant = False
                    break
            if redundant:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                selected.discard(v)
                changed = True
                break
    
    # 1-exchange: try removing v_out and adding v_in to enable further reductions
    improved = True
    max_iterations = 50
    for _ in range(max_iterations):
        if time.time() - _START_TIME > _DEADLINE - 10:
            break
        improved = False
        best_swap = None
        best_potential = 0
        
        for v_out in list(selected):
            if time.time() - _START_TIME > _DEADLINE - 10:
                break
            at_risk = [ci for ci in vertex_cliques[v_out] if cover_count[ci] == 1]
            if not at_risk:
                continue
            
            for v_in in range(n):
                if v_in in selected:
                    continue
                covers_all = True
                for ci in at_risk:
                    if ci not in vertex_cliques[v_in]:
                        covers_all = False
                        break
                if not covers_all:
                    continue
                
                extra = 0
                for ci in vertex_cliques[v_in]:
                    if ci not in vertex_cliques[v_out] and cover_count[ci] >= 1:
                        extra += 1
                
                if extra > best_potential:
                    best_potential = extra
                    best_swap = (v_out, v_in)
        
        if best_swap and best_potential > 0:
            v_out, v_in = best_swap
            for ci in vertex_cliques[v_out]:
                cover_count[ci] -= 1
            for ci in vertex_cliques[v_in]:
                cover_count[ci] += 1
            selected.discard(v_out)
            selected.add(v_in)
            improved = True
        
        # After swap, try to remove redundant vertices again
        changed = True
        while changed and time.time() - _START_TIME < _DEADLINE - 10:
            changed = False
            for v in list(selected):
                if time.time() - _START_TIME > _DEADLINE - 10:
                    break
                redundant = True
                for ci in vertex_cliques[v]:
                    if cover_count[ci] == 1:
                        redundant = False
                        break
                if redundant:
                    for ci in vertex_cliques[v]:
                        cover_count[ci] -= 1
                    selected.discard(v)
                    changed = True
                    break
        
        if not improved:
            break
    
    return sorted(selected)

def _greedy_set_cover(n, cliques):
    """Greedy set cover: select vertices to hit all cliques."""
    selected = set()
    
    vertex_to_cliques = defaultdict(list)
    for i, c in enumerate(cliques):
        for v in c:
            vertex_to_cliques[v].append(i)
    
    uncovered = set(range(len(cliques)))
    
    while uncovered and time.time() - _START_TIME < _DEADLINE - 15:
        best_v = None
        best_count = 0
        for v in range(n):
            count = 0
            for ci in vertex_to_cliques[v]:
                if ci in uncovered:
                    count += 1
                    if count <= best_count:
                        continue
            if count > best_count:
                best_count = count
                best_v = v
                if count == len(uncovered):
                    break
        if best_v is None or best_count == 0:
            break
        selected.add(best_v)
        for ci in vertex_to_cliques[best_v]:
            uncovered.discard(ci)
    
    for ci in uncovered:
        v = next(iter(cliques[ci]))
        selected.add(v)
    
    return selected

# EVOLVE-BLOCK-END
