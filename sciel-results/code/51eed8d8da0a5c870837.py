"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random as _random
import numpy as np

def _generate_graph(n, p, seed):
    """Generate graph edges using the same RNG sequence as the verifier."""
    rng = _random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_bitmask(adj, n):
    """Enumerate all maximal cliques of size >= 2 using Bron-Kerbosch with pivot, bitmask-based."""
    # Precompute adjacency as bitmasks
    adj_mask = [0] * n
    for u in range(n):
        mask = 0
        for v in adj[u]:
            mask |= (1 << v)
        adj_mask[u] = mask
    
    cliques = []
    
    def bk(R_mask, P_mask, X_mask):
        if P_mask == 0 and X_mask == 0:
            if R_mask != 0:
                # Check size >= 2
                if bin(R_mask).count('1') >= 2:
                    cliques.append(R_mask)
            return
        
        U = P_mask | X_mask
        if U == 0:
            return
        
        # Choose pivot with max degree in U
        best_pivot = -1
        best_deg = -1
        u = U
        while u:
            lb = u & (-u)
            v = lb.bit_length() - 1
            deg = bin(adj_mask[v] & U).count('1')
            if deg > best_deg:
                best_deg = deg
                best_pivot = v
            u ^= lb
        
        candidates = P_mask & ~adj_mask[best_pivot]
        
        c = candidates
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            R_new = R_mask | lb
            P_new = P_mask & adj_mask[v]
            X_new = X_mask & adj_mask[v]
            bk(R_new, P_new, X_new)
            P_mask ^= lb
            X_mask |= lb
            c ^= lb
    
    bk(0, (1 << n) - 1, 0)
    return cliques

def _to_bitmasks(cliques, n):
    """Convert list of clique bitmasks to list of frozensets for easier handling if needed."""
    return cliques  # already bitmasks

def _greedy_hitting_set_bitmask(cliques, n):
    """Greedy set cover: repeatedly pick vertex that covers most uncovered cliques (bitmask-based)."""
    if not cliques:
        return set()
    
    m = len(cliques)
    # For each vertex, precompute bitmask of cliques it belongs to
    vertex_clique_mask = [0] * n
    for i, c_mask in enumerate(cliques):
        c = c_mask
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            vertex_clique_mask[v] |= (1 << i)
            c ^= lb
    
    uncovered = (1 << m) - 1
    selected = set()
    
    while uncovered:
        best_v = -1
        best_count = -1
        for v in range(n):
            if v in selected:
                continue
            count = bin(vertex_clique_mask[v] & uncovered).count('1')
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        selected.add(best_v)
        uncovered &= ~vertex_clique_mask[best_v]
    
    return selected

def _local_search_bitmask(selected, cliques, n, adj, time_limit):
    """Local search: try deletions and 2-opt swaps to reduce size (bitmask-based)."""
    start_time = time.time()
    selected = set(selected)
    m = len(cliques)
    
    # Precompute for each vertex, bitmask of cliques it covers
    vertex_clique_mask = [0] * n
    for i, c_mask in enumerate(cliques):
        c = c_mask
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            vertex_clique_mask[v] |= (1 << i)
            c ^= lb
    
    def is_valid(sel):
        """Check if sel covers all cliques."""
        covered = 0
        for v in sel:
            covered |= vertex_clique_mask[v]
        return covered == (1 << m) - 1
    
    # Try deletions
    changed = True
    while changed:
        changed = False
        for v in list(selected):
            new_sel = selected - {v}
            if is_valid(new_sel):
                selected = new_sel
                changed = True
                if time.time() - start_time > time_limit:
                    return selected
    
    # Try 2-opt swaps: remove one vertex, add another, if size doesn't increase and coverage maintained
    changed = True
    while changed:
        changed = False
        if time.time() - start_time > time_limit:
            return selected
        selected_list = list(selected)
        for i in range(len(selected_list)):
            for j in range(i + 1, len(selected_list)):
                v1, v2 = selected_list[i], selected_list[j]
                # Try to find a vertex w not in selected that covers all cliques that v1 or v2 uniquely cover
                # Cliques that need v1 or v2 (i.e., cliques where the only selected vertices are subset of {v1, v2})
                critical_cliques = 0
                for ci, c_mask in enumerate(cliques):
                    sel_in_c = 0
                    for x in selected:
                        if c_mask & (1 << x):
                            sel_in_c |= (1 << x)
                    if v1 in c_mask and v2 in c_mask:
                        pass
                    elif v1 in c_mask:
                        if sel_in_c == (1 << v1):
                            critical_cliques |= (1 << ci)
                    elif v2 in c_mask:
                        if sel_in_c == (1 << v2):
                            critical_cliques |= (1 << ci)
                
                if not critical_cliques:
                    continue
                
                # Find a vertex w not in selected that covers all critical_cliques
                for w in range(n):
                    if w in selected:
                        continue
                    if (vertex_clique_mask[w] & critical_cliques) == critical_cliques:
                        # Check that after removing v1, v2 and adding w, all cliques are still covered
                        new_selected = (selected - {v1, v2}) | {w}
                        if is_valid(new_selected):
                            selected = new_selected
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break
    
    return selected

def _milp_solve(cliques, n, time_limit):
    """Try to solve the hitting set problem using MILP via scipy."""
    m = len(cliques)
    if m == 0:
        return set()
    
    # Build constraint matrix
    A = np.zeros((m, n), dtype=np.int8)
    for i, clique_mask in enumerate(cliques):
        c = clique_mask
        while c:
            lb = c & (-c)
            v = lb.bit_length() - 1
            A[i, v] = 1
            c ^= lb
    
    c = np.ones(n, dtype=np.float64)
    
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        
        constraints = LinearConstraint(A, 1, np.inf)
        bounds_obj = Bounds(0, 1)
        
        res = milp(c=c, constraints=constraints, integrality=np.ones(n, dtype=np.int8), 
                   bounds=bounds_obj, 
                   options={"time_limit": time_limit, "presolve": True})
        
        if res.x is not None:
            sol = set(i for i in range(n) if res.x[i] > 0.5)
            # Verify validity
            valid = True
            for c_clique in cliques:
                if not any(v in c_clique for v in sol):
                    valid = False
                    break
            if valid:
                return sol
    except Exception:
        pass
    
    return None

def construct():
    deadline = time.time() + 140.0  # Safety margin before 150s kill
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 30.0:
            # Not enough time left; return trivial solution for remaining
            adj = _generate_graph(n, p, seed)
            touched = set()
            for u in range(n):
                for v in adj[u]:
                    touched.add(u)
                    touched.add(v)
            results.append(sorted(touched))
            continue
        
        adj = _generate_graph(n, p, seed)
        
        # Enumerate maximal cliques
        cliques = _bron_kerbosch_bitmask(adj, n)
        
        if not cliques:
            results.append([])
            continue
        
        # First try MILP with a reasonable time budget
        time_left = deadline - time.time() - 10.0
        milp_budget = min(20.0, time_left * 0.5)  # Give MILP up to half the remaining time, capped at 20s
        
        selected = _milp_solve(cliques, n, milp_budget)
        
        if selected is None:
            # Fall back to greedy
            selected = _greedy_hitting_set_bitmask(cliques, n)
        
        # Always run local search to polish
        time_left = deadline - time.time() - 10.0
        if time_left > 1.0:
            selected = _local_search_bitmask(selected, cliques, n, adj, time_left)
        
        results.append(sorted(selected))
    
    return results
# EVOLVE-BLOCK-END
