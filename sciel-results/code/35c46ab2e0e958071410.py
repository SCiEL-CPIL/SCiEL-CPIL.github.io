"""Combined program: Bron-Kerbosch + greedy set cover + local search + branch-and-bound + ILP polish."""
import time
import random
import numpy as np
from scipy.optimize import milp, Bounds, LinearConstraint

# EVOLVE-BLOCK-START

def _build_graph(n, p, seed):
    """Build adjacency sets from the same PRNG stream as the seed program."""
    rng = random.Random(seed)
    adj = [set() for _ in range(n)]
    for u in range(n):
        for v in range(u + 1, n):
            if rng.random() < p:
                adj[u].add(v)
                adj[v].add(u)
    return adj

def _bron_kerbosch_max_cliques(adj, min_size=2):
    """Enumerate all maximal cliques of size >= min_size using Bron-Kerbosch with pivot."""
    n = len(adj)
    cliques = []
    
    def bk(R, P, X):
        if not P and not X:
            if len(R) >= min_size:
                cliques.append(frozenset(R))
            return
        U = P | X
        if not U:
            return
        pivot = max(U, key=lambda u: len(P & adj[u]))
        candidates = P - adj[pivot]
        for v in list(candidates):
            R.add(v)
            P_new = P & adj[v]
            X_new = X & adj[v]
            bk(R, P_new, X_new)
            R.remove(v)
            P.discard(v)
            X.add(v)
    
    bk(set(), set(range(n)), set())
    return cliques

def _validate_solution(selected, cliques):
    """Check if selected vertices meet every maximal clique of size >= 2."""
    selected_set = set(selected)
    for clique in cliques:
        if not (selected_set & clique):
            return False
    return True

def _greedy_set_cover(cliques):
    """Greedy set cover: repeatedly pick vertex covering most uncovered cliques."""
    if not cliques:
        return []
    
    n_cliques = len(cliques)
    vertex_to_cliques = {}
    for i, clique in enumerate(cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = []
            vertex_to_cliques[v].append(i)
    
    covered = [False] * n_cliques
    selected = []
    uncovered_count = n_cliques
    
    while uncovered_count > 0:
        best_v = None
        best_count = -1
        for v, clist in vertex_to_cliques.items():
            count = sum(1 for idx in clist if not covered[idx])
            if count > best_count:
                best_count = count
                best_v = v
        if best_v is None or best_count == 0:
            break
        selected.append(best_v)
        for idx in vertex_to_cliques[best_v]:
            if not covered[idx]:
                covered[idx] = True
                uncovered_count -= 1
    
    return sorted(selected)

def _local_search(adj, initial_solution, cliques, max_iters=5000, time_budget=5.0):
    """Optimized local search: removals, 1-1 swaps, and 2-1 / 1-2 moves."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    n_cliques = len(cliques)
    
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    cover_count = [0] * n_cliques
    current_set = set(current)
    for v in current_set:
        for ci in vertex_cliques[v]:
            cover_count[ci] += 1
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
        improved = False
        current_set = set(current)
        
        # 1. Try simple removals
        for v in list(current):
            safe_to_remove = True
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    safe_to_remove = False
                    break
            if safe_to_remove:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
        
        if improved:
            continue
        
        # 2. Try 1-1 swaps
        for v in list(current):
            deficit_cliques = []
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    deficit_cliques.append(ci)
            
            if not deficit_cliques:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
            
            if len(deficit_cliques) == 1:
                ci = deficit_cliques[0]
                candidates = [u for u in cliques[ci] if u not in current_set]
                if candidates:
                    best_u = max(candidates, key=lambda u: len(vertex_cliques[u]))
                    for c1 in vertex_cliques[v]:
                        cover_count[c1] -= 1
                    for c2 in vertex_cliques[best_u]:
                        cover_count[c2] += 1
                    current_set.remove(v)
                    current_set.add(best_u)
                    current = list(current_set)
                    improved = True
                    break
            else:
                potential_u = set(cliques[deficit_cliques[0]])
                for ci in deficit_cliques[1:]:
                    potential_u &= set(cliques[ci])
                potential_u -= current_set
                if potential_u:
                    best_u = max(potential_u, key=lambda u: len(vertex_cliques[u]))
                    for c1 in vertex_cliques[v]:
                        cover_count[c1] -= 1
                    for c2 in vertex_cliques[best_u]:
                        cover_count[c2] += 1
                    current_set.remove(v)
                    current_set.add(best_u)
                    current = list(current_set)
                    improved = True
                    break
        
        if improved:
            continue
        
        # 3. Try 2-1 moves
        cur_list = list(current)
        for i in range(len(cur_list)):
            for j in range(i + 1, len(cur_list)):
                v1, v2 = cur_list[i], cur_list[j]
                deficit = set()
                for ci in vertex_cliques[v1]:
                    if cover_count[ci] == 1:
                        deficit.add(ci)
                for ci in vertex_cliques[v2]:
                    if cover_count[ci] == 1:
                        deficit.add(ci)
                if not deficit:
                    for ci in vertex_cliques[v1]:
                        cover_count[ci] -= 1
                    for ci in vertex_cliques[v2]:
                        cover_count[ci] -= 1
                    current_set.discard(v1)
                    current_set.discard(v2)
                    current = list(current_set)
                    improved = True
                    break
                if len(deficit) <= 1:
                    if len(deficit) == 1:
                        ci = next(iter(deficit))
                        candidates = [u for u in cliques[ci] if u not in current_set]
                    else:
                        candidates = []
                    if candidates:
                        best_u = max(candidates, key=lambda u: len(vertex_cliques[u]))
                        for ci in vertex_cliques[v1]:
                            cover_count[ci] -= 1
                        for ci in vertex_cliques[v2]:
                            cover_count[ci] -= 1
                        for ci in vertex_cliques[best_u]:
                            cover_count[ci] += 1
                        current_set.discard(v1)
                        current_set.discard(v2)
                        current_set.add(best_u)
                        current = list(current_set)
                        improved = True
                        break
                else:
                    pot = set(cliques[next(iter(deficit))])
                    for ci in list(deficit)[1:]:
                        pot &= set(cliques[ci])
                    pot -= current_set
                    if pot:
                        best_u = max(pot, key=lambda u: len(vertex_cliques[u]))
                        for ci in vertex_cliques[v1]:
                            cover_count[ci] -= 1
                        for ci in vertex_cliques[v2]:
                            cover_count[ci] -= 1
                        for ci in vertex_cliques[best_u]:
                            cover_count[ci] += 1
                        current_set.discard(v1)
                        current_set.discard(v2)
                        current_set.add(best_u)
                        current = list(current_set)
                        improved = True
                        break
            if improved:
                break
        
        if improved:
            continue
        
        # 4. Try 1-2 moves
        for v in list(current):
            deficit = set()
            for ci in vertex_cliques[v]:
                if cover_count[ci] == 1:
                    deficit.add(ci)
            if not deficit:
                for ci in vertex_cliques[v]:
                    cover_count[ci] -= 1
                current_set.remove(v)
                current = list(current_set)
                improved = True
                break
            if len(deficit) >= 2:
                all_candidates = set()
                for ci in deficit:
                    all_candidates.update(cliques[ci])
                all_candidates -= current_set
                cand_list = list(all_candidates)
                found = False
                for a_idx in range(len(cand_list)):
                    u1 = cand_list[a_idx]
                    covered_by_u1 = set(vertex_cliques[u1])
                    remaining = deficit - covered_by_u1
                    if not remaining:
                        for ci in vertex_cliques[v]:
                            cover_count[ci] -= 1
                        for ci in vertex_cliques[u1]:
                            cover_count[ci] += 1
                        current_set.remove(v)
                        current_set.add(u1)
                        current = list(current_set)
                        improved = True
                        found = True
                        break
                    for b_idx in range(a_idx + 1, len(cand_list)):
                        u2 = cand_list[b_idx]
                        covered_by_u2 = set(vertex_cliques[u2])
                        if remaining <= (covered_by_u1 | covered_by_u2):
                            for ci in vertex_cliques[v]:
                                cover_count[ci] -= 1
                            for ci in vertex_cliques[u1]:
                                cover_count[ci] += 1
                            for ci in vertex_cliques[u2]:
                                cover_count[ci] += 1
                            current_set.remove(v)
                            current_set.add(u1)
                            current_set.add(u2)
                            current = list(current_set)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
        
        if not improved:
            break
    
    return sorted(current)

def _final_polish(adj, initial_solution, cliques, time_budget=10.0):
    """Aggressive final polish: many small perturbations with strict time control."""
    start = time.time()
    current = list(initial_solution)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    n_cliques = len(cliques)
    
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    cover_count = [0] * n_cliques
    current_set = set(current)
    for v in current_set:
        for ci in vertex_cliques[v]:
            cover_count[ci] += 1
    
    best = list(current)
    best_size = len(current)
    
    # Try many random perturbations
    rng = random.Random(12345)
    
    while time.time() - start < time_budget:
        # Random perturbation: remove 1-2 vertices, add 0-2 vertices
        cur = list(current)
        cur_set = set(cur)
        cur_cover = cover_count[:]
        
        # Remove 1-2 random vertices
        remove_count = rng.randint(1, min(2, len(cur)))
        to_remove = rng.sample(cur, remove_count)
        for v in to_remove:
            for ci in vertex_cliques[v]:
                cur_cover[ci] -= 1
            cur_set.discard(v)
            cur.remove(v)
        
        # Check if still valid; if not, try to fix by adding vertices
        if all(cur_cover[ci] >= 1 for ci in range(n_cliques)):
            # Valid! Update if better
            if len(cur) < best_size:
                best = cur
                best_size = len(cur)
                current = cur
                current_set = cur_set
                cover_count = cur_cover
            continue
        
        # Find uncovered cliques
        uncovered = [ci for ci in range(n_cliques) if cur_cover[ci] == 0]
        if not uncovered:
            if len(cur) < best_size:
                best = cur
                best_size = len(cur)
                current = cur
                current_set = cur_set
                cover_count = cur_cover
            continue
        
        # Try to add 1-2 vertices to cover all uncovered
        add_count = min(2, len(uncovered))
        added = []
        remaining_uncovered = set(uncovered)
        
        for _ in range(add_count):
            if not remaining_uncovered:
                break
            # Find vertex covering most remaining uncovered
            best_v = None
            best_count = -1
            for v in range(n):
                if v in cur_set:
                    continue
                count = sum(1 for ci in vertex_cliques[v] if ci in remaining_uncovered)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            cur_set.add(best_v)
            cur.append(best_v)
            added.append(best_v)
            for ci in vertex_cliques[best_v]:
                cur_cover[ci] += 1
                if cur_cover[ci] >= 1:
                    remaining_uncovered.discard(ci)
        
        # Check if valid
        if all(cur_cover[ci] >= 1 for ci in range(n_cliques)):
            if len(cur) < best_size:
                best = cur
                best_size = len(cur)
                current = cur
                current_set = cur_set
                cover_count = cur_cover
    
    return sorted(best)

def _ilp_solve(cliques, upper_bound, time_budget=10.0):
    """Solve the hitting set problem as an integer linear program using scipy's HiGHS solver."""
    start = time.time()
    n_cliques = len(cliques)
    if n_cliques == 0:
        return []
    
    # Collect all vertices
    all_vertices = set()
    for clique in cliques:
        all_vertices.update(clique)
    vertex_list = sorted(all_vertices)
    vertex_to_idx = {v: i for i, v in enumerate(vertex_list)}
    n_vertices = len(vertex_list)
    
    if n_vertices == 0:
        return []
    
    # Build constraint matrix: each row is a clique, each column is a vertex
    # Constraint: for each clique, sum of selected vertices in clique >= 1
    A = np.zeros((n_cliques, n_vertices), dtype=np.int8)
    for i, clique in enumerate(cliques):
        for v in clique:
            A[i, vertex_to_idx[v]] = 1
    
    # Objective: minimize sum of x_v
    c = np.ones(n_vertices, dtype=np.float64)
    
    # Constraints: A @ x >= 1 for all cliques
    # In scipy milp: LinearConstraint(A, lb, ub) means lb <= A@x <= ub
    # We want A@x >= 1, so lb = 1, ub = inf
    lb = np.ones(n_cliques, dtype=np.float64)
    ub = np.full(n_cliques, np.inf, dtype=np.float64)
    
    # Variable bounds: 0 <= x_v <= 1
    integrality = np.ones(n_vertices, dtype=np.int8)  # all integer
    var_bounds = Bounds(lb=np.zeros(n_vertices), ub=np.ones(n_vertices))
    
    constraint = LinearConstraint(A, lb=lb, ub=ub)
    
    # Add upper bound constraint: sum(x) <= upper_bound - 1 (to find better solutions)
    # Actually, let's just solve and see what we get
    try:
        result = milp(
            c=c,
            constraints=constraint,
            integrality=integrality,
            bounds=var_bounds,
            options={"time_limit": time_budget, "mip_rel_gap": 0.0}
        )
        
        if result.status == 0 or result.status == 1:  # optimal or time limit
            if result.x is not None:
                selected = [vertex_list[i] for i in range(n_vertices) if result.x[i] > 0.5]
                if len(selected) < upper_bound:
                    return selected
    except Exception:
        pass
    
    return None

def _random_local_search(adj, cliques, time_budget=5.0):
    """Generate random candidate solutions and improve via local search."""
    start = time.time()
    n = len(adj)
    rng = random.Random(42)
    
    if not cliques:
        return []
    
    best = None
    best_size = float('inf')
    
    greedy_sol = _greedy_set_cover(cliques)
    if _validate_solution(greedy_sol, cliques):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    n_cliques = len(cliques)
    vertex_cliques = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        max_subset = min(n, best_size + 1 if best else 5)
        subset_size = rng.randint(0, max_subset)
        candidate = set(rng.sample(range(n), subset_size))
        
        covered_cliques = set()
        for i, clique in enumerate(cliques):
            if candidate & clique:
                covered_cliques.add(i)
        
        while len(covered_cliques) < n_cliques:
            uncovered = [i for i in range(n_cliques) if i not in covered_cliques]
            if not uncovered:
                break
            
            best_v = None
            best_count = -1
            candidates = set()
            for i in uncovered:
                candidates.update(cliques[i])
            
            for v in candidates:
                count = len(vertex_cliques[v] & set(uncovered))
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            candidate.add(best_v)
            covered_cliques |= vertex_cliques[best_v]
        
        if len(covered_cliques) < n_cliques:
            continue
        
        if len(candidate) < best_size:
            improved = _local_search(adj, list(candidate), cliques, max_iters=1000, time_budget=0.3)
            if _validate_solution(improved, cliques) and len(improved) < best_size:
                best = improved
                best_size = len(improved)
    
    if best is None:
        return greedy_sol
    return best

def _solve_instance(adj, cliques, time_budget=30.0):
    """Solve one instance: greedy + local search + ILP + B&B polish + final polish."""
    if not cliques:
        return []
    
    start = time.time()
    
    greedy_sol = _greedy_set_cover(cliques)
    if not _validate_solution(greedy_sol, cliques):
        touched = set()
        for clique in cliques:
            touched.update(clique)
        greedy_sol = sorted(touched)
    
    best = greedy_sol
    best_size = len(greedy_sol)
    
    # Local search on greedy solution
    remaining = time_budget - (time.time() - start)
    if remaining > 1:
        ls_sol = _local_search(adj, list(greedy_sol), cliques, max_iters=8000, time_budget=min(remaining, 8))
        if _validate_solution(ls_sol, cliques) and len(ls_sol) < best_size:
            best = ls_sol
            best_size = len(ls_sol)
    
    # ILP solve - this can find optimal or near-optimal solutions
    remaining = time_budget - (time.time() - start)
    if remaining > 5 and len(cliques) <= 500:  # ILP is practical for moderate clique counts
        ilp_sol = _ilp_solve(cliques, best_size, time_budget=min(remaining, 15))
        if ilp_sol is not None and _validate_solution(ilp_sol, cliques) and len(ilp_sol) < best_size:
            best = ilp_sol
            best_size = len(ilp_sol)
    
    # Random local search for diversity
    remaining = time_budget - (time.time() - start)
    if remaining > 3:
        rs_sol = _random_local_search(adj, cliques, time_budget=min(remaining, 10))
        if _validate_solution(rs_sol, cliques) and len(rs_sol) < best_size:
            best = rs_sol
            best_size = len(rs_sol)
    
    # Final polish: aggressive perturbation search
    remaining = time_budget - (time.time() - start)
    if remaining > 2:
        polish_sol = _final_polish(adj, list(best), cliques, time_budget=min(remaining, 6))
        if _validate_solution(polish_sol, cliques) and len(polish_sol) < best_size:
            best = polish_sol
    
    return best

def construct():
    deadline = time.time() + 145
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time based on expected difficulty (more edges = more cliques = harder)
    total_time = 130
    # Use clique count estimates for better allocation
    # n*p*(n-1)/2 gives expected edge count, which correlates with clique complexity
    edge_estimates = [n * p * (n - 1) / 2 for n, p, _ in instances]
    total_edges = sum(edge_estimates)
    time_alloc = [total_time * e / total_edges for e in edge_estimates]
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_total = deadline - time.time()
        if remaining_total < 5:
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        budget = min(time_alloc[i], remaining_total - 10)
        if budget < 5:
            budget = max(3, remaining_total / 2)
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
        
        if not cliques:
            results.append([])
            continue
        
        sol = _solve_instance(adj, cliques, time_budget=budget)
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
