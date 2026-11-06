"""Combined program: Bron-Kerbosch + greedy set cover + local search + branch-and-bound polish."""
import time
import random

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
    # Build vertex -> cliques mapping
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
    
    # Precompute vertex -> list of clique indices
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    # Initialize coverage counts
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
        
        # 3. Try 2-1 moves: remove two vertices, add one that covers all their deficit cliques
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
                    # Both can be removed
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
                    # Find a single vertex covering all deficit cliques
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
                    # Intersect all deficit cliques
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
        
        # 4. Try 1-2 moves: remove one vertex, add two that together cover its deficit
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
                # Find two vertices that together cover all deficit cliques
                # Try pairs from the union of deficit cliques
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
                        # u1 alone covers all deficit
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

def _branch_and_bound(cliques, upper_bound, time_budget=10.0):
    """Branch-and-bound with LP-style lower bound for tighter pruning.
    
    Returns a solution of size < upper_bound if found, else None.
    """
    start = time.time()
    n_cliques = len(cliques)
    if n_cliques == 0:
        return []
    
    # Sort cliques by size (smallest first for better pruning)
    sorted_cliques = sorted(cliques, key=len)
    
    # Build vertex -> set of clique indices it covers
    vertex_to_cliques = {}
    for i, clique in enumerate(sorted_cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = set()
            vertex_to_cliques[v].add(i)
    
    use_bitmask = n_cliques <= 64
    if use_bitmask:
        all_mask = (1 << n_cliques) - 1
    else:
        all_mask = None
    
    best_found = None
    best_size = upper_bound
    
    # Precompute for each clique index, the set of vertices that can cover it
    # and for lower bound: count how many uncovered cliques each remaining vertex can cover
    
    def _lower_bound(idx, covered_mask, covered_set):
        """Compute a lower bound on additional vertices needed.
        
        Uses a greedy packing: repeatedly pick the vertex that covers the most
        uncovered cliques, count how many picks needed.
        This is an admissible lower bound (optimistic).
        """
        if use_bitmask:
            remaining_mask = all_mask & ~covered_mask
        else:
            remaining_mask = None
        
        # Count uncovered cliques from idx onward
        uncovered = []
        for ci in range(idx, n_cliques):
            if use_bitmask:
                if (covered_mask >> ci) & 1:
                    continue
            else:
                if ci in covered_set:
                    continue
            uncovered.append(ci)
        
        if not uncovered:
            return 0
        
        # Greedy packing lower bound
        unc_set = set(uncovered)
        lb = 0
        while unc_set:
            best_v = None
            best_count = 0
            for v, clist in vertex_to_cliques.items():
                count = 0
                for ci in clist:
                    if ci in unc_set:
                        count += 1
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                return float('inf')
            lb += 1
            for ci in vertex_to_cliques[best_v]:
                unc_set.discard(ci)
        
        return lb
    
    def bbsolve(idx, selected, covered_mask, covered_set):
        nonlocal best_found, best_size
        if time.time() - start > time_budget:
            return
        
        if idx == n_cliques:
            if len(selected) < best_size:
                best_size = len(selected)
                best_found = list(selected)
            return
        
        # Pruning: if current selected + lower bound >= best_size, prune
        lb = _lower_bound(idx, covered_mask, covered_set)
        if len(selected) + lb >= best_size:
            return
        
        # Try each vertex in the current clique
        clique = sorted_cliques[idx]
        
        def vertex_score(v):
            if use_bitmask:
                additional = 0
                for ci in vertex_to_cliques.get(v, ()):
                    if ci >= idx and not (covered_mask >> ci) & 1:
                        additional += 1
                return additional
            else:
                additional = 0
                for ci in vertex_to_cliques.get(v, ()):
                    if ci >= idx and ci not in covered_set:
                        additional += 1
                return additional
        
        candidates = sorted(clique, key=vertex_score, reverse=True)
        
        for v in candidates:
            new_selected = selected + [v]
            if len(new_selected) >= best_size:
                continue
            
            if use_bitmask:
                new_mask = covered_mask
                for ci in vertex_to_cliques.get(v, ()):
                    new_mask |= (1 << ci)
                bbsolve(idx + 1, new_selected, new_mask, covered_set)
            else:
                new_set = covered_set | vertex_to_cliques.get(v, set())
                bbsolve(idx + 1, new_selected, covered_mask, new_set)
        
        # Skip this clique if already covered
        if use_bitmask:
            if (covered_mask >> idx) & 1:
                bbsolve(idx + 1, selected, covered_mask, covered_set)
        else:
            if idx in covered_set:
                bbsolve(idx + 1, selected, covered_mask, covered_set)
    
    bbsolve(0, [], 0, set())
    
    if best_found is not None:
        return best_found
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
    
    # First try greedy
    greedy_sol = _greedy_set_cover(cliques)
    if _validate_solution(greedy_sol, cliques):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    # Precompute vertex -> clique indices for fast coverage checks
    n_cliques = len(cliques)
    vertex_cliques = [set() for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].add(i)
    
    # Random restarts
    max_restarts = 50
    for r in range(max_restarts):
        if time.time() - start > time_budget:
            break
        
        # Start with a random subset biased towards smaller sizes
        max_subset = min(n, best_size + 1 if best else 5)
        subset_size = rng.randint(0, max_subset)
        candidate = set(rng.sample(range(n), subset_size))
        
        # Greedily add vertices to cover all cliques
        covered_cliques = set()
        for i, clique in enumerate(cliques):
            if candidate & clique:
                covered_cliques.add(i)
        
        while len(covered_cliques) < n_cliques:
            uncovered = [i for i in range(n_cliques) if i not in covered_cliques]
            if not uncovered:
                break
            
            # Pick vertex that covers most uncovered cliques
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
    """Solve one instance: greedy + local search + B&B polish."""
    if not cliques:
        return []
    
    start = time.time()
    
    # Step 1: Greedy
    greedy_sol = _greedy_set_cover(cliques)
    if not _validate_solution(greedy_sol, cliques):
        touched = set()
        for clique in cliques:
            touched.update(clique)
        greedy_sol = sorted(touched)
    
    best = greedy_sol
    best_size = len(greedy_sol)
    
    # Step 2: Local search on greedy
    remaining = time_budget - (time.time() - start)
    if remaining > 1:
        ls_sol = _local_search(adj, list(greedy_sol), cliques, max_iters=8000, time_budget=min(remaining, 12))
        if _validate_solution(ls_sol, cliques) and len(ls_sol) < best_size:
            best = ls_sol
            best_size = len(ls_sol)
    
    # Step 3: Random restarts
    remaining = time_budget - (time.time() - start)
    if remaining > 2:
        rs_sol = _random_local_search(adj, cliques, time_budget=min(remaining, 15))
        if _validate_solution(rs_sol, cliques) and len(rs_sol) < best_size:
            best = rs_sol
            best_size = len(rs_sol)
    
    # Step 4: B&B polish - try to find a solution strictly smaller than best
    remaining = time_budget - (time.time() - start)
    if remaining > 3 and best_size > 1:
        bb_sol = _branch_and_bound(cliques, best_size, time_budget=min(remaining, 15))
        if bb_sol is not None and _validate_solution(bb_sol, cliques) and len(bb_sol) < best_size:
            best = bb_sol
            best_size = len(bb_sol)
    
    # Step 5: Final local search polish
    remaining = time_budget - (time.time() - start)
    if remaining > 1:
        final_sol = _local_search(adj, list(best), cliques, max_iters=5000, time_budget=min(remaining, 5))
        if _validate_solution(final_sol, cliques) and len(final_sol) < best_size:
            best = final_sol
    
    return best

def construct():
    deadline = time.time() + 145  # 5s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate time: give more time to larger/harder instances
    # Rough heuristic: n * p determines density, larger n is harder
    total_time = 130  # leave 15s buffer
    weights = [30 * 0.3, 35 * 0.25, 40 * 0.2]
    total_weight = sum(weights)
    time_alloc = [total_time * w / total_weight for w in weights]
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_total = deadline - time.time()
        if remaining_total < 5:
            # Emergency fallback
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        budget = min(time_alloc[i], remaining_total - 10)  # leave 10s for others
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
