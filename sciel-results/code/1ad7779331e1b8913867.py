"""Improved program using branch-and-bound for maximal clique transversal."""
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
        if not P:
            return
        # Choose pivot u from P ∪ X that maximizes |P ∩ N(u)|
        pivot = max(P | X, key=lambda u: len(P & adj[u]))
        candidates = [v for v in P if v not in adj[pivot]]
        for v in candidates:
            bk(R | {v}, P & adj[v], X & adj[v])
            P.remove(v)
            X.add(v)
    
    bk(frozenset(), set(range(n)), set())
    return cliques

def _lower_bound_packing(cliques, n):
    """Greedy packing of vertex-disjoint cliques to get a lower bound on transversal size."""
    if not cliques:
        return 0
    used = [False] * n
    count = 0
    # Sort cliques by size (smallest first) for better packing
    sorted_cliques = sorted(cliques, key=len)
    for clique in sorted_cliques:
        if all(not used[v] for v in clique):
            for v in clique:
                used[v] = True
            count += 1
    return count

def _greedy_initial(cliques, n, deadline):
    """Greedy set cover to get an initial upper bound."""
    if not cliques:
        return []
    
    m = len(cliques)
    # Build bitmask for each vertex
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    full_mask = (1 << m) - 1
    remaining = full_mask
    solution = []
    
    while remaining and time.time() < deadline:
        best_v = -1
        best_count = -1
        for v in range(n):
            if vertex_mask[v] == 0:
                continue
            count = bin(remaining & vertex_mask[v]).count('1')
            if count > best_count:
                best_count = count
                best_v = v
        if best_v < 0:
            break
        solution.append(best_v)
        remaining &= ~vertex_mask[best_v]
    
    return sorted(solution)

def _local_search(solution, cliques, n, deadline):
    """Aggressive local search: removals, 1-1 swaps, 2-1 swaps, 3-1 swaps."""
    if not solution:
        return solution
    
    m = len(cliques)
    full_mask = (1 << m) - 1
    
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    def covered(sol):
        c = 0
        for v in sol:
            c |= vertex_mask[v]
        return c
    
    best = list(solution)
    best_set = set(best)
    
    for iteration in range(20000):
        if time.time() >= deadline:
            break
        
        improved = False
        
        # Try removing each vertex
        for idx in range(len(best)):
            v = best[idx]
            new_sol = [u for j, u in enumerate(best) if j != idx]
            if covered(new_sol) == full_mask:
                best = new_sol
                best_set.discard(v)
                improved = True
                break
        
        if improved:
            continue
        
        # Try 1-1 swap
        found = False
        for idx in range(len(best)):
            v = best[idx]
            rest = [u for j, u in enumerate(best) if j != idx]
            rest_cm = 0
            for u in rest:
                rest_cm |= vertex_mask[u]
            missing = full_mask & ~rest_cm
            if missing == 0:
                continue
            for w in range(n):
                if w in best_set:
                    continue
                if (rest_cm | vertex_mask[w]) == full_mask:
                    best[idx] = w
                    best_set.discard(v)
                    best_set.add(w)
                    improved = True
                    found = True
                    break
            if found:
                break
        
        if improved:
            continue
        
        # Try 2-1 swap: remove two, add one
        found = False
        for i1 in range(len(best)):
            for i2 in range(i1 + 1, len(best)):
                rest = [u for j, u in enumerate(best) if j != i1 and j != i2]
                rest_cm = 0
                for u in rest:
                    rest_cm |= vertex_mask[u]
                missing = full_mask & ~rest_cm
                if missing == 0:
                    best = rest
                    best_set = set(best)
                    improved = True
                    found = True
                    break
                for w in range(n):
                    if w in best_set:
                        continue
                    if (rest_cm | vertex_mask[w]) == full_mask:
                        new_best = rest + [w]
                        best = new_best
                        best_set = set(best)
                        improved = True
                        found = True
                        break
                if found:
                    break
            if found:
                break
        
        if improved:
            continue
        
        # Try 3-1 swap: remove three, add one
        found = False
        for i1 in range(len(best)):
            for i2 in range(i1 + 1, len(best)):
                for i3 in range(i2 + 1, len(best)):
                    rest = [u for j, u in enumerate(best) if j != i1 and j != i2 and j != i3]
                    rest_cm = 0
                    for u in rest:
                        rest_cm |= vertex_mask[u]
                    missing = full_mask & ~rest_cm
                    if missing == 0:
                        best = rest
                        best_set = set(best)
                        improved = True
                        found = True
                        break
                    for w in range(n):
                        if w in best_set:
                            continue
                        if (rest_cm | vertex_mask[w]) == full_mask:
                            new_best = rest + [w]
                            best = new_best
                            best_set = set(best)
                            improved = True
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            if found:
                break
        
        if not improved:
            break
    
    return sorted(best)

def _branch_and_bound(cliques, n, initial_solution, deadline):
    """Branch-and-bound to find a smaller transversal."""
    if not cliques:
        return []
    
    m = len(cliques)
    full_mask = (1 << m) - 1
    
    # Build vertex masks
    vertex_mask = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_mask[v] |= bit
    
    # Build clique membership: for each clique, which vertices are in it
    clique_vertices = [list(c) for c in cliques]
    
    # Lower bound: greedy packing
    def compute_lb(selected_mask, remaining_cliques_mask):
        """Lower bound based on packing of remaining cliques."""
        if remaining_cliques_mask == 0:
            return 0
        # Count remaining cliques that need to be hit
        # Use a simple bound: number of remaining cliques with no selected vertex
        count = 0
        used = [False] * n
        # Greedy pack disjoint remaining cliques
        for i in range(m):
            if (remaining_cliques_mask >> i) & 1:
                clique = clique_vertices[i]
                if all(not used[v] for v in clique):
                    for v in clique:
                        used[v] = True
                    count += 1
        return count
    
    best_solution = list(initial_solution)
    best_size = len(best_solution)
    
    # Sort cliques by size (smallest first) for better branching
    clique_order = sorted(range(m), key=lambda i: len(clique_vertices[i]))
    
    # Precompute: for each vertex, which cliques it covers
    # We'll use a recursive B&B
    
    # State: (clique_idx_in_order, selected_vertices, covered_mask)
    # We process cliques in order; for each clique, we must have at least one selected vertex in it
    
    import sys
    sys.setrecursionlimit(100000)
    
    def bb(clique_idx, selected, covered_mask):
        nonlocal best_solution, best_size
        
        if time.time() >= deadline:
            return
        
        # All cliques processed
        if clique_idx == m:
            if len(selected) < best_size:
                best_solution = list(selected)
                best_size = len(selected)
            return
        
        # Lower bound pruning
        remaining_mask = 0
        for j in range(clique_idx, m):
            remaining_mask |= (1 << j)
        lb = compute_lb(0, remaining_mask & ~covered_mask)
        if len(selected) + lb >= best_size:
            return
        
        # Current clique
        ci = clique_order[clique_idx]
        clique = clique_vertices[ci]
        clique_bit = 1 << ci
        
        # Check if already covered
        if covered_mask & clique_bit:
            bb(clique_idx + 1, selected, covered_mask)
            return
        
        # Need to select at least one vertex from this clique
        # Branch on each vertex in the clique
        for v in clique:
            if v in selected:
                continue
            new_selected = selected + [v]
            new_covered = covered_mask | vertex_mask[v]
            bb(clique_idx + 1, new_selected, new_covered)
    
    # Start B&B
    bb(0, [], 0)
    
    return sorted(best_solution)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    results = []
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    
    # Allocate time: more to larger graphs
    total_weight = sum(n * n for n, p, s in instances)
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            results.append([])
            continue
        
        weight = n * n
        inst_time = (deadline - start_time - 15) * (weight / total_weight)
        inst_deadline = min(start_time + inst_time, deadline - 5)
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj)
        
        if not cliques:
            results.append([])
            continue
        
        # Greedy initial solution
        sol = _greedy_initial(cliques, n, inst_deadline - 30)
        
        # Local search to tighten
        sol = _local_search(sol, cliques, n, inst_deadline - 10)
        
        # Branch and bound to try to improve
        sol = _branch_and_bound(cliques, n, sol, inst_deadline)
        
        results.append(sol)
    
    return results

# EVOLVE-BLOCK-END
