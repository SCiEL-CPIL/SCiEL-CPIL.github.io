import time
import random
import sys
from functools import lru_cache

def _build_graph(n, p, seed):
    """Build adjacency lists from the same PRNG stream as the seed program."""
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
        # Pick pivot with max degree in P (heuristic)
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

def _greedy_hitting_set(vertex_covers, num_cliques):
    """Greedy set cover to get an initial hitting set."""
    n = len(vertex_covers)
    sol_mask = 0
    remaining = (1 << num_cliques) - 1
    
    while remaining:
        best_v = -1
        best_count = 0
        for v in range(n):
            if (sol_mask & (1 << v)) != 0:
                continue
            count = (vertex_covers[v] & remaining).bit_count()
            if count > best_count:
                best_count = count
                best_v = v
        if best_v == -1 or best_count == 0:
            break
        sol_mask |= (1 << best_v)
        remaining &= ~vertex_covers[best_v]
    
    return sol_mask

def _solve_instance(n, p, seed, time_budget):
    """
    Solve the minimum hitting set problem for maximal cliques of a random graph.
    Uses branch-and-bound with constraint propagation.
    """
    adj = _build_graph(n, p, seed)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []

    num_cliques = len(cliques)
    
    # Precompute for each vertex, the bitmask of cliques it covers
    vertex_covers = [0] * n
    for i, clique in enumerate(cliques):
        bit = 1 << i
        for v in clique:
            vertex_covers[v] |= bit
    
    # Remove vertices that don't cover any clique
    active_vertices = [v for v in range(n) if vertex_covers[v] != 0]
    if not active_vertices:
        return []
    
    # Remap to compact indices
    v_to_compact = {v: i for i, v in enumerate(active_vertices)}
    compact_n = len(active_vertices)
    compact_covers = [0] * compact_n
    for v in active_vertices:
        compact_covers[v_to_compact[v]] = vertex_covers[v]
    
    # Get initial upper bound from greedy
    greedy_sol = _greedy_hitting_set(compact_covers, num_cliques)
    greedy_size = greedy_sol.bit_count()
    
    # If greedy is very small, try to prove optimality or improve
    if greedy_size <= 1:
        return [active_vertices[i] for i in range(compact_n) if (greedy_sol & (1 << i))]
    
    # Precompute: for each clique, list of vertices that cover it
    clique_vertices = [[] for _ in range(num_cliques)]
    for v in range(compact_n):
        c = compact_covers[v]
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            clique_vertices[i].append(v)
    
    # Sort cliques by size (fewest vertices first) for better pruning
    clique_order = sorted(range(num_cliques), key=lambda i: len(clique_vertices[i]))
    
    best_size = [greedy_size]
    best_sol = [greedy_sol]
    start_time = time.time()
    deadline = start_time + time_budget
    
    def bb(selected_mask, remaining_cliques_mask):
        """
        Branch and bound for minimum hitting set.
        selected_mask: bitmask of vertices already selected
        remaining_cliques_mask: bitmask of cliques not yet hit
        """
        if time.time() > deadline:
            return
        
        if remaining_cliques_mask == 0:
            size = selected_mask.bit_count()
            if size < best_size[0]:
                best_size[0] = size
                best_sol[0] = selected_mask
            return
        
        # Lower bound: at least 1 more vertex needed if any clique remains
        current_size = selected_mask.bit_count()
        if current_size + 1 >= best_size[0]:
            return
        
        # Find the remaining clique with fewest candidate vertices
        best_clique = -1
        best_candidates = None
        best_count = float('inf')
        
        c = remaining_cliques_mask
        while c:
            i = (c & -c).bit_length() - 1
            c &= c - 1
            # Count vertices in clique i not yet selected
            count = 0
            for v in clique_vertices[i]:
                if (selected_mask & (1 << v)) == 0:
                    count += 1
            if count < best_count:
                best_count = count
                best_clique = i
                if count == 1:
                    break
        
        if best_clique == -1:
            return
        
        # If any remaining clique has 0 available vertices, this branch is infeasible
        if best_count == 0:
            return
        
        # Branch on each vertex in the most constrained clique
        candidates = []
        for v in clique_vertices[best_clique]:
            if (selected_mask & (1 << v)) == 0:
                candidates.append(v)
        
        # Sort candidates by how many remaining cliques they cover (most first)
        candidates.sort(key=lambda v: (compact_covers[v] & remaining_cliques_mask).bit_count(), reverse=True)
        
        for v in candidates:
            if time.time() > deadline:
                return
            
            new_selected = selected_mask | (1 << v)
            new_remaining = remaining_cliques_mask & ~compact_covers[v]
            
            # Prune: if we've already selected as many as best, no point continuing
            if new_selected.bit_count() >= best_size[0]:
                continue
            
            bb(new_selected, new_remaining)
    
    # Run branch and bound
    bb(0, (1 << num_cliques) - 1)
    
    # Map back to original vertex indices
    result = []
    for i in range(compact_n):
        if (best_sol[0] & (1 << i)) != 0:
            result.append(active_vertices[i])
    
    return sorted(result)

def construct():
    deadline = time.time() + 145  # Safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for i, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time()
        if remaining_time < 5:
            # Fallback: just return greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            sol = []
            for clique in cliques:
                if not any(v in sol for v in clique):
                    v = next(iter(clique))
                    sol.append(v)
            results.append(sorted(set(sol)))
            continue
         
        # Allocate time: give more to earlier instances, but ensure all get some
        if i == 0:
            alloc = min(remaining_time - 10, 50)
        elif i == 1:
            alloc = min(remaining_time - 5, 40)
        else:
            alloc = min(remaining_time - 2, 30)
        
        sol = _solve_instance(n, p, seed, alloc)
        results.append(sol)
        
    return results
