import random
import time
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
        # Choose pivot u from P union X to reduce recursion
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

def _validate_solution(adj, selected, cliques):
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
    
    n = max(max(c) for c in cliques) + 1
    # Build vertex -> cliques mapping
    vertex_to_cliques = {}
    for i, clique in enumerate(cliques):
        for v in clique:
            if v not in vertex_to_cliques:
                vertex_to_cliques[v] = []
            vertex_to_cliques[v].append(i)
    
    covered = [False] * len(cliques)
    selected = []
    
    while any(not c for c in covered):
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
            covered[idx] = True
    
    return sorted(selected)

def _local_search(adj, initial_solution, cliques, max_iters=5000, time_budget=5.0):
    """Optimized Local Search: Removals and 1-1 Swaps with critical clique analysis."""
    start = time.time()
    current = list(initial_solution)
    current_set = set(current)
    n = len(adj)
    
    if not cliques:
        return sorted(current)
    
    best_size = len(current)
    best_solution = current[:]
    
    sideways_moves = 0
    max_sideways = 100 # Allow more exploration
    
    for _ in range(max_iters):
        if time.time() - start > time_budget:
            break
            
        improved = False
        
        # 1. Try removals
        for v in list(current):
            trial_set = current_set - {v}
            # Check validity
            valid = True
            for clique in cliques:
                if not (trial_set & clique):
                    valid = False
                    break
            if valid:
                current = list(trial_set)
                current_set = trial_set
                improved = True
                sideways_moves = 0
                if len(current) < best_size:
                    best_size = len(current)
                    best_solution = current[:]
                break
        if improved:
            continue
        
        # 2. Try 1-1 swaps
        if sideways_moves < max_sideways:
            found_swap = False
            # To avoid bias, we can shuffle the order of vertices to try
            # But deterministic is fine for now, maybe add randomization in restarts
            
            for v in list(current):
                # Identify cliques that become uncovered if v is removed
                # These are cliques where v is the ONLY vertex from current
                critical_cliques = []
                for clique in cliques:
                    if v in clique:
                        # Check if any other vertex in current is in clique
                        # If not, this clique is critical for v
                        # Optimization: 
                        # if len(clique & current_set) == 1:
                        #    critical_cliques.append(clique)
                        
                        # This check is O(|clique|) or O(1) if we precompute?
                        # We can just do it.
                        if len(clique & current_set) == 1:
                            critical_cliques.append(clique)
                
                if not critical_cliques:
                    # If no critical cliques, removing v should have been valid.
                    # This shouldn't happen if step 1 was exhaustive.
                    # But if it does, it means v is redundant? 
                    # Wait, if v is in a clique, and no other current vertex is in it, 
                    # then removing v uncovers it. So it IS critical.
                    # So critical_cliques is non-empty for any v that is in any clique.
                    # If v is not in any clique, it would have been removed in step 1.
                    pass
                
                if critical_cliques:
                    # Find u not in current that covers ALL critical_cliques
                    # Intersection of all critical cliques
                    candidates = set(critical_cliques[0])
                    for c in critical_cliques[1:]:
                        candidates &= set(c)
                    
                    # Remove current vertices from candidates
                    candidates -= current_set
                    
                    if candidates:
                        # Pick a candidate
                        # Heuristic: Pick one that covers most other cliques? 
                        # For now, just pick one (e.g., min index for determinism, or random)
                        u = min(candidates) # Deterministic choice
                        # Perform swap
                        current_set.remove(v)
                        current_set.add(u)
                        current = list(current_set)
                        improved = True
                        sideways_moves += 1
                        found_swap = True
                        break
            
            if found_swap:
                continue
        
        sideways_moves += 1
        if sideways_moves > max_sideways:
            break
            
    return sorted(best_solution)

def _random_local_search(adj, cliques, time_budget=5.0):
    """Random restarts with kick strategy and improved local search."""
    start = time.time()
    n = len(adj)
    rng = random.Random(42)
    
    if not cliques:
        return []
    
    best = None
    best_size = float('inf')
    
    # First try greedy
    greedy_sol = _greedy_set_cover(cliques)
    if _validate_solution(adj, greedy_sol, cliques):
        best = greedy_sol
        best_size = len(greedy_sol)
    
    # Random restarts
    for i in range(50): # Increased restarts
        if time.time() - start > time_budget:
            break
        
        # Strategy: 
        # 1. Random subset
        # 2. Kick from best (remove k random, repair)
        
        if best and rng.random() < 0.5:
            # Kick from best
            candidate = set(best)
            # Remove k random vertices
            k = rng.randint(1, max(1, len(candidate) // 3))
            if candidate:
                to_remove = rng.sample(list(candidate), min(k, len(candidate)))
                for v in to_remove:
                    candidate.remove(v)
        else:
            # Random start
            subset_size = rng.randint(0, min(n, 10))
            candidate = set(rng.sample(range(n), subset_size))
        
        # Repair: Greedily add vertices to cover all cliques
        covered_cliques = set()
        for i_c, clique in enumerate(cliques):
            if candidate & set(clique):
                covered_cliques.add(i_c)
        
        while len(covered_cliques) < len(cliques):
            uncovered = [cliques[i] for i in range(len(cliques)) if i not in covered_cliques]
            if not uncovered:
                break
            best_v = None
            best_count = -1
            for v in range(n):
                if v in candidate:
                    count = sum(1 for c in uncovered if v in c)
                else:
                    count = sum(1 for c in uncovered if v in c)
                if count > best_count:
                    best_count = count
                    best_v = v
            if best_v is None or best_count == 0:
                break
            candidate.add(best_v)
            for i_c, clique in enumerate(cliques):
                if i_c not in covered_cliques and best_v in clique:
                    covered_cliques.add(i_c)
        
        if len(covered_cliques) < len(cliques):
            continue
        
        # Local search
        # Allocate remaining time / remaining restarts
        remaining_time = time_budget - (time.time() - start)
        ts = max(0.5, remaining_time / (50 - i))
        improved = _local_search(adj, list(candidate), cliques, max_iters=2000, time_budget=ts)
        
        if _validate_solution(adj, improved, cliques) and len(improved) < best_size:
            best = improved
            best_size = len(improved)
    
    if best is None:
        return greedy_sol
    return best

def construct():
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    for n, p, seed in instances:
        if time.time() > deadline - 10:
            # Fallback
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            sol = _greedy_set_cover(cliques)
            results.append(sol)
            continue
        
        adj = _build_graph(n, p, seed)
        cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
        
        if not cliques:
            results.append([])
            continue
        
        greedy_sol = _greedy_set_cover(cliques)
        if not _validate_solution(adj, greedy_sol, cliques):
            touched = set()
            for clique in cliques:
                touched.update(clique)
            greedy_sol = sorted(touched)
        
        remaining = deadline - time.time() - 5
        if remaining > 1:
            better = _random_local_search(adj, cliques, time_budget=min(remaining, 40))
            if _validate_solution(adj, better, cliques) and len(better) < len(greedy_sol):
                greedy_sol = better
        
        results.append(greedy_sol)
    
    return results

# EVOLVE-BLOCK-END
