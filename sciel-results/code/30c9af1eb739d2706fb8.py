import time
import random
import itertools
from collections import deque

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

def _solve_instance(adj, time_limit=40.0):
    """Solve the minimum transversal problem for maximal cliques."""
    n = len(adj)
    cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
    
    if not cliques:
        return []
    
    # Precompute vertex -> clique indices mapping
    num_cliques = len(cliques)
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    def is_valid(cover):
        return all(c >= 1 for c in cover)
    
    def compute_cover(S):
        cover = [0] * num_cliques
        for v in S:
            for ci in vertex_cliques[v]:
                cover[ci] += 1
        return cover
    
    # Greedy Initialization with O(1) updates
    S = set()
    is_covered = [False] * num_cliques
    uncovered_count = [0] * num_cliques  # how many uncovered cliques each vertex covers
    
    # Precompute for each vertex: which cliques it covers
    for v in range(n):
        for ci in vertex_cliques[v]:
            uncovered_count[v] += 1
    
    remaining = num_cliques
    
    while remaining > 0:
        best_v = -1
        best_score = -1
        for v in range(n):
            if v in S:
                continue
            if uncovered_count[v] > best_score:
                best_score = uncovered_count[v]
                best_v = v
        
        if best_v == -1 or best_score == 0:
            break
        
        S.add(best_v)
        for ci in vertex_cliques[best_v]:
            if not is_covered[ci]:
                is_covered[ci] = True
                remaining -= 1
                # Update uncovered_count for other vertices
                for v in cliques[ci]:
                    if v not in S:
                        uncovered_count[v] -= 1
    
    # Local Search: iterative improvement
    current_S = set(S)
    current_cover = compute_cover(current_S)
    best_S = set(current_S)
    best_size = len(current_S)
    
    start_time = time.time()
    
    # Phase 1: Greedy removals until no more possible
    while (time.time() - start_time) < time_limit * 0.3:
        improved = False
        for v in list(current_S):
            # Try removing v
            new_cover = current_cover[:]
            for ci in vertex_cliques[v]:
                new_cover[ci] -= 1
            if is_valid(new_cover):
                current_S.remove(v)
                current_cover = new_cover
                best_S = set(current_S)
                best_size = len(current_S)
                improved = True
                break
        if not improved:
            break
    
    # Phase 2: Systematic swap search
    while (time.time() - start_time) < time_limit * 0.6:
        improved = False
        for v_in in list(current_S):
            for v_out in range(n):
                if v_out in current_S:
                    continue
                # Try swap v_in -> v_out
                new_cover = current_cover[:]
                for ci in vertex_cliques[v_in]:
                    new_cover[ci] -= 1
                for ci in vertex_cliques[v_out]:
                    new_cover[ci] += 1
                if is_valid(new_cover):
                    current_S.remove(v_in)
                    current_S.add(v_out)
                    current_cover = new_cover
                    improved = True
                    break
            if improved:
                break
        
        # After swaps, try removals again
        for v in list(current_S):
            new_cover = current_cover[:]
            for ci in vertex_cliques[v]:
                new_cover[ci] -= 1
            if is_valid(new_cover):
                current_S.remove(v)
                current_cover = new_cover
                best_S = set(current_S)
                best_size = len(current_S)
                improved = True
                break
        
        if not improved:
            break
    
    # Phase 3: Random restarts with perturbation
    while (time.time() - start_time) < time_limit:
        # Start from best_S, perturb, then local search
        perturbed_S = set(best_S)
        # Remove 1-2 random vertices from perturbed_S
        to_remove = random.sample(list(perturbed_S), min(2, len(perturbed_S)))
        for v in to_remove:
            perturbed_S.remove(v)
        
        # Repair: greedily add vertices to make valid
        cover = compute_cover(perturbed_S)
        while not is_valid(cover):
            # Find uncovered cliques
            uncovered_cliques = [ci for ci in range(num_cliques) if cover[ci] == 0]
            if not uncovered_cliques:
                break
            # Pick vertex covering most uncovered cliques
            best_v = -1
            best_score = -1
            for v in range(n):
                if v in perturbed_S:
                    continue
                score = sum(1 for ci in uncovered_cliques if v in cliques[ci])
                if score > best_score:
                    best_score = score
                    best_v = v
            if best_v == -1:
                break
            perturbed_S.add(best_v)
            for ci in vertex_cliques[best_v]:
                cover[ci] += 1
        
        if not is_valid(cover):
            continue
        
        # Local search on perturbed_S
        local_S = set(perturbed_S)
        local_cover = compute_cover(local_S)
        
        # Greedy removals
        for _ in range(20):
            improved = False
            for v in list(local_S):
                new_cover = local_cover[:]
                for ci in vertex_cliques[v]:
                    new_cover[ci] -= 1
                if is_valid(new_cover):
                    local_S.remove(v)
                    local_cover = new_cover
                    improved = True
                    break
            if not improved:
                break
        
        # Systematic swaps
        for _ in range(10):
            improved = False
            for v_in in list(local_S):
                for v_out in range(n):
                    if v_out in local_S:
                        continue
                    new_cover = local_cover[:]
                    for ci in vertex_cliques[v_in]:
                        new_cover[ci] -= 1
                    for ci in vertex_cliques[v_out]:
                        new_cover[ci] += 1
                    if is_valid(new_cover):
                        local_S.remove(v_in)
                        local_S.add(v_out)
                        local_cover = new_cover
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        
        if len(local_S) < best_size:
            best_S = set(local_S)
            best_size = len(local_S)
    
    return sorted(best_S)

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate more time to harder instances
    time_weights = [0.3, 0.35, 0.35]
    total_time = deadline - time.time()
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_time = deadline - time.time() - 3  # 3s safety margin
        if remaining_time < 5:
            # Fallback: simple greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
            # Simple greedy
            S = set()
            is_covered = [False] * len(cliques)
            vertex_cliques = [[] for _ in range(n)]
            for i, c in enumerate(cliques):
                for v in c:
                    vertex_cliques[v].append(i)
                    
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in range(n):
                    if v in S: continue
                    cnt = sum(1 for ci in vertex_cliques[v] if not is_covered[ci])
                    if cnt > best_count:
                        best_count = cnt
                        best_v = v
                if best_v == -1: break
                S.add(best_v)
                for ci in vertex_cliques[best_v]:
                    if not is_covered[ci]:
                        is_covered[ci] = True
                        remaining.remove(ci)
            results.append(sorted(S))
            continue
            
        time_budget = remaining_time * time_weights[idx]
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_budget)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
