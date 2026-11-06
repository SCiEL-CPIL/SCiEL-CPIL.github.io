import time
import random
import sys
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
    
    num_cliques = len(cliques)
    
    # Precompute for each vertex, the list of cliques it covers
    vertex_cliques = [[] for _ in range(n)]
    for i, clique in enumerate(cliques):
        for v in clique:
            vertex_cliques[v].append(i)
    
    relevant_vertices = [v for v in range(n) if vertex_cliques[v]]
    
    if not relevant_vertices:
        return []
    
    # Greedy Initialization with degree-based tie-breaking
    S = set()
    is_covered = [False] * num_cliques
    remaining_cliques = list(range(num_cliques))
    
    # Precompute vertex degrees for tie-breaking
    vertex_degrees = [len(adj[v]) for v in range(n)]
    
    while remaining_cliques:
        best_v = -1
        best_score = -1
        best_degree = -1
        for v in relevant_vertices:
            if v in S:
                continue
            count = 0
            for ci in vertex_cliques[v]:
                if not is_covered[ci]:
                    count += 1
            if count > best_score:
                best_score = count
                best_v = v
                best_degree = vertex_degrees[v]
            elif count == best_score and count > 0:
                # Tie break: prefer vertex with higher degree
                if vertex_degrees[v] > best_degree:
                    best_v = v
                    best_degree = vertex_degrees[v]
        
        if best_v == -1 or best_score == 0:
            break
            
        S.add(best_v)
        for ci in vertex_cliques[best_v]:
            if not is_covered[ci]:
                is_covered[ci] = True
                remaining_cliques.remove(ci)
                
    best_solution = list(S)
    best_size = len(best_solution)
    
    # Local Search / Simulated Annealing
    current_S = set(best_solution)
    current_cover = [0] * num_cliques
    for v in current_S:
        for ci in vertex_cliques[v]:
            current_cover[ci] += 1
            
    start_time = time.time()
    current_size = len(current_S)
    
    # 1. Greedy Removals
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.3:
        improved = False
        for v in list(current_S):
            valid = True
            for ci in vertex_cliques[v]:
                if current_cover[ci] == 1:
                    valid = False
                    break
            if valid:
                current_S.remove(v)
                for ci in vertex_cliques[v]:
                    current_cover[ci] -= 1
                current_size -= 1
                improved = True
                if current_size < best_size:
                    best_size = current_size
                    best_solution = list(current_S)
                break
    
    # 2. Enhanced Simulated Annealing with 2-opt moves
    if time.time() - start_time < time_limit * 0.7:
        T = 2.0
        T_min = 0.05
        cooling = 0.998
        current_S = set(best_solution)
        current_cover = [0] * num_cliques
        for v in current_S:
            for ci in vertex_cliques[v]:
                current_cover[ci] += 1
                
        current_energy = len(current_S)
        best_energy = best_size
        best_S = set(best_solution)
        
        while T > T_min and (time.time() - start_time) < time_limit * 0.7:
            move_type = random.random()
            
            if move_type < 0.4:
                # Try removal
                if current_S:
                    v = random.choice(list(current_S))
                    valid = True
                    for ci in vertex_cliques[v]:
                        if current_cover[ci] == 1:
                            valid = False
                            break
                    if valid:
                        current_S.remove(v)
                        for ci in vertex_cliques[v]:
                            current_cover[ci] -= 1
                        current_energy -= 1
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_S = set(current_S)
                        
            elif move_type < 0.7:
                # Try single swap
                if current_S:
                    v_in = random.choice(list(current_S))
                    candidates_out = [u for u in relevant_vertices if u not in current_S]
                    if candidates_out:
                        v_out = random.choice(candidates_out)
                        valid = True
                        for ci in vertex_cliques[v_in]:
                            if current_cover[ci] == 1:
                                if v_out not in cliques[ci]:
                                    valid = False
                                    break
                        if valid:
                            current_S.remove(v_in)
                            current_S.add(v_out)
                            for ci in vertex_cliques[v_in]:
                                current_cover[ci] -= 1
                            for ci in vertex_cliques[v_out]:
                                current_cover[ci] += 1
                            
            else:
                # Try 2-opt: remove two, add one or two
                if len(current_S) >= 2:
                    v1, v2 = random.sample(list(current_S), 2)
                    # Check if we can remove both
                    valid = True
                    for ci in vertex_cliques[v1]:
                        if current_cover[ci] == 1:
                            valid = False
                            break
                    if valid:
                        for ci in vertex_cliques[v2]:
                            if current_cover[ci] == 1 and v1 not in cliques[ci]:
                                valid = False
                                break
                    if valid:
                        current_S.remove(v1)
                        current_S.remove(v2)
                        for ci in vertex_cliques[v1]:
                            current_cover[ci] -= 1
                        for ci in vertex_cliques[v2]:
                            current_cover[ci] -= 1
                        current_energy -= 2
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_S = set(current_S)
                    else:
                        # Try adding one vertex to enable removal of two
                        candidates_add = [u for u in relevant_vertices if u not in current_S]
                        if candidates_add:
                            v_add = random.choice(candidates_add)
                            # Check if v_add helps
                            can_remove = True
                            for ci in vertex_cliques[v1]:
                                if current_cover[ci] == 1 and v_add not in cliques[ci]:
                                    can_remove = False
                                    break
                            if can_remove:
                                for ci in vertex_cliques[v2]:
                                    if current_cover[ci] == 1 and v_add not in cliques[ci]:
                                        can_remove = False
                                        break
                                if can_remove:
                                    current_S.remove(v1)
                                    current_S.remove(v2)
                                    current_S.add(v_add)
                                    for ci in vertex_cliques[v1]:
                                        current_cover[ci] -= 1
                                    for ci in vertex_cliques[v2]:
                                        current_cover[ci] -= 1
                                    for ci in vertex_cliques[v_add]:
                                        current_cover[ci] += 1
                                    # Energy unchanged
                            
            T *= cooling
    
    # 3. Final Shrink Polish: try multiple random removal orders
    if time.time() - start_time < time_limit * 0.95:
        polish_deadline = time.time() + (time_limit - (time.time() - start_time)) * 0.9
        for _ in range(100):
            if time.time() > polish_deadline:
                break
            # Start from best_S
            trial_S = set(best_S)
            trial_cover = [0] * num_cliques
            for v in trial_S:
                for ci in vertex_cliques[v]:
                    trial_cover[ci] += 1
            
            # Try random removal order
            removed_any = True
            while removed_any:
                removed_any = False
                for v in random.sample(list(trial_S), len(trial_S)):
                    valid = True
                    for ci in vertex_cliques[v]:
                        if trial_cover[ci] == 1:
                            valid = False
                            break
                    if valid:
                        trial_S.remove(v)
                        for ci in vertex_cliques[v]:
                            trial_cover[ci] -= 1
                        removed_any = True
                        break
            
            if len(trial_S) < best_energy:
                best_energy = len(trial_S)
                best_S = set(trial_S)
    
    return sorted(best_S)

def construct():
    deadline = time.time() + 140  # 10s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    time_per_instance = (deadline - time.time()) / len(instances)
    
    for n, p, seed in instances:
        if time.time() > deadline - 5:
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
                    
            relevant_vertices = [v for v in range(n) if vertex_cliques[v]]
            remaining = list(range(len(cliques)))
            while remaining:
                best_v = -1
                best_count = -1
                for v in relevant_vertices:
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
            
        adj = _build_graph(n, p, seed)
        sol = _solve_instance(adj, time_limit=time_per_instance - 1)
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
