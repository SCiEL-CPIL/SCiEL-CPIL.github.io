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
    """Solve the minimum transversal problem for maximal cliques using B&B + heuristics."""
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
    
    # Phase 1: Greedy initialization
    S = set()
    is_covered = [False] * num_cliques
    remaining_cliques = list(range(num_cliques))
    
    while remaining_cliques:
        best_v = -1
        best_score = -1
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
            elif count == best_score and count > 0:
                if v < best_v:
                    best_v = v
        
        if best_v == -1 or best_score == 0:
            break
            
        S.add(best_v)
        for ci in vertex_cliques[best_v]:
            if not is_covered[ci]:
                is_covered[ci] = True
                remaining_cliques.remove(ci)
                
    best_solution = list(S)
    best_size = len(best_solution)
    
    start_time = time.time()
    
    # Phase 2: Greedy removals
    current_S = set(best_solution)
    current_cover = [0] * num_cliques
    for v in current_S:
        for ci in vertex_cliques[v]:
            current_cover[ci] += 1
    
    improved = True
    while improved and (time.time() - start_time) < time_limit * 0.2:
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
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_solution = list(current_S)
                improved = True
                break
    
    # Phase 3: Branch-and-bound with strong pruning
    if time.time() - start_time < time_limit * 0.9:
        # Build incidence: for each clique, which vertices cover it
        clique_vertices = [sorted(set(c)) for c in cliques]
        
        # Precompute: for each vertex, bitmask of cliques it covers (using Python ints as bitsets)
        vertex_clique_mask = [0] * n
        for v in range(n):
            mask = 0
            for ci in vertex_cliques[v]:
                mask |= (1 << ci)
            vertex_clique_mask[v] = mask
        
        # All cliques that need to be hit
        all_cliques_mask = (1 << num_cliques) - 1
        
        # Precompute for each clique, the set of vertices that can cover it
        # and sort by coverage to help with branching
        clique_candidate_list = []
        for ci in range(num_cliques):
            cands = [(v, bin(vertex_clique_mask[v]).count('1')) for v in clique_vertices[ci]]
            cands.sort(key=lambda x: x[1], reverse=True)
            clique_candidate_list.append(cands)
        
        # Precompute max coverage per vertex for lower bound
        max_cov_per_vertex = [bin(vertex_clique_mask[v]).count('1') for v in relevant_vertices]
        
        # B&B state
        best_found = best_size
        best_set = list(best_solution)
        
        # Count nodes visited
        nodes_visited = 0
        deadline_bnb = start_time + time_limit * 0.85
        
        def lower_bound(remaining_mask):
            """Compute a lower bound using greedy packing."""
            if remaining_mask == 0:
                return 0
            
            remaining_count = bin(remaining_mask).count('1')
            if remaining_count == 0:
                return 0
            
            # Greedy packing: pick cliques that don't share vertices
            # This gives a valid lower bound
            packed = 0
            rm = remaining_mask
            while rm:
                # Find clique with fewest uncovered vertices among remaining
                lsb = rm & (-rm)
                ci = lsb.bit_length() - 1
                rm ^= lsb
                
                # Find all remaining cliques that share a vertex with clique ci
                shared_mask = 0
                for v in clique_vertices[ci]:
                    shared_mask |= vertex_clique_mask[v]
                
                # Remove all cliques that share a vertex with ci
                rm &= ~shared_mask
                packed += 1
                
                if packed >= best_found:
                    break
            
            return packed
        
        def branch_and_bound(selected_mask, remaining_mask, depth):
            """
            selected_mask: bitset of vertices already selected
            remaining_mask: bitset of cliques still needing to be hit
            depth: current depth (number of vertices selected)
            """
            nonlocal best_found, best_set, nodes_visited
            
            nodes_visited += 1
            
            if time.time() > deadline_bnb:
                return False  # Timeout
            
            if remaining_mask == 0:
                # All cliques hit
                if depth < best_found:
                    best_found = depth
                    sol = [v for v in range(n) if (selected_mask >> v) & 1]
                    best_set = sol
                return True
            
            # Lower bound check
            lb = lower_bound(remaining_mask)
            if depth + lb >= best_found:
                return True  # Pruned
            
            # Find the most constrained remaining clique
            best_clique = -1
            best_candidates = None
            best_count = float('inf')
            
            rm = remaining_mask
            while rm:
                lsb = rm & (-rm)
                ci = lsb.bit_length() - 1
                rm ^= lsb
                
                # Count available candidates
                cnt = 0
                for v, _ in clique_candidate_list[ci]:
                    if not (selected_mask >> v) & 1:
                        cnt += 1
                
                if cnt < best_count:
                    best_count = cnt
                    best_clique = ci
                    best_candidates = [(v, c) for v, c in clique_candidate_list[ci] if not (selected_mask >> v) & 1]
                    
                    if best_count <= 1:
                        break
            
            if best_clique == -1:
                return True
            
            if best_count == 0:
                return True  # Dead end
            
            # Sort candidates by coverage of remaining cliques
            def candidate_score(v):
                return bin(vertex_clique_mask[v] & remaining_mask).count('1')
            
            best_candidates.sort(key=lambda x: candidate_score(x[0]), reverse=True)
            
            for v, _ in best_candidates:
                new_selected = selected_mask | (1 << v)
                new_remaining = remaining_mask & ~vertex_clique_mask[v]
                
                result = branch_and_bound(new_selected, new_remaining, depth + 1)
                if result:
                    return True
                if time.time() > deadline_bnb:
                    return False
            
            return True
        
        # Run B&B
        branch_and_bound(0, all_cliques_mask, 0)
        
        if best_found < best_size:
            best_size = best_found
            best_solution = best_set
    
    # Phase 4: Final local search polish
    if time.time() - start_time < time_limit * 0.98 and best_solution:
        current_S = set(best_solution)
        current_cover = [0] * num_cliques
        for v in current_S:
            for ci in vertex_cliques[v]:
                current_cover[ci] += 1
        
        # Try swaps to reduce size
        for _ in range(50):
            if time.time() - start_time > time_limit:
                break
            if not current_S:
                break
            
            v_in = random.choice(list(current_S))
            
            unique_cliques = []
            for ci in vertex_cliques[v_in]:
                if current_cover[ci] == 1:
                    unique_cliques.append(ci)
            
            if not unique_cliques:
                current_S.remove(v_in)
                for ci in vertex_cliques[v_in]:
                    current_cover[ci] -= 1
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_solution = list(current_S)
                continue
            
            found_swap = False
            for v_out in relevant_vertices:
                if v_out in current_S:
                    continue
                covers_all = True
                for ci in unique_cliques:
                    if ci not in vertex_cliques[v_out]:
                        covers_all = False
                        break
                if covers_all:
                    current_S.remove(v_in)
                    current_S.add(v_out)
                    for ci in vertex_cliques[v_in]:
                        current_cover[ci] -= 1
                    for ci in vertex_cliques[v_out]:
                        current_cover[ci] += 1
                    found_swap = True
                    break
            
            if found_swap:
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
                        if len(current_S) < best_size:
                            best_size = len(current_S)
                            best_solution = list(current_S)
    
    return sorted(best_solution)

def construct():
    deadline = time.time() + 145  # 5s safety margin
    
    instances = [(30, 0.3, 201), (35, 0.25, 202), (40, 0.2, 203)]
    results = []
    
    # Allocate more time to later instances which are harder
    time_weights = [0.25, 0.35, 0.40]
    total_time = deadline - time.time()
    
    for idx, (n, p, seed) in enumerate(instances):
        remaining_instances = len(instances) - idx
        time_left = deadline - time.time()
        
        if remaining_instances <= 1:
            time_for_this = max(5.0, time_left - 2.0)
        else:
            time_for_this = max(5.0, time_left * time_weights[idx] / sum(time_weights[idx:]))
        
        if time.time() > deadline - 5:
            # Fallback: simple greedy
            adj = _build_graph(n, p, seed)
            cliques = _bron_kerbosch_max_cliques(adj, min_size=2)
            if not cliques:
                results.append([])
                continue
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
        sol = _solve_instance(adj, time_limit=min(time_for_this, deadline - time.time() - 3.0))
        results.append(sol)
        
    return results

# EVOLVE-BLOCK-END
