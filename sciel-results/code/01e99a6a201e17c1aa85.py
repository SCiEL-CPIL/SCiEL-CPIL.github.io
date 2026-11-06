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
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_solution = list(current_S)
                improved = True
                break
    
    # Phase 3: Branch-and-bound with strong pruning
    if time.time() - start_time < time_limit * 0.8:
        # Build incidence: for each clique, which vertices cover it
        clique_vertices = [set(c) for c in cliques]
        
        # For B&B: we need to find minimum set of vertices hitting all cliques
        # Lower bound: maximum number of pairwise disjoint cliques (packing number)
        # Upper bound: current best_size
        
        # Sort cliques by size (smaller first for better branching)
        clique_order = sorted(range(num_cliques), key=lambda i: len(clique_vertices[i]))
        
        # Precompute: for each vertex, bitmask of cliques it covers (using Python ints as bitsets)
        # num_cliques can be up to a few hundred, so use Python int bitsets
        vertex_clique_mask = [0] * n
        for v in range(n):
            mask = 0
            for ci in vertex_cliques[v]:
                mask |= (1 << ci)
            vertex_clique_mask[v] = mask
        
        # All cliques that need to be hit
        all_cliques_mask = (1 << num_cliques) - 1
        
        # B&B state
        best_found = best_size
        best_set = list(best_solution)
        
        # Count nodes visited
        nodes_visited = 0
        deadline_bnb = start_time + time_limit * 0.75
        
        def lower_bound(remaining_mask):
            """Compute a lower bound on the number of additional vertices needed.
            Uses a greedy packing: repeatedly pick the clique with fewest uncovered vertices."""
            if remaining_mask == 0:
                return 0
            
            # Simple bound: max clique in the conflict graph of remaining cliques
            # Two cliques conflict if they share a vertex
            # Actually, for hitting set, the lower bound is the packing number
            # of cliques that are pairwise "disjoint" in the sense that no single vertex hits two of them.
            # But that's expensive. Simpler: just count remaining cliques / max_cliques_per_vertex
            # Even simpler: 1 if any remaining, else 0 (weak but fast)
            
            # Better: greedy packing
            # Pick a clique, remove all cliques that share a vertex with it, repeat
            # This gives a valid packing (independent set in the conflict graph)
            
            # For speed, use a weaker bound: 
            # For each vertex, max cliques it can cover from remaining
            # LB = ceil(remaining_count / max_coverage)
            
            remaining_count = bin(remaining_mask).count('1')
            if remaining_count == 0:
                return 0
            
            max_cov = 0
            for v in relevant_vertices:
                cov = bin(vertex_clique_mask[v] & remaining_mask).count('1')
                if cov > max_cov:
                    max_cov = cov
            
            if max_cov == 0:
                return remaining_count  # Each clique needs its own vertex
            else:
                return (remaining_count + max_cov - 1) // max_cov
        
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
                    # Reconstruct solution from selected_mask
                    sol = [v for v in range(n) if (selected_mask >> v) & 1]
                    best_set = sol
                return True
            
            # Lower bound check
            lb = lower_bound(remaining_mask)
            if depth + lb >= best_found:
                return True  # Pruned
            
            # Find the most constrained remaining clique
            # Pick clique with fewest available vertices that can still cover it
            best_clique = -1
            best_candidates = None
            best_count = float('inf')
            
            # Iterate through remaining cliques
            rm = remaining_mask
            while rm:
                # Get lowest set bit
                lsb = rm & (-rm)
                ci = lsb.bit_length() - 1
                rm ^= lsb
                
                # Vertices that can cover this clique and aren't already selected
                candidates = []
                for v in clique_vertices[ci]:
                    if not (selected_mask >> v) & 1:
                        candidates.append(v)
                
                if len(candidates) < best_count:
                    best_count = len(candidates)
                    best_clique = ci
                    best_candidates = candidates
                    
                    if best_count <= 1:
                        break
            
            if best_clique == -1:
                return True
            
            if best_count == 0:
                return True  # Dead end
            
            # Branch on candidates for best_clique
            # Order candidates by how many other cliques they cover (greedy heuristic)
            def candidate_score(v):
                return bin(vertex_clique_mask[v] & remaining_mask).count('1')
            
            best_candidates.sort(key=candidate_score, reverse=True)
            
            for v in best_candidates:
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
    if time.time() - start_time < time_limit * 0.95 and best_solution:
        current_S = set(best_solution)
        current_cover = [0] * num_cliques
        for v in current_S:
            for ci in vertex_cliques[v]:
                current_cover[ci] += 1
        
        # Try swaps to reduce size
        for _ in range(100):
            if time.time() - start_time > time_limit:
                break
            if not current_S:
                break
            
            v_in = random.choice(list(current_S))
            # Find a vertex not in S that can replace v_in
            # v_in can be removed if all cliques it uniquely covers are also covered by some other vertex
            # OR we can swap: remove v_in, add v_out where v_out covers all cliques that v_in uniquely covers
            
            # Check which cliques are only covered by v_in
            unique_cliques = []
            for ci in vertex_cliques[v_in]:
                if current_cover[ci] == 1:
                    unique_cliques.append(ci)
            
            if not unique_cliques:
                # v_in is redundant
                current_S.remove(v_in)
                for ci in vertex_cliques[v_in]:
                    current_cover[ci] -= 1
                if len(current_S) < best_size:
                    best_size = len(current_S)
                    best_solution = list(current_S)
                continue
            
            # Need to find v_out that covers all unique_cliques
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
                    # Perform swap
                    current_S.remove(v_in)
                    current_S.add(v_out)
                    for ci in vertex_cliques[v_in]:
                        current_cover[ci] -= 1
                    for ci in vertex_cliques[v_out]:
                        current_cover[ci] += 1
                    found_swap = True
                    break
            
            if found_swap:
                # After swap, try to remove any redundant vertices
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
