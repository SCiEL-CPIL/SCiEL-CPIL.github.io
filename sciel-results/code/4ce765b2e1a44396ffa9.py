<<<<<<< SEARCH
import time
import random
from itertools import combinations
=======
import time
import random
from itertools import combinations
from collections import deque
>>>>>>> REPLACE
<<<<<<< SEARCH
def _find_maximal_bicliques(edge_set, deadline, max_bicliques=5000):
    """Find many maximal bicliques using a greedy expansion approach from multiple seeds."""
    n = P['n']
    vertices = list(range(n))
    all_vertices_set = set(vertices)
    
    # Build adjacency bitmasks for fast intersection checks
    adj_mask = [0] * n
    for u, v in edge_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    bicliques = []
    seen_bicliques = set()
    
    # Try expanding from various seed pairs
    seed_pairs = []
    for u, v in edge_set:
        seed_pairs.append((u, v))
    
    random.shuffle(seed_pairs)
    
    for seed_u, seed_v in seed_pairs:
        if time.time() > deadline or len(bicliques) >= max_bicliques:
            break
        
        # Start with L={seed_u}, R={seed_v}
        L_mask = (1 << seed_u)
        R_mask = (1 << seed_v)
        
        # Expand: try to add vertices to L or R using bitmask operations
        changed = True
        while changed:
            changed = False
            
            # Try adding to L: vertex x not in L∪R such that x connects to all in R
            # x must have all R vertices as neighbors
            for x in range(n):
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                if (adj_mask[x] & R_mask) == R_mask:
                    L_mask |= (1 << x)
                    changed = True
            
            # Try adding to R: vertex x not in L∪R such that x connects to all in L
            for x in range(n):
                if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                    continue
                if (adj_mask[x] & L_mask) == L_mask:
                    R_mask |= (1 << x)
                    changed = True
        
        # Extract vertex lists from masks
        L = [i for i in range(n) if (L_mask >> i) & 1]
        R = [i for i in range(n) if (R_mask >> i) & 1]
        
        if L and R:
            key = (frozenset(L), frozenset(R))
            key2 = (frozenset(R), frozenset(L))
            if key not in seen_bicliques and key2 not in seen_bicliques:
                seen_bicliques.add(key)
                seen_bicliques.add(key2)
                bicliques.append((L, R))
    
    return bicliques
=======
def _find_maximal_bicliques(edge_set, deadline, max_bicliques=5000):
    """Find many maximal bicliques using BFS over (L_mask, R_mask) states."""
    n = P['n']
    
    # Build adjacency bitmasks for fast intersection checks
    adj_mask = [0] * n
    for u, v in edge_set:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    
    bicliques = []
    seen_states = set()
    seen_bicliques = set()
    
    # Seed with all edges
    seed_pairs = []
    for u, v in edge_set:
        seed_pairs.append((u, v))
    
    random.shuffle(seed_pairs)
    
    # BFS queue: (L_mask, R_mask)
    queue = deque()
    for seed_u, seed_v in seed_pairs:
        L_mask = (1 << seed_u)
        R_mask = (1 << seed_v)
        state = (L_mask, R_mask)
        if state not in seen_states:
            seen_states.add(state)
            queue.append(state)
    
    while queue:
        if time.time() > deadline or len(bicliques) >= max_bicliques:
            break
        
        L_mask, R_mask = queue.popleft()
        
        # Extract vertex lists from masks
        L = [i for i in range(n) if (L_mask >> i) & 1]
        R = [i for i in range(n) if (R_mask >> i) & 1]
        
        if L and R:
            key = (frozenset(L), frozenset(R))
            key2 = (frozenset(R), frozenset(L))
            if key not in seen_bicliques and key2 not in seen_bicliques:
                seen_bicliques.add(key)
                seen_bicliques.add(key2)
                bicliques.append((L, R))
        
        # Try adding to L: vertex x not in L∪R such that x connects to all in R
        for x in range(n):
            if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                continue
            if (adj_mask[x] & R_mask) == R_mask:
                new_L_mask = L_mask | (1 << x)
                state = (new_L_mask, R_mask)
                if state not in seen_states:
                    seen_states.add(state)
                    queue.append(state)
        
        # Try adding to R: vertex x not in L∪R such that x connects to all in L
        for x in range(n):
            if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                continue
            if (adj_mask[x] & L_mask) == L_mask:
                new_R_mask = R_mask | (1 << x)
                state = (L_mask, new_R_mask)
                if state not in seen_states:
                    seen_states.add(state)
                    queue.append(state)
    
    return bicliques
>>>>>>> REPLACE
<<<<<<< SEARCH
def _greedy_biclique_partition(edge_set, deadline):
    """Greedy: repeatedly find largest biclique, remove its edges, recurse."""
    remaining = set(edge_set)
    bicliques = []
    
    while remaining:
        if time.time() > deadline:
            break
        # Find a good biclique from remaining edges
        best_biclique = None
        best_size = 0
        
        # Get all vertices involved in remaining edges
        verts = set()
        for u, v in remaining:
            verts.add(u)
            verts.add(v)
        verts = list(verts)
        
        # Try each vertex as a starting point for left side
        for start in verts:
            if time.time() > deadline:
                break
            # Collect all neighbors of start in remaining edges
            neighbors = set()
            for u, v in remaining:
                if u == start:
                    neighbors.add(v)
                elif v == start:
                    neighbors.add(u)
            
            if not neighbors:
                continue
            
            # Try to grow a biclique
            L = [start]
            R = list(neighbors)
            
            # Grow L
            changed = True
            while changed:
                changed = False
                candidates = set(verts) - set(L)
                for cand in list(candidates):
                    if cand in R:
                        continue
                    all_connect = True
                    for r in R:
                        if (min(cand, r), max(cand, r)) not in remaining:
                            all_connect = False
                            break
                    if all_connect:
                        L.append(cand)
                        changed = True
            
            # Grow R
            R_set = set(R)
            changed = True
            while changed:
                changed = False
                candidates = set(verts) - set(L)
                for cand in list(candidates):
                    if cand in R_set:
                        continue
                    all_connect = True
                    for l in L:
                        if (min(cand, l), max(cand, l)) not in remaining:
                            all_connect = False
                            break
                    if all_connect:
                        R_set.add(cand)
                        changed = True
            R = list(R_set)
            
            if len(L) * len(R) > best_size:
                best_size = len(L) * len(R)
                best_biclique = (L, R)
        
        if best_biclique is None or best_size == 0:
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
        else:
            L, R = best_biclique
            bicliques.append((L, R))
            for u in L:
                for v in R:
                    remaining.discard((min(u,v), max(u,v)))
    
    return bicliques
=======
def _greedy_biclique_partition(edge_set, deadline):
    """Greedy: repeatedly find largest biclique using bitmask operations, remove its edges, recurse."""
    n = P['n']
    remaining = set(edge_set)
    bicliques = []
    
    # Build full adjacency
    full_adj = [0] * n
    for u, v in edge_set:
        full_adj[u] |= (1 << v)
        full_adj[v] |= (1 << u)
    
    while remaining:
        if time.time() > deadline:
            break
        
        # Build adjacency for remaining edges only
        rem_adj = [0] * n
        for u, v in remaining:
            rem_adj[u] |= (1 << v)
            rem_adj[v] |= (1 << u)
        
        # Find best biclique by trying each edge as seed
        best_biclique = None
        best_size = 0
        
        # Try each edge as a seed
        for seed_u, seed_v in list(remaining):
            if time.time() > deadline:
                break
            
            L_mask = (1 << seed_u)
            R_mask = (1 << seed_v)
            
            # Expand using BFS-like approach
            changed = True
            while changed:
                changed = False
                
                # Try adding to L
                for x in range(n):
                    if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                        continue
                    if (rem_adj[x] & R_mask) == R_mask:
                        L_mask |= (1 << x)
                        changed = True
                
                # Try adding to R
                for x in range(n):
                    if (L_mask >> x) & 1 or (R_mask >> x) & 1:
                        continue
                    if (rem_adj[x] & L_mask) == L_mask:
                        R_mask |= (1 << x)
                        changed = True
            
            L = [i for i in range(n) if (L_mask >> i) & 1]
            R = [i for i in range(n) if (R_mask >> i) & 1]
            
            size = len(L) * len(R)
            if size > best_size:
                best_size = size
                best_biclique = (L, R)
        
        if best_biclique is None or best_size == 0:
            u, v = next(iter(remaining))
            bicliques.append(([u], [v]))
            remaining.discard((u, v))
        else:
            L, R = best_biclique
            bicliques.append((L, R))
            for u in L:
                for v in R:
                    remaining.discard((min(u,v), max(u,v)))
    
    return bicliques
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    edge_set = _build_edge_set()
    num_edges = len(edge_set)
    
    # Phase 1: Greedy partition for initial upper bound
    greedy_result = _greedy_biclique_partition(edge_set, deadline)
    greedy_count = len(greedy_result)
    
    # Phase 2: Find many candidate bicliques
    candidates = _find_maximal_bicliques(edge_set, deadline, max_bicliques=10000)
    
    # Map candidates to edge index sets
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    biclique_edge_sets = []
    biclique_vertex_sets = []
    seen_edge_sets = set()
    for L, R in candidates:
        edges_covered = set()
        valid = True
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in edge_to_idx:
                    edges_covered.add(edge_to_idx[e])
                else:
                    valid = False
                    break
            if not valid:
                break
        if valid and edges_covered:
            key = frozenset(edges_covered)
            if key not in seen_edge_sets:
                seen_edge_sets.add(key)
                biclique_edge_sets.append(key)
                biclique_vertex_sets.append((L, R))
    
    # Phase 3: Set cover search with improved pruning
    best_bicliques = None
    if biclique_edge_sets:
        # Build edge-to-biclique mapping
        num_cand = len(biclique_edge_sets)
        edge_to_bicliques = [set() for _ in range(num_edges)]
        for i, be in enumerate(biclique_edge_sets):
            for e in be:
                edge_to_bicliques[e].add(i)
        
        # Greedy upper bound with multiple strategies
        def greedy_cover_idx():
            remaining = set(range(num_edges))
            chosen = []
            while remaining:
                best_bi = None
                best_count = 0
                for bi in range(num_cand):
                    cnt = len(biclique_edge_sets[bi] & remaining)
                    if cnt > best_count:
                        best_count = cnt
                        best_bi = bi
                if best_bi is None:
                    break
                chosen.append(best_bi)
                remaining -= biclique_edge_sets[best_bi]
            return chosen
        
        greedy_chosen = greedy_cover_idx()
        best_size = len(greedy_chosen)
        best_solution = list(greedy_chosen)
        
        # Try greedy with different edge orderings to get better upper bound
        for trial in range(5):
            random.seed(trial * 12345)
            perm = list(range(num_edges))
            random.shuffle(perm)
            remaining = set(range(num_edges))
            chosen = []
            while remaining:
                best_bi = None
                best_count = 0
                for bi in range(num_cand):
                    cnt = len(biclique_edge_sets[bi] & remaining)
                    if cnt > best_count:
                        best_count = cnt
                        best_bi = bi
                if best_bi is None:
                    break
                chosen.append(best_bi)
                remaining -= biclique_edge_sets[best_bi]
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = list(chosen)
        
        memo = {}
        node_count = [0]
        
        # Precompute coverage counts for each biclique
        biclique_coverage = [len(be) for be in biclique_edge_sets]
        
        def backtrack(uncovered, current_solution, depth):
            nonlocal best_solution, best_size
            
            node_count[0] += 1
            if time.time() > deadline or node_count[0] > 1000000:
                return
            
            if not uncovered:
                if depth < best_size:
                    best_size = depth
                    best_solution = list(current_solution)
                return
            
            if depth >= best_size:
                return
            
            # Stronger lower bound: use maximum coverage
            max_cov = 0
            for i in range(num_cand):
                if biclique_edge_sets[i] & uncovered:
                    cov = len(biclique_edge_sets[i] & uncovered)
                    if cov > max_cov:
                        max_cov = cov
            if max_cov == 0:
                return
            lb = (len(uncovered) + max_cov - 1) // max_cov
            if depth + lb >= best_size:
                return
            
            state = frozenset(uncovered)
            if state in memo:
                if memo[state] <= depth:
                    return
            else:
                memo[state] = depth
            
            # Most constrained edge
            min_count = float('inf')
            min_edge = None
            for e in uncovered:
                actual_cnt = len(edge_to_bicliques[e])
                if actual_cnt < min_count:
                    min_count = actual_cnt
                    min_edge = e
            
            if min_edge is None or min_count == 0:
                return
            
            # Sort candidates by coverage in uncovered, then by total size
            candidates_for_edge = sorted(edge_to_bicliques[min_edge], 
                                         key=lambda bi: (-len(biclique_edge_sets[bi] & uncovered), -biclique_coverage[bi]))
            
            for bi in candidates_for_edge:
                new_uncovered = uncovered - biclique_edge_sets[bi]
                current_solution.append(bi)
                backtrack(new_uncovered, current_solution, depth + 1)
                current_solution.pop()
        
        all_edges_frozen = frozenset(range(num_edges))
        backtrack(all_edges_frozen, [], 0)
        
        if best_solution:
            best_bicliques = [biclique_vertex_sets[i] for i in best_solution]
    
    # Use the best result
    if best_bicliques and len(best_bicliques) < greedy_count:
        result_bicliques = best_bicliques
    else:
        result_bicliques = greedy_result
    
    # Final improvement pass: try to merge bicliques if possible
    # Check if two bicliques can be merged (all cross edges exist)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        new_result = []
        used = [False] * len(result_bicliques)
        for i in range(len(result_bicliques)):
            if used[i]:
                continue
            L_i, R_i = result_bicliques[i]
            L_set_i = set(L_i)
            R_set_i = set(R_i)
            for j in range(i + 1, len(result_bicliques)):
                if used[j]:
                    continue
                L_j, R_j = result_bicliques[j]
                # Try merging: new_L = L_i ∪ L_j, new_R = R_i ∪ R_j
                # Check all cross edges exist
                new_L = list(L_set_i | set(L_j))
                new_R = list(R_set_i | set(R_j))
                valid = True
                for u in new_L:
                    for v in new_R:
                        e = (min(u, v), max(u, v))
                        if e not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that merged biclique doesn't cover edges not in original
                    merged_edges = set()
                    for u in new_L:
                        for v in new_R:
                            merged_edges.add((min(u, v), max(u, v)))
                    orig_edges_i = set()
                    for u in L_i:
                        for v in R_i:
                            orig_edges_i.add((min(u, v), max(u, v)))
                    orig_edges_j = set()
                    for u in L_j:
                        for v in R_j:
                            orig_edges_j.add((min(u, v), max(u, v)))
                    if merged_edges == orig_edges_i | orig_edges_j:
                        result_bicliques[i] = (new_L, new_R)
                        used[j] = True
                        improved = True
                        break
        result_bicliques = [bc for bc, u in zip(result_bicliques, used) if not u]
    
    # Convert to proper format
    result = []
    for L, R in result_bicliques:
        L_list = [int(v) for v in L]
        R_list = [int(v) for v in R]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    return result
=======
def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit
    
    edge_set = _build_edge_set()
    num_edges = len(edge_set)
    
    # Phase 1: Greedy partition for initial upper bound
    greedy_result = _greedy_biclique_partition(edge_set, deadline)
    greedy_count = len(greedy_result)
    
    # Phase 2: Find many candidate bicliques
    candidates = _find_maximal_bicliques(edge_set, deadline, max_bicliques=10000)
    
    # Map candidates to edge index sets
    edge_list = sorted(edge_set)
    edge_to_idx = {e: i for i, e in enumerate(edge_list)}
    
    biclique_edge_sets = []
    biclique_vertex_sets = []
    seen_edge_sets = set()
    for L, R in candidates:
        edges_covered = set()
        valid = True
        for u in L:
            for v in R:
                e = (min(u,v), max(u,v))
                if e in edge_to_idx:
                    edges_covered.add(edge_to_idx[e])
                else:
                    valid = False
                    break
            if not valid:
                break
        if valid and edges_covered:
            key = frozenset(edges_covered)
            if key not in seen_edge_sets:
                seen_edge_sets.add(key)
                biclique_edge_sets.append(key)
                biclique_vertex_sets.append((L, R))
    
    # Phase 3: Set cover search with improved pruning
    best_bicliques = None
    if biclique_edge_sets:
        # Build edge-to-biclique mapping
        num_cand = len(biclique_edge_sets)
        edge_to_bicliques = [set() for _ in range(num_edges)]
        for i, be in enumerate(biclique_edge_sets):
            for e in be:
                edge_to_bicliques[e].add(i)
        
        # Greedy upper bound
        def greedy_cover_idx():
            remaining = set(range(num_edges))
            chosen = []
            while remaining:
                best_bi = None
                best_count = 0
                for bi in range(num_cand):
                    cnt = len(biclique_edge_sets[bi] & remaining)
                    if cnt > best_count:
                        best_count = cnt
                        best_bi = bi
                if best_bi is None:
                    break
                chosen.append(best_bi)
                remaining -= biclique_edge_sets[best_bi]
            return chosen
        
        greedy_chosen = greedy_cover_idx()
        best_size = len(greedy_chosen)
        best_solution = list(greedy_chosen)
        
        # Try greedy with different random orderings to get better upper bound
        for trial in range(10):
            random.seed(trial * 12345)
            remaining = set(range(num_edges))
            chosen = []
            while remaining:
                # Randomly perturb the choice
                best_bi = None
                best_count = 0
                for bi in range(num_cand):
                    cnt = len(biclique_edge_sets[bi] & remaining)
                    # Add small random noise to break ties and explore
                    cnt_perturbed = cnt + random.random() * 0.1
                    if cnt_perturbed > best_count:
                        best_count = cnt_perturbed
                        best_bi = bi
                if best_bi is None:
                    break
                chosen.append(best_bi)
                remaining -= biclique_edge_sets[best_bi]
            if len(chosen) < best_size:
                best_size = len(chosen)
                best_solution = list(chosen)
        
        memo = {}
        node_count = [0]
        
        # Precompute coverage counts for each biclique
        biclique_coverage = [len(be) for be in biclique_edge_sets]
        
        def backtrack(uncovered, current_solution, depth):
            nonlocal best_solution, best_size
            
            node_count[0] += 1
            if time.time() > deadline or node_count[0] > 2000000:
                return
            
            if not uncovered:
                if depth < best_size:
                    best_size = depth
                    best_solution = list(current_solution)
                return
            
            if depth >= best_size:
                return
            
            # Stronger lower bound: use maximum coverage
            max_cov = 0
            for i in range(num_cand):
                if biclique_edge_sets[i] & uncovered:
                    cov = len(biclique_edge_sets[i] & uncovered)
                    if cov > max_cov:
                        max_cov = cov
            if max_cov == 0:
                return
            lb = (len(uncovered) + max_cov - 1) // max_cov
            if depth + lb >= best_size:
                return
            
            state = frozenset(uncovered)
            if state in memo:
                if memo[state] <= depth:
                    return
            else:
                memo[state] = depth
            
            # Most constrained edge
            min_count = float('inf')
            min_edge = None
            for e in uncovered:
                actual_cnt = len(edge_to_bicliques[e])
                if actual_cnt < min_count:
                    min_count = actual_cnt
                    min_edge = e
            
            if min_edge is None or min_count == 0:
                return
            
            # Sort candidates by coverage in uncovered, then by total size
            candidates_for_edge = sorted(edge_to_bicliques[min_edge], 
                                         key=lambda bi: (-len(biclique_edge_sets[bi] & uncovered), -biclique_coverage[bi]))
            
            for bi in candidates_for_edge:
                new_uncovered = uncovered - biclique_edge_sets[bi]
                current_solution.append(bi)
                backtrack(new_uncovered, current_solution, depth + 1)
                current_solution.pop()
        
        all_edges_frozen = frozenset(range(num_edges))
        backtrack(all_edges_frozen, [], 0)
        
        if best_solution:
            best_bicliques = [biclique_vertex_sets[i] for i in best_solution]
    
    # Use the best result
    if best_bicliques and len(best_bicliques) < greedy_count:
        result_bicliques = best_bicliques
    else:
        result_bicliques = greedy_result
    
    # Final improvement pass: try to merge bicliques if possible
    improved = True
    while improved and time.time() < deadline:
        improved = False
        used = [False] * len(result_bicliques)
        for i in range(len(result_bicliques)):
            if used[i]:
                continue
            L_i, R_i = result_bicliques[i]
            L_set_i = set(L_i)
            R_set_i = set(R_i)
            for j in range(i + 1, len(result_bicliques)):
                if used[j]:
                    continue
                L_j, R_j = result_bicliques[j]
                # Try merging: new_L = L_i ∪ L_j, new_R = R_i ∪ R_j
                # Check all cross edges exist
                new_L = list(L_set_i | set(L_j))
                new_R = list(R_set_i | set(R_j))
                valid = True
                for u in new_L:
                    for v in new_R:
                        e = (min(u, v), max(u, v))
                        if e not in edge_set:
                            valid = False
                            break
                    if not valid:
                        break
                if valid:
                    # Check that merged biclique doesn't cover edges not in original
                    merged_edges = set()
                    for u in new_L:
                        for v in new_R:
                            merged_edges.add((min(u, v), max(u, v)))
                    orig_edges_i = set()
                    for u in L_i:
                        for v in R_i:
                            orig_edges_i.add((min(u, v), max(u, v)))
                    orig_edges_j = set()
                    for u in L_j:
                        for v in R_j:
                            orig_edges_j.add((min(u, v), max(u, v)))
                    if merged_edges == orig_edges_i | orig_edges_j:
                        result_bicliques[i] = (new_L, new_R)
                        used[j] = True
                        improved = True
                        break
        result_bicliques = [bc for bc, u in zip(result_bicliques, used) if not u]
    
    # Convert to proper format
    result = []
    for L, R in result_bicliques:
        L_list = [int(v) for v in L]
        R_list = [int(v) for v in R]
        if L_list and R_list:
            result.append([L_list, R_list])
    
    return result
>>>>>>> REPLACE
