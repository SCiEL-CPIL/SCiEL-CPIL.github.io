"""Cochromatic partition solver using branch-and-bound over maximal cliques/independent sets."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _find_all_maximal_sets(adj):
    """Find all maximal cliques and maximal independent sets using Bron-Kerbosch variant."""
    n = P['n']
    full = set(range(n))
    
    # Find maximal cliques
    max_cliques = []
    def bronk_clique(R, P_set, X):
        if not P_set and not X:
            if R:
                max_cliques.append(frozenset(R))
            return
        # Pivot selection: choose vertex with max connections in P_set
        pivot_candidates = P_set | X
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(adj[v] & P_set))
            candidates = P_set - adj[pivot]
        else:
            candidates = P_set()
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P_set & adj[v]
            new_X = X & adj[v]
            bronk_clique(new_R, new_P, new_X)
            P_set.remove(v)
            X.add(v)
    
    bronk_clique(set(), full, set())
    
    # Find maximal independent sets (cliques in complement)
    max_indep = []
    comp_adj = [set() for _ in range(n)]
    for v in range(n):
        comp_adj[v] = full - {v} - adj[v]
    
    def bronk_indep(R, P_set, X):
        if not P_set and not X:
            if R:
                max_indep.append(frozenset(R))
            return
        pivot_candidates = P_set | X
        if pivot_candidates:
            pivot = max(pivot_candidates, key=lambda v: len(comp_adj[v] & P_set))
            candidates = P_set - comp_adj[pivot]
        else:
            candidates = P_set()
        
        for v in list(candidates):
            new_R = R | {v}
            new_P = P_set & comp_adj[v]
            new_X = X & comp_adj[v]
            bronk_indep(new_R, new_P, new_X)
            P_set.remove(v)
            X.add(v)
    
    bronk_indep(set(), full, set())
    
    # Deduplicate and combine
    seen = set()
    all_sets = []
    for s in max_cliques + max_indep:
        if s not in seen:
            seen.add(s)
            all_sets.append(s)
    
    return all_sets

def _to_mask(fs):
    m = 0
    for v in fs:
        m |= (1 << v)
    return m

def _popcount(m):
    return bin(m).count('1')

def _solve_cochromatic(adj):
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Find all maximal sets
    max_sets = _find_all_maximal_sets(adj)
    
    # Convert to bitmasks
    candidate_masks = [_to_mask(s) for s in max_sets]
    
    # Also add all single vertices as candidates (safety)
    for v in range(n):
        vm = 1 << v
        if vm not in candidate_masks:
            candidate_masks.append(vm)
    
    # Sort by size descending
    candidate_masks.sort(key=lambda m: -_popcount(m))
    
    num_candidates = len(candidate_masks)
    
    # Precompute: for each vertex, which candidate indices contain it
    vertex_candidates = [[] for _ in range(n)]
    for i, cm in enumerate(candidate_masks):
        for v in range(n):
            if cm & (1 << v):
                vertex_candidates[v].append(i)
    
    # Greedy initial solution
    best_count = n
    best_solution = None
    remaining = full_mask
    greedy_sol = []
    while remaining:
        best_idx = -1
        best_covered = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining
            cnt = _popcount(covered)
            if cnt > best_covered:
                best_covered = cnt
                best_idx = i
        if best_idx >= 0 and best_covered > 0:
            # Reconstruct the actual set from the mask
            part = [v for v in range(n) if candidate_masks[best_idx] & (1 << v)]
            greedy_sol.append(part)
            remaining &= ~candidate_masks[best_idx]
        else:
            # Fallback: pick any uncovered vertex
            for v in range(n):
                if remaining & (1 << v):
                    greedy_sol.append([v])
                    remaining &= ~(1 << v)
                    break
    if greedy_sol and len(greedy_sol) < best_count:
        best_count = len(greedy_sol)
        best_solution = greedy_sol[:]
    
    # Memoization
    memo = {}
    
    # Time limit
    deadline = time.time() + 55
    
    def lower_bound(remaining_mask):
        """Compute a lower bound on the number of additional sets needed."""
        if remaining_mask == 0:
            return 0
        uncovered = _popcount(remaining_mask)
        if uncovered == 0:
            return 0
        # Find max possible single-set coverage from remaining
        max_cover = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining_mask
            cnt = _popcount(covered)
            if cnt > max_cover:
                max_cover = cnt
        if max_cover == 0:
            return uncovered
        lb = (uncovered + max_cover - 1) // max_cover
        return lb
    
    # Stronger lower bound using matching-like argument
    def refined_lower_bound(remaining_mask):
        """Try to find a set of vertices where no single candidate can cover two."""
        if remaining_mask == 0:
            return 0
        uncovered = _popcount(remaining_mask)
        if uncovered == 0:
            return 0
        
        # Basic bound
        max_cover = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining_mask
            cnt = _popcount(covered)
            if cnt > max_cover:
                max_cover = cnt
        if max_cover == 0:
            return uncovered
        lb1 = (uncovered + max_cover - 1) // max_cover
        
        # Try to find a packing: vertices that must be in separate parts
        # For each pair of vertices, check if any candidate covers both
        vertices = [v for v in range(n) if remaining_mask & (1 << v)]
        if len(vertices) <= 1:
            return lb1
        
        # Greedy packing: repeatedly pick a vertex, remove all vertices that can share a part with it
        packing_size = 0
        remaining_v = set(vertices)
        while remaining_v:
            v = next(iter(remaining_v))
            packing_size += 1
            # Find all vertices that can be in the same part as v
            compatible = {v}
            for w in remaining_v:
                if w == v:
                    continue
                for i in vertex_candidates[v]:
                    if candidate_masks[i] & (1 << w):
                        compatible.add(w)
                        break
            remaining_v -= compatible
        
        lb2 = packing_size
        return max(lb1, lb2)
    
    def dfs(remaining_mask, current_count, current_sol):
        nonlocal best_count, best_solution
        if time.time() > deadline:
            return
        
        if remaining_mask == 0:
            if current_count < best_count:
                best_count = current_count
                best_solution = current_sol[:]
            return
        
        # Check memo
        if remaining_mask in memo:
            if memo[remaining_mask] <= current_count:
                return
        else:
            memo[remaining_mask] = current_count
        
        # Lower bound pruning
        lb = refined_lower_bound(remaining_mask)
        if current_count + lb >= best_count:
            return
        
        # Pick the hardest vertex (fewest usable candidate options)
        hardest_v = -1
        min_cnt = float('inf')
        for v in range(n):
            if remaining_mask & (1 << v):
                cnt = 0
                for i in vertex_candidates[v]:
                    if candidate_masks[i] & remaining_mask:
                        cnt += 1
                if cnt < min_cnt:
                    min_cnt = cnt
                    hardest_v = v
                    if cnt <= 1:
                        break
        
        if hardest_v < 0:
            return
        
        # Try all candidates that contain hardest_v and cover at least one remaining vertex
        # Sort candidates by coverage to try best first
        tried = set()
        cand_list = []
        for i in vertex_candidates[hardest_v]:
            if candidate_masks[i] & remaining_mask:
                if i not in tried:
                    tried.add(i)
                    covered = _popcount(candidate_masks[i] & remaining_mask)
                    cand_list.append((covered, i))
        
        cand_list.sort(key=lambda x: -x[0])
        
        for covered, i in cand_list:
            if current_count + 1 >= best_count:
                break
            new_remaining = remaining_mask & ~candidate_masks[i]
            part = [v for v in range(n) if candidate_masks[i] & (1 << v)]
            current_sol.append(part)
            dfs(new_remaining, current_count + 1, current_sol)
            current_sol.pop()
    
    dfs(full_mask, 0, [])
    
    if best_solution is None:
        return [[v] for v in range(n)]
    return best_solution

def construct():
    adj = _build_adj()
    solution = _solve_cochromatic(adj)
    
    # Validate: every part must be a clique or independent set
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
    valid = True
    for part in solution:
        part_set = set(part)
        is_clique = True
        is_indep = True
        for i in range(len(part)):
            for j in range(i + 1, len(part)):
                if (part[i], part[j]) in edge_set:
                    is_indep = False
                else:
                    is_clique = False
        if not is_clique and not is_indep:
            valid = False
            break
    
    # Ensure all vertices are covered exactly once
    all_vertices = set()
    for part in solution:
        for v in part:
            if v in all_vertices:
                valid = False
                break
            all_vertices.add(v)
        if not valid:
            break
    
    if not valid or all_vertices != set(range(P['n'])):
        # Fallback: greedy construction
        parts = []
        for v in range(P['n']):
            placed = False
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in edge_set or (w, u) in edge_set 
                       for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    placed = True
                    break
            if not placed:
                parts.append([v])
        return [list(p) for p in parts]
    
    return [list(p) for p in solution]
# EVOLVE-BLOCK-END
