"""Cochromatic partition solver using branch-and-bound over maximal cliques/independent sets."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_adj():
    n = P['n']
    adj = [set() for _ in range(n)]
    for a, b in P['edges']:
        adj[a].add(b)
        adj[b].add(a)
    return adj

def _is_clique(S, adj):
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[j] not in adj[S[i]]:
                return False
    return True

def _is_independent(S, adj):
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            if S[j] in adj[S[i]]:
                return False
    return True

def _find_maximal_sets(adj, is_type):
    """Find all maximal sets that are either cliques (is_type=True) or independent sets (is_type=False)."""
    n = P['n']
    all_sets = []
    all_vertices = set(range(n))
    
    # Enumerate all subsets and check maximality
    # For n=14, 2^14 = 16384 subsets, manageable
    for mask in range(1, 1 << n):
        S = tuple(v for v in range(n) if mask & (1 << v))
        if len(S) < 1:
            continue
        if is_type:
            if not _is_clique(S, adj):
                continue
        else:
            if not _is_independent(S, adj):
                continue
        # Check maximality: no vertex outside S can be added
        S_set = set(S)
        maximal = True
        for v in range(n):
            if v in S_set:
                continue
            trial = S_set | {v}
            if is_type:
                if _is_clique(tuple(trial), adj):
                    maximal = False
                    break
            else:
                if _is_independent(tuple(trial), adj):
                    maximal = False
                    break
        if maximal:
            all_sets.append(S)
    return all_sets

def _solve_cochromatic(adj):
    """Solve cochromatic partition using branch-and-bound."""
    n = P['n']
    full_mask = (1 << n) - 1
    
    # Find all maximal cliques and maximal independent sets
    max_cliques = _find_maximal_sets(adj, True)
    max_indep = _find_maximal_sets(adj, False)
    
    # All candidate sets
    candidates = []
    for S in max_cliques:
        candidates.append((S, frozenset(S)))
    for S in max_indep:
        candidates.append((S, frozenset(S)))
    
    # Deduplicate (a set could theoretically be both, though unlikely)
    seen = set()
    unique_candidates = []
    for S, fs in candidates:
        if fs not in seen:
            seen.add(fs)
            unique_candidates.append((S, fs))
    
    # Sort candidates by size descending (larger sets cover more)
    unique_candidates.sort(key=lambda x: -len(x[0]))
    
    # Convert to bitmask representation for fast intersection
    candidate_masks = []
    candidate_sets = []
    for S, fs in unique_candidates:
        mask = 0
        for v in S:
            mask |= (1 << v)
        candidate_masks.append(mask)
        candidate_sets.append(S)
    
    num_candidates = len(candidate_masks)
    
    # Greedy initial solution
    best_count = n  # worst case: each vertex alone
    best_solution = None
    remaining = full_mask
    greedy_sol = []
    while remaining:
        best_idx = -1
        best_covered = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining
            cnt = bin(covered).count('1')
            if cnt > best_covered:
                best_covered = cnt
                best_idx = i
        if best_idx >= 0:
            greedy_sol.append(candidate_sets[best_idx])
            remaining &= ~candidate_masks[best_idx]
    if len(greedy_sol) < best_count:
        best_count = len(greedy_sol)
        best_solution = greedy_sol[:]
    
    # Lower bound: for each vertex, count how many candidates cover it
    # Simple LB: ceil(uncovered / max_candidate_size)
    max_cand_size = max(len(S) for S in candidate_sets) if candidate_sets else 1
    
    # Branch and bound
    # State: (remaining_mask, current_count, current_solution)
    # We'll use iterative deepening or simple B&B
    
    # Sort vertices by degree for ordering
    vertex_order = sorted(range(n), key=lambda v: -len(adj[v]))
    
    def lower_bound(remaining_mask):
        """Compute a lower bound on the number of additional sets needed."""
        if remaining_mask == 0:
            return 0
        uncovered = bin(remaining_mask).count('1')
        # Each set can cover at most max_cand_size vertices from remaining
        lb = (uncovered + max_cand_size - 1) // max_cand_size
        return lb
    
    # More refined LB: find a vertex with fewest covering candidates
    def refined_lb(remaining_mask):
        if remaining_mask == 0:
            return 0
        uncovered = bin(remaining_mask).count('1')
        lb1 = (uncovered + max_cand_size - 1) // max_cand_size
        # For each uncovered vertex, find min number of sets needed to cover it
        # Actually, find a vertex that appears in the fewest candidate sets
        min_cover_count = float('inf')
        for v in range(n):
            if remaining_mask & (1 << v):
                cnt = 0
                for i in range(num_candidates):
                    if candidate_masks[i] & (1 << v):
                        cnt += 1
                if cnt < min_cover_count:
                    min_cover_count = cnt
        # This gives us that at least 1 set is needed if min_cover_count > 0
        # Not very strong. Let's use a packing-based LB.
        return max(lb1, 1) if uncovered > 0 else 0
    
    # Use a simple recursive B&B with memoization
    # Since n=14, the state space is manageable
    from functools import lru_cache
    
    # Precompute: for each mask, which candidate indices cover it
    # Actually, let's just do a DFS with pruning
    
    best_count_val = best_count
    best_sol_val = best_solution[:]
    
    # Try to improve with B&B
    # We'll search for solutions with count < best_count_val
    
    def dfs(remaining_mask, current_count, current_sol, deadline):
        nonlocal best_count_val, best_sol_val
        if time.time() > deadline:
            return
        if remaining_mask == 0:
            if current_count < best_count_val:
                best_count_val = current_count
                best_sol_val = current_sol[:]
            return
        if current_count + lower_bound(remaining_mask) >= best_count_val:
            return
        
        # Choose the next set to add
        # Heuristic: pick the candidate that covers the most remaining vertices
        best_idx = -1
        best_covered = 0
        for i in range(num_candidates):
            covered = candidate_masks[i] & remaining_mask
            cnt = bin(covered).count('1')
            if cnt > best_covered:
                best_covered = cnt
                best_idx = i
        if best_idx < 0 or best_covered == 0:
            return
        
        # Try adding this set
        new_remaining = remaining_mask & ~candidate_masks[best_idx]
        current_sol.append(candidate_sets[best_idx])
        dfs(new_remaining, current_count + 1, current_sol, deadline)
        current_sol.pop()
        
        # Also try other candidates that cover the "hardest" vertex
        # Find a vertex in remaining that appears in fewest candidates
        hardest_v = -1
        min_cnt = float('inf')
        for v in range(n):
            if remaining_mask & (1 << v):
                cnt = 0
                for i in range(num_candidates):
                    if candidate_masks[i] & (1 << v):
                        cnt += 1
                if cnt < min_cnt:
                    min_cnt = cnt
                    hardest_v = v
        if hardest_v >= 0:
            # Try all candidates that cover hardest_v
            for i in range(num_candidates):
                if i == best_idx:
                    continue
                if candidate_masks[i] & (1 << hardest_v):
                    new_remaining = remaining_mask & ~candidate_masks[i]
                    current_sol.append(candidate_sets[i])
                    dfs(new_remaining, current_count + 1, current_sol, deadline)
                    current_sol.pop()
    
    deadline = time.time() + 50  # 50 second budget
    dfs(full_mask, 0, [], deadline)
    
    if best_sol_val is None:
        return best_solution
    return best_sol_val

def construct():
    adj = _build_adj()
    solution = _solve_cochromatic(adj)
    
    # Validate: every part must be a clique or independent set
    edge_set = set()
    for a, b in P['edges']:
        edge_set.add((a, b))
        edge_set.add((b, a))
    
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
            # Fallback: use greedy
            parts = []
            for v in range(P['n']):
                for part in parts:
                    trial = part + [v]
                    hits = [(u, w) in edge_set or (w, u) in edge_set 
                           for i, u in enumerate(trial) for w in trial[i+1:]]
                    if all(hits) or not any(hits):
                        part.append(v)
                        break
                else:
                    parts.append([v])
            return [list(p) for p in parts]
    
    # Ensure all vertices are covered exactly once
    all_vertices = set()
    for part in solution:
        for v in part:
            if v in all_vertices:
                # Duplicate, fallback
                parts = []
                for v2 in range(P['n']):
                    for part in parts:
                        trial = part + [v2]
                        hits = [(u, w) in edge_set or (w, u) in edge_set 
                               for i, u in enumerate(trial) for w in trial[i+1:]]
                        if all(hits) or not any(hits):
                            part.append(v2)
                            break
                    else:
                        parts.append([v2])
                return [list(p) for p in parts]
            all_vertices.add(v)
    
    if all_vertices != set(range(P['n'])):
        # Missing vertices, fallback
        parts = []
        for v in range(P['n']):
            for part in parts:
                trial = part + [v]
                hits = [(u, w) in edge_set or (w, u) in edge_set 
                       for i, u in enumerate(trial) for w in trial[i+1:]]
                if all(hits) or not any(hits):
                    part.append(v)
                    break
            else:
                parts.append([v])
        return [list(p) for p in parts]
    
    return [list(p) for p in solution]
# EVOLVE-BLOCK-END
