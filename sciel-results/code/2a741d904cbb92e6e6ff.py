import time
import random
import sys
from itertools import combinations

def _construct_polarity_graph(q):
    """
    Construct the polarity graph of the projective plane PG(2, q).
    Vertices are points of the plane.
    Two points p and r are adjacent if p lies on the polar line of r.
    This graph is C4-free.
    """
    points = []
    seen = set()
    for x in range(q):
        for y in range(q):
            for z in range(q):
                if x == 0 and y == 0 and z == 0:
                    continue
                if x != 0:
                    inv_x = pow(x, q-2, q)
                    key = (1, (y * inv_x) % q, (z * inv_x) % q)
                elif y != 0:
                    inv_y = pow(y, q-2, q)
                    key = (0, 1, (z * inv_y) % q)
                else:
                    key = (0, 0, 1)
                
                if key not in seen:
                    seen.add(key)
                    points.append(key)
    
    n_pts = len(points)
    adj = [set() for _ in range(n_pts)]
    
    for i in range(n_pts):
        x1, x2, x3 = points[i]
        for j in range(i + 1, n_pts):
            y1, y2, y3 = points[j]
            if (x1 * y1 + x2 * y2 + x3 * y3) % q == 0:
                adj[i].add(j)
                adj[j].add(i)
                
    edges = []
    for i in range(n_pts):
        for j in adj[i]:
            if i < j:
                edges.append([i, j])
                
    return n_pts, edges, adj

def _verify_c4_free(n, adj):
    """Check that no two vertices have >= 2 common neighbours."""
    for u in range(n):
        for v in range(u + 1, n):
            common = adj[u] & adj[v]
            if len(common) >= 2:
                return False
    return True

def _has_independent_set_fast(adj_mask, n, limit=11, deadline=None):
    """
    Check if there is an independent set of size >= limit.
    Uses bitmasks with aggressive pruning and deadline checking.
    """
    if deadline is not None and time.time() > deadline:
        return True  # Assume it exists to be safe
    
    all_mask = (1 << n) - 1
    visited = set()
    
    def _branch(rem_mask, current_size, path):
        if deadline is not None and time.time() > deadline:
            return True
        if current_size >= limit:
            return True
        if rem_mask == 0:
            return False
        if current_size + rem_mask.bit_count() < limit:
            return False
        
        # Memoization key
        key = (rem_mask, current_size)
        if key in visited:
            return False
        visited.add(key)
        
        # Pick vertex with max degree in rem_mask
        best_v = -1
        best_deg = -1
        m = rem_mask
        while m:
            v_bit = m & -m
            v = v_bit.bit_length() - 1
            m ^= v_bit
            deg = (adj_mask[v] & rem_mask).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                
        if best_v == -1:
            return False
            
        # Branch 1: Include best_v
        rem_include = rem_mask & ~(adj_mask[best_v] | (1 << best_v))
        if _branch(rem_include, current_size + 1, path + [best_v]):
            return True
            
        # Branch 2: Exclude best_v
        rem_exclude = rem_mask & ~(1 << best_v)
        if _branch(rem_exclude, current_size, path):
            return True
            
        return False

    return _branch(all_mask, 0, [])

def _greedy_build_valid_graph(n_full, adj_full, target_n, limit, deadline):
    """
    Greedily build a valid subgraph of size target_n by adding vertices one at a time.
    A subgraph is valid if it's C4-free (inherited from parent) and has no independent set of size >= limit.
    """
    # Start with a random seed of small size
    rng = random.Random(42)
    
    # Try multiple random starts
    for attempt in range(50):
        if time.time() > deadline - 2:
            break
            
        # Start with 5 random vertices
        current = set(rng.sample(range(n_full), 5))
        
        # Check if initial set is valid
        current_list = list(current)
        sub_adj_mask = [0] * len(current_list)
        old_to_new = {old: new for new, old in enumerate(current_list)}
        for old_i in current_list:
            new_i = old_to_new[old_i]
            for old_j in adj_full[old_i]:
                if old_j in current:
                    new_j = old_to_new[old_j]
                    sub_adj_mask[new_i] |= (1 << new_j)
        
        if _has_independent_set_fast(sub_adj_mask, len(current_list), limit, deadline):
            continue
        
        # Now greedily add vertices
        added = 0
        for _ in range(target_n - 5):
            if time.time() > deadline - 1:
                break
                
            # Find candidate vertices not in current
            candidates = [v for v in range(n_full) if v not in current]
            if not candidates:
                break
                
            # Score candidates: prefer those with high degree in current subgraph
            scored_candidates = []
            for v in candidates:
                # Count neighbors in current
                neigh_in_current = sum(1 for u in current if u in adj_full[v])
                scored_candidates.append((neigh_in_current, v))
            
            # Sort by descending degree in current
            scored_candidates.sort(reverse=True)
            
            # Try top candidates
            found = False
            for _, v in scored_candidates[:10]:
                new_current = current | {v}
                new_list = list(new_current)
                new_sub_adj_mask = [0] * len(new_list)
                new_old_to_new = {old: new for new, old in enumerate(new_list)}
                for old_i in new_list:
                    new_i = new_old_to_new[old_i]
                    for old_j in adj_full[old_i]:
                        if old_j in new_current:
                            new_j = new_old_to_new[old_j]
                            new_sub_adj_mask[new_i] |= (1 << new_j)
                
                if not _has_independent_set_fast(new_sub_adj_mask, len(new_list), limit, deadline):
                    current = new_current
                    current_list = new_list
                    sub_adj_mask = new_sub_adj_mask
                    old_to_new = new_old_to_new
                    found = True
                    added += 1
                    break
            
            if not found:
                # Try random candidates
                for v in rng.sample(candidates, min(5, len(candidates))):
                    new_current = current | {v}
                    new_list = list(new_current)
                    new_sub_adj_mask = [0] * len(new_list)
                    new_old_to_new = {old: new for new, old in enumerate(new_list)}
                    for old_i in new_list:
                        new_i = new_old_to_new[old_i]
                        for old_j in adj_full[old_i]:
                            if old_j in new_current:
                                new_j = new_old_to_new[old_j]
                                new_sub_adj_mask[new_i] |= (1 << new_j)
                    
                    if not _has_independent_set_fast(new_sub_adj_mask, len(new_list), limit, deadline):
                        current = new_current
                        current_list = new_list
                        sub_adj_mask = new_sub_adj_mask
                        old_to_new = new_old_to_new
                        found = True
                        added += 1
                        break
            
            if not found:
                break
        
        if len(current) >= target_n:
            # Found a valid subset of at least target_n
            selected = list(current)[:target_n]
            return selected
    
    return None

def _find_max_subset_with_alpha_bound(n_full, adj_full, target_n, limit, time_limit, rng):
    """
    Find a subset of size target_n from n_full vertices such that the induced subgraph
    has no independent set of size >= limit.
    Uses greedy construction as primary strategy.
    """
    deadline = time.time() + time_limit
    
    # First try greedy construction
    selected = _greedy_build_valid_graph(n_full, adj_full, target_n, limit, deadline)
    if selected is not None:
        return selected
    
    # Fallback to random sampling if greedy fails
    best_subset = None
    
    attempts = 0
    max_attempts = 500
    
    while attempts < max_attempts and time.time() < deadline:
        attempts += 1
        
        selected = rng.sample(range(n_full), target_n)
        selected_set = set(selected)
        
        sub_adj_mask = [0] * target_n
        old_to_new = {old: new for new, old in enumerate(selected)}
        for old_i in selected:
            new_i = old_to_new[old_i]
            for old_j in adj_full[old_i]:
                if old_j in selected_set:
                    new_j = old_to_new[old_j]
                    sub_adj_mask[new_i] |= (1 << new_j)
        
        if not _has_independent_set_fast(sub_adj_mask, target_n, limit, deadline):
            best_subset = selected
            break
    
    return best_subset

def construct():
    deadline = time.time() + 140.0
    
    best_n = 0
    best_edges = []
    
    qs_to_try = [7, 5]
    
    for q in qs_to_try:
        if time.time() > deadline - 10:
            break
            
        try:
            n_full, edges_full, adj_full = _construct_polarity_graph(q)
            
            if not _verify_c4_free(n_full, adj_full):
                continue
                
            adj_mask = [0] * n_full
            for i in range(n_full):
                mask = 0
                for j in adj_full[i]:
                    mask |= (1 << j)
                adj_mask[i] = mask
            
            if n_full <= 43:
                if not _has_independent_set_fast(adj_mask, n_full, limit=11, deadline=deadline):
                    if n_full > best_n:
                        best_n = n_full
                        best_edges = edges_full
                        if best_n == 43:
                            return {'n': best_n, 'edges': best_edges}
            
            target_sizes = [43, 42, 41, 40, 39]
            
            for target_n in target_sizes:
                if target_n > n_full:
                    continue
                if target_n <= best_n:
                    continue
                
                if time.time() > deadline - 5:
                    break
                    
                rng = random.Random(42)
                
                time_remaining = deadline - time.time()
                time_for_search = min(15.0, time_remaining - 2.0)
                if time_for_search <= 0:
                    break
                    
                selected = _find_max_subset_with_alpha_bound(n_full, adj_full, target_n, 11, time_for_search, rng)
                
                if selected is not None:
                    selected_set = set(selected)
                    old_to_new = {old: new for new, old in enumerate(selected)}
                    candidate_edges = []
                    for old_i in selected:
                        new_i = old_to_new[old_i]
                        for old_j in adj_full[old_i]:
                            if old_j in selected_set and old_i < old_j:
                                new_j = old_to_new[old_j]
                                if new_i < new_j:
                                    candidate_edges.append([new_i, new_j])
                    best_n = target_n
                    best_edges = candidate_edges
                    break
                        
                if best_n == 43:
                    return {'n': best_n, 'edges': best_edges}
                    
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            continue
            
    if best_n == 0:
        best_n = 3
        best_edges = [[0, 1], [1, 2], [0, 2]]
        
    return {'n': best_n, 'edges': best_edges}
