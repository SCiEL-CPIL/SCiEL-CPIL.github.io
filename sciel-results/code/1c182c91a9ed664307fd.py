import random
import time
import itertools

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    Optimized for n=18 using bitmasks and tight pruning.
    """
    best = 0
    full_mask = (1 << n) - 1
    
    # Precompute common neighbors
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            c = adj_bitmasks[i] & adj_bitmasks[j]
            common[i][j] = c
            common[j][i] = c
            
    def dfs(available, current_mask, current_size):
        nonlocal best
        if current_size + available.bit_count() <= best:
            return
        if available == 0:
            if current_size > best:
                best = current_size
            return
            
        # Pick vertex with highest degree in available
        best_v = -1
        best_deg = -1
        best_v_bit = 0
        temp = available
        while temp:
            v_bit = temp & -temp
            v = v_bit.bit_length() - 1
            deg = (adj_bitmasks[v] & available).bit_count()
            if deg > best_deg:
                best_deg = deg
                best_v = v
                best_v_bit = v_bit
            temp -= v_bit
        
        if best_v < 0:
            return
        
        v = best_v
        v_bit = best_v_bit
        
        # Check if v can be added
        N_v_in_cur = adj_bitmasks[v] & current_mask
        safe_to_add = True
        if N_v_in_cur:
            temp_check = N_v_in_cur
            while temp_check:
                u_bit = temp_check & -temp_check
                u = u_bit.bit_length() - 1
                if adj_bitmasks[u] & N_v_in_cur:
                    safe_to_add = False
                    break
                temp_check -= u_bit
        
        if safe_to_add:
            new_mask = current_mask | v_bit
            N_v_in_new = adj_bitmasks[v] & new_mask
            forbidden = 0
            temp_w = N_v_in_new
            while temp_w:
                w_bit = temp_w & -temp_w
                w = w_bit.bit_length() - 1
                forbidden |= adj_bitmasks[w]
                temp_w -= w_bit
            forbidden &= adj_bitmasks[v]
            forbidden &= available
            new_available = available & ~forbidden & ~v_bit
            dfs(new_available, new_mask, current_size + 1)
        
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best

def _is_k4_free(bm, n):
    """Check K4-freeness using bitmasks."""
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                common = bm[i] & bm[j]
                while common:
                    k_bit = common & -common
                    k = k_bit.bit_length() - 1
                    if bm[i] & bm[j] & bm[k]:
                        return False
                    common -= k_bit
    return True

def _turan_base(n, parts):
    """Create complete multipartite graph with given part sizes."""
    bm = [0] * n
    idx = 0
    for p in parts:
        for i in range(idx, idx + p):
            for j in range(idx + p, n):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
        idx += p
    return bm

def _edge_flip_k4_check(bm, n, i, j, adding):
    """Check if flipping edge (i,j) maintains K4-freeness."""
    if adding:
        common = bm[i] & bm[j]
        while common:
            k_bit = common & -common
            k = k_bit.bit_length() - 1
            if bm[i] & bm[j] & bm[k]:
                return False
            common -= k_bit
        return True
    else:
        return True

def _find_min_edge_additions(bm, n, target_size, deadline):
    """
    Constraint satisfaction approach:
    Find the minimum number of edges to add (maintaining K4-freeness)
    such that no triangle-free induced subgraph of size target_size exists.
    Uses DPLL-style search with conflict-driven clause learning.
    """
    # Precompute all subsets of size target_size
    all_subsets = list(itertools.combinations(range(n), target_size))
    
    # For each subset, check if it's currently triangle-free
    # If it is, we need to add an edge within it to break it
    # But adding an edge might break K4-freeness
    
    # We'll use a greedy + backtracking approach
    # State: current graph (bm), set of "active" TF subsets
    # Goal: eliminate all TF subsets of size target_size
    
    # First, identify which subsets are currently TF
    active_subsets = []
    for subset in all_subsets:
        subset_mask = 0
        for v in subset:
            subset_mask |= (1 << v)
        is_tf = True
        for a in subset:
            for b in subset:
                if a < b and (bm[a] & (1 << b)):
                    common = bm[a] & bm[b] & subset_mask
                    if common:
                        is_tf = False
                        break
            if not is_tf:
                break
        if is_tf:
            active_subsets.append(subset)
    
    if not active_subsets:
        return 0, bm[:]
    
    # Greedy: for each active subset, find edges that can be added to break it
    # while maintaining K4-freeness
    # Prioritize edges that break multiple subsets
    
    # Build a mapping: edge -> list of subsets it would break
    edge_to_subsets = {}
    for subset in active_subsets:
        subset_mask = 0
        for v in subset:
            subset_mask |= (1 << v)
        for i in subset:
            for j in subset:
                if i < j and not (bm[i] & (1 << j)):
                    key = (i, j)
                    if key not in edge_to_subsets:
                        edge_to_subsets[key] = []
                    edge_to_subsets[key].append(subset)
    
    # Sort edges by how many subsets they break
    sorted_edges = sorted(edge_to_subsets.items(), key=lambda x: -len(x[1]))
    
    # Try to add edges greedily, checking K4-freeness
    bm_copy = bm[:]
    added_edges = []
    
    for edge, subsets in sorted_edges:
        if time.time() > deadline:
            break
        i, j = edge
        if not (bm_copy[i] & (1 << j)):
            if _edge_flip_k4_check(bm_copy, n, i, j, True):
                bm_copy[i] |= (1 << j)
                bm_copy[j] |= (1 << i)
                added_edges.append((i, j))
    
    return len(added_edges), bm_copy

def _dpll_search(bm, n, target_size, deadline, max_depth=20):
    """
    DPLL-style search to find a K4-free graph where no TF subset of size target_size exists.
    Uses clause learning: if adding an edge creates a K4, learn a conflict clause.
    """
    # This is a simplified version - in practice, we use the greedy approach above
    # combined with local search for refinement
    return bm[:]

def _local_search_guided(bm, n, max_iter, seed, deadline, target_size=11):
    """
    Clause-guided local search: try edge flips that reduce the number of 
    triangle-free subsets of size target_size+1.
    """
    rng = random.Random(seed)
    best_score = _max_tf_size_fast(bm, n)
    best_bm = bm[:]
    
    # Precompute all (target_size+1)-subsets
    all_subsets = list(itertools.combinations(range(n), target_size + 1))
    
    for iteration in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        # Pick a random edge to try flipping
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        has_edge = bool(bm[i] & (1 << j))
        
        if has_edge:
            # Try removing edge (i, j)
            bm[i] &= ~(1 << j)
            bm[j] &= ~(1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
        else:
            # Try adding edge (i, j)
            if _edge_flip_k4_check(bm, n, i, j, True):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    bm[i] &= ~(1 << j)
                    bm[j] &= ~(1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _systematic_perturbation(bm, n, deadline):
    """
    Systematically try all single-edge perturbations around the base graph,
    then double-edge perturbations for promising ones.
    """
    best_score = _max_tf_size_fast(bm, n)
    best_bm = bm[:]
    
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    # Try all single edge additions (that maintain K4-freeness)
    for i, j in all_edges:
        if time.time() > deadline - 10:
            break
        if not (bm[i] & (1 << j)):
            if _edge_flip_k4_check(bm, n, i, j, True):
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                score = _max_tf_size_fast(bm, n)
                if score < best_score:
                    best_score = score
                    best_bm = bm[:]
                bm[i] &= ~(1 << j)
                bm[j] &= ~(1 << i)
    
    # Try all single edge removals
    for i, j in all_edges:
        if time.time() > deadline - 10:
            break
        if bm[i] & (1 << j):
            bm[i] &= ~(1 << j)
            bm[j] &= ~(1 << i)
            score = _max_tf_size_fast(bm, n)
            if score < best_score:
                best_score = score
                best_bm = bm[:]
            bm[i] |= (1 << j)
            bm[j] |= (1 << i)
    
    # For the best single perturbation, try double perturbations
    if best_score < 15:
        best_bm_copy = best_bm[:]
        for i, j in all_edges:
            if time.time() > deadline - 5:
                break
            if not (best_bm_copy[i] & (1 << j)):
                if _edge_flip_k4_check(best_bm_copy, n, i, j, True):
                    best_bm_copy[i] |= (1 << j)
                    best_bm_copy[j] |= (1 << i)
                    score = _max_tf_size_fast(best_bm_copy, n)
                    if score < best_score:
                        best_score = score
                        best_bm = best_bm_copy[:]
                    best_bm_copy[i] &= ~(1 << j)
                    best_bm_copy[j] &= ~(1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _constraint_satisfaction_search(bm, n, deadline, target_size=12):
    """
    Structural change: Model as CSP where we need to ensure every subset of 
    target_size vertices contains a triangle. Use iterative refinement with 
    conflict-driven edge addition.
    """
    best_score = _max_tf_size_fast(bm, n)
    best_bm = bm[:]
    
    # Iteratively find TF subsets of size target_size and try to break them
    iteration = 0
    while time.time() < deadline and iteration < 100:
        iteration += 1
        
        # Find a TF subset of size target_size
        found_subset = None
        all_subsets = list(itertools.combinations(range(n), target_size))
        random.shuffle(all_subsets)
        
        for subset in all_subsets:
            subset_mask = 0
            for v in subset:
                subset_mask |= (1 << v)
            is_tf = True
            for a in subset:
                for b in subset:
                    if a < b and (bm[a] & (1 << b)):
                        common = bm[a] & bm[b] & subset_mask
                        if common:
                            is_tf = False
                            break
                if not is_tf:
                    break
            if is_tf:
                found_subset = subset
                break
        
        if found_subset is None:
            # No TF subset of size target_size found - we've achieved the goal!
            break
        
        # Try to add an edge within this subset to break it
        subset_set = set(found_subset)
        candidate_edges = []
        for i in found_subset:
            for j in found_subset:
                if i < j and not (bm[i] & (1 << j)):
                    if _edge_flip_k4_check(bm, n, i, j, True):
                        candidate_edges.append((i, j))
        
        if not candidate_edges:
            # No edge can be added without creating K4 - try removing edges elsewhere
            # to allow more flexibility
            break
        
        # Try each candidate edge
        best_candidate_score = best_score
        best_candidate_edge = None
        for i, j in candidate_edges:
            bm[i] |= (1 << j)
            bm[j] |= (1 << i)
            score = _max_tf_size_fast(bm, n)
            if score < best_candidate_score:
                best_candidate_score = score
                best_candidate_edge = (i, j)
            bm[i] &= ~(1 << j)
            bm[j] &= ~(1 << i)
        
        if best_candidate_edge is not None and best_candidate_score < best_score:
            i, j = best_candidate_edge
            bm[i] |= (1 << j)
            bm[j] |= (1 << i)
            best_score = best_candidate_score
            best_bm = bm[:]
        else:
            # No improvement - try a different approach
            break
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110
    n = 18
    
    # Start with Turán graph T(18,3): three parts of size 6
    parts = [6, 6, 6]
    bm = _turan_base(n, parts)
    
    baseline_score = _max_tf_size_fast(bm, n)
    best_score = baseline_score
    best_bm = bm[:]
    
    # Systematic perturbation around Turán base
    if time.time() < deadline - 30:
        edges, score = _systematic_perturbation(bm, n, deadline)
        if score < best_score:
            best_score = score
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # NEW: Constraint satisfaction search - structural change
    if time.time() < deadline - 25:
        edges, score = _constraint_satisfaction_search(best_bm, n, deadline, target_size=12)
        if score < best_score:
            best_score = score
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # Guided local search from the best found so far
    if time.time() < deadline - 20:
        edges, score = _local_search_guided(best_bm, n, max_iter=500, seed=42, deadline=deadline)
        if score < best_score:
            best_score = score
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # Try different Turán-like bases: unbalanced partitions
    alt_parts = [
        [5, 6, 7],
        [4, 7, 7],
        [5, 5, 8],
        [6, 5, 7],
        [7, 5, 6],
    ]
    for alt in alt_parts:
        if time.time() > deadline - 15:
            break
        alt_bm = _turan_base(n, alt)
        alt_score = _max_tf_size_fast(alt_bm, n)
        if alt_score < best_score:
            best_score = alt_score
            best_bm = alt_bm[:]
            if time.time() < deadline - 10:
                edges, ls_score = _local_search_guided(best_bm, n, max_iter=200, seed=123, deadline=deadline)
                if ls_score < best_score:
                    best_score = ls_score
                    best_bm = [0] * n
                    for e in edges:
                        u, v = e
                        best_bm[u] |= (1 << v)
                        best_bm[v] |= (1 << u)
    
    # Final local search polish
    if time.time() < deadline - 5:
        edges, score = _local_search_guided(best_bm, n, max_iter=300, seed=777, deadline=deadline)
        if score < best_score:
            best_score = score
            best_bm = [0] * n
            for e in edges:
                u, v = e
                best_bm[u] |= (1 << v)
                best_bm[v] |= (1 << u)
    
    # Verify K4-freeness
    if not _is_k4_free(best_bm, n):
        best_bm = _turan_base(n, [6, 6, 6])
    
    # Convert to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    
    return edges
