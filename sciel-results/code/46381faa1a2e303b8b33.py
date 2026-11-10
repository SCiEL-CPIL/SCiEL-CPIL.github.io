# EVOLVE-BLOCK-START
import random
import time

def _max_tf_size_fast(adj_bitmasks, n):
    """
    Fast branch and bound to find the largest triangle-free induced subgraph.
    adj_bitmasks: list of ints, where adj_bitmasks[i] is a bitmask of neighbors of i.
    Returns the size of the largest triangle-free induced subgraph.
    """
    best = 0
    full_mask = (1 << n) - 1
    
    # Precompute N(i) & N(j) for all pairs to check triangle existence quickly
    common = [[0]*n for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            common[i][j] = adj_bitmasks[i] & adj_bitmasks[j]
            common[j][i] = common[i][j]
            
    def dfs(available, current_mask, current_size):
        nonlocal best
        # Pruning
        if current_size + available.bit_count() <= best:
            return
            
        if available == 0:
            if current_size > best:
                best = current_size
            return
            
        # Heuristic: pick a vertex with high degree in available for better pruning
        best_v = -1
        best_deg = -1
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
        
        # Option 1: Include v
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
            
        # Option 2: Exclude v
        dfs(available & ~v_bit, current_mask, current_size)
        
    dfs(full_mask, 0, 0)
    return best

def _adj_to_bitmasks(adj, n):
    """Convert adjacency matrix to list of bitmasks."""
    bm = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        bm[i] = mask
    return bm

def _check_k4_free_bitmask(bm, n):
    """Check if graph represented by bitmasks is K4-free."""
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

def _random_k4_free_graph(n, seed):
    """Generate a random K4-free graph using bitmask representation."""
    rng = random.Random(seed)
    bm = [0] * n
    
    edges_to_try = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rng.shuffle(edges_to_try)
    
    for i, j in edges_to_try:
        if rng.random() < 0.5:
            common_ij = bm[i] & bm[j]
            creates_k4 = False
            temp_common = common_ij
            while temp_common:
                k_bit = temp_common & -temp_common
                k = k_bit.bit_length() - 1
                if bm[i] & bm[j] & bm[k]:
                    creates_k4 = True
                    break
                temp_common -= k_bit
            
            if not creates_k4:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if bm[i] & (1 << j):
                edges.append([i, j])
    return bm, edges

def _compute_clauses(bm, n, target_tf_size=11):
    """
    Compute clauses for Min-SAT formulation.
    Each clause represents a constraint that should be satisfied.
    Returns a list of clauses, where each clause is a dict with:
    - 'type': 'k4' or 'triangle_free'
    - 'vertices': list of vertex indices involved
    - 'satisfied': whether the clause is currently satisfied
    """
    clauses = []
    
    # K4-freeness clauses: for each potential K4 (4 vertices), 
    # at least one edge must be missing
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                for l in range(k+1, n):
                    # Check if all 6 edges exist (would form K4)
                    has_all_edges = (bm[i] & (1<<j) and bm[i] & (1<<k) and bm[i] & (1<<l) and
                                    bm[j] & (1<<k) and bm[j] & (1<<l) and
                                    bm[k] & (1<<l))
                    if has_all_edges:
                        clauses.append({
                            'type': 'k4',
                            'vertices': [i, j, k, l],
                            'satisfied': False
                        })
    
    # Triangle-free constraint: for each subset of size (target_tf_size+1),
    # it must contain a triangle (so no triangle-free subgraph of that size exists)
    # We use a sampling approach for efficiency
    import itertools
    rng = random.Random(42)
    all_subsets = list(itertools.combinations(range(n), target_tf_size + 1))
    # Sample a subset of these to keep computation manageable
    sample_size = min(len(all_subsets), 5000)
    sampled_subsets = rng.sample(all_subsets, sample_size) if len(all_subsets) > sample_size else all_subsets
    
    for subset in sampled_subsets:
        subset_set = set(subset)
        # Check if this subset induces a triangle-free subgraph
        has_triangle = False
        for a in subset:
            for b in subset:
                if a < b and (bm[a] & (1<<b)):
                    # Check if there's a common neighbor in subset
                    common = bm[a] & bm[b] & sum([1<<x for x in subset if x != a and x != b])
                    if common:
                        has_triangle = True
                        break
            if has_triangle:
                break
        
        if not has_triangle:
            clauses.append({
                'type': 'triangle_free',
                'vertices': list(subset),
                'satisfied': False
            })
    
    return clauses

def _min_sat_local_search(bm, n, max_iter=1000, seed=0, deadline=None, target_tf_size=11):
    """
    Min-SAT inspired local search.
    Tries to flip edges to satisfy more clauses (reduce K4s and create triangles in large subsets).
    """
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
        
        # Compute current clauses
        clauses = _compute_clauses(bm, n, target_tf_size)
        
        if not clauses:
            break
        
        # Try to find a beneficial edge flip
        best_flip = None
        best_improvement = 0
        
        # Sample a few random edge flips
        for _ in range(20):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            i, j = min(i, j), max(i, j)
            
            # Check if flip is valid (maintains K4-freeness)
            if not (bm[i] & (1 << j)):
                # Try adding edge
                common_ij = bm[i] & bm[j]
                creates_k4 = False
                temp_common = common_ij
                while temp_common:
                    k_bit = temp_common & -temp_common
                    k = k_bit.bit_length() - 1
                    if bm[i] & bm[j] & bm[k]:
                        creates_k4 = True
                        break
                    temp_common -= k_bit
                
                if creates_k4:
                    continue
                
                # Apply flip temporarily
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                
                # Check improvement
                new_score = _max_tf_size_fast(bm, n)
                improvement = current_score - new_score
                
                # Undo flip
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_flip = (i, j, True)  # True = add edge
            else:
                # Try removing edge
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
                new_score = _max_tf_size_fast(bm, n)
                improvement = current_score - new_score
                
                # Undo flip
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_flip = (i, j, False)  # False = remove edge
        
        if best_flip:
            i, j, add = best_flip
            if add:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
            else:
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
            
            current_score = _max_tf_size_fast(bm, n)
            if current_score < best_score:
                best_score = current_score
                best_bm = bm[:]
        else:
            # No improvement found, break early
            break
    
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def _local_search_bitmask(bm, n, max_iter=500, seed=0, deadline=None):
    """Local search on bitmask representation."""
    rng = random.Random(seed)
    current_score = _max_tf_size_fast(bm, n)
    best_score = current_score
    best_bm = bm[:]
    
    for _ in range(max_iter):
        if deadline and time.time() > deadline:
            break
            
        i = rng.randint(0, n - 1)
        j = rng.randint(0, n - 1)
        if i == j:
            continue
        i, j = min(i, j), max(i, j)
        
        if not (bm[i] & (1 << j)):
            # Try adding edge
            common_ij = bm[i] & bm[j]
            creates_k4 = False
            temp_common = common_ij
            while temp_common:
                k_bit = temp_common & -temp_common
                k = k_bit.bit_length() - 1
                if bm[i] & bm[j] & bm[k]:
                    creates_k4 = True
                    break
                temp_common -= k_bit
            
            if not creates_k4:
                bm[i] |= (1 << j)
                bm[j] |= (1 << i)
                new_score = _max_tf_size_fast(bm, n)
                if new_score < best_score:
                    best_score = new_score
                    best_bm = bm[:]
                else:
                    bm[i] ^= (1 << j)
                    bm[j] ^= (1 << i)
        else:
            # Try removing edge
            bm[i] ^= (1 << j)
            bm[j] ^= (1 << i)
            new_score = _max_tf_size_fast(bm, n)
            if new_score < best_score:
                best_score = new_score
                best_bm = bm[:]
            else:
                bm[i] ^= (1 << j)
                bm[j] ^= (1 << i)
                
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_bm[i] & (1 << j):
                edges.append([i, j])
    return edges, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n = 18
    
    # 1. Baseline: Turan graph T(18, 3)
    parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
    baseline_bm = [0] * n
    for i in range(3):
        for j in range(i + 1, 3):
            for a in parts[i]:
                for b in parts[j]:
                    baseline_bm[a] |= (1 << b)
                    baseline_bm[b] |= (1 << a)
    
    baseline_score = _max_tf_size_fast(baseline_bm, n)
    best_score = baseline_score
    best_edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if baseline_bm[i] & (1 << j):
                best_edges.append([i, j])
    
    # 2. Random restarts with local search
    for seed in range(200):
        if time.time() > deadline - 5:
            break
        
        bm, edges = _random_k4_free_graph(n, seed)
        score = _max_tf_size_fast(bm, n)
        
        if score < best_score:
            best_score = score
            best_edges = edges
        
        # Local search
        if time.time() > deadline - 10:
            break
            
        ls_edges, ls_score = _local_search_bitmask(bm, n, max_iter=200, seed=seed, deadline=deadline)
        if ls_score < best_score:
            best_score = ls_score
            best_edges = ls_edges
            
        # Min-SAT inspired search
        if time.time() > deadline - 15:
            break
            
        msat_edges, msat_score = _min_sat_local_search(bm, n, max_iter=100, seed=seed + 1000, deadline=deadline)
        if msat_score < best_score:
            best_score = msat_score
            best_edges = msat_edges
    
    # 3. Final Polish: Try to improve best_edges with aggressive local search
    if time.time() < deadline - 5:
        # Convert best_edges to bm
        best_bm = [0] * n
        for e in best_edges:
            u, v = e
            best_bm[u] |= (1 << v)
            best_bm[v] |= (1 << u)
            
        # Run a dedicated local search on the best candidate
        polish_edges, polish_score = _local_search_bitmask(best_bm, n, max_iter=1000, seed=999, deadline=deadline)
        if polish_score < best_score:
            best_score = polish_score
            best_edges = polish_edges
        
        # Also try Min-SAT polish
        if time.time() < deadline - 3:
            msat_polish_edges, msat_polish_score = _min_sat_local_search(best_bm, n, max_iter=500, seed=2000, deadline=deadline)
            if msat_polish_score < best_score:
                best_score = msat_polish_score
                best_edges = msat_polish_edges

    return best_edges
# EVOLVE-BLOCK-END
