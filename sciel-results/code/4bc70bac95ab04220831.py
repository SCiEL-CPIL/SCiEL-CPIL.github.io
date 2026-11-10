import time
import random
import itertools

def _precompute_tri_free(adj_mask, n):
    """Precompute which subsets are triangle-free using DP with bitmasks.
    adj_mask[i] is a bitmask of neighbors of vertex i.
    Returns a list of length 2^n where tri_free[mask] is True if the subset is triangle-free."""
    size = 1 << n
    tri_free = [False] * size
    tri_free[0] = True
    
    # Precompute: for each mask, is it an independent set?
    # indep[mask] = True if no two vertices in mask are adjacent
    indep = [False] * size
    indep[0] = True
    for mask in range(1, size):
        # Get lowest set bit
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        # mask is independent iff prev is independent AND v has no neighbors in prev
        indep[mask] = indep[prev] and (adj_mask[v] & prev) == 0
    
    # DP for triangle-free
    for mask in range(1, size):
        lb = mask & (-mask)
        v = lb.bit_length() - 1
        prev = mask ^ lb
        if not tri_free[prev]:
            tri_free[mask] = False
        else:
            # Check if v forms a triangle with any pair in prev
            # v forms triangle with (a,b) iff adj[v][a] and adj[v][b] and adj[a][b]
            # i.e., the neighbors of v in prev must be an independent set
            neigh_in_prev = adj_mask[v] & prev
            tri_free[mask] = indep[neigh_in_prev]
    
    return tri_free

def _max_tf_size(adj_mask, n, tri_free):
    """Find the largest triangle-free subset size."""
    size = 1 << n
    best = 0
    for mask in range(size):
        if tri_free[mask]:
            c = mask.bit_count()
            if c > best:
                best = c
    return best

def _is_k4_free_mask(adj_mask, n):
    """Check if graph is K4-free using bitmasks."""
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_mask[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common = adj_mask[a] & adj_mask[b]
            # For each c in common, check if c has any neighbor in common
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                if adj_mask[c] & common:
                    return False
                c_bits ^= lb
    return True

def _is_k4_free_after_add(adj_mask, n, u, v):
    """Fast check: after adding edge (u,v), is graph still K4-free?
    Only need to check K4s containing the new edge (u,v).
    A K4 containing edge (u,v) has two other vertices c,d that are:
    - common neighbors of u and v (i.e., adjacent to both u and v)
    - c and d are adjacent to each other
    So: among common neighbors of u,v, is there an edge?"""
    common = adj_mask[u] & adj_mask[v]
    c_bits = common
    while c_bits:
        lb = c_bits & (-c_bits)
        c = lb.bit_length() - 1
        if adj_mask[c] & common:
            return False
        c_bits ^= lb
    return True

def _compute_score(adj_mask, n):
    """Compute the largest triangle-free induced subgraph size."""
    tri_free = _precompute_tri_free(adj_mask, n)
    return _max_tf_size(adj_mask, n, tri_free)

def _count_k4s(adj_mask, n):
    """Count number of K4s in the graph using bitmasks."""
    count = 0
    for a in range(n):
        for b in range(a+1, n):
            if not (adj_mask[a] >> b) & 1:
                continue
            common = adj_mask[a] & adj_mask[b]
            # Count edges within common neighbors (each K4 counted once per edge in it)
            c_bits = common
            while c_bits:
                lb = c_bits & (-c_bits)
                c = lb.bit_length() - 1
                # Count neighbors of c in common that are > c to avoid double counting
                neigh_c = adj_mask[c] & common
                neigh_c &= ~((1 << (c+1)) - 1)  # only count vertices > c
                count += neigh_c.bit_count()
                c_bits ^= lb
    return count

def _count_tri_free_12subsets(adj_mask, n, target=12):
    """Count how many subsets of size `target` are triangle-free.
    Returns (count, list_of_some_tf_subsets) for diagnostics."""
    tri_free = _precompute_tri_free(adj_mask, n)
    count = 0
    examples = []
    # Iterate over all masks of size `target`
    for mask in range(1 << n):
        if mask.bit_count() == target and tri_free[mask]:
            count += 1
            if len(examples) < 5:
                examples.append(mask)
    return count, examples

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin for 150s limit
    n = 18
    size = 1 << n
    
    def time_ok():
        return time.time() < deadline
    
    best_score = 18  # worst case
    best_adj_mask = [0] * n
    
    def evaluate(adj_mask):
        """Return (score, adj_mask) if K4-free, else None."""
        if not _is_k4_free_mask(adj_mask, n):
            return None
        score = _compute_score(adj_mask, n)
        return (score, adj_mask)
    
    # Construction 1: K6,6,6 (complete tripartite)
    adj1 = [0] * n
    for i in range(6):
        for j in range(6, 12):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
        for j in range(12, 18):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    for i in range(6, 12):
        for j in range(12, 18):
            adj1[i] |= (1 << j)
            adj1[j] |= (1 << i)
    
    result = evaluate(adj1)
    if result:
        best_score, best_adj_mask = result
    
    # Construction 2: Random dense K4-free graphs
    random.seed(42)
    
    for trial in range(200):
        if not time_ok():
            break
        adj = [0] * n
        edges = list(itertools.combinations(range(n), 2))
        random.shuffle(edges)
        for u, v in edges:
            if not time_ok():
                break
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            if not _is_k4_free_after_add(adj, n, u, v):
                adj[u] &= ~(1 << v)
                adj[v] &= ~(1 << u)
        result = evaluate(adj)
        if result:
            score, _ = result
            if score < best_score:
                best_score = score
                best_adj_mask = adj[:]
    
    # Construction 3: Min-SAT style local search
    # Objective: minimize (K4_count * 100 + tri_free_12_count)
    # K4-free means K4_count = 0; we want tri_free_12_count = 0 for score <= 11
    if time_ok():
        adj = best_adj_mask[:]
        
        def min_sat_objective(adj):
            k4_count = _count_k4s(adj, n)
            tf12_count, _ = _count_tri_free_12subsets(adj, n, 12)
            return k4_count * 100 + tf12_count
        
        # Local search: flip edges to reduce objective
        all_edges = list(itertools.combinations(range(n), 2))
        current_obj = min_sat_objective(adj)
        
        for iteration in range(5000):
            if not time_ok():
                break
            
            # Try single edge flips
            random.shuffle(all_edges)
            improved = False
            for u, v in all_edges:
                if not time_ok():
                    break
                # Flip edge (u,v)
                had_edge = (adj[u] >> v) & 1
                if had_edge:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                
                new_obj = min_sat_objective(adj)
                
                if new_obj < current_obj:
                    current_obj = new_obj
                    improved = True
                    # Check if K4-free and update best
                    if _is_k4_free_mask(adj, n):
                        score = _compute_score(adj, n)
                        if score < best_score:
                            best_score = score
                            best_adj_mask = adj[:]
                else:
                    # Revert
                    if had_edge:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
            
            if not improved:
                # Try 2-flips (swap: remove one edge, add another)
                existing = [(u,v) for u,v in all_edges if (adj[u]>>v)&1]
                missing = [(u,v) for u,v in all_edges if not (adj[u]>>v)&1]
                random.shuffle(existing)
                random.shuffle(missing)
                improved = False
                for rem_u, rem_v in existing[:20]:
                    if not time_ok():
                        break
                    for add_u, add_v in missing[:20]:
                        if not time_ok():
                            break
                        if (rem_u, rem_v) == (add_u, add_v) or (rem_u, rem_v) == (add_v, add_u):
                            continue
                        # Remove rem edge, add add edge
                        adj[rem_u] &= ~(1 << rem_v)
                        adj[rem_v] &= ~(1 << rem_u)
                        adj[add_u] |= (1 << add_v)
                        adj[add_v] |= (1 << add_u)
                        
                        new_obj = min_sat_objective(adj)
                        
                        if new_obj < current_obj:
                            current_obj = new_obj
                            improved = True
                            if _is_k4_free_mask(adj, n):
                                score = _compute_score(adj, n)
                                if score < best_score:
                                    best_score = score
                                    best_adj_mask = adj[:]
                        else:
                            # Revert
                            adj[rem_u] |= (1 << rem_v)
                            adj[rem_v] |= (1 << rem_u)
                            adj[add_u] &= ~(1 << add_v)
                            adj[add_v] &= ~(1 << add_u)
                    
                    if not time_ok():
                        break
                
                if not improved:
                    break
        
        # Restore best_adj_mask from local search if it's better
        if time_ok():
            if _is_k4_free_mask(adj, n):
                score = _compute_score(adj, n)
                if score < best_score:
                    best_score = score
                    best_adj_mask = adj[:]
    
    # Construction 4: Final polish - try all single-edge changes to best graph
    if time_ok():
        adj = best_adj_mask[:]
        for u in range(n):
            for v in range(u+1, n):
                if not time_ok():
                    break
                if (adj[u] >> v) & 1:
                    # Try removing
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    score = _compute_score(adj, n)
                    if score < best_score:
                        best_score = score
                        best_adj_mask = adj[:]
                    else:
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                else:
                    # Try adding
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    if _is_k4_free_after_add(adj, n, u, v):
                        score = _compute_score(adj, n)
                        if score < best_score:
                            best_score = score
                            best_adj_mask = adj[:]
                        else:
                            adj[u] &= ~(1 << v)
                            adj[v] &= ~(1 << u)
                    else:
                        adj[u] &= ~(1 << v)
                        adj[v] &= ~(1 << u)
    
    # Convert to edge list
    best_edges = []
    for u in range(n):
        for v in range(u+1, n):
            if (best_adj_mask[u] >> v) & 1:
                best_edges.append([u, v])
    
    return best_edges
