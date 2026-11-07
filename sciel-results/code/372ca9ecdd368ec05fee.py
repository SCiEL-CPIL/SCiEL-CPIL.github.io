"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

P = {'n': 14, 'edges': [[0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 10], [0, 12], [1, 4], [1, 9], [1, 11], [2, 3], [2, 5], [2, 6], [2, 7], [2, 12], [2, 13], [3, 4], [3, 5], [3, 8], [3, 11], [3, 12], [3, 13], [4, 5], [4, 7], [4, 13], [5, 8], [5, 9], [5, 10], [5, 12], [5, 13], [6, 8], [6, 11], [6, 13], [7, 12], [8, 9], [8, 13], [9, 11], [9, 12], [11, 13]]}

def _build_graph():
    n = P['n']
    adj_mask = [0] * n
    for u, v in P['edges']:
        adj_mask[u] |= (1 << v)
        adj_mask[v] |= (1 << u)
    return adj_mask

def _max_clique_size(mask, adj_mask):
    """Compute maximum clique size in the subgraph induced by mask."""
    if mask == 0:
        return 0
    # Bron-Kerbosch with pivot
    best = [0]
    def bk(R, P_set, X):
        if not P_set and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        # Choose pivot
        pivot_pool = P_set | X
        pivot = -1
        max_deg = -1
        m = pivot_pool
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (adj_mask[v] & P_set).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
            m ^= lsb
        if pivot == -1:
            # No pivot, expand on all
            candidates = P_set
        else:
            # Candidates are P_set minus neighbors of pivot
            candidates = P_set & ~(adj_mask[pivot] | (1 << pivot))
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            bk(R | (1 << v), P_set & adj_mask[v] & ~(1 << v), X & adj_mask[v] & ~(1 << v))
            P_set &= ~(1 << v)
            X |= (1 << v)
    bk(0, mask, 0)
    return best[0]

def _max_independent_size(mask, adj_mask):
    """Maximum independent set = maximum clique in complement."""
    if mask == 0:
        return 0
    n = P['n']
    full = (1 << n) - 1
    # Complement adjacency
    comp_adj = [full ^ adj_mask[v] ^ (1 << v) for v in range(n)]
    best = [0]
    def bk(R, P_set, X):
        if not P_set and not X:
            if len(R) > best[0]:
                best[0] = len(R)
            return
        pivot_pool = P_set | X
        pivot = -1
        max_deg = -1
        m = pivot_pool
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            deg = (comp_adj[v] & P_set).bit_count()
            if deg > max_deg:
                max_deg = deg
                pivot = v
            m ^= lsb
        if pivot == -1:
            candidates = P_set
        else:
            candidates = P_set & ~(comp_adj[pivot] | (1 << pivot))
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            bk(R | (1 << v), P_set & comp_adj[v] & ~(1 << v), X & comp_adj[v] & ~(1 << v))
            P_set &= ~(1 << v)
            X |= (1 << v)
    bk(0, mask, 0)
    return best[0]

def _get_maximal_parts(remaining, pivot_v, adj_mask):
    """Get all maximal homogeneous sets (clique or independent) containing pivot_v within remaining."""
    n = P['n']
    full = (1 << n) - 1
    
    # For clique: neighbors of pivot within remaining
    clique_candidates = remaining & adj_mask[pivot_v] & ~(1 << pivot_v)
    # For independent set: non-neighbors of pivot within remaining
    indep_candidates = remaining & ~(adj_mask[pivot_v] | (1 << pivot_v))
    
    parts = []
    
    # Generate maximal cliques containing pivot
    # A maximal clique containing pivot = {pivot} U maximal clique in clique_candidates
    # We want all maximal cliques in the subgraph, but we need those that are maximal in the full remaining
    # Simpler: enumerate all subsets and check, but that's expensive. Use Bron-Kerbosch to find maximal cliques.
    
    # Find all maximal cliques in clique_candidates that, when combined with pivot, are maximal in remaining
    # A clique C containing pivot is maximal in remaining if no vertex in remaining\C can be added
    # Equivalently, C is a maximal clique in the subgraph induced by (remaining) that contains pivot
    
    # Use Bron-Kerbosch to find maximal cliques in the subgraph induced by remaining that contain pivot
    # We can do this by: find all maximal cliques in (remaining & (adj_mask[pivot_v] | (1 << pivot_v))) that contain pivot
    # Actually, let's just enumerate maximal cliques in the subgraph and filter
    
    # For efficiency with n=14, let's use a different approach:
    # Generate all subsets of clique_candidates, check if they form a clique with pivot, and check maximality
    
    # Better: use Bron-Kerbosch on the subgraph to get maximal cliques, then filter those containing pivot
    
    def bk_maximal(P_set, X, R):
        """Bron-Kerbosch to find all maximal cliques."""
        if not P_set and not X:
            yield R
            return
        pivot_pool = P_set | X
        if pivot_pool:
            pivot = pivot_pool & (-pivot_pool)
            pivot_v = pivot.bit_length() - 1
            candidates = P_set & ~(adj_mask[pivot_v] | pivot)
        else:
            candidates = P_set
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            yield from bk_maximal(P_set & adj_mask[v] & ~(1 << v), X & adj_mask[v] & ~(1 << v), R | (1 << v))
            P_set &= ~(1 << v)
            X |= (1 << v)
    
    # Find maximal cliques in remaining that contain pivot_v
    # The subgraph for cliques containing pivot: vertices that are neighbors of pivot in remaining
    subgraph = remaining & (adj_mask[pivot_v] | (1 << pivot_v))
    
    # We want maximal cliques in 'remaining' that contain pivot_v
    # These are maximal cliques in the subgraph induced by (remaining) that include pivot
    # Use BK with R starting at pivot
    
    def bk_containing(P_set, X, R):
        if not P_set and not X:
            yield R
            return
        pivot_pool = P_set | X
        if pivot_pool:
            pivot = pivot_pool & (-pivot_pool)
            pv = pivot.bit_length() - 1
            candidates = P_set & ~(adj_mask[pv] | pivot)
        else:
            candidates = P_set
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            yield from bk_containing(P_set & adj_mask[v] & ~(1 << v), X & adj_mask[v] & ~(1 << v), R | (1 << v))
            P_set &= ~(1 << v)
            X |= (1 << v)
    
    # Maximal cliques in remaining containing pivot_v
    # Start with R = {pivot_v}, P = remaining & adj_mask[pivot_v], X = 0
    # But we need them to be maximal in remaining, so we should run BK on the full remaining subgraph
    # and filter
    
    # Actually, let's just run BK on remaining and collect those containing pivot
    cliques = set()
    for c in bk_maximal(remaining, 0, 0):
        if c & (1 << pivot_v):
            cliques.add(c)
    
    # Similarly for independent sets: maximal independent sets containing pivot
    # Independent set in G = clique in complement
    comp_adj = [full ^ adj_mask[v] ^ (1 << v) for v in range(n)]
    
    def bk_maximal_comp(P_set, X, R):
        if not P_set and not X:
            yield R
            return
        pivot_pool = P_set | X
        if pivot_pool:
            pivot = pivot_pool & (-pivot_pool)
            pv = pivot.bit_length() - 1
            candidates = P_set & ~(comp_adj[pv] | pivot)
        else:
            candidates = P_set
        m = candidates
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            yield from bk_maximal_comp(P_set & comp_adj[v] & ~(1 << v), X & comp_adj[v] & ~(1 << v), R | (1 << v))
            P_set &= ~(1 << v)
            X |= (1 << v)
    
    indep_sets = set()
    for c in bk_maximal_comp(remaining, 0, 0):
        if c & (1 << pivot_v):
            indep_sets.add(c)
    
    parts = list(cliques) + list(indep_sets)
    # Sort by size descending for better pruning
    parts.sort(key=lambda m: m.bit_count(), reverse=True)
    return parts

def _lower_bound(remaining, adj_mask):
    """Lower bound on number of parts needed to cover remaining."""
    if remaining == 0:
        return 0
    n = P['n']
    full = (1 << n) - 1
    # The largest homogeneous set we can pick has size at most max(max_clique, max_independent)
    # So we need at least ceil(popcount(remaining) / max_size) parts
    max_c = _max_clique_size(remaining, adj_mask)
    max_i = _max_independent_size(remaining, adj_mask)
    max_size = max(max_c, max_i)
    if max_size == 0:
        return remaining.bit_count()
    import math
    return math.ceil(remaining.bit_count() / max_size)

def _greedy_upper_bound(adj_mask, n):
    """Greedy: repeatedly pick the largest homogeneous set."""
    remaining = (1 << n) - 1
    parts = 0
    while remaining:
        parts += 1
        # Find largest clique or independent set in remaining
        # Try all subsets? No, use BK to find maximal ones
        # For simplicity, greedy: pick a vertex, extend to maximal clique or independent set
        # Better: find the actual largest homogeneous set
        
        # Find all maximal cliques and independent sets, pick the largest
        best_mask = 0
        best_size = 0
        
        # Maximal cliques
        def bk_maximal(P_set, X, R):
            if not P_set and not X:
                if R.bit_count() > best_size:
                    nonlocal best_mask, best_size
                    best_mask = R
                    best_size = R.bit_count()
                return
            pivot_pool = P_set | X
            if pivot_pool:
                pivot = pivot_pool & (-pivot_pool)
                pv = pivot.bit_length() - 1
                candidates = P_set & ~(adj_mask[pv] | pivot)
            else:
                candidates = P_set
            m = candidates
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                bk_maximal(P_set & adj_mask[v] & ~(1 << v), X & adj_mask[v] & ~(1 << v), R | (1 << v))
                P_set &= ~(1 << v)
                X |= (1 << v)
        
        bk_maximal(remaining, 0, 0)
        
        # Maximal independent sets
        full = (1 << n) - 1
        comp_adj = [full ^ adj_mask[v] ^ (1 << v) for v in range(n)]
        
        def bk_maximal_comp(P_set, X, R):
            if not P_set and not X:
                if R.bit_count() > best_size:
                    nonlocal best_mask, best_size
                    best_mask = R
                    best_size = R.bit_count()
                return
            pivot_pool = P_set | X
            if pivot_pool:
                pivot = pivot_pool & (-pivot_pool)
                pv = pivot.bit_length() - 1
                candidates = P_set & ~(comp_adj[pv] | pivot)
            else:
                candidates = P_set
            m = candidates
            while m:
                lsb = m & (-m)
                v = lsb.bit_length() - 1
                m ^= lsb
                bk_maximal_comp(P_set & comp_adj[v] & ~(1 << v), X & comp_adj[v] & ~(1 << v), R | (1 << v))
                P_set &= ~(1 << v)
                X |= (1 << v)
        
        bk_maximal_comp(remaining, 0, 0)
        
        if best_mask == 0:
            # Fallback: single vertex
            lsb = remaining & (-remaining)
            best_mask = lsb
        remaining ^= best_mask
    return parts

def _backtrack_min_parts(adj_mask, n, deadline):
    full_mask = (1 << n) - 1
    best_k = [float('inf')]
    best_partition = [None]
    nodes = [0]
    
    def dfs(remaining, k_used, path):
        nodes[0] += 1
        if time.time() > deadline or nodes[0] > 5000000:
            return
        if remaining == 0:
            if k_used < best_k[0]:
                best_k[0] = k_used
                best_partition[0] = list(path)
            return
        if k_used >= best_k[0]:
            return
        lb = _lower_bound(remaining, adj_mask)
        if k_used + lb >= best_k[0]:
            return
        
        # Choose pivot: vertex with fewest possible parts
        best_v = -1
        best_count = float('inf')
        m = remaining
        while m:
            lsb = m & (-m)
            v = lsb.bit_length() - 1
            m ^= lsb
            # Count how many maximal homogeneous sets contain v
            count = 0
            # Quick estimate: degree-based
            deg = (adj_mask[v] & remaining).bit_count()
            non_deg = remaining.bit_count() - 1 - deg
            count = deg + non_deg + 1  # upper bound on number of parts containing v
            if count < best_count:
                best_count = count
                best_v = v
                if count <= 2:
                    break
        
        if best_v == -1:
            return
        
        parts = _get_maximal_parts(remaining, best_v, adj_mask)
        
        # Sort parts: larger ones first
        parts.sort(key=lambda m: m.bit_count(), reverse=True)
        
        for part_mask in parts:
            if (part_mask & remaining) != part_mask:
                continue
            if part_mask.bit_count() < 1:
                continue
            path.append(part_mask)
            dfs(remaining ^ part_mask, k_used + 1, path)
            path.pop()
    
    ub = _greedy_upper_bound(adj_mask, n)
    best_k[0] = ub
    dfs(full_mask, 0, [])
    
    if best_partition[0] is not None:
        parts = []
        for m in best_partition[0]:
            parts.append([i for i in range(n) if m & (1 << i)])
        return parts
    return None

def construct():
    adj_mask = _build_graph()
    n = P['n']
    deadline = time.time() + 55
    result = _backtrack_min_parts(adj_mask, n, deadline)
    if result is not None:
        parts = []
        for part in result:
            parts.append([int(v) for v in part])
        return parts
    # Fallback: greedy
    es = {tuple(e) for e in P['edges']}
    parts = []
    for v in range(n):
        placed = False
        for part in parts:
            trial = part + [v]
            hits = [(u, w) in es for i, u in enumerate(trial) for w in trial[i + 1:]]
            if all(hits) or not any(hits):
                part.append(v)
                placed = True
                break
        if not placed:
            parts.append([v])
    return parts

# EVOLVE-BLOCK-END
