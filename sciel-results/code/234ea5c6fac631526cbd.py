import time
import random
import sys

def _paley_37():
    """Return adjacency bitsets for Paley graph on 37 vertices."""
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    adj = [0] * q
    for i in range(q):
        for j in range(i + 1, q):
            if (j - i) % q in residues:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
    return q, adj

def _get_k_cliques(adj, n, k):
    """Return list of tuples representing all cliques of size k."""
    cliques = []
    if k <= 1:
        return [(i,) for i in range(n)]
    if k == 2:
        for i in range(n):
            for j in range(i+1, n):
                if adj[i] & (1 << j):
                    cliques.append((i, j))
        return cliques
    
    # Recursive search for k-cliques
    def _search(candidates, depth, current_clique):
        if depth == 0:
            cliques.append(tuple(sorted(current_clique)))
            return
        while candidates:
            v = (candidates & -candidates).bit_length() - 1
            candidates &= candidates - 1
            _search(candidates & adj[v], depth - 1, current_clique + [v])
            
    _search((1 << n) - 1, k, [])
    return cliques

def _get_k_independents(adj, n, k):
    """Return list of tuples representing all independent sets of size k."""
    comp_adj = [0] * n
    full_mask = (1 << n) - 1
    for i in range(n):
        comp_adj[i] = full_mask & ~(adj[i] | (1 << i))
    return _get_k_cliques(comp_adj, n, k)

def _find_valid_neighbor(adj, n):
    """
    Find a subset S of {0..n-1} such that adding a new vertex connected to S
    results in a (5,5)-free graph.
    S must be K4-free.
    V\S must be Kbar4-free (i.e., no independent set of size 4 in G[V\S]).
    """
    # Precompute all K4s and Kbar4s
    k4s = _get_k_cliques(adj, n, 4)
    kbar4s = _get_k_independents(adj, n, 4)
    
    # Convert to sets of frozensets or just lists of tuples for fast intersection check?
    # Since n is small, we can represent S as a bitmask.
    # Condition: For all c in k4s, (S_mask & c_mask) != c_mask.
    # Condition: For all c in kbar4s, (N_mask & c_mask) != c_mask, where N_mask = full_mask ^ S_mask.
    
    full_mask = (1 << n) - 1
    
    # Backtracking search for S_mask
    # We decide for each vertex 0..n-1 whether it is in S or not.
    # Order: Randomize to avoid bias, or use degree heuristic.
    # Let's use a simple recursive backtracker.
    
    # To speed up, we can prune if S already contains a K4 or N already contains a Kbar4.
    # But checking "contains a K4" against a list of K4s is O(|K4s|).
    # |K4s| can be large.
    
    # Alternative: Just generate random S and check?
    # With n=37, 2^37 is too big.
    # But the constraints are tight.
    
    # Let's use a randomized local search / hill climbing on the bitmask S.
    # Start with S=0.
    # Violations:
    # v1 = number of K4s fully contained in S.
    # v2 = number of Kbar4s fully contained in N (which is full_mask ^ S).
    # Minimize v1 + v2.
    
    best_S = 0
    best_cost = float('inf')
    
    # Initial random starts
    for _ in range(50):
        S = 0
        # Random initialization
        for i in range(n):
            if random.random() < 0.5:
                S |= (1 << i)
        
        # Local search
        # Flip bits to reduce cost
        stuck = False
        for _ in range(100):
            if stuck: break
            stuck = True
            cost = 0
            # Calculate cost
            # Check K4s in S
            for c in k4s:
                mask = 0
                for v in c: mask |= (1 << v)
                if (S & mask) == mask:
                    cost += 1
            # Check Kbar4s in N
            N = full_mask ^ S
            for c in kbar4s:
                mask = 0
                for v in c: mask |= (1 << v)
                if (N & mask) == mask:
                    cost += 1
            
            if cost == 0:
                return S
            
            if cost < best_cost:
                best_cost = cost
                best_S = S
            
            # Try flipping a random bit
            i = random.randint(0, n-1)
            S_new = S ^ (1 << i)
            
            # Calculate new cost efficiently?
            # For now, full recalculation is fine for n=40, k=4.
            # Number of K4s is likely in the hundreds/thousands.
            # 100 iterations * 50 starts * (cost calc) should be fast enough.
            
            # Heuristic: Only flip if it reduces cost?
            # Let's try a few flips and pick the best.
            min_cost_neighbor = cost
            best_neighbor_S = S
            for _ in range(10):
                j = random.randint(0, n-1)
                S_try = S ^ (1 << j)
                # Calc cost for S_try
                c_cost = 0
                for c in k4s:
                    mask = 0
                    for v in c: mask |= (1 << v)
                    if (S_try & mask) == mask:
                        c_cost += 1
                N_try = full_mask ^ S_try
                for c in kbar4s:
                    mask = 0
                    for v in c: mask |= (1 << v)
                    if (N_try & mask) == mask:
                        c_cost += 1
                
                if c_cost < min_cost_neighbor:
                    min_cost_neighbor = c_cost
                    best_neighbor_S = S_try
                    stuck = False
                    break
            
            if not stuck:
                S = best_neighbor_S
            else:
                # Perturb
                for _ in range(5):
                    j = random.randint(0, n-1)
                    S ^= (1 << j)
                stuck = False

    if best_cost == 0:
        return best_S
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 290
    
    n, adj = _paley_37()
    best_n = n
    best_adj = adj[:]
    
    # Try to add vertices
    while time.time() < deadline:
        S = _find_valid_neighbor(adj, n)
        if S is not None:
            # Add new vertex n connected to S
            new_adj = adj + [S]
            for i in range(n):
                if S & (1 << i):
                    new_adj[i] |= (1 << n)
            n += 1
            adj = new_adj
            best_n = n
            best_adj = adj[:]
        else:
            # Stuck. Try to restart or perturb.
            # Simple strategy: Just break and return best.
            # Or try a different random seed?
            # Let's try to remove a random edge and see if we can re-add?
            # That's complex. Let's just try a few more times with different random states.
            # If we fail 10 times in a row, break.
            fails = 0
            for _ in range(10):
                if time.time() >= deadline: break
                S = _find_valid_neighbor(adj, n)
                if S is not None:
                    new_adj = adj + [S]
                    for i in range(n):
                        if S & (1 << i):
                            new_adj[i] |= (1 << n)
                    n += 1
                    adj = new_adj
                    best_n = n
                    best_adj = adj[:]
                    fails = 0
                    break
                else:
                    fails += 1
            if fails == 10:
                break

    # Convert adj bitsets back to edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if best_adj[i] & (1 << j):
                edges.append((i, j))
    
    return best_n, edges
