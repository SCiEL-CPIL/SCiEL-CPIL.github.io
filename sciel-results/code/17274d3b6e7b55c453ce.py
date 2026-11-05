import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    n = 40
    best_edges = []
    best_count = 0
    best_adj = [0] * n
    
    # Fast C6 check using bitsets
    # Checks if adding edge (u,v) creates a C6.
    # Equivalent to: is there a simple path of length 5 from u to v in the current graph?
    # Path: u-a-b-c-d-v
    def creates_c6(adj, u, v):
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)
        
        if nu == 0 or nv == 0:
            return False
        
        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a
            
            # d must be a neighbor of v, not a, not u
            bits_d = nv & ~(1 << a) & ~(1 << u)
            if bits_d == 0:
                continue
            
            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d
                
                # b must be neighbor of a, not u, not v, not a, not d
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue
                
                # c must be neighbor of d, not u, not v, not a, not d
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue
                
                # c must also be neighbor of b
                # So we need intersection of N(b) and c_mask
                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b
                    
                    # Check if any c in c_mask is also neighbor of b
                    # c_mask already excludes u, v, a, d.
                    # We need c != b (implied if b not in c_mask? No, b could be in c_mask if b~d and b not in {u,v,a,d})
                    # But c is in N(d). If b is in N(d), then b is in c_mask.
                    # However, c must be distinct from b.
                    # So check (adj[b] & c_mask) & ~(1<<b)
                    if adj[b] & c_mask & ~(1 << b):
                        return True
        return False

    def get_edges(adj):
        edges = []
        for i in range(n):
            bits = adj[i]
            while bits:
                lsb = bits & -bits
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                bits ^= lsb
        return edges

    def count_edges(adj):
        return sum(adj[i].bit_count() for i in range(n)) // 2

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def greedy_construct(rng):
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)
        
        for u, v in all_edges:
            if not creates_c6(adj, u, v):
                add_edge(adj, u, v)
        return adj

    def local_search(adj, rng):
        """
        Greedily add valid edges until no more can be added.
        """
        changed = True
        while changed:
            changed = False
            missing = []
            for u in range(n):
                for v in range(u + 1, n):
                    if not (adj[u] & (1 << v)):
                        missing.append((u, v))
            
            if not missing:
                break
                
            rng.shuffle(missing)
            for u, v in missing:
                if time.time() > deadline:
                    return adj
                if not creates_c6(adj, u, v):
                    add_edge(adj, u, v)
                    changed = True
        return adj

    def swap_search(adj, rng):
        """
        Try removing an edge and adding others to see if we can increase edge count.
        """
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return adj
            
            # Get list of current edges
            current_edges = []
            for i in range(n):
                for j in range(i + 1, n):
                    if (1 << j) & adj[i]:
                        current_edges.append((i, j))
            
            if not current_edges:
                return adj
                
            rng.shuffle(current_edges)
            
            for u, v in current_edges:
                if time.time() > deadline:
                    return adj
                
                # Try removing (u, v)
                remove_edge(adj, u, v)
                old_count = count_edges(adj)
                
                # Try to add as many missing edges as possible
                added_edges = []
                for i in range(n):
                    for j in range(i + 1, n):
                        if not ((1 << j) & adj[i]):
                            if not creates_c6(adj, i, j):
                                add_edge(adj, i, j)
                                added_edges.append((i, j))
                
                new_count = count_edges(adj)
                
                if new_count > old_count + 1:  # Net gain > 0
                    improved = True
                    # Keep the changes
                else:
                    # Revert
                    for i, j in added_edges:
                        remove_edge(adj, i, j)
                    add_edge(adj, u, v)
                    
        return adj

    rng = random.Random(42)
    
    # Phase 1: Random Restarts + Local Search
    # We will run multiple restarts.
    # Time budget for phase 1: ~30s
    phase1_deadline = start_time + 30.0
    
    while time.time() < phase1_deadline:
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        m = count_edges(adj)
        if m > best_count:
            best_count = m
            best_adj = adj[:]
            best_edges = get_edges(adj)
            
    current_adj = best_adj[:]
    current_m = best_count

    # Phase 2: Iterated Local Search (ILS) with Swaps
    # Perturb, then swap_search, then local_search
    while time.time() < deadline:
        # Perturb current best
        perturbed = current_adj[:]
        
        # Randomly remove some edges
        num_removals = rng.randint(3, 10)
        existing = []
        for i in range(n):
            for j in range(i + 1, n):
                if (1 << j) & perturbed[i]:
                    existing.append((i, j))
                    
        if existing:
            rng.shuffle(existing)
            for i in range(min(num_removals, len(existing))):
                u, v = existing[i]
                remove_edge(perturbed, u, v)
        
        # Swap Search
        perturbed = swap_search(perturbed, rng)
        
        # Local Search (fill in gaps)
        perturbed = local_search(perturbed, rng)
        
        m = count_edges(perturbed)
        
        if m > best_count:
            best_count = m
            best_adj = perturbed[:]
            best_edges = get_edges(perturbed)
            current_adj = perturbed[:]
            current_m = m
        else:
            # With some probability, move to the worse solution to escape local optima
            # Or just keep current best as the base for next perturbation?
            # Standard ILS: accept if better, or with prob p if worse.
            # Here we simply keep the best found so far as the base, 
            # but we could also accept worse to explore.
            # Let's try: if m > current_m - 2, accept as new current (annealing-like)
            # But simpler: always perturb the global best.
            # However, if we always perturb global best, we might get stuck in a local basin of the global best.
            # Let's try perturbing the current solution (which might be worse than best) to explore.
            if m >= current_m - 2:
                current_adj = perturbed[:]
                current_m = m
            else:
                # Revert to best if too bad
                current_adj = best_adj[:]
                current_m = best_count

    # Final validation
    # We should ensure the returned graph is C6-free.
    # The local search and swap search only add edges if they don't create C6.
    # So the result should be valid.
    # But just in case, we can do a quick check.
    # For now, we trust the incremental construction.
    
    return best_edges
