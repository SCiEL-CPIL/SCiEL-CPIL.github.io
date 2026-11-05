import random
import time

def construct():
    deadline = time.time() + 115
    n = 40

    # Bitmask utilities
    FULL = (1 << n) - 1

    def make_adj(edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def adj_to_edges(adj):
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

    def creates_c6(adj, u, v):
        """
        Check if adding edge (u, v) creates a C6.
        Path: u - a - b - c - d - v
        """
        nu = adj[u] & ~(1 << v)
        nv = adj[v] & ~(1 << u)

        if nu == 0 or nv == 0:
            return False

        bits_a = nu
        while bits_a:
            lsb_a = bits_a & -bits_a
            a = lsb_a.bit_length() - 1
            bits_a ^= lsb_a

            bits_d = nv & ~(1 << a)
            if bits_d == 0:
                continue

            bits_d_cur = bits_d
            while bits_d_cur:
                lsb_d = bits_d_cur & -bits_d_cur
                d = lsb_d.bit_length() - 1
                bits_d_cur ^= lsb_d

                # b in N(a) \ {u, v, a, d}
                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue

                # c in N(d) \ {u, v, a, d}
                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue

                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b

                    # c in N(b) & c_mask, c != b
                    if adj[b] & c_mask & ~(1 << b):
                        return True

        return False

    def is_c6_free(adj):
        """Full validation: check that no C6 exists in the graph."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_c6(adj, u, v):
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
                        return False
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
        return True

    def get_missing_edges(adj):
        missing = []
        for u in range(n):
            for v in range(u + 1, n):
                if not (adj[u] & (1 << v)):
                    missing.append((u, v))
        return missing

    def get_existing_edges(adj):
        existing = []
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    existing.append((u, v))
        return existing

    # Greedy construction
    def greedy_construct(rng):
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)

        for u, v in all_edges:
            if not creates_c6(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng):
        """
        Greedily add valid edges until no more can be added.
        Uses a prioritized order: edges with lower 'risk' first.
        Risk is approximated by the number of 2-paths that could extend to C6s.
        """
        changed = True
        while changed:
            changed = False
            missing = get_missing_edges(adj)
            if not missing:
                break
            
            # Score missing edges: prefer those that are 'easier' to add
            # A simple heuristic: shuffle is okay, but let's try sorting by degree sum
            # Actually, just shuffling and iterating is robust.
            # To improve, we can do multiple passes or sort by some metric.
            # Let's stick to shuffle for speed, but iterate until stable.
            rng.shuffle(missing)
            
            # Optimization: Iterate in batches
            for u, v in missing:
                if not creates_c6(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    changed = True
        return adj

    def block_swap(adj, rng, k_remove, k_add_target):
        """
        Try to remove k_remove edges and add at least k_add_target edges.
        Returns (new_adj, improved)
        """
        existing = get_existing_edges(adj)
        if len(existing) < k_remove:
            return adj, False
        
        # Select k_remove edges to remove
        # Heuristic: Remove edges that are in many C6s? 
        # For now, random selection is a good baseline for ILS.
        removal_candidates = rng.sample(existing, k_remove)
        
        # Create a copy
        trial_adj = adj[:]
        
        # Remove edges
        for u, v in removal_candidates:
            trial_adj[u] &= ~(1 << v)
            trial_adj[v] &= ~(1 << u)
            
        # Local search to fill up
        trial_adj = local_search(trial_adj, rng)
        
        m_orig = count_edges(adj)
        m_new = count_edges(trial_adj)
        
        if m_new > m_orig:
            return trial_adj, True
        else:
            return adj, False

    rng = random.Random(42)

    best_adj = None
    best_m = 0

    # Phase 1: Random Restarts + Local Search
    initial_time = time.time()
    while time.time() < initial_time + 10:
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_m

    # Phase 2: Iterated Local Search with Block Swaps
    time_start = time.time()
    no_improve_count = 0

    while time.time() < deadline:
        # Perturb current best
        perturbed = current_adj[:]
        
        # Choose a random block swap size
        # k_remove: 2 to 6
        # k_add_target: k_remove + 1 (net gain of 1)
        k_remove = rng.randint(2, 6)
        
        # Try block swap
        perturbed, improved = block_swap(perturbed, rng, k_remove, k_remove + 1)
        
        if not improved:
             # If block swap didn't improve, try a simpler random perturbation + local search
             # to ensure we are exploring
             num_removals = rng.randint(1, 4)
             existing = get_existing_edges(perturbed)
             if existing:
                 rng.shuffle(existing)
                 for i in range(min(num_removals, len(existing))):
                     u, v = existing[i]
                     perturbed[u] &= ~(1 << v)
                     perturbed[v] &= ~(1 << u)
             perturbed = local_search(perturbed, rng)

        m = count_edges(perturbed)

        if m > best_m:
            best_m = m
            best_adj = perturbed[:]
            current_adj = perturbed[:]
            current_m = m
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Accept worse solutions to escape local optima
            # Probability decays with no_improve_count
            if no_improve_count % 5 == 0 and m > current_m - 4:
                current_adj = perturbed[:]
                current_m = m

    # Final validation
    if not is_c6_free(best_adj):
        adj = greedy_construct(rng)
        if is_c6_free(adj):
            best_adj = adj

    return adj_to_edges(best_adj)
