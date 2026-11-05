import random
import time

def construct():
    deadline = time.time() + 115  # Safety margin
    n = 40

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

    def creates_c6_fast(adj, u, v):
        """
        Fast C6 check: is there a path u-a-b-c-d-v of length 5?
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

                b_mask = adj[a] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if b_mask == 0:
                    continue

                c_mask = adj[d] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << d)
                if c_mask == 0:
                    continue

                bits_b = b_mask
                while bits_b:
                    lsb_b = bits_b & -bits_b
                    b = lsb_b.bit_length() - 1
                    bits_b ^= lsb_b

                    if adj[b] & c_mask & ~(1 << b):
                        return True

        return False

    def is_c6_free(adj):
        """Full validation."""
        for u in range(n):
            for v in range(u + 1, n):
                if adj[u] & (1 << v):
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                    if creates_c6_fast(adj, u, v):
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

    def greedy_construct(rng):
        adj = [0] * n
        all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
        rng.shuffle(all_edges)

        for u, v in all_edges:
            if not creates_c6_fast(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
        return adj

    def local_search(adj, rng):
        """
        Greedily add valid edges until no more can be added.
        """
        improved = True
        while improved:
            improved = False
            missing = get_missing_edges(adj)
            if not missing:
                break
            rng.shuffle(missing)
            for u, v in missing:
                if not creates_c6_fast(adj, u, v):
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                    improved = True
        return adj

    def try_2opt_swap(adj, rng):
        """
        Attempt a 2-opt swap: remove (u,v) and (x,y), add (u,x) and (v,y) or (u,y) and (v,x).
        Returns True if a swap was successfully applied.
        """
        existing = get_existing_edges(adj)
        if len(existing) < 4:
            return False

        # Pick two distinct edges
        i, j = rng.sample(range(len(existing)), 2)
        u, v = existing[i]
        x, y = existing[j]

        # Ensure vertices are distinct enough
        # If u==x or u==y etc, the swap might be trivial or invalid
        # We just try the permutations

        # Permutation 1: (u,x) and (v,y)
        # Remove (u,v) and (x,y)
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] &= ~(1 << y)
        adj[y] &= ~(1 << x)

        success = False
        if u != x and v != y:
            if not creates_c6_fast(adj, u, x):
                adj[u] |= (1 << x)
                adj[x] |= (1 << u)
                if not creates_c6_fast(adj, v, y):
                    adj[v] |= (1 << y)
                    adj[y] |= (1 << v)
                    success = True
                else:
                    # Revert (u,x)
                    adj[u] &= ~(1 << x)
                    adj[x] &= ~(1 << u)
            # If u==x or v==y or C6 found, we need to revert the removals if we didn't succeed
            if not success:
                # Restore original edges
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                adj[x] |= (1 << y)
                adj[y] |= (1 << x)
        else:
            # Invalid swap structure, restore
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            adj[x] |= (1 << y)
            adj[y] |= (1 << x)

        if success:
            return True

        # Permutation 2: (u,y) and (v,x)
        # Remove (u,v) and (x,y) again (they are currently removed if success was False and we restored? 
        # Wait, if success was False in Perm 1, we restored. So edges are present.
        # We need to remove them again to try Perm 2.
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)
        adj[x] &= ~(1 << y)
        adj[y] &= ~(1 << x)

        if u != y and v != x:
            if not creates_c6_fast(adj, u, y):
                adj[u] |= (1 << y)
                adj[y] |= (1 << u)
                if not creates_c6_fast(adj, v, x):
                    adj[v] |= (1 << x)
                    adj[x] |= (1 << v)
                    return True
                else:
                    # Revert (u,y)
                    adj[u] &= ~(1 << y)
                    adj[y] &= ~(1 << u)
        
        # Restore original edges
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        adj[x] |= (1 << y)
        adj[y] |= (1 << x)
        return False

    rng = random.Random(42)

    best_adj = None
    best_m = 0

    # Phase 1: Multiple random greedy constructions
    phase1_deadline = time.time() + 10
    while time.time() < phase1_deadline:
        adj = greedy_construct(rng)
        adj = local_search(adj, rng)
        m = count_edges(adj)
        if m > best_m:
            best_m = m
            best_adj = adj[:]

    current_adj = best_adj[:]
    current_m = best_m

    # Phase 2: Iterated Local Search with 2-Opt and Edge Removals
    while time.time() < deadline - 3:
        perturbed = current_adj[:]
        pert_type = rng.random()

        if pert_type < 0.5:
            # 2-Opt Swaps
            # Try a few swaps
            for _ in range(rng.randint(1, 3)):
                try_2opt_swap(perturbed, rng)
        else:
            # Edge Removals
            num_removals = rng.randint(2, 6)
            existing = get_existing_edges(perturbed)
            if existing:
                rng.shuffle(existing)
                for i in range(min(num_removals, len(existing))):
                    u, v = existing[i]
                    perturbed[u] &= ~(1 << v)
                    perturbed[v] &= ~(1 << u)

        # Local search on perturbed
        perturbed = local_search(perturbed, rng)
        m = count_edges(perturbed)

        # Acceptance
        if m >= best_m:
            if m > best_m:
                best_m = m
                best_adj = perturbed[:]
            current_adj = perturbed[:]
            current_m = m
        else:
            gap = best_m - m
            if gap <= 1:
                accept_prob = 0.3
            elif gap <= 2:
                accept_prob = 0.1
            else:
                accept_prob = 0.02

            if rng.random() < accept_prob:
                current_adj = perturbed[:]
                current_m = m

    # Final validation
    if not is_c6_free(best_adj):
        adj = greedy_construct(rng)
        if is_c6_free(adj):
            best_adj = adj
        else:
            return []

    return adj_to_edges(best_adj)
