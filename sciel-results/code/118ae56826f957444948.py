import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def popcount(x):
        return bin(x).count('1')

    def count_edges_in_mask(adj, mask):
        """Count edges among vertices in mask."""
        total = 0
        k = mask
        while k:
            lsb = k & -k
            idx = lsb.bit_length() - 1
            k -= lsb
            total += popcount(adj[idx] & mask)
        return total // 2

    def find_k4_fast(adj, n):
        """Find a K4 using bitmask operations. Returns tuple or None."""
        for i in range(n):
            for j in range(i + 1, n):
                if (adj[i] >> j) & 1:
                    common = adj[i] & adj[j]
                    common &= ~(1 << i)
                    common &= ~(1 << j)
                    # Check if there's an edge among common neighbors
                    k = common
                    while k:
                        lsb = k & -k
                        w = lsb.bit_length() - 1
                        k -= lsb
                        rest = common & ~(1 << w)
                        if adj[w] & rest:
                            # Find the z
                            z_mask = adj[w] & rest
                            lsb2 = z_mask & -z_mask
                            z = lsb2.bit_length() - 1
                            return (i, j, w, z)
        return None

    def find_i6_fast(adj, n):
        """Find an independent set of size 6 using recursive backtracking with pruning."""
        # Build non-neighbor masks for faster checking
        non_adj = [((1 << n) - 1) ^ adj[i] ^ (1 << i) for i in range(n)]
        
        def backtrack(current, start):
            if len(current) == 6:
                return current[:]
            remaining = 6 - len(current)
            # Pruning: not enough vertices left
            if n - start < remaining:
                return None
            for i in range(start, n):
                if not (1 << i) & non_adj[current[0]] if current else True:
                    pass
                # Check if i is non-adjacent to all in current
                ok = True
                for v in current:
                    if (adj[i] >> v) & 1:
                        ok = False
                        break
                if ok:
                    result = backtrack(current + [i], i + 1)
                    if result:
                        return result
            return None
        
        return backtrack([], 0)

    def find_i6_iterative(adj, n):
        """Find I6 using iterative method with bitmask."""
        import itertools
        for combo in itertools.combinations(range(n), 6):
            is_indep = True
            for i in range(6):
                for j in range(i + 1, 6):
                    u, v = combo[i], combo[j]
                    if (adj[u] >> v) & 1:
                        is_indep = False
                        break
                if not is_indep:
                    break
            if is_indep:
                return list(combo)
        return None

    def is_valid(adj, n):
        """Check if graph has no K4 and no I6."""
        if find_k4_fast(adj, n) is not None:
            return False
        if find_i6_fast(adj, n) is not None:
            return False
        return True

    def local_search(adj, n, max_steps=5000, deadline=deadline):
        """Run local search to eliminate violations."""
        for step in range(max_steps):
            if time.time() > deadline:
                break
            
            # Find a violation
            k4 = find_k4_fast(adj, n)
            if k4:
                # Flip a random edge in the K4
                u, v = random.sample(k4, 2)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            i6 = find_i6_fast(adj, n)
            if i6:
                # Flip a random edge in the I6
                u, v = random.sample(i6, 2)
                if (adj[u] >> v) & 1:
                    adj[u] &= ~(1 << v)
                    adj[v] &= ~(1 << u)
                else:
                    adj[u] |= (1 << v)
                    adj[v] |= (1 << u)
                continue
            
            # No violations found
            return True
        
        return False

    # Start with Paley graph of order 17 as baseline
    def make_paley(p):
        """Create Paley graph of prime order p."""
        adj = [0] * p
        quadratic_residues = set()
        for x in range(1, p):
            quadratic_residues.add((x * x) % p)
        for i in range(p):
            for j in range(i + 1, p):
                if (j - i) % p in quadratic_residues or (i - j) % p in quadratic_residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        return adj

    # Best known valid graph
    best_n = 17
    best_adj = make_paley(17)

    # Try to find valid graphs for increasing n
    for n in range(18, 36):
        if time.time() > deadline - 10:
            break
        
        found = False
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random graph with density ~0.5
            adj = [0] * n
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < 0.5:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            if local_search(adj, n, max_steps=3000):
                if is_valid(adj, n):
                    best_n = n
                    best_adj = adj[:]
                    found = True
                    break
        
        if not found and time.time() > deadline - 15:
            break

    # Build edge list
    edges = []
    for i in range(best_n):
        for j in range(i + 1, best_n):
            if (best_adj[i] >> j) & 1:
                edges.append([i, j])

    return {'n': best_n, 'edges': edges}
