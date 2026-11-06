"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit

    # Known valid graph: 8 disjoint K4's -> n=32, alpha=8 < 9, no C5
    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }

    def build_adj(n, edges):
        """Build adjacency bitmasks from edge list."""
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_fast(adj, n):
        """Fast C5 detection: for each edge (u,v), check if N(u)∩N(v) has
        a path of length 2 through a vertex not in {u,v}.
        C5 = u-a-b-c-v where a∈N(u), b∈N(a)∩N(c), c∈N(v), all distinct.
        Key insight: a C5 exists iff there's an edge (u,v) such that
        there exist a∈N(u)\\{v}, c∈N(v)\\{u}, a≠c, and N(a)∩N(c)∩(V\\{u,v,a,c})≠∅.
        Even faster: check if there's a common neighbor pair.
        """
        for u in range(n):
            nu = adj[u]
            v_mask = nu
            while v_mask:
                v_bit = v_mask & (-v_mask)
                v = v_bit.bit_length() - 1
                v_mask ^= v_bit
                nv = adj[v]
                # Common neighbors of u and v, excluding u and v themselves
                common = nu & nv & ~(1 << u) & ~(1 << v)
                if common == 0:
                    continue
                # For each pair (a,c) in common, check if they share a neighbor
                # But that's O(k^2) per edge. Instead:
                # C5: u-a-b-c-v. a∈N(u)\\{v}, c∈N(v)\\{u}, a≠c, b∈N(a)∩N(c)\\{u,v,a,c}
                # Equivalent: ∃a∈N(u)\\{v}, ∃c∈N(v)\\{u}, a≠c, N(a)∩N(c) has element not in {u,v,a,c}
                # Even simpler check: if |common| >= 2, pick two common neighbors a,c.
                # Then N(a)∩N(c) includes u and v. We need another common neighbor of a and c.
                # So: ∃a,c ∈ common, a≠c, such that (N(a)∩N(c)) \\ {u,v,a,c} ≠ ∅
                # But N(a)∩N(c) always contains u and v (since a,c both adjacent to both).
                # We need a THIRD common neighbor or a neighbor of a and c outside {u,v}.
                # Let's just do the direct check but optimized:
                a_mask = nu & ~(1 << v)
                while a_mask:
                    a_bit = a_mask & (-a_mask)
                    a = a_bit.bit_length() - 1
                    a_mask ^= a_bit
                    # c must be in N(v)\\{u,a} and N(a)∩N(c) must have element outside {u,v,a,c}
                    c_mask = nv & ~(1 << u) & ~(1 << a)
                    while c_mask:
                        c_bit = c_mask & (-c_mask)
                        c = c_bit.bit_length() - 1
                        c_mask ^= c_bit
                        # Check if N(a) and N(c) share a vertex outside {u,v,a,c}
                        shared = adj[a] & adj[c] & ~(1 << u) & ~(1 << v) & ~(1 << a) & ~(1 << c)
                        if shared:
                            return True
        return False

    def has_c5(adj, n):
        """Check if graph contains a 5-cycle. Returns True if found."""
        # Use the fast method
        return has_c5_fast(adj, n)

    def has_independent_set_9(adj, n):
        """Check if graph has an independent set of size 9 using bitset backtracking."""

        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False

            # Pick vertex with minimum degree in available (excludes fewest)
            best_v = -1
            best_excluded = 0
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = bin(adj_in_avail).count('1')
                if best_v == -1 or excluded < best_excluded:
                    best_excluded = excluded
                    best_v = v
                if excluded == 0:
                    break

            if best_v == -1:
                return False

            v = best_v
            v_bit = 1 << v

            # Option 1: include v
            new_avail = available & ~adj[v] & ~v_bit
            if can_find(new_avail, count - 1):
                return True

            # Option 2: exclude v
            if can_find(available & ~v_bit, count):
                return True

            return False

        return can_find((1 << n) - 1, 9)

    def is_valid(n, adj):
        """Check if graph is C5-free and has no independent set of 9."""
        if has_c5(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        """Convert adjacency bitmasks to edge list."""
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    best_result = known_32

    # Strategy 0: Deterministic algebraic construction
    # Try to build a C5-free graph on 33-35 vertices using structured patterns
    if time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break
            found = False
            # Pattern: 8 blocks of K4 + extra vertices connected to specific blocks
            # Key: extra vertices must connect to at most 2 vertices per block to avoid C5
            for block_pattern in range(20):
                if time.time() > deadline:
                    break
                adj = [0] * target_n
                # 8 blocks of K4
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                extra = target_n - 32
                # Connect extra vertices to specific blocks
                for ex_idx in range(extra):
                    ex = 32 + ex_idx
                    # Connect to 2-3 vertices in 1-2 blocks
                    # Use deterministic pattern based on block_pattern
                    b1 = (ex_idx * 3 + block_pattern) % 8
                    b2 = (ex_idx * 5 + block_pattern + 1) % 8
                    if b2 == b1:
                        b2 = (b2 + 1) % 8

                    # Connect to 3 vertices in block b1
                    for t in range(3):
                        v = 4 * b1 + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)

                    # Connect to 2 vertices in block b2
                    for t in range(2):
                        v = 4 * b2 + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)

                # Also connect extra vertices to each other if multiple
                if extra > 1:
                    for i in range(extra):
                        for j in range(i + 1, extra):
                            adj[32 + i] |= (1 << (32 + j))
                            adj[32 + j] |= (1 << (32 + i))

                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break
            if found:
                break

    # Strategy 1: Extend 8xK4 by adding vertices with careful connections
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break

            extra = target_n - 32
            num_attempts = 5000
            for attempt in range(num_attempts):
                if time.time() > deadline:
                    break

                # Initialize: 8xK4 structure
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                # Add edges from extra vertices to existing vertices
                for ex in range(32, target_n):
                    # Connect to vertices from 1-2 blocks to maintain C5-freeness
                    # Key constraint: to avoid C5, each extra vertex should connect
                    # to at most 2 vertices per block, and total connections limited
                    block = random.randint(0, 7)
                    base = 4 * block
                    # Connect to 2-3 vertices in this block
                    num_conn = random.randint(2, 3)
                    targets = random.sample(range(4), num_conn)
                    for t in targets:
                        v = base + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)

                    # Optionally connect to a second block (only 1-2 vertices)
                    if random.random() < 0.3:
                        block2 = random.randint(0, 7)
                        if block2 != block:
                            base2 = 4 * block2
                            num_conn2 = random.randint(1, 2)
                            targets2 = random.sample(range(4), num_conn2)
                            for t in targets2:
                                v = base2 + t
                                adj[ex] |= (1 << v)
                                adj[v] |= (1 << ex)

                # Connect extra vertices to each other
                if extra > 1:
                    for i in range(extra):
                        for j in range(i + 1, extra):
                            if random.random() < 0.5:
                                adj[32 + i] |= (1 << (32 + j))
                                adj[32 + j] |= (1 << (32 + i))

                # Check validity
                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                break

    # Strategy 2: Local search - start from valid n=32 graph and modify
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break

            found = False
            for attempt in range(5000):
                if time.time() > deadline:
                    break

                # Start with 8xK4 + extra vertices with limited connections
                adj = [0] * target_n
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)

                # Add random edges from extra vertices (conservative)
                for ex in range(32, target_n):
                    # Connect to 2-4 random vertices from at most 2 blocks
                    num_blocks = random.randint(1, 2)
                    chosen_blocks = random.sample(range(8), num_blocks)
                    for blk in chosen_blocks:
                        num_conn = random.randint(1, 3)
                        targets = random.sample(range(4), num_conn)
                        for t in targets:
                            v = 4 * blk + t
                            adj[ex] |= (1 << v)
                            adj[v] |= (1 << ex)

                # Check if valid directly
                if has_c5(adj, target_n):
                    # Try removing edges to fix C5
                    adj_copy = adj[:]
                    for _ in range(30):
                        if time.time() > deadline:
                            break
                        if not has_c5(adj_copy, target_n):
                            break
                        # Remove a random edge
                        all_edges = []
                        for i in range(target_n):
                            for j in range(i + 1, target_n):
                                if adj_copy[i] & (1 << j):
                                    all_edges.append((i, j))
                        if not all_edges:
                            break
                        u, v = random.choice(all_edges)
                        adj_copy[u] &= ~(1 << v)
                        adj_copy[v] &= ~(1 << u)

                    adj = adj_copy

                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break

            if found:
                break

    # Strategy 3: Pure random search with low density
    if best_result['n'] == 32 and time.time() < deadline:
        for target_n in [33, 34, 35]:
            if time.time() > deadline:
                break

            found = False
            for attempt in range(8000):
                if time.time() > deadline:
                    break

                # Random graph with low density
                p = random.uniform(0.06, 0.18)

                adj = [0] * target_n
                for i in range(target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < p:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)

                if has_c5(adj, target_n):
                    continue
                if has_independent_set_9(adj, target_n):
                    continue

                edges = adj_to_edges(adj, target_n)
                best_result = {'n': target_n, 'edges': edges}
                found = True
                break

            if found:
                break

    return best_result

# EVOLVE-BLOCK-END
