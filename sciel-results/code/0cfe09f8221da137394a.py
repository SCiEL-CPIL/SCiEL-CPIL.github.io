import time
import random
import sys

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    known_32 = {
        'n': 32,
        'edges': [[4 * b + i, 4 * b + j] for b in range(8) for i in range(4) for j in range(i + 1, 4)]
    }

    def build_adj(n, edges):
        adj = [0] * n
        for u, v in edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
        return adj

    def has_c5_fast(adj, n):
        # Precompute popcount? No, just bit ops.
        # Check for C5: a-b-c-d-e-a
        # For each pair (a, c) with dist 2, check if they share a neighbor b,
        # and if N(a)\{b} and N(c)\{b} have an edge d-e where d,e not in {a,b,c}.
        # Actually, the previous implementation was decent.
        # Let's use a slightly different approach:
        # Iterate over all pairs (u, v) that are NOT adjacent.
        # If they share 2+ common neighbors, and those neighbors are not adjacent? No.
        # Standard C5 detection is O(n^4) or O(n^3) with bitsets.
        
        for a in range(n):
            na = adj[a]
            # b is a neighbor of a
            b_mask = na
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                nb = adj[b]
                # c is a neighbor of b, c != a
                c_mask = nb & ~(1 << a)
                while c_mask:
                    c_bit = c_mask & (-c_mask)
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    nc = adj[c]
                    # d is a neighbor of c, d != b
                    d_mask = nc & ~(1 << b)
                    while d_mask:
                        d_bit = d_mask & (-d_mask)
                        d = d_bit.bit_length() - 1
                        d_mask ^= d_bit
                        
                        nd = adj[d]
                        # e is a neighbor of d, e != c
                        # e must be a neighbor of a
                        # e != a, b, c, d
                        e_mask = nd & na & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if e_mask:
                            return True
        return False

    def has_independent_set_9(adj, n):
        # Branch and bound
        # Use a greedy coloring bound? Or just simple backtracking with pruning.
        # For n=35, depth 9 is small.
        
        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False
            
            # Pick vertex with max degree in available?
            # Or min degree?
            # Min degree in available is good for independent set (excludes fewest).
            best_v = -1
            best_excluded = n + 1
            avail = available
            while avail:
                v_bit = avail & (-avail)
                v = v_bit.bit_length() - 1
                avail ^= v_bit
                adj_in_avail = available & adj[v]
                excluded = bin(adj_in_avail).count('1')
                if excluded < best_excluded:
                    best_excluded = excluded
                    best_v = v
                if excluded == 0:
                    break

            if best_v == -1:
                return False

            v = best_v
            v_bit = 1 << v

            # Include v
            new_avail = available & ~adj[v] & ~v_bit
            if can_find(new_avail, count - 1):
                return True

            # Exclude v
            if can_find(available & ~v_bit, count):
                return True

            return False

        return can_find((1 << n) - 1, 9)

    def is_valid(n, adj):
        if has_c5_fast(adj, n):
            return False
        if has_independent_set_9(adj, n):
            return False
        return True

    def adj_to_edges(adj, n):
        edges = []
        for i in range(n):
            mask = adj[i]
            for j in range(i + 1, n):
                if mask & (1 << j):
                    edges.append([i, j])
        return edges

    best_result = known_32

    # Try to find n=35, 34, 33
    for target_n in [35, 34, 33]:
        if time.time() > deadline:
            break
            
        found = False
        # Random restarts
        for attempt in range(500):
            if time.time() > deadline:
                break
                
            adj = [0] * target_n
            # Start with 8 K4s if possible, plus random connections for extras
            if target_n >= 32:
                for b in range(8):
                    for i in range(4):
                        for j in range(i + 1, 4):
                            u = 4 * b + i
                            v = 4 * b + j
                            adj[u] |= (1 << v)
                            adj[v] |= (1 << u)
                
                # Connect extra vertices
                for ex in range(32, target_n):
                    for b in range(8):
                        # Connect to 1-2 random vertices in each block
                        num_conn = random.randint(1, 2)
                        for _ in range(num_conn):
                            t = random.randint(0, 3)
                            v = 4 * b + t
                            if random.random() < 0.8: # Probability to connect
                                adj[ex] |= (1 << v)
                                adj[v] |= (1 << ex)
                
                # Random edges between extras
                for i in range(32, target_n):
                    for j in range(i + 1, target_n):
                        if random.random() < 0.5:
                            adj[i] |= (1 << j)
                            adj[j] |= (1 << i)
            
            # Check validity
            if has_c5_fast(adj, target_n):
                # Try to fix by removing edges
                adj_copy = adj[:]
                for _ in range(20):
                    if not has_c5_fast(adj_copy, target_n):
                        break
                    # Remove random edge
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
                
            if has_c5_fast(adj, target_n):
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
