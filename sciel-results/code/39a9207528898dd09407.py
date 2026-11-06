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
        """Check for C5 using bitset operations. Returns True if C5 exists."""
        for a in range(n):
            na = adj[a]
            b_mask = na
            while b_mask:
                b_bit = b_mask & (-b_mask)
                b = b_bit.bit_length() - 1
                b_mask ^= b_bit
                
                nb = adj[b]
                c_mask = nb & ~(1 << a)
                while c_mask:
                    c_bit = c_mask & (-c_mask)
                    c = c_bit.bit_length() - 1
                    c_mask ^= c_bit
                    
                    nc = adj[c]
                    d_mask = nc & ~(1 << b)
                    while d_mask:
                        d_bit = d_mask & (-d_mask)
                        d = d_bit.bit_length() - 1
                        d_mask ^= d_bit
                        
                        nd = adj[d]
                        e_mask = nd & na & ~(1 << a) & ~(1 << b) & ~(1 << c) & ~(1 << d)
                        if e_mask:
                            return True
        return False

    def has_independent_set_9(adj, n):
        """Check for independent set of size 9 using branch and bound with bitsets."""
        def can_find(available, count):
            if count == 0:
                return True
            if available == 0:
                return False
            if bin(available).count('1') < count:
                return False
            
            # Pick vertex with minimum degree in available
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
        for attempt in range(1000):
            if time.time() > deadline:
                break
                
            adj = [0] * target_n
            
            # Start with 8 K4s
            for b in range(8):
                for i in range(4):
                    for j in range(i + 1, 4):
                        u = 4 * b + i
                        v = 4 * b + j
                        adj[u] |= (1 << v)
                        adj[v] |= (1 << u)
            
            # Connect extra vertices (32 to target_n-1)
            for ex in range(32, target_n):
                # Connect to a random subset of vertices in each K4
                for b in range(8):
                    # Connect to 1-3 random vertices in this block
                    num_conn = random.randint(1, 3)
                    chosen = random.sample(range(4), num_conn)
                    for t in chosen:
                        v = 4 * b + t
                        adj[ex] |= (1 << v)
                        adj[v] |= (1 << ex)
            
            # Connect extras to each other (make them a clique or near-clique)
            for i in range(32, target_n):
                for j in range(i + 1, target_n):
                    if random.random() < 0.7:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Check C5
            if has_c5_fast(adj, target_n):
                # Try to remove edges to fix C5
                adj_copy = adj[:]
                fixed = False
                for _ in range(30):
                    if not has_c5_fast(adj_copy, target_n):
                        fixed = True
                        break
                    # Find and remove a random edge
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
                if fixed:
                    adj = adj_copy
                else:
                    continue
            
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
