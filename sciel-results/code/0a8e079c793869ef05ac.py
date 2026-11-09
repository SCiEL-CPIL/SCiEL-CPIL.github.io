"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def _has_k4_fast(adj, n):
    """Fast K4 check using bitmasks."""
    # adj[i] is a bitmask of neighbors of i
    for a in range(n):
        for b in range(a+1, n):
            if not (adj[a] >> b) & 1:
                continue
            # Common neighbors of a and b
            common_ab = adj[a] & adj[b]
            # Remove a and b
            common_ab &= ~(1 << a)
            common_ab &= ~(1 << b)
            # For each pair of common neighbors, check if they're adjacent
            # Actually, we need c, d both in common_ab and adj[c] & (1<<d)
            c_bits = common_ab
            while c_bits:
                c = c_bits & (-c_bits)
                c_bits ^= c
                c_idx = c.bit_length() - 1
                # d must be in common_ab, greater than c_idx, and adj[c] has d
                d_bits = common_ab & (adj[c] & ~(1 << c_idx))
                d_bits &= ~((1 << (c_idx + 1)) - 1)  # d > c
                if d_bits:
                    return True
    return False

def _has_indep6_fast(adj, n):
    """Check for independent 6-set using complement graph bitmasks."""
    # non_adj[i] = bitmask of non-neighbors of i (excluding i itself)
    full = (1 << n) - 1
    non_adj = [full ^ adj[i] ^ (1 << i) for i in range(n)]
    
    for a in range(n):
        for b in range(a+1, n):
            if (adj[a] >> b) & 1:
                continue
            # Common non-neighbors of a and b
            common = non_adj[a] & non_adj[b]
            common &= ~(1 << a)
            common &= ~(1 << b)
            for c in range(b+1, n):
                if not (common >> c) & 1:
                    continue
                common2 = common & non_adj[c]
                common2 &= ~(1 << c)
                for d in range(c+1, n):
                    if not (common2 >> d) & 1:
                        continue
                    common3 = common2 & non_adj[d]
                    common3 &= ~(1 << d)
                    for e in range(d+1, n):
                        if not (common3 >> e) & 1:
                            continue
                        common4 = common3 & non_adj[e]
                        common4 &= ~(1 << e)
                        for f in range(e+1, n):
                            if (common4 >> f) & 1:
                                return True
    return False

def construct():
    start_time = time.time()
    deadline = start_time + 135.0
    
    # Fallback: Paley graph on 17 vertices
    squares = {x * x % 17 for x in range(1, 17)}
    best_result = {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
    
    # Try random + local search for larger n
    for n in range(18, 36):
        if time.time() > deadline - 10:
            break
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            seed = 42 + attempt * 1000 + n
            random.seed(seed)
            
            # Choose edge probability based on n
            # For K4-free with alpha <= 5, need moderate density
            p = 0.35 + 0.05 * min(attempt, 2)
            
            # Build random graph
            adj = [0] * n  # bitmask adjacency
            for i in range(n):
                for j in range(i+1, n):
                    if random.random() < p:
                        adj[i] |= (1 << j)
                        adj[j] |= (1 << i)
            
            # Local search: remove K4s, add edges to break indep 6-sets
            for _ in range(50):
                if time.time() > deadline - 3:
                    break
                # Check and fix K4s
                if _has_k4_fast(adj, n):
                    # Find a K4 and remove an edge
                    found = False
                    for a in range(n):
                        if found:
                            break
                        for b in range(a+1, n):
                            if found:
                                break
                            if not (adj[a] >> b) & 1:
                                continue
                            common_ab = adj[a] & adj[b] & ~((1 << a) | (1 << b))
                            c_bits = common_ab
                            while c_bits and not found:
                                c = c_bits & (-c_bits)
                                c_bits ^= c
                                c_idx = c.bit_length() - 1
                                d_bits = common_ab & adj[c] & ~((1 << c_idx) | (1 << a) | (1 << b))
                                d_bits &= ~((1 << (c_idx + 1)) - 1)
                                if d_bits:
                                    d = d_bits & (-d_bits)
                                    d_idx = d.bit_length() - 1
                                    # Remove random edge from K4 {a,b,c,d}
                                    edges = [(a,b),(a,c),(a,d),(b,c),(b,d),(c,d)]
                                    u,v = random.choice(edges)
                                    adj[u] &= ~(1 << v)
                                    adj[v] &= ~(1 << u)
                                    found = True
                                    break
                    continue
                
                # Check and fix independent 6-sets
                if _has_indep6_fast(adj, n):
                    # Try to add an edge
                    full = (1 << n) - 1
                    non_adj = [full ^ adj[i] ^ (1 << i) for i in range(n)]
                    added = False
                    # Find an independent 6-set and try to add an edge in it
                    for a in range(n):
                        if added or time.time() > deadline - 2:
                            break
                        for b in range(a+1, n):
                            if added:
                                break
                            if (adj[a] >> b) & 1:
                                continue
                            common = non_adj[a] & non_adj[b] & ~((1 << a) | (1 << b))
                            c_bits = common
                            while c_bits and not added:
                                c = c_bits & (-c_bits)
                                c_bits ^= c
                                c_idx = c.bit_length() - 1
                                common2 = common & non_adj[c] & ~((1 << a) | (1 << b) | (1 << c_idx))
                                d_bits = common2
                                while d_bits and not added:
                                    d = d_bits & (-d_bits)
                                    d_bits ^= d
                                    d_idx = d.bit_length() - 1
                                    common3 = common2 & non_adj[d] & ~((1 << a) | (1 << b) | (1 << c_idx) | (1 << d_idx))
                                    e_bits = common3
                                    while e_bits and not added:
                                        e = e_bits & (-e_bits)
                                        e_bits ^= e
                                        e_idx = e.bit_length() - 1
                                        common4 = common3 & non_adj[e] & ~((1 << a) | (1 << b) | (1 << c_idx) | (1 << d_idx) | (1 << e_idx))
                                        f_bits = common4
                                        while f_bits and not added:
                                            f = f_bits & (-f_bits)
                                            f_bits ^= f
                                            f_idx = f.bit_length() - 1
                                            # Try adding edge between two of {a,b,c,d,e,f}
                                            candidates = [(a,b),(a,c),(a,d),(a,e),(a,f),(b,c),(b,d),(b,e),(b,f),(c,d),(c,e),(c,f),(d,e),(d,f),(e,f)]
                                            random.shuffle(candidates)
                                            for u,v in candidates:
                                                if (adj[u] >> v) & 1:
                                                    continue
                                                # Check if adding (u,v) creates K4
                                                adj[u] |= (1 << v)
                                                adj[v] |= (1 << u)
                                                if not _has_k4_fast(adj, n):
                                                    added = True
                                                    break
                                                else:
                                                    adj[u] &= ~(1 << v)
                                                    adj[v] &= ~(1 << u)
                                            break
                                    break
                                break
                            break
                        break
                    if not added:
                        break
                else:
                    # No K4 and no indep 6-set!
                    edges = [[i, j] for i in range(n) for j in range(i+1, n) if (adj[i] >> j) & 1]
                    result = {'n': n, 'edges': edges}
                    if n > best_result['n']:
                        best_result = result
                    if n >= 35:
                        return best_result
                    break
            
            if n > best_result['n'] and n >= 35:
                return best_result
    
    return best_result
# EVOLVE-BLOCK-END
