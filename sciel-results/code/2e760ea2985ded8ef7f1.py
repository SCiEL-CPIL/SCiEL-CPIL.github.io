"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
from math import gcd

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin before 90s kill
    
    N = 512
    
    # Precompute adjacency
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            if i * j // g <= N:
                mask |= (1 << (j - 1))
        adj[i] = mask
    
    degree = [bin(adj[i]).count('1') for i in range(N + 1)]
    
    # Multiple greedy strategies
    def greedy_clique(order):
        current = []
        current_mask = 0
        for v in order:
            if (adj[v] & current_mask) == current_mask:
                current.append(v)
                current_mask |= (1 << (v - 1))
        return current, current_mask
    
    # Strategy 1: degree descending
    order1 = list(range(1, N + 1))
    order1.sort(key=lambda x: degree[x], reverse=True)
    c1, m1 = greedy_clique(order1)
    
    # Strategy 2: natural order (small numbers are more compatible)
    order2 = list(range(1, N + 1))
    c2, m2 = greedy_clique(order2)
    
    # Strategy 3: by number of distinct prime factors
    def omega(n):
        c = 0
        d = 2
        while d * d <= n:
            if n % d == 0:
                c += 1
                while n % d == 0:
                    n //= d
            d += 1
        if n > 1:
            c += 1
        return c
    
    order3 = list(range(1, N + 1))
    order3.sort(key=lambda x: (omega(x), x))
    c3, m3 = greedy_clique(order3)
    
    # Strategy 4: by value (small first)
    order4 = list(range(1, N + 1))
    c4, m4 = greedy_clique(order4)
    
    candidates = [(len(c1), c1, m1), (len(c2), c2, m2), (len(c3), c3, m3), (len(c4), c4, m4)]
    candidates.sort(key=lambda x: -x[0])
    best_size, best_clique, best_mask = candidates[0]
    
    # Local search: add vertices greedily
    def local_add(clique, mask):
        nonlocal best_clique, best_mask, best_size
        improved = True
        while improved:
            if time.time() > deadline - 3:
                break
            improved = False
            for v in range(1, N + 1):
                if time.time() > deadline - 3:
                    break
                if (mask >> (v - 1)) & 1:
                    continue
                if (adj[v] & mask) == mask:
                    clique.append(v)
                    mask |= (1 << (v - 1))
                    improved = True
                    if len(clique) > best_size:
                        best_size = len(clique)
                        best_clique = list(clique)
                        best_mask = mask
        return clique, mask
    
    best_clique, best_mask = local_add(list(best_clique), best_mask)
    
    # Bron-Kerbosch with pivot and bitsets (time-limited)
    def bron_kerbosch(R_mask, P_mask, X_mask):
        nonlocal best_clique, best_mask, best_size
        
        if time.time() > deadline - 5:
            return
        
        if P_mask == 0 and X_mask == 0:
            size = bin(R_mask).count('1')
            if size > best_size:
                best_size = size
                best_mask = R_mask
                best_clique = [i for i in range(1, N + 1) if (R_mask >> (i - 1)) & 1]
            return
        
        union_mask = P_mask | X_mask
        best_u = 0
        best_count = -1
        temp = union_mask
        while temp:
            low = temp & (-temp)
            u = low.bit_length()
            count = bin(P_mask & adj[u]).count('1')
            if count > best_count:
                best_count = count
                best_u = u
            temp ^= low
        
        if best_u == 0:
            return
        
        candidates = P_mask & ~adj[best_u]
        
        while candidates:
            low = candidates & (-candidates)
            v = low.bit_length()
            candidates ^= low
            P_mask_next = P_mask & adj[v] & ~low
            bron_kerbosch(R_mask | low, P_mask_next, X_mask & adj[v])
            P_mask = P_mask & ~low
            X_mask |= low
    
    if time.time() < deadline - 15:
        all_mask = (1 << N) - 1
        bron_kerbosch(0, all_mask, 0)
    
    # Final local add
    best_clique, best_mask = local_add(list(best_clique), best_mask)
    
    result = [i for i in range(1, N + 1) if (best_mask >> (i - 1)) & 1]
    if not result:
        result = [1]
    
    return result
# EVOLVE-BLOCK-END
