import time
import math

def construct():
    start_time = time.perf_counter()
    LIMIT = 512
    N = LIMIT
    deadline = start_time + 85.0

    # Precompute adjacency bitmasks
    adj = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = math.gcd(i, j)
            l = (i // g) * j
            if l <= LIMIT:
                adj[i] |= (1 << (j - 1))
                adj[j] |= (1 << (i - 1))

    best_clique_size = 0
    best_clique_mask = 0

    def bron_kerbosch_opt(R, P, X):
        nonlocal best_clique_size, best_clique_mask, deadline
        
        if time.perf_counter() > deadline:
            return False
        
        # Pruning
        r_size = R.bit_count()
        p_size = P.bit_count()
        if r_size + p_size <= best_clique_size:
            return True
            
        if P == 0:
            if r_size > best_clique_size:
                best_clique_size = r_size
                best_clique_mask = R
            return True
        
        PX = P | X
        if PX == 0:
            return True

        # Select pivot u from P U X to minimize candidates
        # Heuristic: max degree in P
        max_pop = -1
        pivot = -1
        temp = PX
        while temp:
            lsb = temp & -temp
            u_idx = (lsb.bit_length() - 1)
            u = u_idx + 1
            pop = (adj[u] & P).bit_count()
            if pop > max_pop:
                max_pop = pop
                pivot = u
            temp ^= lsb
        
        if pivot == -1:
            return True

        candidates = P & ~adj[pivot]
        
        while candidates:
            if time.perf_counter() > deadline:
                return False
            
            lsb = candidates & -candidates
            v_idx = (lsb.bit_length() - 1)
            v = v_idx + 1
            
            new_R = R | lsb
            new_P = P & ~adj[v]
            new_X = X & ~adj[v]
            
            if not bron_kerbosch_opt(new_R, new_P, new_X):
                return False
            
            P = P & ~lsb
            X = X | lsb
            candidates = candidates & ~lsb
            
        return True

    # Greedy Seed
    def greedy_seed():
        # Order by degree descending
        degrees = [(adj[i].bit_count(), i) for i in range(1, N + 1)]
        degrees.sort(reverse=True)
        
        R_mask = 0
        P_mask = (1 << N) - 1
        
        while P_mask:
            best_v = -1
            best_deg = -1
            temp = P_mask
            while temp:
                lsb = temp & -temp
                v_idx = (lsb.bit_length() - 1)
                v = v_idx + 1
                d = (adj[v] & P_mask).bit_count()
                if d > best_deg:
                    best_deg = d
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            R_mask |= (1 << (best_v - 1))
            P_mask = P_mask & adj[best_v]
            P_mask &= ~(1 << (best_v - 1))
            
        return R_mask

    seed_mask = greedy_seed()
    best_clique_size = seed_mask.bit_count()
    best_clique_mask = seed_mask
    
    success = bron_kerbosch_opt(0, (1 << N) - 1, 0)
    
    result = [i + 1 for i in range(N) if (best_clique_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result
