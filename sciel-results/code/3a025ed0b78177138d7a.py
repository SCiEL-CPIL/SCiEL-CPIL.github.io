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

    # Improved Greedy Seed with multiple strategies
    def greedy_seed_min_conflict():
        """Greedy: pick element that minimizes conflicts with remaining candidates"""
        R_mask = 0
        P_mask = (1 << N) - 1
        
        while P_mask:
            best_v = -1
            best_score = -1
            temp = P_mask
            while temp:
                lsb = temp & -temp
                v_idx = (lsb.bit_length() - 1)
                v = v_idx + 1
                # Score: prefer high degree within P (more connections)
                d = (adj[v] & P_mask).bit_count()
                if d > best_score:
                    best_score = d
                    best_v = v
                temp ^= lsb
            
            if best_v == -1:
                break
            
            R_mask |= (1 << (best_v - 1))
            P_mask = P_mask & adj[best_v]
            P_mask &= ~(1 << (best_v - 1))
            
        return R_mask

    def greedy_seed_small_first():
        """Greedy: prefer smaller numbers (they tend to have lower LCMs)"""
        R_mask = 0
        P_mask = (1 << N) - 1
        
        # Process in order of increasing value
        for v in range(1, N + 1):
            bit = 1 << (v - 1)
            if P_mask & bit:
                # Check if v is compatible with all already selected
                compatible = True
                r_temp = R_mask
                while r_temp:
                    lsb = r_temp & -r_temp
                    u_idx = (lsb.bit_length() - 1)
                    u = u_idx + 1
                    if not (adj[v] & (1 << (u - 1))):
                        compatible = False
                        break
                    r_temp ^= lsb
                
                if compatible:
                    R_mask |= bit
                    P_mask &= adj[v]
                    P_mask &= ~bit
        
        return R_mask

    def local_search_improve(mask):
        """Try to add elements or swap to improve the clique"""
        improved = True
        current_mask = mask
        current_size = current_mask.bit_count()
        
        while improved:
            improved = False
            # Try adding each element not in current
            for v in range(1, N + 1):
                bit = 1 << (v - 1)
                if current_mask & bit:
                    continue
                # Check if v is compatible with all in current
                compatible = True
                c_temp = current_mask
                while c_temp:
                    lsb = c_temp & -c_temp
                    u_idx = (lsb.bit_length() - 1)
                    u = u_idx + 1
                    if not (adj[v] & (1 << (u - 1))):
                        compatible = False
                        break
                    c_temp ^= lsb
                
                if compatible:
                    current_mask |= bit
                    current_size += 1
                    improved = True
                    break
            
            # Try swaps: remove one, add one
            if not improved and time.perf_counter() < deadline - 5:
                for v_remove in range(1, N + 1):
                    bit_rm = 1 << (v_remove - 1)
                    if not (current_mask & bit_rm):
                        continue
                    
                    for v_add in range(1, N + 1):
                        bit_add = 1 << (v_add - 1)
                        if current_mask & bit_add:
                            continue
                        
                        # Check if v_add is compatible with (current - v_remove)
                        temp_mask = current_mask & ~bit_rm
                        compatible = True
                        t_temp = temp_mask
                        while t_temp:
                            lsb = t_temp & -t_temp
                            u_idx = (lsb.bit_length() - 1)
                            u = u_idx + 1
                            if not (adj[v_add] & (1 << (u - 1))):
                                compatible = False
                                break
                            t_temp ^= lsb
                        
                        if compatible:
                            current_mask = temp_mask | bit_add
                            improved = True
                            break
                    if improved:
                        break
        
        return current_mask

    # Generate multiple seeds and pick the best
    seed1 = greedy_seed_min_conflict()
    seed2 = greedy_seed_small_first()
    
    best_seed = seed1 if seed1.bit_count() >= seed2.bit_count() else seed2
    
    # Local search on the best seed
    improved_seed = local_search_improve(best_seed)
    
    best_clique_size = improved_seed.bit_count()
    best_clique_mask = improved_seed
    
    # Run Bron-Kerbosch to find optimal clique
    success = bron_kerbosch_opt(0, (1 << N) - 1, 0)
    
    # Final local search on the result
    if time.perf_counter() < deadline - 3:
        final_mask = local_search_improve(best_clique_mask)
        if final_mask.bit_count() > best_clique_size:
            best_clique_size = final_mask.bit_count()
            best_clique_mask = final_mask
    
    result = [i + 1 for i in range(N) if (best_clique_mask >> i) & 1]
    
    if not result:
        result = [1]
        
    return result
