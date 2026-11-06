import time
import random
import math

def construct():
    start_time = time.time()
    timeout = 55.0  # Safety margin before the 60 s evaluator limit

    n = 256
    # adj[i] is a bitmask of vertices that conflict with i
    adj = [0] * (n + 1)
    
    # Build conflict graph
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)

    # Helper: Get list of set bits in a mask
    def get_bits(mask):
        bits = []
        while mask:
            lsb = mask & -mask
            bits.append(lsb.bit_length() - 1)
            mask ^= lsb
        return bits

    def greedy_mis():
        remaining = (1 << (n + 1)) - 2 # bits 1..n set
        result = 0
        
        while remaining:
            min_v = -1
            min_deg = float('inf')
            
            # Iterate through set bits in remaining
            r = remaining
            while r:
                lsb = r & -r
                v = lsb.bit_length() - 1
                deg = (adj[v] & remaining).bit_count()
                if deg < min_deg:
                    min_deg = deg
                    min_v = v
                r ^= lsb
            
            if min_v == -1:
                break
                
            result |= (1 << min_v)
            # Remove min_v and its neighbors from remaining
            remaining &= ~(adj[min_v] | (1 << min_v))
            
        return result

    # Exact MIS for a small subgraph defined by mask P
    # Returns the mask of the MIS
    def mis_small(P):
        if P == 0:
            return 0
        
        # If P is small enough, we can use a simple recursive B&B
        # Heuristic: Order vertices by degree in P descending? Or just pick one.
        # Tomita's algorithm is better, but for small P, simple B&B with pruning is fine.
        
        # Simple B&B
        # We want to maximize the number of vertices.
        # Upper bound: number of vertices in P.
        
        best_mask = 0
        best_size = 0
        
        # To speed up, we can sort vertices by degree in P
        verts = get_bits(P)
        # Sort by degree in P descending to find conflicts early? 
        # Actually, for MIS, picking high degree vertices first in the "exclude" branch 
        # or "include" branch matters.
        
        # Let's use a standard recursive approach with memoization? 
        # P is a mask, so we can memoize results for submasks if they repeat, 
        # but P is changing. 
        # Given P is small (neighbors of a vertex), we don't need heavy optimization.
        
        def solve(sub_P, current_mask, current_size):
            nonlocal best_mask, best_size
            
            if sub_P == 0:
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
                return
            
            # Upper bound: current_size + popcount(sub_P)
            if current_size + sub_P.bit_count() <= best_size:
                return
            
            # Pick a vertex v from sub_P
            # Heuristic: pick vertex with max degree in sub_P
            v = -1
            max_deg = -1
            r = sub_P
            while r:
                lsb = r & -r
                u = lsb.bit_length() - 1
                deg = (adj[u] & sub_P).bit_count()
                if deg > max_deg:
                    max_deg = deg
                    v = u
                r ^= lsb
            
            if v == -1:
                return
                
            # Branch 1: Include v
            # New sub_P is sub_P without v and its neighbors
            new_sub_P = sub_P & ~(adj[v] | (1 << v))
            solve(new_sub_P, current_mask | (1 << v), current_size + 1)
            
            # Branch 2: Exclude v
            new_sub_P2 = sub_P & ~(1 << v)
            solve(new_sub_P2, current_mask, current_size)

        solve(P, 0, 0)
        return best_mask

    def try_improve(S_mask):
        # Local search: remove a vertex, add exact MIS of its valid neighbors
        S_mask = S_mask
        verts = get_bits(S_mask)
        random.shuffle(verts)
        
        improved = False
        for v in verts:
            if time.time() - start_time > timeout:
                break
                
            # Remove v
            S_new = S_mask & ~(1 << v)
            
            # Find candidates: neighbors of v not in S_new, and no neighbor in S_new
            # C = N(v) \ S_new
            # Valid if N(u) \cap S_new == 0
            cands_mask = adj[v] & ~S_new
            valid_mask = 0
            c = cands_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                if (adj[u] & S_new) == 0:
                    valid_mask |= lsb
                c ^= lsb
            
            # Find MIS of valid_mask
            add_mask = mis_small(valid_mask)
            
            S_cand = S_new | add_mask
            
            if S_cand.bit_count() > S_mask.bit_count():
                S_mask = S_cand
                improved = True
                # Don't break immediately, might find more improvements in same pass?
                # Or break to restart the shuffle?
                # Let's continue the loop, but the verts list is stale.
                # It's safer to break and restart the outer while loop if we improved.
                break
                
        return S_mask, improved

    def anneal(S_mask, time_budget):
        S_mask = S_mask
        T0 = 1.2
        T_end = 0.015
        cooling = 0.9995
        T = T0
        
        while time.time() - start_time < time_budget:
            if T < T_end:
                break
                
            # Pick random v in S
            verts = get_bits(S_mask)
            if not verts:
                break
            v = random.choice(verts)
            
            S_new = S_mask & ~(1 << v)
            
            # Find valid candidates
            cands_mask = adj[v] & ~S_new
            valid_mask = 0
            c = cands_mask
            while c:
                lsb = c & -c
                u = lsb.bit_length() - 1
                if (adj[u] & S_new) == 0:
                    valid_mask |= lsb
                c ^= lsb
            
            # For anneal, use a fast greedy for add_mask to keep it fast
            # Or use mis_small if valid_mask is very small
            if valid_mask.bit_count() < 20:
                add_mask = mis_small(valid_mask)
            else:
                # Greedy
                add_mask = 0
                rem = valid_mask
                while rem:
                    # Pick min degree in rem
                    min_v = -1
                    min_deg = float('inf')
                    r = rem
                    while r:
                        lsb = r & -r
                        u = lsb.bit_length() - 1
                        deg = (adj[u] & rem).bit_count()
                        if deg < min_deg:
                            min_deg = deg
                            min_v = u
                        r ^= lsb
                    if min_v == -1:
                        break
                    add_mask |= (1 << min_v)
                    rem &= ~(adj[min_v] | (1 << min_v))
            
            S_cand = S_new | add_mask
            d = S_cand.bit_count() - S_mask.bit_count()
            
            if d >= 0 or random.random() < math.exp(d / T):
                S_mask = S_cand
                
            T *= cooling
            
        return S_mask

    # Initial solution
    best_mask = greedy_mis()
    best_size = best_mask.bit_count()
    
    # Main Loop
    while time.time() - start_time < timeout - 2:
        # Local search phase
        improved = True
        while improved and time.time() - start_time < timeout - 2:
            sol_mask, improved = try_improve(best_mask)
            if sol_mask.bit_count() > best_size:
                best_size = sol_mask.bit_count()
                best_mask = sol_mask
        
        # Simulated annealing refinement
        anneal_budget = min(10.0, max(2.0, timeout - (time.time() - start_time) - 2))
        if time.time() - start_time + anneal_budget <= timeout - 1:
            sol2_mask = anneal(best_mask, timeout - (time.time() - start_time) - 2)
            if sol2_mask.bit_count() > best_size:
                best_size = sol2_mask.bit_count()
                best_mask = sol2_mask
                # Polish with local search
                improved = True
                while improved and time.time() - start_time < timeout - 2:
                    sol3_mask, improved = try_improve(best_mask)
                    if sol3_mask.bit_count() > best_size:
                        best_size = sol3_mask.bit_count()
                        best_mask = sol3_mask
        
        # Random restart
        if random.random() < 0.3:
            new_mask = greedy_mis()
            # Improve the new one
            improved = True
            while improved and time.time() - start_time < timeout - 2:
                new_mask, improved = try_improve(new_mask)
            if new_mask.bit_count() > best_size:
                best_size = new_mask.bit_count()
                best_mask = new_mask

    return get_bits(best_mask)
