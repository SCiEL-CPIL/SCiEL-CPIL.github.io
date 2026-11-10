import numpy as np
import random
import time

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    # Forbidden differences
    forbidden = set()
    for x in range(1, m):
        forbidden.add(pow(x, 2, m))
    forbidden.discard(0)
    
    # Build adjacency bitmasks
    # adj[i] is a bitmask of nodes j such that (i-j) % m in forbidden
    adj = [0] * m
    for i in range(m):
        mask = 0
        for d in forbidden:
            j = (i + d) % m
            mask |= (1 << j)
        adj[i] = mask

    best_set = 0
    best_size = 0
    
    # Simulated Annealing
    def run_sa(start_state, iterations, t_start, t_end):
        nonlocal best_set, best_size
        
        S = start_state
        # Calculate initial conflict count
        # Conflict count is number of pairs (i,j) in S such that j in adj[i]
        # It's easier to just check validity or calculate energy
        # Energy = -popcount(S) + penalty * conflict_count
        
        penalty = 20
        
        def get_conflicts(S):
            count = 0
            # Iterate over set bits
            # Efficient way: for each i in S, count bits in S & adj[i]
            # This counts each edge twice.
            temp = S
            while temp:
                # Get lowest set bit
                lsb = temp & -temp
                i = lsb.bit_length() - 1
                count += (S & adj[i]).bit_count()
                temp ^= lsb
            return count // 2

        def get_energy(S):
            size = S.bit_count()
            conf = get_conflicts(S)
            return -size + penalty * conf

        current_energy = get_energy(S)
        
        for i in range(iterations):
            # Cooling
            t = t_start + (t_end - t_start) * (i / iterations)
            
            # Pick random move
            # Flip a random bit
            idx = random.randint(0, m - 1)
            bit = 1 << idx
            
            if S & bit:
                # Removing idx
                # Conflicts decrease by number of neighbors in S
                neighbors_in_S = (S & adj[idx]).bit_count()
                delta_conf = -neighbors_in_S
                delta_size = -1
            else:
                # Adding idx
                # Conflicts increase by number of neighbors in S
                neighbors_in_S = (S & adj[idx]).bit_count()
                delta_conf = neighbors_in_S
                delta_size = 1
                
            delta_energy = -delta_size + penalty * delta_conf
            
            if delta_energy < 0 or random.random() < pow(2.718281828, -delta_energy / max(t, 1e-9)):
                S ^= bit
                current_energy += delta_energy
                
                # Check if valid and better
                if delta_conf == 0 and neighbors_in_S == 0: # Wait, this logic is tricky for removal
                    pass # Just check full validity for best update to be safe, or track conflicts properly
            
            # To be safe and simple, check validity periodically or on improvement
            # Actually, calculating full conflicts every step is O(N). 
            # With N=205, it's fast enough for 10M steps?
            # 10M * 205 ops = 2B ops. Too slow in Python.
            
            # Optimization: Maintain conflict count incrementally.
            # But we need to know if the state is VALID (conflict count == 0).
            
            # Let's restart the SA logic to maintain conflict count correctly.
            pass

    # Revised SA with incremental conflict tracking
    def run_sa_optimized(start_state, iterations, t_start, t_end):
        nonlocal best_set, best_size
        
        S = start_state
        penalty = 20
        
        # Initial conflict count
        # We need a function to calculate conflicts for a given S
        # And update it when flipping a bit.
        
        def calc_conflicts(S):
            c = 0
            temp = S
            while temp:
                lsb = temp & -temp
                i = lsb.bit_length() - 1
                c += (S & adj[i]).bit_count()
                temp ^= lsb
            return c // 2

        current_conf = calc_conflicts(S)
        current_size = S.bit_count()
        current_energy = -current_size + penalty * current_conf
        
        # Update best if valid
        if current_conf == 0 and current_size > best_size:
            best_size = current_size
            best_set = S
            
        for i in range(iterations):
            t = t_start + (t_end - t_start) * (i / iterations)
            
            idx = random.randint(0, m - 1)
            bit = 1 << idx
            
            if S & bit:
                # Remove idx
                # Neighbors of idx in S
                neigh = (S & adj[idx]).bit_count()
                delta_conf = -neigh
                delta_size = -1
            else:
                # Add idx
                # Neighbors of idx in S
                neigh = (S & adj[idx]).bit_count()
                delta_conf = neigh
                delta_size = 1
                
            delta_energy = -delta_size + penalty * delta_conf
            
            accept = False
            if delta_energy < 0:
                accept = True
            else:
                # Avoid division by zero or overflow
                if t > 1e-9:
                    accept = random.random() < pow(2.718281828, -delta_energy / t)
            
            if accept:
                S ^= bit
                current_conf += delta_conf
                current_size += delta_size
                current_energy += delta_energy
                
                # Check if valid
                if current_conf == 0:
                    if current_size > best_size:
                        best_size = current_size
                        best_set = S

    # Run multiple restarts
    start_time = time.time()
    timeout = 500 # 500 seconds limit
    
    restart_count = 0
    while time.time() - start_time < timeout:
        restart_count += 1
        # Start with random state or greedy
        # Greedy: pick random nodes, add if no conflict
        S = 0
        nodes = list(range(m))
        random.shuffle(nodes)
        for n in nodes:
            if (S & adj[n]) == 0:
                S |= (1 << n)
        
        run_sa_optimized(S, 200000, 10.0, 0.1)
        
        # Optional: Local search polish
        # Try to add any feasible node
        changed = True
        while changed:
            changed = False
            for n in range(m):
                if not (best_set & (1 << n)) and (best_set & adj[n]) == 0:
                    best_set |= (1 << n)
                    best_size += 1
                    changed = True
                    break # Restart loop to check again? Or just continue
            # If we found one, we should re-check all, but simple loop is fine for polish
            # Actually, just one pass is not enough.
            # Let's do a simple greedy fill on the best_set found so far in each restart
            # But best_set is global. We should polish the local best before updating global?
            # Or just let SA find it.
        
        # Polish the global best_set
        changed = True
        while changed:
            changed = False
            for n in range(m):
                if not (best_set & (1 << n)) and (best_set & adj[n]) == 0:
                    best_set |= (1 << n)
                    best_size += 1
                    changed = True
                    break # Break to restart the for loop? No, just continue to next n?
                    # If we add n, it might block others, but it won't unblock.
                    # So we can just iterate once? No, adding one node doesn't prevent adding others unless they conflict.
                    # But since we check `(best_set & adj[n]) == 0`, and best_set grows,
                    # it's safer to restart the loop or just do multiple passes.
                    # Given N=205, multiple passes are cheap.
                    
        # Actually, a simple greedy fill:
        # for n in range(m):
        #   if not in best_set and no conflict: add
        # This is O(N^2) per restart. Negligible.
        for n in range(m):
            if not (best_set & (1 << n)) and (best_set & adj[n]) == 0:
                best_set |= (1 << n)
                best_size += 1

    # Convert best_set to array
    res = []
    for i in range(m):
        if best_set & (1 << i):
            res.append(i)
            
    return np.array(res, dtype=np.int64)
