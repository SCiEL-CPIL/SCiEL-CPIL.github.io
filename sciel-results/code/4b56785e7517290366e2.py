import time
import numpy as np

def construct():
    n = 256
    start_time = time.time()
    # Evaluator kills at 90s, task says 60s. Use 58s to be safe.
    time_limit = 58.0

    # 1. Precomputation of 4-AP conflicts and 3-AP contributions
    
    # Bit values
    bit_val = [1 << i for i in range(n)]
    
    # Precompute 3-AP contributions for each element u.
    # pairs_for_x[x] is a list of masks M such that (S & M) == M implies x is in a 3-AP.
    pairs_for_x = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        # x is 1st: x, x+d, x+2d
        for d in range(1, (n - x) // 2 + 1):
            u, v = x + d, x + 2 * d
            if v <= n:
                pairs_for_x[x].append(bit_val[u - 1] | bit_val[v - 1])
        # x is 2nd: x-d, x, x+d
        for d in range(1, min(x - 1, n - x) + 1):
            u, v = x - d, x + d
            if u >= 1:
                pairs_for_x[x].append(bit_val[u - 1] | bit_val[v - 1])
        # x is 3rd: x-2d, x-d, x
        for d in range(1, (x - 1) // 2 + 1):
            u, v = x - 2 * d, x - d
            if u >= 1:
                pairs_for_x[x].append(bit_val[u - 1] | bit_val[v - 1])
        
    # Precompute 4-AP constraints for validity.
    forbidden_4 = [[] for _ in range(n + 1)]
    for x in range(1, n + 1):
        masks = []
        # x is 1st: x, x+d, x+2d, x+3d
        for d in range(1, (n - x) // 3 + 1):
            if x + 3 * d <= n:
                m = bit_val[x + d - 1] | bit_val[x + 2 * d - 1] | bit_val[x + 3 * d - 1]
                masks.append(m)
        # x is 2nd: x-d, x, x+d, x+2d
        for d in range(1, n // 3 + 1):
            if x - d >= 1 and x + 2 * d <= n:
                m = bit_val[x - d - 1] | bit_val[x + d - 1] | bit_val[x + 2 * d - 1]
                masks.append(m)
        # x is 3rd: x-2d, x-d, x, x+d
        for d in range(1, n // 3 + 1):
            if x - 2 * d >= 1 and x + d <= n:
                m = bit_val[x - 2 * d - 1] | bit_val[x - d - 1] | bit_val[x + d - 1]
                masks.append(m)
        # x is 4th: x-3d, x-2d, x-d, x
        for d in range(1, n // 3 + 1):
            if x - 3 * d >= 1:
                m = bit_val[x - 3 * d - 1] | bit_val[x - 2 * d - 1] | bit_val[x - d - 1]
                masks.append(m)
        forbidden_4[x] = masks

    def is_valid_addition(S_mask, x):
        for m in forbidden_4[x]:
            if (S_mask & m) == m:
                return False
        return True

    def count_3aps(S_mask):
        total = 0
        S_set = {i + 1 for i in range(n) if (S_mask >> i) & 1}
        for x in S_set:
            for m in pairs_for_x[x]:
                if (S_mask & m) == m:
                    total += 1
        return total // 3

    # 2. Greedy Initialization
    # Start with empty set, add elements one by one if valid.
    # To maximize 3-APs, we should prioritize elements that have high potential for 3-APs.
    # Or just random order? Random order is often good for diversity.
    # Let's try a few random greedy starts and keep the best.
    
    best_S_mask = 0
    best_count = 0
    
    # Run multiple greedy restarts
    num_restarts = 20
    for r in range(num_restarts):
        S_mask = 0
        candidates = list(range(1, n + 1))
        np.random.shuffle(candidates)
        
        for x in candidates:
            if is_valid_addition(S_mask, x):
                S_mask |= bit_val[x - 1]
                
        cnt = count_3aps(S_mask)
        if cnt > best_count:
            best_count = cnt
            best_S_mask = S_mask
            
    # 3. Parallel Tempering (Replica Exchange) Simulated Annealing
    
    # We maintain a set of replicas (chains) at different temperatures.
    # Low temperature chains refine the solution, high temperature chains explore.
    # Periodically, we swap states between chains based on Metropolis criterion.
    
    num_replicas = 10
    temps = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0])
    
    # Initialize replicas from the best greedy solution, with some perturbation for diversity
    replicas = []
    for i in range(num_replicas):
        if i == 0:
            rep_mask = best_S_mask
        else:
            # Perturb the best solution
            rep_mask = best_S_mask
            # Flip a few bits randomly
            for _ in range(i * 2):
                idx = np.random.randint(0, n)
                rep_mask ^= bit_val[idx]
        replicas.append(rep_mask)
        
    # Track best found overall
    current_best_mask = best_S_mask
    current_best_count = best_count
    
    # SA loop
    steps_per_temp = 500
    total_steps = steps_per_temp * num_replicas
    
    # Precompute list of all possible moves (add, remove, swap) for speed?
    # No, generate on the fly but vectorize where possible.
    # For n=256, generating moves is fast.
    
    for step in range(total_steps):
        if time.time() - start_time > time_limit - 2.0:
            break
            
        # Update each replica
        for i in range(num_replicas):
            T = temps[i]
            S_mask = replicas[i]
            
            # Generate a random move
            # 50% add, 50% remove? Or just swap?
            # Swaps preserve cardinality, which might be good.
            # Adds/Removes change cardinality.
            # Let's try: 10% add, 10% remove, 80% swap.
            
            move_type = np.random.rand()
            
            if move_type < 0.1:
                # Add a random element not in S
                N_set = [x for x in range(1, n+1) if not (S_mask & bit_val[x-1])]
                if not N_set:
                    continue
                v = N_set[np.random.randint(len(N_set))]
                new_S_mask = S_mask | bit_val[v-1]
                # Check validity
                if not is_valid_addition(S_mask, v):
                    continue
                # Calculate delta
                old_count = count_3aps(S_mask)
                new_count = count_3aps(new_S_mask)
                delta = new_count - old_count
            elif move_type < 0.2:
                # Remove a random element in S
                S_set = [x for x in range(1, n+1) if (S_mask & bit_val[x-1])]
                if not S_set:
                    continue
                u = S_set[np.random.randint(len(S_set))]
                new_S_mask = S_mask & ~bit_val[u-1]
                # Always valid to remove
                old_count = count_3aps(S_mask)
                new_count = count_3aps(new_S_mask)
                delta = new_count - old_count
            else:
                # Swap u in S with v not in S
                S_set = [x for x in range(1, n+1) if (S_mask & bit_val[x-1])]
                N_set = [x for x in range(1, n+1) if not (S_mask & bit_val[x-1])]
                if not S_set or not N_set:
                    continue
                u = S_set[np.random.randint(len(S_set))]
                v = N_set[np.random.randint(len(N_set))]
                
                S_mask_without_u = S_mask & ~bit_val[u-1]
                # Check validity of v in S\{u}
                if not is_valid_addition(S_mask_without_u, v):
                    continue
                
                new_S_mask = S_mask_without_u | bit_val[v-1]
                old_count = count_3aps(S_mask)
                new_count = count_3aps(new_S_mask)
                delta = new_count - old_count
                
            # Metropolis acceptance
            if delta >= 0:
                accept = True
            else:
                # delta is negative (we want to maximize, so energy increases)
                # Accept with probability exp(delta / T)
                # Since delta is negative, exp(delta/T) < 1
                accept = np.random.rand() < np.exp(delta / T)
                
            if accept:
                replicas[i] = new_S_mask
                # Update best
                if new_count > current_best_count:
                    current_best_count = new_count
                    current_best_mask = new_S_mask
                    
        # Replica Exchange
        # Try to swap adjacent replicas if it improves the "energy" (or is accepted)
        # For simplicity, do random swaps between all pairs?
        # Or just adjacent? Adjacent is standard.
        for i in range(num_replicas - 1):
            if np.random.rand() < 0.5: # 50% chance to attempt swap
                T1 = temps[i]
                T2 = temps[i+1]
                S1 = replicas[i]
                S2 = replicas[i+1]
                
                # Calculate energies (counts)
                E1 = count_3aps(S1)
                E2 = count_3aps(S2)
                
                # Delta E = E2/T1 - E1/T2 ?
                # Standard criterion: accept if exp(-E1/T2 - E2/T1) > exp(-E2/T2 - E1/T1)
                # Which is exp(E1/T1 + E2/T2 - E2/T1 - E1/T2) > 1 ?
                # Wait, the energy in SA is usually -Objective.
                # Let Energy = -Count.
                # E1 = -E1_count, E2 = -E2_count.
                # Accept swap if exp( (E1/T2 - E2/T1) - (E2/T2 - E1/T1) ) > 1?
                # Actually, let's just use the Metropolis criterion on the swap move.
                # Delta = (E1_new/T1 + E2_new/T2) - (E1_old/T1 + E2_old/T2)
                # Where E1_new is energy of S2 at T1, etc.
                # E1_new = -E2_count, E2_new = -E1_count.
                # Delta = (-E2_count/T1 - E1_count/T2) - (-E1_count/T1 - E2_count/T2)
                # Delta = E1_count*(1/T1 - 1/T2) + E2_count*(1/T2 - 1/T1)
                # Delta = (E1_count - E2_count) * (1/T1 - 1/T2)
                
                delta_swap = (E1 - E2) * (1.0/T1 - 1.0/T2)
                
                if delta_swap <= 0:
                    # Always accept if delta <= 0 (improvement in energy)
                    replicas[i], replicas[i+1] = replicas[i+1], replicas[i]
                else:
                    # Accept with probability exp(-delta_swap)
                    if np.random.rand() < np.exp(-delta_swap):
                        replicas[i], replicas[i+1] = replicas[i+1], replicas[i]

    # Final result
    res_list = sorted([i + 1 for i in range(n) if (current_best_mask >> i) & 1])
    if not res_list:
        return [1]
    return res_list

if __name__ == "__main__":
    res = construct()
    print(len(res), res[:10])
