import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    P = 1000.0  # Penalty for 4-APs

    # Precompute 3-APs and 4-APs
    # 0-based indices
    threes = []
    fours = []
    
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            threes.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            fours.append((a, a + d, a + 2 * d, a + 3 * d))

    # Map index -> list of other indices in 3-APs and 4-APs
    index_3aps = [[] for _ in range(N)]
    for a, b, c in threes:
        index_3aps[a].append((b, c))
        index_3aps[b].append((a, c))
        index_3aps[c].append((a, b))
        
    index_4aps = [[] for _ in range(N)]
    for a, b, c, d in fours:
        index_4aps[a].append((b, c, d))
        index_4aps[b].append((a, c, d))
        index_4aps[c].append((a, b, d))
        index_4aps[d].append((a, b, c))

    def calc_scores(in_set):
        s3 = 0
        s4 = 0
        for a, b, c in threes:
            if in_set[a] and in_set[b] and in_set[c]:
                s3 += 1
        for a, b, c, d in fours:
            if in_set[a] and in_set[b] and in_set[c] and in_set[d]:
                s4 += 1
        return s3, s4

    def get_delta(state, x, new_val):
        """
        Calculate change in s3 and s4 if state[x] becomes new_val.
        state is the CURRENT state before flip.
        """
        delta_s3 = 0
        delta_s4 = 0
        
        sign = 1 if new_val else -1
        
        # 3-APs
        for o1, o2 in index_3aps[x]:
            if state[o1] and state[o2]:
                delta_s3 += sign
                
        # 4-APs
        for o1, o2, o3 in index_4aps[x]:
            if state[o1] and state[o2] and state[o3]:
                delta_s4 += sign
                
        return delta_s3, delta_s4

    def init_greedy():
        in_set = [False] * N
        # Greedy: add x if it doesn't create a 4-AP
        for x in range(N):
            creates_4ap = False
            for o1, o2, o3 in index_4aps[x]:
                if in_set[o1] and in_set[o2] and in_set[o3]:
                    creates_4ap = True
                    break
            
            if not creates_4ap:
                in_set[x] = True
        return in_set

    def init_random():
        return [random.random() < 0.5 for _ in range(N)]

    def fast_sa(state, s3, s4, time_limit, T_start=10.0, T_end=0.5):
        """
        Run Simulated Annealing on a copy of state.
        Returns best valid set found and its s3 count.
        """
        state = state[:]
        best_valid = None
        best_valid_s3 = -1
        
        # Check initial
        if s4 == 0:
            best_valid = state[:]
            best_valid_s3 = s3
            
        T = T_start
        alpha = T_end / T_start
        steps_per_sec = 50000 # Rough estimate
        total_steps = int((time_limit) * steps_per_sec)
        
        # Use a simple loop with time check
        end_time = time.time() + time_limit
        
        # Pre-compute log for speed? No, math.exp is fine.
        # Optimization: local variables for speed
        idx3 = index_3aps
        idx4 = index_4aps
        P_local = P
        
        while time.time() < end_time:
            x = random.randint(0, N - 1)
            new_val = not state[x]
            
            d_s3, d_s4 = get_delta(state, x, new_val)
            
            new_s3 = s3 + d_s3
            new_s4 = s4 + d_s4
            new_score = new_s3 - P_local * new_s4
            old_score = s3 - P_local * s4
            
            delta_E = new_score - old_score
            
            if delta_E > 0 or random.random() < math.exp(delta_E / T):
                state[x] = new_val
                s3 = new_s3
                s4 = new_s4
                
                if s4 == 0 and s3 > best_valid_s3:
                    best_valid_s3 = s3
                    best_valid = state[:]
            
            T *= alpha
            if T < T_end:
                T = T_end

        return best_valid, best_valid_s3

    # Strategy:
    # 1. Generate a diverse set of initial candidates (Greedy + Random perturbations)
    # 2. Run parallel SA chains on these candidates
    # 3. Track the global best valid set
    
    candidates = []
    
    # Add greedy start
    g = init_greedy()
    s3_g, s4_g = calc_scores(g)
    candidates.append((g, s3_g, s4_g))
    
    # Add perturbed greedy
    for _ in range(10):
        g = init_greedy()
        # Randomly flip 10-20 bits
        for _ in range(random.randint(10, 20)):
            idx = random.randint(0, N-1)
            g[idx] = not g[idx]
        s3_g, s4_g = calc_scores(g)
        candidates.append((g, s3_g, s4_g))
        
    # Add random starts
    for _ in range(10):
        r = init_random()
        s3_r, s4_r = calc_scores(r)
        candidates.append((r, s3_r, s4_r))

    # Sort by score (s3 - P*s4) to prioritize good starts?
    # Or just run them all.
    # Let's run SA on each candidate with a time slice.
    
    best_global_set = None
    best_global_s3 = -1
    
    # Check initial candidates
    for state, s3, s4 in candidates:
        if s4 == 0 and s3 > best_global_s3:
            best_global_s3 = s3
            best_global_set = state[:]

    # Determine time slice for each SA run
    # We have ~50s left.
    # Let's do 20 SA runs, each with ~2.5s.
    num_runs = 20
    time_per_run = (deadline - time.time()) / num_runs - 0.5 # buffer
    
    # Shuffle candidates to ensure diversity in time allocation
    random.shuffle(candidates)
    
    # If we have fewer candidates than runs, reuse the best ones or just run what we have
    # Actually, let's just run SA on the top 20 candidates by initial score
    candidates.sort(key=lambda x: x[1] - P * x[2], reverse=True)
    
    runs_to_do = candidates[:num_runs]
    
    for i, (state, s3, s4) in enumerate(runs_to_do):
        if time.time() > deadline - 2.0:
            break
            
        # Allocate slightly more time to earlier runs? Or equal.
        # Equal is fine.
        t_slice = min(time_per_run, deadline - time.time() - 1.0)
        if t_slice < 0.1:
            break
            
        # Vary temperature based on index?
        # Higher index -> lower T (more exploitation)
        T_start = 10.0
        T_end = 0.5 + (i / num_runs) * 5.0 # Later runs have higher T_end? No, let's keep it simple.
        
        # Actually, let's just use standard cooling.
        best_set, best_s3 = fast_sa(state, s3, s4, t_slice)
        
        if best_set is not None and best_s3 > best_global_s3:
            best_global_s3 = best_s3
            best_global_set = best_set[:]

    # Final polish: Greedy add on the best set
    if best_global_set:
        # Try to add elements that create 3-APs and no 4-APs
        for x in range(N):
            if not best_global_set[x]:
                creates_4ap = False
                for o1, o2, o3 in index_4aps[x]:
                    if best_global_set[o1] and best_global_set[o2] and best_global_set[o3]:
                        creates_4ap = True
                        break
                if not creates_4ap:
                    # Check if it adds 3-APs
                    adds_3ap = False
                    for o1, o2 in index_3aps[x]:
                        if best_global_set[o1] and best_global_set[o2]:
                            adds_3ap = True
                            break
                    if adds_3ap:
                        best_global_set[x] = True

    # Fallback if no valid set found (shouldn't happen with greedy)
    if best_global_set is None:
        g = init_greedy()
        best_global_set = g
        best_global_s3, _ = calc_scores(g)

    # Convert to 1-based indices
    return [i + 1 for i in range(N) if best_global_set[i]]

if __name__ == "__main__":
    res = construct()
    print(f"Size: {len(res)}")
