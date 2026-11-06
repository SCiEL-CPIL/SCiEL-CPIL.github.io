import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    N = 256
    P = 100.0  # Penalty for 4-APs

    # Precompute 3-APs and 4-APs
    # A 3-AP is (a, a+d, a+2d). We store the indices (0-based) as a tuple.
    # To speed up delta calculation, we just need to know which 3-APs a specific element 'x' participates in.
    # However, calculating delta exactly is O(num_aps_involving_x).
    # Total 3-APs in [1, 256] is sum_{d=1}^{127} (256 - 2d) approx 256*127 - 2*127*128/2 = 32512 - 16256 = 16256.
    # Total 4-APs is sum_{d=1}^{85} (256 - 3d) approx 256*85 - 3*85*86/2 = 21760 - 10965 = 10795.
    
    # For efficiency in SA, we can precompute for each element x, the list of 3-APs and 4-APs it belongs to.
    # But checking the state of the other 2 or 3 elements is still needed.
    
    # Let's stick to a robust SA with bitmasks.
    
    # Helper to count 3-APs and 4-APs given a mask
    # This is O(N^2) if done naively, but we can optimize.
    # Actually, for N=256, naive counting is fast enough if we do it sparingly.
    # But in SA, we need delta.
    
    # Let's precompute the "neighbors" for delta calculation.
    # For a 3-AP (a, b, c), if we flip b, the AP exists if a and c are present.
    # If we flip a, the AP exists if b and c are present.
    
    # Let's create a list of all 3-APs and 4-APs as tuples of 0-based indices.
    threes = []
    fours = []
    
    for d in range(1, N // 2 + 1):
        for a in range(0, N - 2 * d):
            threes.append((a, a + d, a + 2 * d))
            
    for d in range(1, N // 3 + 1):
        for a in range(0, N - 3 * d):
            fours.append((a, a + d, a + 2 * d, a + 3 * d))

    # For each index i, list the indices of 3-APs and 4-APs that contain i.
    # And for each such AP, which other indices are involved.
    # This allows O(1) or O(k) delta calculation where k is the number of APs containing i.
    
    # Actually, simpler: just precompute the list of APs.
    # In the move evaluation, we iterate through the APs containing the moved element.
    
    # Map index -> list of (ap_index, other_indices_tuple)
    # To save memory and time, let's just store the APs themselves.
    
    # Let's try a different strategy: 
    # Use a dense array for membership.
    # But bitmasks are faster for checking.
    
    # Let's use a list of bools for current state, and update counts incrementally.
    
    def init_greedy():
        in_set = [False] * N
        # Greedy: add x if it doesn't create a 4-AP
        # We want to maximize 3-APs, so maybe we should prioritize adding elements that create many 3-APs?
        # The seed just adds if no 4-AP is created. This results in a maximal 4-AP-free set.
        # Let's do a greedy that maximizes immediate 3-AP gain while avoiding 4-APs.
        
        # Simple greedy:
        for x in range(N):
            creates_4ap = False
            # Check 4-APs containing x
            # x as 0th, 1st, 2nd, 3rd
            # 0th: x, x+d, x+2d, x+3d
            max_d = (N - 1 - x) // 3
            for d in range(1, max_d + 1):
                if in_set[x+d] and in_set[x+2*d] and in_set[x+3*d]:
                    creates_4ap = True
                    break
            if not creates_4ap:
                # 1st: x-d, x, x+d, x+2d
                max_d = min(x, (N - 1 - x) // 2)
                for d in range(1, max_d + 1):
                    if in_set[x-d] and in_set[x+d] and in_set[x+2*d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # 2nd: x-2d, x-d, x, x+d
                max_d = min(x // 2, N - 1 - x)
                for d in range(1, max_d + 1):
                    if in_set[x-2*d] and in_set[x-d] and in_set[x+d]:
                        creates_4ap = True
                        break
            if not creates_4ap:
                # 3rd: x-3d, x-2d, x-d, x
                max_d = x // 3
                for d in range(1, max_d + 1):
                    if in_set[x-3*d] and in_set[x-2*d] and in_set[x-d]:
                        creates_4ap = True
                        break
            
            if not creates_4ap:
                in_set[x] = True
        return in_set

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

    # Parallel Tempering Setup
    num_replicas = 16
    # Temperatures: geometric progression
    T_min = 0.5
    T_max = 10.0
    temperatures = [T_min * (T_max / T_min) ** (i / (num_replicas - 1)) for i in range(num_replicas)]
    
    replicas = []
    for _ in range(num_replicas):
        # Start with slightly perturbed greedy or random
        if random.random() < 0.5:
            r = init_greedy()
            # Perturb
            for _ in range(10):
                idx = random.randint(0, N-1)
                r[idx] = not r[idx]
        else:
            r = [random.random() < 0.5 for _ in range(N)]
        
        s3, s4 = calc_scores(r)
        score = s3 - P * s4
        replicas.append({
            'state': r,
            's3': s3,
            's4': s4,
            'score': score,
            'T': T_max # Will be assigned
        })
    
    # Assign temperatures
    for i, r in enumerate(replicas):
        r['T'] = temperatures[i]

    best_valid_set = None
    best_valid_score = -1

    # Initial check
    for r in replicas:
        if r['s4'] == 0 and r['s3'] > best_valid_score:
            best_valid_score = r['s3']
            best_valid_set = r['state'].copy()

    # Pre-compute for each index, the list of 3-APs and 4-APs it participates in.
    # And for each AP, the other indices.
    # This is crucial for speed.
    
    # index_3aps[i] = list of (other1, other2)
    # index_4aps[i] = list of (other1, other2, other3)
    
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

    def get_delta(s3, s4, state, x, new_val):
        # Calculate change in s3 and s4 if state[x] becomes new_val
        # new_val is True (add) or False (remove)
        delta_s3 = 0
        delta_s4 = 0
        
        sign = 1 if new_val else -1
        
        # 3-APs
        for o1, o2 in index_3aps[x]:
            if state[o1] and state[o2]:
                # If we are adding, and the other two are present, we gain 1.
                # If we are removing, and the other two are present, we lose 1.
                # Wait, if state[x] is currently False and we set to True:
                # The AP exists iff state[o1] and state[o2] are True.
                # So delta is +1 if state[o1] and state[o2] are True.
                # If state[x] is currently True and we set to False:
                # The AP existed iff state[o1] and state[o2] are True.
                # So delta is -1 if state[o1] and state[o2] are True.
                # In both cases, the condition is the same.
                delta_s3 += sign
                
        # 4-APs
        for o1, o2, o3 in index_4aps[x]:
            if state[o1] and state[o2] and state[o3]:
                delta_s4 += sign
                
        return delta_s3, delta_s4

    steps = 0
    last_swap_check = 0
    
    while time.time() < deadline:
        # Select a random replica
        i = random.randint(0, num_replicas - 1)
        r = replicas[i]
        state = r['state']
        T = r['T']
        
        # Generate a move
        # Mostly flip, sometimes swap
        if random.random() < 0.9:
            x = random.randint(0, N - 1)
            new_val = not state[x]
            
            d_s3, d_s4 = get_delta(r['s3'], r['s4'], state, x, new_val)
            
            new_s3 = r['s3'] + d_s3
            new_s4 = r['s4'] + d_s4
            new_score = new_s3 - P * new_s4
            
            delta_E = new_score - r['score']
            
            # Acceptance
            if delta_E > 0 or random.random() < math.exp(delta_E / T):
                state[x] = new_val
                r['s3'] = new_s3
                r['s4'] = new_s4
                r['score'] = new_score
                
                if new_s4 == 0 and new_s3 > best_valid_score:
                    best_valid_score = new_s3
                    best_valid_set = state.copy()
        else:
            # Swap move: pick two distinct indices x, y
            # Try to swap their values.
            # This preserves cardinality.
            x, y = random.sample(range(N), 2)
            if state[x] == state[y]:
                continue # No change
            
            # Calculate delta for flipping x then y?
            # No, it's a joint move.
            # Let's just do two sequential flips for simplicity, or calculate joint delta.
            # Joint delta:
            # If state[x]=1, state[y]=0. New: state[x]=0, state[y]=1.
            # This is equivalent to removing x and adding y.
            # d_s3 = d_s3_remove(x) + d_s3_add(y, with x removed)
            
            # Let's just do a simple flip for now, swap is complex to calculate delta correctly without side effects.
            # Or just do two flips.
            # Flip x
            new_val_x = not state[x]
            d_s3_1, d_s4_1 = get_delta(r['s3'], r['s4'], state, x, new_val_x)
            
            # Temporarily apply x
            state[x] = new_val_x
            s3_temp = r['s3'] + d_s3_1
            s4_temp = r['s4'] + d_s4_1
            
            # Flip y
            new_val_y = not state[y]
            d_s3_2, d_s4_2 = get_delta(s3_temp, s4_temp, state, y, new_val_y)
            
            new_s3 = s3_temp + d_s3_2
            new_s4 = s4_temp + d_s4_2
            new_score = new_s3 - P * new_s4
            
            delta_E = new_score - r['score']
            
            if delta_E > 0 or random.random() < math.exp(delta_E / T):
                state[y] = new_val_y
                r['s3'] = new_s3
                r['s4'] = new_s4
                r['score'] = new_score
                
                if new_s4 == 0 and new_s3 > best_valid_score:
                    best_valid_score = new_s3
                    best_valid_set = state.copy()
            else:
                # Revert x
                state[x] = not new_val_x

        steps += 1
        
        # Replica Exchange
        # Check for swap every 100 steps
        if steps - last_swap_check >= 100:
            last_swap_check = steps
            # Pick two random replicas
            i1, i2 = random.sample(range(num_replicas), 2)
            r1 = replicas[i1]
            r2 = replicas[i2]
            
            # Calculate probability of swap
            # P(swap) = min(1, exp( (1/T_low - 1/T_high) * (E_high - E_low) ))
            # Where T_low < T_high, so E_high is the score of the higher temp replica.
            if r1['T'] < r2['T']:
                T_low = r1['T']
                T_high = r2['T']
                E_low = r1['score']
                E_high = r2['score']
                inv_diff = (1.0 / T_low - 1.0 / T_high)
                delta_E = E_high - E_low
            else:
                T_low = r2['T']
                T_high = r1['T']
                E_low = r2['score']
                E_high = r1['score']
                inv_diff = (1.0 / T_low - 1.0 / T_high)
                delta_E = E_high - E_low
                
            # Note: inv_diff is negative because T_low < T_high -> 1/T_low > 1/T_high.
            # So exponent is negative * (E_high - E_low).
            # If E_high < E_low (high temp replica has lower score), exponent is positive -> accept.
            # This allows the high temp replica to "take" the low energy state.
            
            exp_arg = inv_diff * delta_E
            if exp_arg > 0:
                p_swap = 1.0
            else:
                p_swap = math.exp(exp_arg)
                
            if random.random() < p_swap:
                # Swap states
                replicas[i1]['state'], replicas[i2]['state'] = replicas[i2]['state'], replicas[i1]['state']
                replicas[i1]['s3'], replicas[i2]['s3'] = replicas[i2]['s3'], replicas[i1]['s3']
                replicas[i1]['s4'], replicas[i2]['s4'] = replicas[i2]['s4'], replicas[i1]['s4']
                replicas[i1]['score'], replicas[i2]['score'] = replicas[i2]['score'], replicas[i1]['score']
                
                # Update best if needed
                if replicas[i1]['s4'] == 0 and replicas[i1]['s3'] > best_valid_score:
                    best_valid_score = replicas[i1]['s3']
                    best_valid_set = replicas[i1]['state'].copy()
                if replicas[i2]['s4'] == 0 and replicas[i2]['s3'] > best_valid_score:
                    best_valid_score = replicas[i2]['s3']
                    best_valid_set = replicas[i2]['state'].copy()

    # Final polish: Run a few steps of local search on the best set
    if best_valid_set:
        # Try to remove elements that don't contribute to 3-APs but might allow adding others?
        # Or just try to add elements that create 3-APs and no 4-APs.
        # Let's do a simple greedy add pass.
        for x in range(N):
            if not best_valid_set[x]:
                # Check if adding x creates a 4-AP
                creates_4ap = False
                for o1, o2, o3 in index_4aps[x]:
                    if best_valid_set[o1] and best_valid_set[o2] and best_valid_set[o3]:
                        creates_4ap = True
                        break
                if not creates_4ap:
                    # Check if it adds 3-APs
                    adds_3ap = False
                    for o1, o2 in index_3aps[x]:
                        if best_valid_set[o1] and best_valid_set[o2]:
                            adds_3ap = True
                            break
                    if adds_3ap:
                        best_valid_set[x] = True
                        # Update best score? We just want the set.
                        # The evaluator calculates the score.
                        
    # Convert to 1-based indices
    return [i + 1 for i in range(N) if best_valid_set[i]]

# Test locally? No, just output.
if __name__ == "__main__":
    # Quick sanity check
    res = construct()
    print(f"Size: {len(res)}")
