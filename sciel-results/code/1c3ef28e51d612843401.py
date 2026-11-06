import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 82  # 8s safety margin before 90s kill
    
    N = 256
    
    # Precomputation
    # 3-APs: (a, b, c) such that a+c=2b
    three_aps = []
    threes_involving = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            idx = len(three_aps)
            three_aps.append((a, b, c))
            threes_involving[a].append(idx)
            threes_involving[b].append(idx)
            threes_involving[c].append(idx)
            
    # 4-APs: (a, b, c, d) such that a, a+d, a+2d, a+3d
    four_aps = []
    fours_involving = [[] for _ in range(N + 1)]
    
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            idx = len(four_aps)
            four_aps.append((a, b, c, e))
            fours_involving[a].append(idx)
            fours_involving[b].append(idx)
            fours_involving[c].append(idx)
            fours_involving[e].append(idx)

    # Helper functions using bitsets
    # S_bits: bit i is 1 if i in S
    
    def gain_add(S_bits, x):
        """Count how many 3-APs are completed by adding x to S."""
        gain = 0
        for idx in threes_involving[x]:
            a, b, c = three_aps[idx]
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    gain += 1
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1):
                    gain += 1
            else:  # c == x
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1):
                    gain += 1
        return gain

    def loss_remove(S_bits, x):
        """Count how many 3-APs are destroyed by removing x from S."""
        loss = 0
        for idx in threes_involving[x]:
            a, b, c = three_aps[idx]
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    loss += 1
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1):
                    loss += 1
            else:  # c == x
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1):
                    loss += 1
        return loss

    def can_add(S_bits, x):
        """Check if x can be added to S without creating a 4-AP."""
        for idx in fours_involving[x]:
            a, b, c, d = four_aps[idx]
            if a == x:
                if ((S_bits >> b) & 1) and ((S_bits >> c) & 1) and ((S_bits >> d) & 1):
                    return False
            elif b == x:
                if ((S_bits >> a) & 1) and ((S_bits >> c) & 1) and ((S_bits >> d) & 1):
                    return False
            elif c == x:
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1) and ((S_bits >> d) & 1):
                    return False
            else:  # d == x
                if ((S_bits >> a) & 1) and ((S_bits >> b) & 1) and ((S_bits >> c) & 1):
                    return False
        return True

    def greedy_construct(ordering_seed=None):
        """Greedy construction with a given ordering."""
        S_bits = 0
        current_3ap = 0
        
        if ordering_seed is None:
            ordering_seed = random.randint(0, 2**32 - 1)
        
        # Create different orderings based on seed
        rng = random.Random(ordering_seed)
        
        # Different strategies for ordering
        strategy = rng.randint(0, 3)
        
        if strategy == 0:
            # Random order
            remaining = list(range(1, N + 1))
            rng.shuffle(remaining)
        elif strategy == 1:
            # Sort by number of 4-AP constraints (fewer first)
            remaining = sorted(range(1, N + 1), key=lambda x: len(fours_involving[x]))
            rng.shuffle(remaining)  # Add some randomness
        elif strategy == 2:
            # Sort by number of 3-AP potential (more first)
            remaining = sorted(range(1, N + 1), key=lambda x: -len(threes_involving[x]))
            rng.shuffle(remaining)
        else:
            # Weighted random: prefer elements with more 3-AP potential
            remaining = list(range(1, N + 1))
            weights = [len(threes_involving[x]) + 1 for x in range(1, N + 1)]
            # Custom weighted shuffle
            for i in range(len(remaining) - 1, 0, -1):
                total_w = sum(weights[j] for j in range(i + 1))
                r = rng.random() * total_w
                cum = 0
                for j in range(i + 1):
                    cum += weights[j]
                    if r <= cum:
                        remaining[i], remaining[j] = remaining[j], remaining[i]
                        break
        
        for x in remaining:
            if time.time() > deadline:
                break
            if can_add(S_bits, x):
                S_bits |= (1 << x)
                current_3ap += gain_add(S_bits, x)
        
        return S_bits, current_3ap

    def local_search(S_bits, current_3ap, max_iter=3000):
        """Hill climbing with add, remove, and swap moves."""
        best_S = S_bits
        best_3ap = current_3ap
        current_S = S_bits
        current_3ap_val = current_3ap
        
        # Track elements in S for fast random selection
        in_S = [i for i in range(1, N + 1) if (current_S >> i) & 1]
        not_in_S = [i for i in range(1, N + 1) if not (current_S >> i) & 1]
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            move_type = random.randint(0, 2)
            
            if move_type == 0 and not_in_S:
                # Try Add
                x = random.choice(not_in_S)
                if can_add(current_S, x):
                    new_3ap = current_3ap_val + gain_add(current_S, x)
                    if new_3ap >= best_3ap:
                        current_S |= (1 << x)
                        current_3ap_val = new_3ap
                        in_S.append(x)
                        not_in_S.remove(x)
                        if new_3ap > best_3ap:
                            best_S = current_S
                            best_3ap = new_3ap
            
            elif move_type == 1 and in_S:
                # Try Remove (only if it helps enable future adds)
                x = random.choice(in_S)
                loss = loss_remove(current_S, x)
                new_3ap = current_3ap_val - loss
                # Accept with some probability to escape local optima
                if new_3ap >= best_3ap or random.random() < 0.05:
                    current_S &= ~(1 << x)
                    current_3ap_val = new_3ap
                    in_S.remove(x)
                    not_in_S.append(x)
                    if new_3ap > best_3ap:
                        best_S = current_S
                        best_3ap = new_3ap
            
            elif move_type == 2 and in_S and not_in_S:
                # Try Swap
                x_out = random.choice(in_S)
                x_in = random.choice(not_in_S)
                
                S_temp = current_S & ~(1 << x_out)
                if can_add(S_temp, x_in):
                    loss = loss_remove(current_S, x_out)
                    gain = gain_add(S_temp, x_in)
                    new_3ap = current_3ap_val - loss + gain
                    
                    if new_3ap > best_3ap:
                        current_S = S_temp | (1 << x_in)
                        current_3ap_val = new_3ap
                        in_S.remove(x_out)
                        in_S.append(x_in)
                        not_in_S.remove(x_in)
                        not_in_S.append(x_out)
                        best_S = current_S
                        best_3ap = new_3ap
                    elif new_3ap == best_3ap and random.random() < 0.1:
                        current_S = S_temp | (1 << x_in)
                        current_3ap_val = new_3ap
                        in_S.remove(x_out)
                        in_S.append(x_in)
                        not_in_S.remove(x_in)
                        not_in_S.append(x_out)
                        best_S = current_S
                        best_3ap = new_3ap
        
        return best_S, best_3ap

    # Run multiple trials with different orderings
    best_S_global = 0
    best_3ap_global = 0
    
    num_trials = 0
    for trial in range(30):
        if time.time() > deadline:
            break
        
        # Greedy start with different ordering
        S_bits, current_3ap = greedy_construct(ordering_seed=trial * 7919 + 1)
        
        # Local search
        S_bits, current_3ap = local_search(S_bits, current_3ap, max_iter=1500)
        
        if current_3ap > best_3ap_global:
            best_3ap_global = current_3ap
            best_S_global = S_bits
            num_trials = trial
        
        # If we're running low on time, do a final quick local search on best
        if time.time() > deadline - 10 and best_3ap_global > 0:
            # Quick polish
            S_bits, current_3ap = local_search(best_S_global, best_3ap_global, max_iter=500)
            if current_3ap > best_3ap_global:
                best_3ap_global = current_3ap
                best_S_global = S_bits
            break

    # Convert bitset to list
    result_set = []
    for i in range(1, N + 1):
        if (best_S_global >> i) & 1:
            result_set.append(i)
            
    if not result_set:
        result_set = [1]
        
    return sorted(result_set)
