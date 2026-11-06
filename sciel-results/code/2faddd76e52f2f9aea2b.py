import time
import random
import math

def construct():
    deadline = time.time() + 82  # Safety margin before 90s kill
    
    N = 256
    M = N + 1  # 1-indexed
    
    # Precompute 3-APs and 4-APs
    three_aps = []
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            three_aps.append((a, a + d, a + 2 * d))
    
    four_aps = []
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            four_aps.append((a, a + d, a + 2 * d, a + 3 * d))
    
    # Data structures for fast lookup
    elem_to_3aps = [[] for _ in range(M)]
    for i, (a, b, c) in enumerate(three_aps):
        elem_to_3aps[a].append((i, 0))
        elem_to_3aps[b].append((i, 1))
        elem_to_3aps[c].append((i, 2))
        
    elem_to_4aps = [[] for _ in range(M)]
    for i, (a, b, c, d) in enumerate(four_aps):
        elem_to_4aps[a].append((i, 0))
        elem_to_4aps[b].append((i, 1))
        elem_to_4aps[c].append((i, 2))
        elem_to_4aps[d].append((i, 3))
        
    # State
    in_set = [False] * M
    current_score = 0
    
    # Helper to check validity of adding x
    def try_add(x):
        if in_set[x]:
            return False, 0
        # Check 4-AP constraints
        for ap_idx, pos in elem_to_4aps[x]:
            a, b, c, d = four_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c] and in_set[d]:
                    return False, 0
            elif pos == 1:
                if in_set[a] and in_set[c] and in_set[d]:
                    return False, 0
            elif pos == 2:
                if in_set[a] and in_set[b] and in_set[d]:
                    return False, 0
            else:
                if in_set[a] and in_set[b] and in_set[c]:
                    return False, 0
                    
        # Calculate score delta
        delta = 0
        for ap_idx, pos in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c]:
                    delta += 1
            elif pos == 1:
                if in_set[a] and in_set[c]:
                    delta += 1
            else:
                if in_set[a] and in_set[b]:
                    delta += 1
        return True, delta

    def try_remove(x):
        if not in_set[x]:
            return False, 0
        delta = 0
        for ap_idx, pos in elem_to_3aps[x]:
            a, b, c = three_aps[ap_idx]
            if pos == 0:
                if in_set[b] and in_set[c]:
                    delta += 1
            elif pos == 1:
                if in_set[a] and in_set[c]:
                    delta += 1
            else:
                if in_set[a] and in_set[b]:
                    delta += 1
        return True, -delta

    # Greedy Initialization with lookahead
    # Sort by potential: number of 3-APs minus number of 4-APs
    potential = []
    for x in range(1, N + 1):
        pot_3 = len(elem_to_3aps[x])
        pot_4 = len(elem_to_4aps[x])
        # Heuristic: prefer high 3-AP potential, low 4-AP potential
        potential.append((pot_3 - 0.5 * pot_4, x))
    
    potential.sort(reverse=True)
    
    for _, x in potential:
        valid, delta = try_add(x)
        if valid:
            in_set[x] = True
            current_score += delta

    best_score = current_score
    best_in_set = in_set[:]
    
    # Simulated Annealing with better move generation
    random.seed(42)
    T = 10.0
    cooling = 0.99995
    min_T = 0.005
    all_elements = list(range(1, N + 1))
    
    # Precompute neighbor lists for faster sampling? No, just use lists.
    
    while time.time() < deadline:
        if T < min_T:
            T = min_T
            
        move_type = random.random()
        
        if move_type < 0.3:  # Add
            candidates = [x for x in all_elements if not in_set[x]]
            if candidates:
                # Weighted random selection? Or just uniform.
                # Let's try uniform first, it's simpler.
                x = random.choice(candidates)
                valid, delta = try_add(x)
                if valid:
                    # Acceptance criterion
                    if delta > 0:
                        accept = True
                    elif delta == 0:
                        accept = random.random() < 0.2 # Less likely to add 0-delta
                    else:
                        accept = random.random() < math.exp(delta / T)
                    
                    if accept:
                        in_set[x] = True
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_in_set = in_set[:]
                            
        elif move_type < 0.5:  # Remove
            in_list = [x for x in all_elements if in_set[x]]
            if in_list:
                x = random.choice(in_list)
                valid, delta = try_remove(x)
                if valid:
                    if random.random() < math.exp(delta / T):
                        in_set[x] = False
                        current_score += delta
                        
        else:  # Swap
            in_list = [x for x in all_elements if in_set[x]]
            out_list = [x for x in all_elements if not in_set[x]]
            if in_list and out_list:
                x_out = random.choice(in_list)
                x_in = random.choice(out_list)
                
                # 1. Calculate loss from x_out in current state
                loss = 0
                for ap_idx, pos in elem_to_3aps[x_out]:
                    a, b, c = three_aps[ap_idx]
                    if pos == 0:
                        if in_set[b] and in_set[c]:
                            loss += 1
                    elif pos == 1:
                        if in_set[a] and in_set[c]:
                            loss += 1
                    else:
                        if in_set[a] and in_set[b]:
                            loss += 1
                            
                # 2. Temporarily remove x_out
                in_set[x_out] = False
                
                # 3. Check validity of x_in
                valid_add, delta_add = try_add(x_in)
                
                if valid_add:
                    net_delta = delta_add - loss
                    # Acceptance
                    if net_delta > 0:
                        accept = True
                    elif net_delta == 0:
                        accept = random.random() < 0.3
                    else:
                        accept = random.random() < math.exp(net_delta / T)
                    
                    if accept:
                        in_set[x_in] = True
                        current_score += net_delta
                        if current_score > best_score:
                            best_score = current_score
                            best_in_set = in_set[:]
                    else:
                        # Revert
                        in_set[x_out] = True
                else:
                    # Revert
                    in_set[x_out] = True

        T *= cooling

    # Final Polish: Hill Climbing
    improved = True
    while improved and time.time() < deadline - 2:
        improved = False
        # Try additions
        for x in all_elements:
            if not in_set[x]:
                valid, delta = try_add(x)
                if valid and delta > 0:
                    in_set[x] = True
                    current_score += delta
                    improved = True
                    if current_score > best_score:
                        best_score = current_score
                        best_in_set = in_set[:]
        
        # Try swaps
        in_set_list = [x for x in all_elements if in_set[x]]
        not_in_set_list = [x for x in all_elements if not in_set[x]]
        
        # Optimize swap search: only check promising swaps?
        # For now, just iterate.
        found_improvement = False
        for x_out in in_set_list:
            # Calculate loss for x_out once? No, it depends on x_in potentially?
            # Loss is independent of x_in.
            loss = 0
            for ap_idx, pos in elem_to_3aps[x_out]:
                a, b, c = three_aps[ap_idx]
                if pos == 0:
                    if in_set[b] and in_set[c]:
                        loss += 1
                elif pos == 1:
                    if in_set[a] and in_set[c]:
                        loss += 1
                else:
                    if in_set[a] and in_set[b]:
                        loss += 1
                        
            for x_in in not_in_set_list:
                # Check validity
                in_set[x_out] = False
                valid_add, delta_add = try_add(x_in)
                if valid_add:
                    net_delta = delta_add - loss
                    if net_delta > 0:
                        in_set[x_in] = True
                        current_score += net_delta
                        improved = True
                        found_improvement = True
                        if current_score > best_score:
                            best_score = current_score
                            best_in_set = in_set[:]
                        break
                    else:
                        in_set[x_out] = True
                else:
                    in_set[x_out] = True
            if found_improvement:
                break

    # Convert best_in_set to sorted list
    result = []
    for x in range(1, N + 1):
        if best_in_set[x]:
            result.append(x)
    return sorted(result)
