import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs and 4-APs
    # 3-APs: (a, b, c) where b-a = c-b
    # 4-APs: (a, b, c, e) where b-a = c-b = e-c
    
    three_aps = []
    four_aps = []
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            three_aps.append((a, b, c))
            
            if a + 3 * d <= N:
                e = a + 3 * d
                four_aps.append((a, b, c, e))
                
    num_3aps = len(three_aps)
    num_4aps = len(four_aps)
    
    # Map element to list of 3-AP indices and 4-AP indices
    # Also map element to its position in the AP tuple for fast count update?
    # Actually, we just need to update cnt[ap_idx] by +/- 1.
    # We don't need to know which position x was in, just that it was part of that AP.
    
    aps3_of = [[] for _ in range(N + 1)]
    for idx, (a, b, c) in enumerate(three_aps):
        aps3_of[a].append(idx)
        aps3_of[b].append(idx)
        aps3_of[c].append(idx)
        
    aps4_of = [[] for _ in range(N + 1)]
    for idx, (a, b, c, e) in enumerate(four_aps):
        aps4_of[a].append(idx)
        aps4_of[b].append(idx)
        aps4_of[c].append(idx)
        aps4_of[e].append(idx)

    # State variables
    cnt3 = [0] * num_3aps
    cnt4 = [0] * num_4aps
    in_set = [False] * (N + 1)
    
    # Helper to calculate current score (number of completed 3-APs)
    # This is sum(1 for c in cnt3 if c == 3)
    # But we can maintain this incrementally.
    # Let's maintain current_score directly.
    
    def add_element(x):
        nonlocal current_score
        if in_set[x]:
            return False
        
        # Check 4-AP validity
        for ap_idx in aps4_of[x]:
            if cnt4[ap_idx] == 3:
                return False
        
        # Calculate gain in 3-APs
        gain = 0
        for ap_idx in aps3_of[x]:
            if cnt3[ap_idx] == 2:
                gain += 1
                
        # Commit
        in_set[x] = True
        current_score += gain
        
        # Update counts
        for ap_idx in aps3_of[x]:
            cnt3[ap_idx] += 1
        for ap_idx in aps4_of[x]:
            cnt4[ap_idx] += 1
            
        return True

    def remove_element(x):
        nonlocal current_score
        if not in_set[x]:
            return False
            
        # Calculate loss in 3-APs
        loss = 0
        for ap_idx in aps3_of[x]:
            if cnt3[ap_idx] == 3:
                loss += 1
                
        # Commit
        in_set[x] = False
        current_score -= loss
        
        # Update counts
        for ap_idx in aps3_of[x]:
            cnt3[ap_idx] -= 1
        for ap_idx in aps4_of[x]:
            cnt4[ap_idx] -= 1
            
        return True

    # Initialization: Greedy
    # Sort elements by potential (degree in 3-APs / degree in 4-APs)
    deg3 = [len(aps3_of[i]) for i in range(N + 1)]
    deg4 = [len(aps4_of[i]) for i in range(N + 1)]
    
    elements = list(range(1, N + 1))
    # Heuristic: prefer elements with high 3-AP density and low 4-AP constraint
    elements.sort(key=lambda x: (deg3[x] / (deg4[x] + 1), deg3[x]), reverse=True)
    
    def greedy_init():
        nonlocal current_score
        # Reset state
        for i in range(N + 1):
            in_set[i] = False
        for i in range(num_3aps):
            cnt3[i] = 0
        for i in range(num_4aps):
            cnt4[i] = 0
        current_score = 0
        
        # Add elements in sorted order if valid
        # We can randomize the order slightly to allow different starts
        shuffled_elements = elements[:]
        random.shuffle(shuffled_elements)
        
        for x in shuffled_elements:
            if add_element(x):
                pass # Added
                
        return current_score

    # Simulated Annealing
    best_score = 0
    best_set = [False] * (N + 1)
    
    # Run multiple instances
    while time.time() < deadline - 5:
        greedy_init()
        current_score = best_score if current_score < best_score else current_score # Actually greedy_init returns score
        
        # Reset best tracking for this instance if we want, but we keep global best
        if current_score > best_score:
            best_score = current_score
            best_set = in_set[:]
            
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        
        # Track elements in set for faster random removal/swap
        # We can maintain a list of active elements
        active_elements = [i for i in range(1, N + 1) if in_set[i]]
        
        while time.time() < deadline - 5:
            if T < T_min:
                break
                
            move_type = random.random()
            accepted = False
            delta = 0
            x_add = -1
            x_rem = -1
            
            if move_type < 0.3: # Add
                x = random.randint(1, N)
                if not in_set[x]:
                    # Check validity
                    valid = True
                    for ap_idx in aps4_of[x]:
                        if cnt4[ap_idx] == 3:
                            valid = False
                            break
                    
                    if valid:
                        # Calculate gain
                        gain = 0
                        for ap_idx in aps3_of[x]:
                            if cnt3[ap_idx] == 2:
                                gain += 1
                            
                        delta = gain
                        if delta > 0 or random.random() < math.exp(delta / T):
                            add_element(x)
                            active_elements.append(x)
                            accepted = True
                            
            elif move_type < 0.6: # Remove
                if active_elements:
                    x = random.choice(active_elements)
                    # Calculate loss
                    loss = 0
                    for ap_idx in aps3_of[x]:
                        if cnt3[ap_idx] == 3:
                            loss += 1
                            
                    delta = -loss
                    if delta > 0 or random.random() < math.exp(delta / T):
                        remove_element(x)
                        active_elements.remove(x)
                        accepted = True
                        
            else: # Swap
                if active_elements:
                    x_rem = random.choice(active_elements)
                    x_add = random.randint(1, N)
                    if not in_set[x_add]:
                        # Tentative removal
                        # We need to check if x_add is valid AFTER removing x_rem
                        # This is tricky with counts.
                        # Simpler: 
                        # 1. Calculate loss of removing x_rem from current set.
                        # 2. Calculate gain of adding x_add to (current set - x_rem).
                        
                        # Loss of x_rem:
                        loss = 0
                        for ap_idx in aps3_of[x_rem]:
                            if cnt3[ap_idx] == 3:
                                loss += 1
                                
                        # To check validity of x_add in (A - x_rem):
                        # A 4-AP is violated if it has 3 other elements in A - x_rem.
                        # Count of other elements in 4-AP j containing x_add:
                        # cnt4[j] includes x_rem if x_rem is in j.
                        # If x_rem is in j, then cnt4[j] - 1 is the count in A - x_rem.
                        # If x_rem is not in j, cnt4[j] is the count in A - x_rem.
                        
                        valid = True
                        for ap_idx in aps4_of[x_add]:
                            # Check if x_rem is in this 4-AP
                            # We can precompute a bitset or set of elements for each 4-AP?
                            # Or just check if x_rem is one of the elements.
                            # Since N is small, we can just check the 4 elements.
                            # But we don't have the elements list handy in the loop.
                            # Let's precompute `four_ap_elements` as a list of tuples.
                            
                            # Optimization: Precompute `four_ap_set`? No, memory.
                            # We can check if x_rem is in the 4-AP by checking if it's in the tuple.
                            # But we don't have the tuple here.
                            
                            # Alternative: 
                            # If x_rem is in ap_idx, current_cnt = cnt4[ap_idx] - 1
                            # Else current_cnt = cnt4[ap_idx]
                            # If current_cnt == 3, invalid.
                            
                            # We need to know if x_rem is in ap_idx.
                            # We can precompute `is_in_4ap[x][ap_idx]`? 
                            # Or just iterate through the 4 elements of the AP.
                            # Let's access the precomputed `four_aps` list.
                            a, b, c, e = four_aps[ap_idx]
                            if x_rem in (a, b, c, e):
                                current_cnt = cnt4[ap_idx] - 1
                            else:
                                current_cnt = cnt4[ap_idx]
                            
                            if current_cnt == 3:
                                valid = False
                                break
                                
                        if valid:
                            # Calculate gain of adding x_add to A - x_rem
                            gain = 0
                            for ap_idx in aps3_of[x_add]:
                                # Count of other elements in 3-AP j containing x_add in A - x_rem
                                a, b, c = three_aps[ap_idx]
                                if x_rem in (a, b, c):
                                    current_cnt = cnt3[ap_idx] - 1
                                else:
                                    current_cnt = cnt3[ap_idx]
                                    
                                if current_cnt == 2:
                                    gain += 1
                            
                            delta = gain - loss
                            if delta > 0 or random.random() < math.exp(delta / T):
                                # Execute swap
                                remove_element(x_rem)
                                active_elements.remove(x_rem)
                                add_element(x_add)
                                active_elements.append(x_add)
                                accepted = True
            
            if accepted:
                if current_score > best_score:
                    best_score = current_score
                    best_set = in_set[:]
            
            T *= cooling

    # Convert best_set to list
    result = []
    for i in range(1, N + 1):
        if best_set[i]:
            result.append(i)
            
    return result
