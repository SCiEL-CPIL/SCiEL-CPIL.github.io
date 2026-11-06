import time
import random
import math

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    N = 256
    
    # Precompute all 3-APs and 4-APs data
    # all_3aps_data: list of (mask, a, b, c)
    # all_4aps_data: list of (mask, a, b, c, e)
    all_3aps_data = []
    all_4aps_data = []
    
    for a in range(1, N + 1):
        max_d_3 = (N - a) // 2
        for d in range(1, max_d_3 + 1):
            b = a + d
            c = a + 2 * d
            mask3 = (1 << a) | (1 << b) | (1 << c)
            all_3aps_data.append((mask3, a, b, c))
            
            if a + 3 * d <= N:
                e = a + 3 * d
                mask4 = (1 << a) | (1 << b) | (1 << c) | (1 << e)
                all_4aps_data.append((mask4, a, b, c, e))
                
    # Build 'others' masks for fast incremental updates
    # three_ap_others[x] = list of masks of the other 2 elements in 3-APs with x
    three_ap_others = [[] for _ in range(N + 1)]
    for mask3, a, b, c in all_3aps_data:
        three_ap_others[a].append((1 << b) | (1 << c))
        three_ap_others[b].append((1 << a) | (1 << c))
        three_ap_others[c].append((1 << a) | (1 << b))
        
    # four_ap_others[x] = list of masks of the other 3 elements in 4-APs with x
    four_ap_others = [[] for _ in range(N + 1)]
    for mask4, a, b, c, e in all_4aps_data:
        four_ap_others[a].append((1 << b) | (1 << c) | (1 << e))
        four_ap_others[b].append((1 << a) | (1 << c) | (1 << e))
        four_ap_others[c].append((1 << a) | (1 << b) | (1 << e))
        four_ap_others[e].append((1 << a) | (1 << b) | (1 << c))

    # Helper: Check if adding x is valid (no 4-AP created)
    def is_valid_add(A_mask, x):
        if A_mask & (1 << x):
            return False
        for other_mask in four_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                return False
        return True

    # Helper: Calculate gain in 3-APs if x is added
    def gain_3aps(A_mask, x):
        if A_mask & (1 << x):
            return 0
        gain = 0
        for other_mask in three_ap_others[x]:
            if (A_mask & other_mask) == other_mask:
                gain += 1
        return gain

    # Helper: Calculate loss in 3-APs if x is removed
    def loss_3aps(A_mask, x):
        if not (A_mask & (1 << x)):
            return 0
        loss = 0
        A_mask_without_x = A_mask & ~(1 << x)
        for other_mask in three_ap_others[x]:
            if (A_mask_without_x & other_mask) == other_mask:
                loss += 1
        return loss

    # Count total 3-APs in A_mask (full scan)
    def count_3aps(A_mask):
        count = 0
        for mask3, a, b, c in all_3aps_data:
            if (A_mask & mask3) == mask3:
                count += 1
        return count

    # Initialization: Degree-based Greedy
    # Calculate degree (number of 3-APs) and constraint (number of 4-APs)
    deg3 = [0] * (N + 1)
    deg4 = [0] * (N + 1)
    for x in range(1, N + 1):
        deg3[x] = len(three_ap_others[x])
        deg4[x] = len(four_ap_others[x])
        
    # Sort by density potential: high 3-AP degree, low 4-AP degree
    elements = list(range(1, N + 1))
    elements.sort(key=lambda x: (deg3[x] / (deg4[x] + 1), deg3[x]), reverse=True)
    
    def greedy_init():
        A_mask = 0
        # Use sorted order for deterministic high-quality start
        for x in elements:
            if is_valid_add(A_mask, x):
                # Add if it increases 3-AP count, or with some probability to allow variation
                g = gain_3aps(A_mask, x)
                if g > 0 or random.random() < 0.1:
                    A_mask |= (1 << x)
        return A_mask

    def get_elements_in_mask(mask):
        elems = []
        for i in range(1, N + 1):
            if mask & (1 << i):
                elems.append(i)
        return elems

    # Simulated Annealing
    def simulated_annealing(initial_mask, initial_score, time_limit):
        current_mask = initial_mask
        current_score = initial_score
        best_mask = current_mask
        best_score = current_score
        
        T = 10.0
        T_min = 0.1
        cooling = 0.999
        
        # Precompute list of elements in mask for faster removal/swap selection
        # We update this list incrementally or rebuild if it gets too stale
        # For simplicity and robustness against bugs, we rebuild or use a helper.
        # Given N=256, rebuilding is fast enough if not done every single step.
        # However, for SA we need random access to elements in A.
        # Let's maintain a list `elems_in_A` and update it.
        
        elems_in_A = get_elements_in_mask(current_mask)
        
        while time.time() < time_limit:
            if T < T_min:
                break
                
            move_type = random.random()
            accepted = False
            delta = 0
            new_mask = current_mask
            new_score = current_score
            new_elems_in_A = elems_in_A
            
            if move_type < 0.3: # Add
                x = random.randint(1, N)
                if not (current_mask & (1 << x)):
                    if is_valid_add(current_mask, x):
                        delta = gain_3aps(current_mask, x)
                        if delta > 0 or random.random() < math.exp(delta / T):
                            new_mask = current_mask | (1 << x)
                            new_score = current_score + delta
                            new_elems_in_A = elems_in_A + [x]
                            accepted = True
                            
            elif move_type < 0.6: # Remove
                if elems_in_A:
                    x = random.choice(elems_in_A)
                    delta = -loss_3aps(current_mask, x)
                    # Removal is always valid
                    if delta > 0 or random.random() < math.exp(delta / T):
                        new_mask = current_mask & ~(1 << x)
                        new_score = current_score + delta
                        # Remove x from list
                        new_elems_in_A = [e for e in elems_in_A if e != x]
                        accepted = True
                        
            else: # Swap
                if elems_in_A:
                    x_rem = random.choice(elems_in_A)
                    x_add = random.randint(1, N)
                    if not (current_mask & (1 << x_add)):
                        # Check validity of swap
                        # A_temp = current_mask without x_rem
                        A_temp = current_mask & ~(1 << x_rem)
                        valid = True
                        for other_mask in four_ap_others[x_add]:
                            if (A_temp & other_mask) == other_mask:
                                valid = False
                                break
                                
                        if valid:
                            # Calculate net gain
                            gain = 0
                            for other_mask in three_ap_others[x_add]:
                                if (A_temp & other_mask) == other_mask:
                                    gain += 1
                                    
                            loss = loss_3aps(current_mask, x_rem)
                            delta = gain - loss
                            
                            if delta > 0 or random.random() < math.exp(delta / T):
                                new_mask = A_temp | (1 << x_add)
                                new_score = current_score + delta
                                # Update list: remove x_rem, add x_add
                                new_elems_in_A = [e for e in elems_in_A if e != x_rem]
                                new_elems_in_A.append(x_add)
                                accepted = True
            
            if accepted:
                current_mask = new_mask
                current_score = new_score
                elems_in_A = new_elems_in_A
                
                if current_score > best_score:
                    best_score = current_score
                    best_mask = current_mask
            
            T *= cooling
            
        return best_mask, best_score

    # Main Logic
    best_overall_mask = 0
    best_overall_score = 0
    
    # Run multiple SA instances
    # We allocate time to each instance, or run until deadline
    # Strategy: Run one long SA, or multiple shorter ones with perturbations?
    # ILS approach: Run SA, then perturb, run SA again.
    
    while time.time() < deadline - 2.0:
        # Initialize
        if best_overall_mask == 0:
            current_mask = greedy_init()
            current_score = count_3aps(current_mask)
        else:
            # Perturb the best solution found so far
            # Remove 2-5 random elements to escape local optimum
            elems_in_best = get_elements_in_mask(best_overall_mask)
            if len(elems_in_best) > 5:
                k = random.randint(2, 5)
                to_remove = random.sample(elems_in_best, k)
                current_mask = best_overall_mask
                for x in to_remove:
                    current_mask &= ~(1 << x)
                current_score = count_3aps(current_mask)
            else:
                # If set is small, just re-initialize
                current_mask = greedy_init()
                current_score = count_3aps(current_mask)
        
        # Run SA with a sub-deadline
        # Leave 2 seconds for final conversion and safety
        sa_deadline = min(deadline - 2.0, time.time() + 10.0) 
        # If we have less than 10s left, use all of it. Otherwise, chunk it.
        # Actually, just run until close to deadline is safer for finding improvements.
        # But if we restart, we might lose progress.
        # Let's run SA until time.time() > deadline - 2.0
        
        l_mask, l_score = simulated_annealing(current_mask, current_score, deadline - 2.0)
        
        if l_score > best_overall_score:
            best_overall_score = l_score
            best_overall_mask = l_mask
            
    # Convert best mask to sorted list
    result = []
    for i in range(1, N + 1):
        if best_overall_mask & (1 << i):
            result.append(i)
            
    return result
