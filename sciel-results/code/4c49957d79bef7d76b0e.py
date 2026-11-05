import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 5s safety margin
    
    n = 512
    
    # Precompute all 4-APs
    # For each x, store a list of masks representing the other 3 elements of APs containing x
    conflict_triples = [[] for _ in range(n + 1)]
    
    # Generate all 4-APs (a, a+d, a+2d, a+3d)
    for a in range(1, n + 1):
        max_d = (n - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            term4 = a + 3 * d
            
            mask_a = (1 << (b - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[a].append(mask_a)
            
            mask_b = (1 << (a - 1)) | (1 << (c - 1)) | (1 << (term4 - 1))
            conflict_triples[b].append(mask_b)
            
            mask_c = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (term4 - 1))
            conflict_triples[c].append(mask_c)
            
            mask_d = (1 << (a - 1)) | (1 << (b - 1)) | (1 << (c - 1))
            conflict_triples[term4].append(mask_d)
    
    # Convert to tuples for faster iteration
    conflict_triples = [tuple(lst) for lst in conflict_triples]
    
    # Precompute bit positions as a list for faster indexing (1-indexed)
    bit_positions = [0] * (n + 1)
    for i in range(1, n + 1):
        bit_positions[i] = 1 << (i - 1)
    
    def can_add(x, S_mask):
        """Check if x can be added to S without forming a 4-AP."""
        for mask in conflict_triples[x]:
            if (mask & S_mask) == mask:
                return False
        return True

    def try_add_all(S_mask, S_list):
        """Try to add all possible elements in natural order. Returns (new_mask, new_list)."""
        added = 0
        for x in range(1, n + 1):
            if not (S_mask & bit_positions[x]):
                if can_add(x, S_mask):
                    S_mask |= bit_positions[x]
                    S_list.append(x)
                    added += 1
        return S_mask, S_list

    best_mask = 0
    best_size = 0
    all_elements = list(range(1, n + 1))
    
    # 1. Initial Greedy Constructions
    # Try natural, reverse, and many random orders
    greedy_orders = [
        list(range(1, n + 1)),
        list(range(n, 0, -1)),
    ]
    # Increase number of random starts to find a better initial point
    for _ in range(30):
        shuffled = all_elements[:]
        random.shuffle(shuffled)
        greedy_orders.append(shuffled)
        
    for order in greedy_orders:
        S_mask = 0
        S_list = []
        for x in order:
            if can_add(x, S_mask):
                S_mask |= bit_positions[x]
                S_list.append(x)
        
        # Polish
        S_mask, S_list = try_add_all(S_mask, S_list)
        current_size = len(S_list)
        if current_size > best_size:
            best_size = current_size
            best_mask = S_mask
    
    # If we haven't found anything (shouldn't happen), return trivial
    if best_size == 0:
        return [1]

    # 2. Iterated Local Search (ILS) with Simulated Annealing
    # We will run SA, and if stuck, perturb the best solution and restart.
    
    def run_sa(initial_mask, initial_list, max_time):
        """Run SA on the given state until max_time or stuck."""
        S_mask = initial_mask
        S_list = initial_list[:]
        current_size = len(S_list)
        
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.999
        sa_iter = 0
        no_improve = 0
        max_no_improve = 500
        
        # Local best within this SA run
        local_best_mask = S_mask
        local_best_size = current_size
        
        end_time = time.time() + max_time
        
        while time.time() < end_time:
            if not S_list:
                break
                
            # Choose move: 1-opt (swap)
            # We want to find a swap that allows adding more elements later.
            # Strategy: Pick a random y in S. Pick a random x not in S.
            # Check if swapping y->x is valid (i.e., x can be added to S\{y}).
            # If valid, perform swap.
            
            y_idx = random.randint(0, len(S_list) - 1)
            y = S_list[y_idx]
            y_bit = bit_positions[y]
            
            x_added = False
            # Try a few random x candidates
            for _ in range(20):
                x = random.randint(1, n)
                if S_mask & bit_positions[x]:
                    continue
                    
                S_mask_without_y = S_mask & ~y_bit
                if can_add(x, S_mask_without_y):
                    S_mask_new = S_mask_without_y | bit_positions[x]
                    S_list[y_idx] = x
                    S_mask = S_mask_new
                    x_added = True
                    break
            
            if x_added:
                # After swap, try to add more
                # We do a light pass: try to add a few random elements
                # This is faster than full try_add_all
                improved = False
                for _ in range(10):
                    x = random.randint(1, n)
                    if not (S_mask & bit_positions[x]):
                        if can_add(x, S_mask):
                            S_mask |= bit_positions[x]
                            S_list.append(x)
                            current_size += 1
                            improved = True
                            
                if improved:
                    if current_size > local_best_size:
                        local_best_size = current_size
                        local_best_mask = S_mask
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
            
            # Full greedy fill every 100 iterations
            if sa_iter % 100 == 0:
                S_mask, S_list = try_add_all(S_mask, S_list)
                current_size = len(S_list)
                if current_size > local_best_size:
                    local_best_size = current_size
                    local_best_mask = S_mask
                    no_improve = 0
            
            sa_iter += 1
            if no_improve > max_no_improve:
                break
                
        return local_best_mask, local_best_size

    # ILS Loop
    current_best_mask = best_mask
    current_best_size = best_size
    current_list = [i+1 for i in range(n) if (current_best_mask >> i) & 1]
    
    while time.time() < deadline:
        # Run SA for a short duration, e.g., 5 seconds
        sa_time = 5.0
        if time.time() + sa_time > deadline:
            sa_time = deadline - time.time() - 1.0
            if sa_time < 1.0:
                break
        
        new_mask, new_size = run_sa(current_best_mask, current_list, sa_time)
        
        if new_size > current_best_size:
            current_best_size = new_size
            current_best_mask = new_mask
            current_list = [i+1 for i in range(n) if (current_best_mask >> i) & 1]
            # If we improved, keep searching from here
        else:
            # If no improvement, perturb the best solution
            # Remove 5-10 random elements
            perturb_k = random.randint(5, 10)
            if len(current_list) > perturb_k:
                indices_to_remove = random.sample(range(len(current_list)), perturb_k)
                for idx in sorted(indices_to_remove, reverse=True):
                    x = current_list.pop(idx)
                    current_best_mask &= ~bit_positions[x]
                current_best_size = len(current_list)
                # Update global best if this perturbed state is somehow better? No, it's worse.
                # But we restart SA from this perturbed state to explore new basins.
            else:
                # If set is too small, just restart from a random greedy
                # (Unlikely to happen with good initialization)
                break

    # Final Polish on the global best
    final_list = [i+1 for i in range(n) if (current_best_mask >> i) & 1]
    temp_mask, temp_list = try_add_all(current_best_mask, final_list)
    if len(temp_list) > len(final_list):
        final_list = temp_list

    if not final_list:
        return [1]
        
    return final_list
