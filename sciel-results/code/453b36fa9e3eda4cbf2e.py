import time
import random
import numpy as np

def construct():
    n = 256
    # Safe margin: 150s limit, aim for 140s
    deadline = time.time() + 140.0
    
    # 1. Precomputation
    # Generate all 4-term APs (a, a+d, a+2d, a+3d) within [0, n-1]
    ap_list = []
    for d in range(1, (n - 1) // 3 + 1):
        max_a = n - 1 - 3 * d
        for a in range(max_a + 1):
            ap_list.append((a, a + d, a + 2 * d, a + 3 * d))
    
    num_aps = len(ap_list)
    
    # For each bit index, store the indices of APs it participates in
    bit_aps = [[] for _ in range(n)]
    for idx, ap in enumerate(ap_list):
        for bit in ap:
            bit_aps[bit].append(idx)

    def count_mono(coloring):
        """Count monochromatic 4-APs."""
        count = 0
        for ap in ap_list:
            if coloring[ap[0]] == coloring[ap[1]] == coloring[ap[2]] == coloring[ap[3]]:
                count += 1
        return count

    def get_delta(coloring, bit):
        """
        Calculate the change in the number of monochromatic APs 
        if we flip the bit at index 'bit'.
        """
        old_val = coloring[bit]
        new_val = 1 - old_val
        delta = 0
        
        # Iterate over all APs containing this bit
        for ap_idx in bit_aps[bit]:
            ap = ap_list[ap_idx]
            
            # Check if this AP is currently monochromatic
            # Since 'bit' is in ap, and coloring[bit] == old_val,
            # it is mono if all other elements are also old_val
            is_mono_now = True
            for x in ap:
                if x == bit:
                    continue
                if coloring[x] != old_val:
                    is_mono_now = False
                    break
            
            # Check if this AP would be monochromatic after flip
            # 'bit' becomes new_val. So all other elements must be new_val
            is_mono_new = True
            for x in ap:
                if x == bit:
                    continue
                if coloring[x] != new_val:
                    is_mono_new = False
                    break
            
            if is_mono_now and not is_mono_new:
                delta -= 1
            elif not is_mono_now and is_mono_new:
                delta += 1
                
        return delta

    # 2. Initialization Strategies
    def init_random():
        return [random.randint(0, 1) for _ in range(n)]
    
    def init_thue_morse():
        colors = []
        for i in range(n):
            t = i
            cnt = 0
            while t:
                cnt += t & 1
                t >>= 1
            colors.append(cnt % 2)
        return colors

    def init_blocks():
        """Generate a coloring with random blocks of 0s and 1s."""
        colors = []
        i = 0
        while i < n:
            length = random.randint(1, 12)
            val = random.randint(0, 1)
            for _ in range(min(length, n - i)):
                colors.append(val)
            i += length
        return colors

    # 3. Local Search: Greedy Descent
    def greedy_descent(coloring, count):
        """
        Perform greedy descent: flip any bit that reduces the count.
        Continue until no improving move is found.
        """
        improved = True
        # Limit iterations to prevent infinite loops in case of bugs (shouldn't happen)
        max_iter = 1000
        iter_count = 0
        
        while improved and iter_count < max_iter:
            improved = False
            iter_count += 1
            # Shuffle bit order to avoid bias towards lower indices
            bits = list(range(n))
            random.shuffle(bits)
            
            for bit in bits:
                if time.time() > deadline - 2.0:
                    return coloring, count
                
                delta = get_delta(coloring, bit)
                if delta < 0:
                    coloring[bit] = 1 - coloring[bit]
                    count += delta
                    improved = True
                    # Restart the search for improvements from this new state
                    # This ensures we don't miss a different improvement
                    break 
                    
        return coloring, count

    # 4. Tabu Search
    def tabu_search(initial_coloring, initial_count, max_steps=2000):
        """
        Perform Tabu Search starting from a local minimum.
        """
        coloring = initial_coloring[:]
        count = initial_count
        tabu = [0] * n  # tabu[i] = step at which i is no longer tabu
        tabu_tenure = random.randint(30, 60)
        
        current_step = 0
        best_local_count = count
        best_local_coloring = coloring[:]
        
        while current_step < max_steps:
            if time.time() > deadline - 5.0:
                break
            
            best_delta = float('inf')
            best_bit = -1
            found_non_tabu = False
            
            # Find the best move (min delta)
            # We check all 256 bits
            for bit in range(n):
                delta = get_delta(coloring, bit)
                is_tabu = (current_step < tabu[bit])
                
                if not is_tabu:
                    if delta < best_delta:
                        best_delta = delta
                        best_bit = bit
                        found_non_tabu = True
                else:
                    # If we haven't found a non-tabu move yet, track best tabu move
                    # as a fallback (aspiration-like behavior if we allow it, 
                    # but standard tabu forbids unless it improves global best)
                    # For simplicity, we only consider non-tabu moves unless none exist
                    pass
            
            # If all moves are tabu, pick the best one anyway (escape mechanism)
            if not found_non_tabu:
                for bit in range(n):
                    delta = get_delta(coloring, bit)
                    if delta < best_delta:
                        best_delta = delta
                        best_bit = bit
            
            # Apply move
            if best_bit != -1:
                coloring[best_bit] = 1 - coloring[best_bit]
                count += best_delta
                tabu[best_bit] = current_step + tabu_tenure
                
                if count < best_local_count:
                    best_local_count = count
                    best_local_coloring = coloring[:]
            
            current_step += 1
            
        return best_local_coloring, best_local_count

    # 5. Main Optimization Loop
    best_global_count = float('inf')
    best_global_coloring = [0] * n
    
    # Initial runs with different strategies
    strategies = [init_random, init_thue_morse, init_blocks]
    
    # We will run multiple iterations
    # Each iteration:
    # 1. Generate start (or perturb best)
    # 2. Greedy Descent
    # 3. Tabu Search
    
    num_restarts = 50
    
    for i in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Determine starting point
        if i < len(strategies):
            # First few runs: use specific strategies
            strat = strategies[i]
            start_col = strat()
        else:
            # Subsequent runs: 60% perturb best, 40% random
            if random.random() < 0.6 and best_global_count != float('inf'):
                start_col = best_global_coloring[:]
                # Perturb 5-15 bits
                num_flips = random.randint(5, 15)
                for _ in range(num_flips):
                    bit = random.randint(0, n - 1)
                    start_col[bit] = 1 - start_col[bit]
            else:
                start_col = init_random()
        
        start_count = count_mono(start_col)
        
        # Step 1: Greedy Descent
        col, cnt = greedy_descent(start_col, start_count)
        
        if cnt < best_global_count:
            best_global_count = cnt
            best_global_coloring = col[:]
            
        # Step 2: Tabu Search
        # Estimate time budget for Tabu
        time_left = deadline - time.time()
        # Assume ~1000 steps/sec for Tabu
        max_steps = int(time_left * 800)
        if max_steps < 100:
            continue
            
        col_tabu, cnt_tabu = tabu_search(col, cnt, max_steps)
        
        if cnt_tabu < best_global_count:
            best_global_count = cnt_tabu
            best_global_coloring = col_tabu[:]

    return best_global_coloring
