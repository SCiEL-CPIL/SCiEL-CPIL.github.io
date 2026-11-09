import time
import numpy as np
import random

def construct():
    n = 256
    deadline = time.time() + 115.0  # 5s safety margin under 120s limit
    
    # Precompute all 4-term APs: (a, a+d, a+2d, a+3d) with 0-indexed
    aps = []
    for d in range(1, (n - 3) // 3 + 1):
        for a in range(n - 3 * d):
            aps.append((a, a + d, a + 2 * d, a + 3 * d))
    aps = np.array(aps, dtype=np.int32)  # shape (M, 4)
    M = len(aps)
    
    # Build for each position x, the list of AP indices it participates in
    # and its role (0,1,2,3) in that AP
    pos_aps = [[] for _ in range(n)]
    for idx, (a, b, c, dd) in enumerate(aps):
        pos_aps[a].append((idx, 0))
        pos_aps[b].append((idx, 1))
        pos_aps[c].append((idx, 2))
        pos_aps[dd].append((idx, 3))
    
    # Convert to numpy arrays for speed
    pos_ap_indices = []
    pos_ap_roles = []
    for x in range(n):
        if pos_aps[x]:
            arr = np.array(pos_aps[x], dtype=np.int32)
            pos_ap_indices.append(arr[:, 0])
            pos_ap_roles.append(arr[:, 1])
        else:
            pos_ap_indices.append(np.array([], dtype=np.int32))
            pos_ap_roles.append(np.array([], dtype=np.int32))
    
    # Precompute for fast delta_flip: for each position x, store the AP indices
    # and precompute the "other 3 positions" arrays
    pos_other_indices = []  # For position x, list of arrays of the other 3 positions in each AP
    pos_ap_count = []
    for x in range(n):
        idxs = pos_ap_indices[x]
        roles = pos_ap_roles[x]
        if len(idxs) == 0:
            pos_other_indices.append(np.empty((0, 3), dtype=np.int32))
            pos_ap_count.append(0)
        else:
            # For each AP involving x, get the other 3 positions
            # We need to mask out position x from the 4 positions
            all_pos = aps[idxs]  # shape (K, 4)
            # Create a mask: for each row, which column is x?
            # roles tells us the position of x in the AP
            # We can create a mask of shape (K, 4) where mask[k, role[k]] = True
            mask = np.zeros((len(idxs), 4), dtype=bool)
            rows = np.arange(len(idxs))
            mask[rows, roles] = True
            # The other 3 positions are those not masked
            # We need to extract them as a (K, 3) array
            other = np.empty((len(idxs), 3), dtype=np.int32)
            for col in range(4):
                # Positions where x is NOT at column col
                not_at_col = ~mask[:, col]
                # For each such row, the other 3 positions are the ones not at col
                # This is tricky to vectorize simply. Let's do it differently.
                pass
            
            # Simpler: for each AP, the other 3 are just the 4 positions with x removed
            # Since we have the full 4-tuple and know which is x, we can construct
            other = np.empty((len(idxs), 3), dtype=np.int32)
            for k in range(len(idxs)):
                r = roles[k]
                # Get the 4 positions
                p0, p1, p2, p3 = all_pos[k]
                # Remove the one at role r
                if r == 0:
                    other[k] = [p1, p2, p3]
                elif r == 1:
                    other[k] = [p0, p2, p3]
                elif r == 2:
                    other[k] = [p0, p1, p3]
                else:
                    other[k] = [p0, p1, p2]
            pos_other_indices.append(other)
            pos_ap_count.append(len(idxs))
    
    def count_mono(colors):
        """Count monochromatic 4-APs."""
        c0 = colors[aps[:, 0]]
        c1 = colors[aps[:, 1]]
        c2 = colors[aps[:, 2]]
        c3 = colors[aps[:, 3]]
        mono = (c0 == c1) & (c1 == c2) & (c2 == c3)
        return int(mono.sum())
    
    def delta_flip_fast(colors, x):
        """Compute change in cost if we flip position x, using precomputed other positions."""
        if pos_ap_count[x] == 0:
            return 0
        others = pos_other_indices[x]  # shape (K, 3)
        # Get colors of the other 3 positions
        c_others = colors[others]  # shape (K, 3)
        # Check if the other 3 are all the same color
        # And whether that color matches the current color of x
        old_color = colors[x]
        # The other 3 are mono if all 3 are equal
        o0 = c_others[:, 0]
        o1 = c_others[:, 1]
        o2 = c_others[:, 2]
        others_mono = (o0 == o1) & (o1 == o2)
        # If others are mono and same as old_color, then AP was mono before
        # If others are mono and different from old_color, then AP was NOT mono before
        # After flip, new_color = 1 - old_color
        # If others are mono and same as new_color, then AP is mono after
        # If others are mono and different from new_color, then AP is NOT mono after
        
        # Before: mono if others_mono AND (o0 == old_color)
        before_mono = others_mono & (o0 == old_color)
        # After: mono if others_mono AND (o0 == (1 - old_color))
        after_mono = others_mono & (o0 != old_color)
        
        return int(after_mono.sum() - before_mono.sum())
    
    def thue_morse(n):
        return np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)
    
    def block_pattern(n, block_size):
        return np.array([(i // block_size) % 2 for i in range(n)], dtype=np.int8)
    
    def greedy_local_search(colors, max_iters=5000):
        """Greedy: repeatedly flip the position with the best (most negative) delta."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for _ in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            # Compute delta for all positions
            best_delta = 0
            best_x = -1
            for x in range(n):
                d = delta_flip_fast(colors, x)
                if d < best_delta:
                    best_delta = d
                    best_x = x
            if best_x == -1:
                break
            colors[best_x] = 1 - colors[best_x]
            current_cost += best_delta
            if current_cost < best_cost:
                best_cost = current_cost
                best_colors = colors.copy()
        
        return best_colors, best_cost
    
    def simulated_annealing(colors, start_temp=3.0, end_temp=0.05, max_iters=5000):
        """SA: accept worse moves with probability exp(-delta/temp)."""
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors.copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            x = random.randint(0, n - 1)
            d = delta_flip_fast(colors, x)
            if d < 0 or random.random() < np.exp(-d / max(temp, 1e-10)):
                colors[x] = 1 - colors[x]
                current_cost += d
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors.copy()
        
        return best_colors, best_cost
    
    def vectorized_annealing(pop_size=64, start_temp=3.0, end_temp=0.05, max_iters=3000):
        """Vectorized Annealing: evolve a population of configurations simultaneously."""
        # pop: shape (pop_size, n)
        current_costs = np.array([count_mono(pop[i]) for i in range(pop_size)])
        best_cost = int(current_costs.min())
        best_idx = int(np.argmin(current_costs))
        best_colors = pop[best_idx].copy()
        
        for i in range(max_iters):
            if time.time() > deadline - 2.0:
                break
            temp = start_temp * (end_temp / start_temp) ** (i / max_iters)
            
            # For each individual, pick a random position to flip
            flip_positions = np.random.randint(0, n, size=pop_size)
            
            # Compute deltas for all individuals
            deltas = np.zeros(pop_size, dtype=np.int32)
            for k in range(pop_size):
                deltas[k] = delta_flip_fast(pop[k], flip_positions[k])
            
            # Acceptance: accept if delta < 0 or random < exp(-delta/temp)
            accept = (deltas < 0) | (np.random.random(pop_size) < np.exp(-deltas / max(temp, 1e-10)))
            
            # Apply accepted flips
            for k in range(pop_size):
                if accept[k]:
                    pop[k, flip_positions[k]] = 1 - pop[k, flip_positions[k]]
                    current_costs[k] += deltas[k]
                    if current_costs[k] < best_cost:
                        best_cost = int(current_costs[k])
                        best_idx = k
                        best_colors = pop[k].copy()
            
            # Occasionally: copy best to worst to maintain diversity pressure
            if i % 50 == 0 and i > 0:
                worst_idx = int(np.argmax(current_costs))
                pop[worst_idx] = pop[best_idx].copy()
                current_costs[worst_idx] = current_costs[best_idx]
        
        return best_colors, best_cost
    
    # Multi-start: Thue-Morse + block patterns + random
    tm = thue_morse(n)
    best_overall = tm.copy()
    best_cost_overall = count_mono(best_overall)
    
    # Generate diverse initial solutions
    initial_solutions = [tm.copy()]
    for bs in [2, 4, 8, 16, 32, 64]:
        initial_solutions.append(block_pattern(n, bs))
    for _ in range(4):
        initial_solutions.append(np.random.randint(0, 2, size=n).astype(np.int8))
    
    # Phase 1: Greedy + SA on each initial solution
    for sol in initial_solutions:
        if time.time() > deadline - 30.0:
            break
        c, cost = greedy_local_search(sol.copy(), max_iters=2000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
        if time.time() > deadline - 25.0:
            break
        c, cost = simulated_annealing(best_overall.copy(), start_temp=3.0, end_temp=0.05, max_iters=1500)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Phase 2: Vectorized Annealing from best solution
    if time.time() < deadline - 15.0:
        # Create population centered around best solution
        pop_size = 64
        pop = np.empty((pop_size, n), dtype=np.int8)
        pop[0] = best_overall
        for k in range(1, pop_size):
            # Perturb best solution with random flips
            sol = best_overall.copy()
            num_flips = random.randint(1, 15)
            flip_positions = random.sample(range(n), num_flips)
            for x in flip_positions:
                sol[x] = 1 - sol[x]
            pop[k] = sol
        
        c, cost = vectorized_annealing(pop, pop_size=pop_size, start_temp=3.0, end_temp=0.05, max_iters=2000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Phase 3: Final greedy polish
    if time.time() < deadline - 3.0:
        c, cost = greedy_local_search(best_overall.copy(), max_iters=3000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    # Phase 4: Short SA polish
    if time.time() < deadline - 2.0:
        c, cost = simulated_annealing(best_overall.copy(), start_temp=1.0, end_temp=0.01, max_iters=1000)
        if cost < best_cost_overall:
            best_cost_overall = cost
            best_overall = c
    
    return [int(x) for x in best_overall]
