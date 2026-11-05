"""
Search-based optimizer for two-colourings of [1,256] minimizing monochromatic 4-APs.
Combines multi-restart simulated annealing with greedy polish and fast delta computation.
"""
import time
import random
import math

def construct():
    N = 256
    deadline = time.time() + 110  # Safety margin before 120s limit

    # Precompute all 4-APs and their adjacency structure
    # For fast delta calculation, store for each position the "others" in each AP it belongs to
    aps_at = [[] for _ in range(N)]
    
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(max_a):
            idxs = (a, a + d, a + 2 * d, a + 3 * d)
            for i in range(4):
                cur = idxs[i]
                others = [idxs[j] for j in range(4) if j != i]
                aps_at[cur].append(others)

    def count_mono(cols):
        """Count monochromatic 4-APs."""
        cnt = 0
        for d in range(1, N // 3 + 1):
            max_a = N - 3 * d
            for a in range(max_a):
                i0, i1, i2, i3 = a, a + d, a + 2 * d, a + 3 * d
                if cols[i0] == cols[i1] == cols[i2] == cols[i3]:
                    cnt += 1
        return cnt

    def delta_for_flip(cols, i):
        """Calculate change in mono count if we flip color at position i."""
        c_i = cols[i]
        delta = 0
        for others in aps_at[i]:
            j, k, l = others
            c_j = cols[j]
            c_k = cols[k]
            c_l = cols[l]
            # If the other three are all the same color:
            if c_j == c_k == c_l:
                if c_i == c_j:
                    # Was monochromatic, won't be after flip
                    delta -= 1
                else:
                    # Wasn't monochromatic, will be after flip
                    delta += 1
        return delta

    def greedy_polish(colors, current_cost):
        """Iteratively apply any single flip that reduces cost until no improvement."""
        improved = True
        while improved:
            improved = False
            for i in range(N):
                d = delta_for_flip(colors, i)
                if d < 0:
                    colors[i] = 1 - colors[i]
                    current_cost += d
                    improved = True
        return colors, current_cost

    def simulated_annealing(initial_colors, T_start, cooling, min_T, time_limit):
        """Run simulated annealing with the given parameters."""
        colors = initial_colors[:]
        current_cost = count_mono(colors)
        best_cost = current_cost
        best_colors = colors[:]
        
        T = T_start
        end_time = time.time() + time_limit
        
        while time.time() < end_time:
            i = random.randint(0, N - 1)
            delta = delta_for_flip(colors, i)
            
            if delta < 0:
                # Always accept improvements
                colors[i] = 1 - colors[i]
                current_cost += delta
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_colors = colors[:]
            elif delta == 0:
                # Accept neutral moves to explore
                colors[i] = 1 - colors[i]
            else:
                # Probabilistically accept worse moves
                if T > min_T:
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        colors[i] = 1 - colors[i]
                        current_cost += delta
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_colors = colors[:]
            
            T *= cooling
            if T < min_T:
                T = min_T
        
        return best_colors, best_cost

    # Global best tracking
    global_best_colors = None
    global_best_cost = float('inf')

    # Multiple restarts with diverse initializations
    num_restarts = 8
    time_per_restart = 110.0 / num_restarts

    for restart in range(num_restarts):
        if time.time() > deadline:
            break

        # Diverse initial colorings
        if restart == 0:
            initial = [0] * N
        elif restart == 1:
            initial = [1] * N
        elif restart == 2:
            initial = [i % 2 for i in range(N)]
        elif restart == 3:
            initial = [(i // 8) % 2 for i in range(N)]
        elif restart == 4:
            initial = [random.randint(0, 1) for _ in range(N)]
        else:
            # Perturb previous best for diversity
            if global_best_colors is not None:
                initial = global_best_colors[:]
                for _ in range(N // 4):
                    idx = random.randint(0, N - 1)
                    initial[idx] = 1 - initial[idx]
            else:
                initial = [random.randint(0, 1) for _ in range(N)]

        # Vary SA parameters slightly for diversity
        T_start = 1.0 + 0.5 * (restart % 3)
        cooling = 0.9999 - 0.00001 * (restart % 5)
        min_T = 0.005

        # Run SA with 80% of time budget, leaving room for polish
        sa_colors, sa_cost = simulated_annealing(
            initial, T_start, cooling, min_T, time_per_restart * 0.8
        )

        # Greedy polish
        polished_colors, polished_cost = greedy_polish(sa_colors, sa_cost)

        if polished_cost < global_best_cost:
            global_best_cost = polished_cost
            global_best_colors = polished_colors[:]

    # Final polish on the best result
    if global_best_colors is not None:
        final_colors, final_cost = greedy_polish(global_best_colors, global_best_cost)
        if final_cost < global_best_cost:
            global_best_cost = final_cost
            global_best_colors = final_colors

    return global_best_colors

# For quick testing
if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    t1 = time.time()
    print(f"Time: {t1-t0:.2f}s")
    print(f"Colors: {result[:20]}...")
    # Verify length and values
    assert len(result) == 256
    assert all(c in (0, 1) for c in result)
