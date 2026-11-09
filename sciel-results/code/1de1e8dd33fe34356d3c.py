import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s/150s kill
    n = 256

    # Precompute all 4-term APs as index tuples (0-indexed)
    aps = []
    for d in range(1, 86):
        max_a = n - 3 * d
        for a in range(1, max_a + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            aps.append((i0, i1, i2, i3))
            
    M = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)

    # Build per-position list of AP indices and the indices of the OTHER elements in those APs
    # pos_aps[p] = list of ap indices where p appears
    # pos_others[p] = list of tuples of the other 3 indices for each ap in pos_aps[p]
    pos_aps = [[] for _ in range(n)]
    pos_others = [[] for _ in range(n)]
    
    for idx, ap in enumerate(aps):
        for k in range(4):
            p = ap[k]
            pos_aps[p].append(idx)
            others = [ap[j] for j in range(4) if j != k]
            pos_others[p].append(others)

    # Precompute flat arrays for vectorized delta computation
    # For each position, store the "others" as a 2D array of shape (num_aps, 3)
    pos_others_arrays = [None] * n
    for p in range(n):
        if pos_others[p]:
            pos_others_arrays[p] = np.array(pos_others[p], dtype=np.int32)
        else:
            pos_others_arrays[p] = np.empty((0, 3), dtype=np.int32)

    # Thue-Morse initialization: c[i] = popcount(i) % 2
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(col):
        a0 = col[aps_arr[:, 0]]
        a1 = col[aps_arr[:, 1]]
        a2 = col[aps_arr[:, 2]]
        a3 = col[aps_arr[:, 3]]
        mono = (a0 == a1) & (a1 == a2) & (a2 == a3)
        return int(np.sum(mono))

    def get_delta(pos, col):
        """Calculate delta in mono count if we flip col[pos]."""
        delta = 0
        others_list = pos_others[pos]
        for others in others_list:
            c1 = col[others[0]]
            c2 = col[others[1]]
            c3 = col[others[2]]
            if c1 == c2 and c2 == c3:
                if col[pos] == c1:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def vectorized_deltas(col):
        """Compute delta for flipping each position, vectorized using precomputed arrays."""
        all_deltas = np.zeros(n, dtype=np.int32)
        
        for pos in range(n):
            others_arr = pos_others_arrays[pos]
            if len(others_arr) == 0:
                continue
            
            # Get the colors of the other elements
            o1 = col[others_arr[:, 0]]
            o2 = col[others_arr[:, 1]]
            o3 = col[others_arr[:, 2]]
            
            # Check if all three others are the same color
            all_same = (o1 == o2) & (o2 == o3)
            
            # If all same, delta depends on whether col[pos] matches
            match = (col[pos] == o1)
            
            # delta = -1 if match (removing mono), +1 if not match (creating mono)
            delta = np.where(all_same, np.where(match, -1, 1), 0)
            all_deltas[pos] = int(np.sum(delta))
        
        return all_deltas

    def greedy_polish(col, time_limit=None):
        """Run greedy local search until no improvement or time limit."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        while improved and (time.time() - start_time) < (deadline - 5.0 if time_limit is None else time_limit):
            improved = False
            # Compute all deltas at once for efficiency
            deltas = vectorized_deltas(col)
            # Find the most negative delta
            min_delta = int(np.min(deltas))
            if min_delta < 0:
                # Flip the position with the most negative delta
                pos = int(np.argmin(deltas))
                col[pos] = 1 - col[pos]
                cur_count += min_delta
                improved = True
        return col, cur_count

    # Phase 1: Greedy polish from Thue-Morse
    c, best_count = greedy_polish(c)
    best_c = c.copy()

    # Phase 2: Simulated Annealing with improved restart strategy
    T = 2.0
    cooling = 0.9998
    min_T = 0.03
    no_improve_steps = 0
    max_no_improve = 1500
    restart_count = 0
    max_restarts = 15
    
    while (time.time() - start_time) < deadline - 3.0:
        # Pick random position
        pos = random.randint(0, n - 1)
        d = get_delta(pos, c)

        accept = False
        if d < 0:
            accept = True
            no_improve_steps = 0
        elif d == 0:
            accept = True
        else:
            if T > min_T:
                prob = math.exp(-d / T)
                if random.random() < prob:
                    accept = True
                    no_improve_steps = 0

        if accept:
            c[pos] = 1 - c[pos]
            best_count += d
            if best_count < count_mono(best_c):
                best_c = c.copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1
        else:
            no_improve_steps += 1

        T *= cooling

        # Restart strategy if stuck
        if no_improve_steps > max_no_improve or restart_count >= max_restarts:
            # Reset to best
            c = best_c.copy()
            best_count = count_mono(c)
            T = 1.5
            no_improve_steps = 0
            restart_count += 1
            
            # Adaptive perturbation: flip more positions as we go deeper
            num_flips = 3 + min(restart_count, 7)
            for _ in range(num_flips):
                p2 = random.randint(0, n - 1)
                c[p2] = 1 - c[p2]
            
            # Polish to get back on track
            c, best_count = greedy_polish(c, time_limit=deadline - 10.0)
            if best_count < count_mono(best_c):
                best_c = c.copy()
            
            # Reset restart counter if we made progress
            if restart_count >= max_restarts:
                break

    # Phase 3: Multiple final polish attempts with different random perturbations
    if (time.time() - start_time) < deadline - 2.0:
        # Try a few different perturbations of the best solution
        for attempt in range(3):
            if (time.time() - start_time) > deadline - 2.0:
                break
            perturbed = best_c.copy()
            num_flips = 2 + attempt
            for _ in range(num_flips):
                p2 = random.randint(0, n - 1)
                perturbed[p2] = 1 - perturbed[p2]
            polished, polished_count = greedy_polish(perturbed, time_limit=deadline - 2.0)
            if polished_count < best_count:
                best_c = polished
                best_count = polished_count

    # Final greedy polish
    if (time.time() - start_time) < deadline - 1.0:
        final_c, final_count = greedy_polish(best_c)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
