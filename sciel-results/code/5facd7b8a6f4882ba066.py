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
    # For each position, we need the "others" as a 2D array
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
            if others_arr.shape[0] == 0:
                continue
            c1 = col[others_arr[:, 0]]
            c2 = col[others_arr[:, 1]]
            c3 = col[others_arr[:, 2]]
            all_same = (c1 == c2) & (c2 == c3)
            if np.any(all_same):
                same_color = c1[all_same]
                matches = (col[pos] == same_color)
                all_deltas[pos] = np.sum(matches) - np.sum(~matches)
        
        return all_deltas

    def greedy_polish(col):
        """Run greedy local search until no improvement, using vectorized deltas."""
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        while improved and (time.time() - start_time) < deadline - 5.0:
            improved = False
            deltas = vectorized_deltas(col)
            for pos in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = deltas[pos]
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
                    # Recompute deltas since one flip affects multiple positions
                    deltas = vectorized_deltas(col)
        return col, cur_count

    # Phase 1: Greedy polish from Thue-Morse
    c, best_count = greedy_polish(c)
    best_c = c.copy()

    # Phase 2: Simulated Annealing
    T = 2.0
    cooling = 0.9999
    min_T = 0.05
    no_improve_steps = 0
    max_no_improve = 2000

    # Precompute log of random for speed? No, math.exp is fast enough.
    
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
                # exp(-d/T) can underflow, but d is small (max ~10) and T > 0.05, so -d/T > -200
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
        if no_improve_steps > max_no_improve:
            # Reset to best
            c = best_c.copy()
            best_count = count_mono(c)
            T = 1.0
            no_improve_steps = 0
            # Perturb: flip random positions with varying intensity
            n_flips = random.randint(3, 8)
            for _ in range(n_flips):
                p2 = random.randint(0, n - 1)
                c[p2] = 1 - c[p2]
            best_count = count_mono(c)
            # Polish a bit to get back on track
            c, best_count = greedy_polish(c)
            if best_count < count_mono(best_c):
                best_c = c.copy()
            # If still stuck, try a more aggressive perturbation
            if no_improve_steps > max_no_improve * 2:
                c = best_c.copy()
                # Flip a larger random block
                start = random.randint(0, n - 32)
                for i in range(start, min(start + random.randint(8, 20), n)):
                    c[i] = 1 - c[i]
                best_count = count_mono(c)
                c, best_count = greedy_polish(c)
                if best_count < count_mono(best_c):
                    best_c = c.copy()

    # Phase 3: Final greedy polish
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    # Verify and return
    actual_count = count_mono(best_c)
    if actual_count < best_count:
        best_count = actual_count

    return best_c.tolist()
