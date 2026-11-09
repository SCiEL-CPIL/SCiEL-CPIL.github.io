import time
import numpy as np
import random
import math

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
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

    # For each position, store the indices of the other 3 positions in each AP containing it
    pos_aps_info = [[] for _ in range(n)]
    for ap in aps:
        i0, i1, i2, i3 = ap
        pos_aps_info[i0].append((i1, i2, i3))
        pos_aps_info[i1].append((i0, i2, i3))
        pos_aps_info[i2].append((i0, i1, i3))
        pos_aps_info[i3].append((i0, i1, i2))

    # Thue-Morse initialization
    c = np.array([bin(i).count('1') % 2 for i in range(n)], dtype=np.int8)

    def count_mono(col):
        count = 0
        for ap in aps:
            i0, i1, i2, i3 = ap
            if col[i0] == col[i1] == col[i2] == col[i3]:
                count += 1
        return count

    def get_delta(pos, col):
        delta = 0
        c_pos = col[pos]
        for others in pos_aps_info[pos]:
            if col[others[0]] == col[others[1]] == col[others[2]]:
                if col[others[0]] == c_pos:
                    delta -= 1
                else:
                    delta += 1
        return delta

    def greedy_polish(col):
        col = col.copy()
        cur_count = count_mono(col)
        improved = True
        while improved and (time.time() - start_time) < deadline - 5.0:
            improved = False
            for pos in range(n):
                if (time.time() - start_time) > deadline - 5.0:
                    break
                d = get_delta(pos, col)
                if d < 0:
                    col[pos] = 1 - col[pos]
                    cur_count += d
                    improved = True
        return col, cur_count

    # Phase 1: Greedy polish from Thue-Morse
    c, cur_count = greedy_polish(c)
    best_c = c.copy()
    best_count = cur_count

    # Phase 2: Simulated Annealing
    T = 1.0
    cooling = 0.999
    min_T = 0.01
    
    while (time.time() - start_time) < deadline - 3.0:
        pos = random.randint(0, n - 1)
        d = get_delta(pos, c)
        
        accept = False
        if d < 0:
            accept = True
        elif d == 0:
            accept = True
        else:
            if T > 0.05:
                prob = math.exp(-d / T)
                if random.random() < prob:
                    accept = True

        if accept:
            c[pos] = 1 - c[pos]
            cur_count += d
            if cur_count < best_count:
                best_c = c.copy()
                best_count = cur_count

        T *= cooling
        if T < min_T:
            T = 1.0
            c = best_c.copy()
            cur_count = best_count
            # Perturb
            for _ in range(5):
                p2 = random.randint(0, n - 1)
                c[p2] = 1 - c[p2]
            # Polish the perturbed state
            c, cur_count = greedy_polish(c)
            if cur_count < best_count:
                best_c = c.copy()
                best_count = cur_count

    # Phase 3: Final greedy polish
    if (time.time() - start_time) < deadline - 2.0:
        final_c, final_count = greedy_polish(best_c)
        if final_count < best_count:
            best_c = final_c
            best_count = final_count

    return best_c.tolist()
