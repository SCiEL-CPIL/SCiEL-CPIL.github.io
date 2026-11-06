import numpy as np
import time

def construct():
    N = 256
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Precompute APs
    aps = []
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a - 1, a + d - 1, a + 2 * d - 1, a + 3 * d - 1))
    
    num_aps = len(aps)
    ap_array = np.array(aps, dtype=np.int32)
    
    # Incidence lists
    pos_to_aps_idx = [[] for _ in range(N)]
    pos_to_aps_others = [[] for _ in range(N)]
    
    for idx, ap in enumerate(aps):
        for k in range(4):
            pos = ap[k]
            pos_to_aps_idx[pos].append(idx)
            others = [ap[j] for j in range(4) if j != k]
            pos_to_aps_others[pos].append(others)
            
    # 2. Helper functions
    def count_mono(coloring):
        # coloring is np.array of int8
        count = 0
        for ap in ap_array:
            if coloring[ap[0]] == coloring[ap[1]] == coloring[ap[2]] == coloring[ap[3]]:
                count += 1
        return count

    def delta_for_flip(coloring, pos):
        delta = 0
        cur_val = coloring[pos]
        new_val = 1 - cur_val
        idxs = pos_to_aps_idx[pos]
        others_list = pos_to_aps_others[pos]
        
        for i in range(len(idxs)):
            o1, o2, o3 = others_list[i]
            c1 = coloring[o1]
            c2 = coloring[o2]
            c3 = coloring[o3]
            
            before = 1 if (c1 == cur_val and c2 == cur_val and c3 == cur_val) else 0
            after = 1 if (c1 == new_val and c2 == new_val and c3 == new_val) else 0
            
            delta += after - before
        return delta

    def coordinate_descent(coloring, max_rounds=100):
        coloring = coloring.copy()
        count = count_mono(coloring)
        for _ in range(max_rounds):
            if time.time() > deadline - 2.0:
                break
            improved = False
            for pos in range(N):
                d = delta_for_flip(coloring, pos)
                if d < 0:
                    coloring[pos] = 1 - coloring[pos]
                    count += d
                    improved = True
            if not improved:
                break
        return coloring, count

    def simulated_annealing(initial, t_start, t_end, steps):
        coloring = initial.copy()
        count = count_mono(coloring)
        best_local = coloring.copy()
        best_local_count = count
        
        for i in range(steps):
            if time.time() > deadline - 5.0:
                break
            pos = np.random.randint(N)
            d = delta_for_flip(coloring, pos)
            
            T = t_start * (t_end / t_start) ** (i / steps)
            if d < 0 or np.random.random() < np.exp(-d / T):
                coloring[pos] = 1 - coloring[pos]
                count += d
                if count < best_local_count:
                    best_local_count = count
                    best_local = coloring.copy()
                    
        return best_local, best_local_count

    # 3. Main Search Loop
    best_coloring = None
    best_count = float('inf')
    
    # Initial candidates
    def thue_morse(n):
        t = [0]
        for _ in range(n.bit_length()):
            t = t + [1 - x for x in t]
        return t[:n]
            
    initial_strategies = [
        np.array(thue_morse(N), dtype=np.int8),
        np.random.randint(0, 2, size=N, dtype=np.int8),
        np.array([0 if i % 4 < 2 else 1 for i in range(N)], dtype=np.int8),
        np.array([0 if i % 3 < 1 else 1 for i in range(N)], dtype=np.int8),
    ]
    
    # Evaluate initial strategies
    for init in initial_strategies:
        c, cnt = coordinate_descent(init, max_rounds=50)
        if cnt < best_count:
            best_count = cnt
            best_coloring = c.copy()
            
    # SA Loop
    np.random.seed(42)
    sa_step_time = 0.1 # Approx time per step? No, just use fixed steps and check time.
    
    while time.time() < deadline - 10.0:
        # Choose start
        if np.random.random() < 0.5:
            start = best_coloring.copy()
            # Perturb
            num_flips = np.random.randint(5, 20)
            flip_pos = np.random.choice(N, size=num_flips, replace=False)
            start[flip_pos] = 1 - start[flip_pos]
        else:
            start = np.random.randint(0, 2, size=N, dtype=np.int8)
            
        # Run SA
        # Use more steps, but check time
        sa_coloring, sa_count = simulated_annealing(start, t_start=2.0, t_end=0.01, steps=20000)
        
        if sa_count < best_count:
            best_count = sa_count
            best_coloring = sa_coloring.copy()
            
        # Polish
        cd_coloring, cd_count = coordinate_descent(sa_coloring, max_rounds=200)
        if cd_count < best_count:
            best_count = cd_count
            best_coloring = cd_coloring.copy()
            
    # Final Polish
    cd_coloring, cd_count = coordinate_descent(best_coloring, max_rounds=500)
    if cd_count < best_count:
        best_count = cd_count
        best_coloring = cd_coloring.copy()
        
    return [int(c) for c in best_coloring]
