import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    best_A = None
    best_max = float('inf')
    best_sum_sq = float('inf')

    def compute_counts_full(ind):
        conv = np.convolve(ind, ind, mode='full')
        counts = np.empty(R, dtype=np.int32)
        for i, s in enumerate(range(T_min, T_max + 1)):
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                n_eq = 1 if (half < N_max and ind[half] == 1) else 0
            else:
                n_eq = 0
            counts[i] = (c_ord + n_eq) // 2
        return counts

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def get_metrics(counts):
        return int(np.max(counts)), float(np.sum(counts * counts))

    def try_improve(A_initial, time_budget):
        nonlocal best_A, best_max, best_sum_sq
        
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
            
        # Initial counts
        counts = compute_counts_full(ind)
        
        # Repair if invalid
        if not is_valid(counts):
            # Simple greedy repair
            for _ in range(50):
                if is_valid(counts): break
                gaps = np.where(counts == 0)[0]
                best_y = None
                best_gain = -1
                for y in range(N_max):
                    if ind[y] == 1: continue
                    gain = 0
                    for gi in gaps:
                        s = T_min + gi
                        a = s - y
                        if 0 <= a < N_max and (ind[a] == 1 or a == y):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                if best_y is not None:
                    A_set.add(best_y)
                    ind[best_y] = 1
                    counts = compute_counts_full(ind)
                else:
                    break
            if not is_valid(counts):
                return

        cur_max, cur_sum_sq = get_metrics(counts)
        if cur_max < best_max or (cur_max == best_max and cur_sum_sq < best_sum_sq):
            best_max = cur_max
            best_sum_sq = cur_sum_sq
            best_A = set(A_set)

        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = cur_max
        sum_sq_cur = cur_sum_sq
        
        # Local Search
        no_improve = 0
        # Temperature for simulated annealing-like acceptance
        T = 1.0
        
        while time.time() < start_time + time_budget:
            if len(A_cur) == 0: break
            
            x = random.choice(list(A_cur))
            y = random.randint(0, N_max - 1)
            
            if y == x or ind_cur[y] == 1:
                continue

            # Incremental update
            # delta[s] = new_count[s] - old_count[s]
            # We only care about sums in [T_min, T_max]
            # Let's create a temporary delta array of zeros, size R
            # Actually, we can just compute the new max and sum_sq directly by updating a copy?
            # No, R is 129. Copying is fast. But O(|A|) update is better for logic clarity if we just track changes.
            # Let's stick to copying the counts array and modifying it, as R is small (129).
            
            new_counts = counts_cur.copy()
            
            # Remove x
            # Pairs involving x in A_cur
            # 1. x + x = 2x
            s_2x = 2 * x
            if T_min <= s_2x <= T_max:
                new_counts[s_2x - T_min] -= 1
            
            # 2. x + a for a in A_cur \ {x}
            for a in A_cur:
                if a == x: continue
                s_xa = x + a
                if T_min <= s_xa <= T_max:
                    new_counts[s_xa - T_min] -= 1
            
            # Add y
            # Pairs involving y in A_new = (A_cur \ {x}) U {y}
            # 1. y + y = 2y
            s_2y = 2 * y
            if T_min <= s_2y <= T_max:
                new_counts[s_2y - T_min] += 1
            
            # 2. y + a for a in A_cur \ {x}
            for a in A_cur:
                if a == x: continue
                s_ya = y + a
                if T_min <= s_ya <= T_max:
                    new_counts[s_ya - T_min] += 1
            
            # Check validity
            if not np.all(new_counts > 0):
                no_improve += 1
                if no_improve > 500:
                    break
                continue

            new_max = int(np.max(new_counts))
            new_sum_sq = float(np.sum(new_counts * new_counts))
            
            # Acceptance Criteria
            accepted = False
            if new_max < max_cur:
                accepted = True
            elif new_max == max_cur:
                if new_sum_sq < sum_sq_cur:
                    accepted = True
                elif random.random() < 0.1: # Random walk on plateau
                    accepted = True
            else:
                # Simulated annealing: accept worse max with small prob
                # Delta max = new_max - max_cur
                # Prob = exp(-Delta / T)
                if T > 0.01:
                    prob = np.exp(-(new_max - max_cur) / T)
                    if random.random() < prob:
                        accepted = True
                    T *= 0.99 # Cool down slowly
                
            if accepted:
                A_cur = A_cur - {x}
                A_cur.add(y)
                ind_cur[x] = 0
                ind_cur[y] = 1
                counts_cur = new_counts
                max_cur = new_max
                sum_sq_cur = new_sum_sq
                no_improve = 0
                
                if max_cur < best_max or (max_cur == best_max and sum_sq_cur < best_sum_sq):
                    best_max = max_cur
                    best_sum_sq = sum_sq_cur
                    best_A = set(A_cur)
            else:
                no_improve += 1
                if no_improve > 500:
                    break

    # Initializations
    starts = []
    # 1. Upper half
    starts.append(list(range(64, 129)))
    # 2. Evens
    starts.append(list(range(0, 129, 2)))
    # 3. Odds
    starts.append(list(range(1, 129, 2)))
    # 4. Random sets of various sizes
    for sz in [40, 50, 60, 70]:
        for _ in range(5):
            starts.append(random.sample(range(0, N_max), sz))
    
    # Time allocation
    total_time = 85.0
    time_per_start = total_time / len(starts)
    
    for i, s in enumerate(starts):
        if time.time() > start_time + total_time - 2.0:
            break
        # Give more time to earlier starts? Or equal?
        # Let's give equal time, but ensure we don't exceed deadline
        remaining = (start_time + total_time) - time.time()
        budget = min(time_per_start, remaining - 1.0)
        if budget > 0.5:
            try_improve(s, budget)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
