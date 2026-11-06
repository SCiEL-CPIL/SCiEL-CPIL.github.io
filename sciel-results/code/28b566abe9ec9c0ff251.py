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

    def fast_remove(ind, x, counts):
        """Update counts after removing element x from A."""
        new_counts = counts.copy()
        s_2x = 2 * x
        if T_min <= s_2x <= T_max:
            new_counts[s_2x - T_min] -= 1
        for a in range(N_max):
            if ind[a] == 1 and a != x:
                s_xa = x + a
                if T_min <= s_xa <= T_max:
                    new_counts[s_xa - T_min] -= 1
        return new_counts

    def fast_add(ind, x, counts):
        """Update counts after adding element x to A."""
        new_counts = counts.copy()
        s_2x = 2 * x
        if T_min <= s_2x <= T_max:
            new_counts[s_2x - T_min] += 1
        for a in range(N_max):
            if ind[a] == 1 and a != x:
                s_xa = x + a
                if T_min <= s_xa <= T_max:
                    new_counts[s_xa - T_min] += 1
        return new_counts

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
            for _ in range(100):
                if is_valid(counts): break
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0: break
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
        T = 1.0
        steps = 0
        
        while time.time() < start_time + time_budget:
            if len(A_cur) == 0: break
            
            # 1. Try to remove an element (prune)
            if len(A_cur) > 20:
                removable = [x for x in A_cur if random.random() < 0.1]
                if removable:
                    x = random.choice(removable)
                    new_counts = fast_remove(ind_cur, x, counts_cur)
                    
                    if is_valid(new_counts):
                        new_max = int(np.max(new_counts))
                        new_sum_sq = float(np.sum(new_counts * new_counts))
                        if new_max < max_cur or (new_max == max_cur and new_sum_sq < sum_sq_cur):
                            A_cur.remove(x)
                            ind_cur[x] = 0
                            counts_cur = new_counts
                            max_cur = new_max
                            sum_sq_cur = new_sum_sq
                            no_improve = 0
                            if max_cur < best_max or (max_cur == best_max and sum_sq_cur < best_sum_sq):
                                best_max = max_cur
                                best_sum_sq = sum_sq_cur
                                best_A = set(A_cur)
                            continue
                        else:
                            no_improve += 1
                            if no_improve > 100:
                                # Reset no_improve to allow more search
                                no_improve = 0
                            continue

            # 2. Swap
            x = random.choice(list(A_cur))
            y = random.randint(0, N_max - 1)
            
            if y == x or ind_cur[y] == 1:
                continue

            # Remove x, then add y
            temp_counts = fast_remove(ind_cur, x, counts_cur)
            ind_cur[x] = 0
            new_counts = fast_add(ind_cur, y, temp_counts)
            ind_cur[x] = 1  # restore for next iteration
            
            if not is_valid(new_counts):
                no_improve += 1
                if no_improve > 500:
                    break
                continue

            new_max = int(np.max(new_counts))
            new_sum_sq = float(np.sum(new_counts * new_counts))
            
            accepted = False
            if new_max < max_cur:
                accepted = True
            elif new_max == max_cur:
                if new_sum_sq < sum_sq_cur:
                    accepted = True
                elif random.random() < 0.05:
                    accepted = True
            else:
                if T > 0.01:
                    prob = np.exp(-(new_max - max_cur) / T)
                    if random.random() < prob:
                        accepted = True
                    T *= 0.995
                
            if accepted:
                A_cur.remove(x)
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
    # 1. Greedy construction to ensure coverage
    # Start with empty, add elements that cover most gaps
    ind_greedy = np.zeros(N_max, dtype=np.int32)
    counts_greedy = np.zeros(R, dtype=np.int32)
    A_greedy = set()
    for _ in range(50):
        if is_valid(counts_greedy): break
        best_y = None
        best_gain = -1
        for y in range(N_max):
            if ind_greedy[y] == 1: continue
            gain = 0
            for i in range(R):
                s = T_min + i
                if counts_greedy[i] == 0:
                    a = s - y
                    if 0 <= a < N_max and (ind_greedy[a] == 1 or a == y):
                        gain += 1
            if gain > best_gain:
                best_gain = gain
                best_y = y
        if best_y is not None:
            A_greedy.add(best_y)
            ind_greedy[best_y] = 1
            counts_greedy = compute_counts_full(ind_greedy)
        else:
            break
    if is_valid(counts_greedy):
        starts.append(list(A_greedy))
    
    # 2. Random sets with varying densities
    for sz in [30, 40, 50, 60, 70]:
        for _ in range(3):
            starts.append(random.sample(range(0, N_max), sz))
            
    # 3. Structured sets
    starts.append(list(range(64, 129)))
    starts.append(list(range(0, 129, 2)))
    starts.append(list(range(1, 129, 2)))
    
    # Time allocation
    total_time = 85.0
    time_per_start = total_time / len(starts)
    
    for i, s in enumerate(starts):
        if time.time() > start_time + total_time - 2.0:
            break
        remaining = (start_time + total_time) - time.time()
        budget = min(time_per_start, remaining - 1.0)
        if budget > 0.5:
            try_improve(s, budget)

    # Final polish: targeted reduction of high-multiplicity sums
    if best_A is not None and time.time() < start_time + total_time - 5.0:
        A_polish = set(best_A)
        ind_polish = np.zeros(N_max, dtype=np.int32)
        for x in A_polish:
            ind_polish[x] = 1
        counts_polish = compute_counts_full(ind_polish)
        max_polish = int(np.max(counts_polish))
        sum_sq_polish = float(np.sum(counts_polish * counts_polish))
        
        polish_time = start_time + total_time - 3.0
        no_improve_polish = 0
        
        while time.time() < polish_time:
            # Find sums with highest multiplicity
            if max_polish <= 2:
                break
            high_sums = np.where(counts_polish >= max_polish)[0]
            if len(high_sums) == 0:
                break
            
            # Try to reduce by swapping elements that contribute to high-multiplicity sums
            improved = False
            for _ in range(50):
                if time.time() > polish_time:
                    break
                if max_polish <= 2:
                    break
                
                x = random.choice(list(A_polish))
                # Choose y that would reduce high-multiplicity sums
                best_y = None
                best_new_max = max_polish + 1
                best_new_sum_sq = sum_sq_polish + 1
                
                for _attempt in range(20):
                    y = random.randint(0, N_max - 1)
                    if y == x or ind_polish[y] == 1:
                        continue
                    
                    temp_counts = fast_remove(ind_polish, x, counts_polish)
                    ind_polish[x] = 0
                    new_counts = fast_add(ind_polish, y, temp_counts)
                    ind_polish[x] = 1
                    
                    if not is_valid(new_counts):
                        continue
                    
                    new_max = int(np.max(new_counts))
                    new_sum_sq = float(np.sum(new_counts * new_counts))
                    
                    if new_max < best_new_max or (new_max == best_new_max and new_sum_sq < best_new_sum_sq):
                        best_new_max = new_max
                        best_new_sum_sq = new_sum_sq
                        best_y = y
                        best_new_counts = new_counts
                
                if best_y is not None and (best_new_max < max_polish or (best_new_max == max_polish and best_new_sum_sq < sum_sq_polish)):
                    A_polish.remove(x)
                    A_polish.add(best_y)
                    ind_polish[x] = 0
                    ind_polish[best_y] = 1
                    counts_polish = best_new_counts
                    max_polish = best_new_max
                    sum_sq_polish = best_new_sum_sq
                    no_improve_polish = 0
                    improved = True
                    
                    if max_polish < best_max or (max_polish == best_max and sum_sq_polish < best_sum_sq):
                        best_max = max_polish
                        best_sum_sq = sum_sq_polish
                        best_A = set(A_polish)
                else:
                    no_improve_polish += 1
                    if no_improve_polish > 200:
                        break
            
            if not improved:
                break

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
