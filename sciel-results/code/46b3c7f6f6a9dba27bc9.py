import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin for 90s limit

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129 targets

    best_A = None
    best_max = float('inf')

    def compute_initial_counts(ind):
        """Compute unordered pair-sum representation counts for [T_min, T_max]."""
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

    def try_improve(A_initial):
        nonlocal best_A, best_max
        
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        
        counts = compute_initial_counts(ind)

        # Repair if invalid
        if not is_valid(counts):
            for _ in range(100):
                gaps = np.where(counts == 0)[0]
                if len(gaps) == 0:
                    break
                best_y = None
                best_gain = -1
                # Find y that covers most gaps
                for y in range(N_max):
                    if ind[y] == 1:
                        continue
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
                    # Incremental update for addition
                    for x in list(A_set):
                        if x == best_y: continue
                        s1 = x + best_y
                        if T_min <= s1 <= T_max:
                            counts[s1 - T_min] += 1
                        s2 = best_y + best_y # handled by x loop if we iterate over all pairs? 
                        # Wait, the loop above `for x in list(A_set)` includes best_y if we added it.
                        # Let's do a simpler incremental update:
                        # For each a in A (including new y), target = a + y.
                        # If a != y, count increases by 1.
                        # If a == y, count increases by 1 (since (y,y) is 1 rep).
                        # Actually, adding y increases count for T=a+y by 1 for each a in A.
                        # My loop above does exactly that if A_set contains best_y.
                        # But I added best_y to A_set before the loop.
                        # So `for x in list(A_set)` iterates over all elements including best_y.
                        # s = x + best_y.
                        # This is correct.
                    # However, I just recomputed counts from scratch in the previous draft logic? 
                    # No, I should use incremental update for speed.
                    # Let's re-implement the repair update correctly.
                    # The loop above is O(|A|).
                else:
                    break
            
            if not is_valid(counts):
                return

        current_max = int(np.max(counts))
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)

        # Local Search
        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts.copy()
        max_cur = current_max
        no_improve = 0
        
        # Precompute list of elements in A for fast iteration
        A_list = list(A_cur)
        
        while time.time() < deadline - 2.0:
            if len(A_cur) == 0:
                break
            
            # Random move: swap x (in A) with y (not in A)
            x = random.choice(A_list)
            y = random.randint(0, N_max - 1)
            
            if y == x or ind_cur[y] == 1:
                continue
            
            # Simulate swap: Remove x, Add y
            # 1. Remove x:
            # For each a in A_cur, target = x + a.
            # If a != x, count[target] -= 1
            # If a == x, count[target] -= 1 (since (x,x) is 1 rep)
            # So for all a in A_cur, count[x+a] -= 1
            
            # 2. Add y:
            # For each a in A_cur, target = y + a.
            # count[y+a] += 1
            
            # We need to be careful if y is added, the set becomes A_cur - {x} + {y}.
            # The "Add y" step should consider the new set?
            # No, standard incremental:
            # Delta = (Add y to A_cur) - (Remove x from A_cur)
            # But A_cur changes.
            # Let's do:
            # Step 1: Remove x from A_cur.
            for a in A_list:
                s = x + a
                if T_min <= s <= T_max:
                    counts_cur[s - T_min] -= 1
            
            # Step 2: Add y to (A_cur - {x}).
            # The set is now A_cur - {x}.
            # We iterate over a in A_cur - {x}.
            # If a == x, skip (since x is removed).
            # If a != x, count[y+a] += 1.
            # Also, what about (y,y)? It's added when a=y? No, y is not in A_cur - {x}.
            # So (y,y) is not added by the loop `for a in A_cur - {x}`.
            # We must manually add (y,y) if 2y is in range.
            
            # Wait, the formula for count[T] is number of pairs {a,b} in A such that a+b=T.
            # When we add y to set S, the new pairs involving y are {y, a} for a in S, and {y, y}.
            # So count[T] increases by 1 for each a in S such that a+y=T, plus 1 if 2y=T.
            
            # So:
            for a in A_list:
                if a == x: continue
                s = y + a
                if T_min <= s <= T_max:
                    counts_cur[s - T_min] += 1
            
            # Handle (y,y)
            s_ee = 2 * y
            if T_min <= s_ee <= T_max:
                counts_cur[s_ee - T_min] += 1
            
            # Check validity
            if not is_valid(counts_cur):
                # Rollback
                # Reverse Add y
                for a in A_list:
                    if a == x: continue
                    s = y + a
                    if T_min <= s <= T_max:
                        counts_cur[s - T_min] -= 1
                s_ee = 2 * y
                if T_min <= s_ee <= T_max:
                    counts_cur[s_ee - T_min] -= 1
                # Reverse Remove x
                for a in A_list:
                    s = x + a
                    if T_min <= s <= T_max:
                        counts_cur[s - T_min] += 1
                
                no_improve += 1
                if no_improve > 1000:
                    break
                continue

            m_new = int(np.max(counts_cur))

            if m_new < max_cur:
                # Accept
                A_cur.remove(x)
                A_cur.add(y)
                A_list.remove(x)
                A_list.append(y)
                ind_cur[x] = 0
                ind_cur[y] = 1
                max_cur = m_new
                no_improve = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                if random.random() < 0.1:
                    A_cur.remove(x)
                    A_cur.add(y)
                    A_list.remove(x)
                    A_list.append(y)
                    ind_cur[x] = 0
                    ind_cur[y] = 1
                    no_improve = 0
            else:
                # Rollback
                for a in A_list:
                    if a == x: continue
                    s = y + a
                    if T_min <= s <= T_max:
                        counts_cur[s - T_min] -= 1
                s_ee = 2 * y
                if T_min <= s_ee <= T_max:
                    counts_cur[s_ee - T_min] -= 1
                for a in A_list:
                    s = x + a
                    if T_min <= s <= T_max:
                        counts_cur[s - T_min] += 1
                
                no_improve += 1
                if no_improve > 1000:
                    break

    # Diverse starting points
    starts = []
    starts.append(list(range(64, 129))) # Upper half
    starts.append(list(range(0, 129, 2))) # Evens
    starts.append(list(range(1, 129, 2))) # Odds
    starts.append(list(range(0, 129, 3)) + list(range(2, 129, 3)))
    
    # Random starts
    for sz in [35, 40, 45, 50, 55, 60]:
        for trial in range(3):
            random.seed(42 + trial * 1000 + sz)
            starts.append(random.sample(range(0, N_max), sz))

    for s in starts:
        if time.time() > deadline - 3.0:
            break
        try_improve(s)

    if best_A is None:
        return list(range(64, 129))
    return list(best_A)
