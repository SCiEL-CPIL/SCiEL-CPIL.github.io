import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    T_min = 128
    T_max = 256
    N_max = 129
    R = T_max - T_min + 1  # 129

    def compute_counts(ind):
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

    def max_count(counts):
        return int(np.max(counts))

    def is_valid(counts):
        return bool(np.all(counts > 0))

    def mixed_radix_basis():
        """
        Construct a mixed-radix basis for [0,128] that covers [128,256]
        with unique representations (peak=1).
        
        Strategy: Use a base-2 style decomposition where we pick elements
        such that every target sum has exactly one representation.
        
        We use a greedy approach: for each target sum from 128 to 256,
        ensure it's covered, preferring to add elements that create
        minimal additional representations.
        """
        A = set()
        ind = np.zeros(N_max, dtype=np.int32)
        covered = np.zeros(R, dtype=np.int32)
        
        # Greedy: for each uncovered sum, find the best element to add
        for s in range(T_min, T_max + 1):
            if covered[s - T_min] > 0:
                continue
            
            # Find pairs (a,b) with a<=b, a+b=s, a,b in [0,128]
            best_y = None
            best_score = float('inf')
            
            for a in range(max(0, s - 128), min(128, s // 2) + 1):
                b = s - a
                if b > 128 or b < a:
                    continue
                
                # Score: how many new representations would be created
                # If we add y, it creates pairs (y, s-y) for all s in range
                # We want to minimize total new representations
                
                for y in [a, b]:
                    if y in A:
                        continue
                    # Count how many new sums this would create
                    new_reps = 0
                    for ss in range(T_min, T_max + 1):
                        if covered[ss - T_min] > 0:
                            continue
                        partner = ss - y
                        if 0 <= partner < N_max and (partner in A or partner == y):
                            new_reps += 1
                    if new_reps < best_score:
                        best_score = new_reps
                        best_y = y
            
            if best_y is not None:
                A.add(best_y)
                ind[best_y] = 1
                # Update covered
                for ss in range(T_min, T_max + 1):
                    partner = ss - best_y
                    if 0 <= partner < N_max and (partner in A or partner == best_y):
                        covered[ss - T_min] += 1
        
        return A

    def simulated_annealing(A_initial, time_budget):
        """Simulated annealing to minimize peak multiplicity."""
        A_cur = set(A_initial)
        ind_cur = np.zeros(N_max, dtype=np.int32)
        for x in A_cur:
            ind_cur[x] = 1
        counts_cur = compute_counts(ind_cur)
        
        if not is_valid(counts_cur):
            return A_cur, max_count(counts_cur) if len(counts_cur) > 0 else float('inf')
        
        current_max = max_count(counts_cur)
        best_A = set(A_cur)
        best_max = current_max
        
        temperature = 5.0
        cooling = 0.995
        sa_deadline = start_time + time_budget
        
        while time.time() < sa_deadline:
            if len(A_cur) == 0:
                break
            
            # Choose move type: swap, add, or remove
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap: remove x, add y
                x = random.choice(list(A_cur))
                y = random.randint(0, N_max - 1)
                if y == x or ind_cur[y] == 1:
                    continue
                
                # Compute delta
                ind_test = ind_cur.copy()
                ind_test[x] = 0
                ind_test[y] = 1
                counts_test = compute_counts(ind_test)
                
                if not is_valid(counts_test):
                    temperature *= cooling
                    continue
                
                new_max = max_count(counts_test)
                delta = new_max - current_max
                
                if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                    A_cur.discard(x)
                    A_cur.add(y)
                    ind_cur = ind_test
                    counts_cur = counts_test
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A = set(A_cur)
                        
            elif move_type < 0.85 and len(A_cur) < 80:
                # Add element
                y = random.randint(0, N_max - 1)
                if ind_cur[y] == 1:
                    continue
                ind_test = ind_cur.copy()
                ind_test[y] = 1
                counts_test = compute_counts(ind_test)
                new_max = max_count(counts_test)
                delta = new_max - current_max
                if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                    A_cur.add(y)
                    ind_cur = ind_test
                    counts_cur = counts_test
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A = set(A_cur)
                        
            else:
                # Remove element (only if still valid)
                if len(A_cur) <= 20:
                    continue
                x = random.choice(list(A_cur))
                ind_test = ind_cur.copy()
                ind_test[x] = 0
                counts_test = compute_counts(ind_test)
                if not is_valid(counts_test):
                    temperature *= cooling
                    continue
                new_max = max_count(counts_test)
                delta = new_max - current_max
                if delta <= 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                    A_cur.discard(x)
                    ind_cur = ind_test
                    counts_cur = counts_test
                    current_max = new_max
                    if current_max < best_max:
                        best_max = current_max
                        best_A = set(A_cur)
            
            temperature *= cooling
        
        return best_A, best_max

    # Phase 1: Try mixed-radix basis construction
    A_mr = mixed_radix_basis()
    ind_mr = np.zeros(N_max, dtype=np.int32)
    for x in A_mr:
        ind_mr[x] = 1
    counts_mr = compute_counts(ind_mr)
    
    if is_valid(counts_mr):
        peak_mr = max_count(counts_mr)
        print(f"Mixed radix basis: |A|={len(A_mr)}, peak={peak_mr}")
        
        if peak_mr == 1:
            # Already optimal! Return immediately
            return list(A_mr)
        
        # Refine with simulated annealing
        A_refined, peak_refined = simulated_annealing(A_mr, 40.0)
        if peak_refined < peak_mr:
            A_mr = A_refined
            peak_mr = peak_refined
        return list(A_mr)
    
    # Phase 2: If mixed-radix failed, try greedy construction + SA
    A_greedy = set()
    ind_g = np.zeros(N_max, dtype=np.int32)
    
    # Greedy add elements to cover all sums
    for s in range(T_min, T_max + 1):
        # Check if s is already covered
        covered_s = False
        for a in A_greedy:
            b = s - a
            if 0 <= b < N_max and (b in A_greedy or b == a):
                covered_s = True
                break
        if covered_s:
            continue
        
        # Find best element to add
        best_y = None
        best_gain = -1
        for y in range(N_max):
            if ind_g[y] == 1:
                continue
            gain = 0
            for ss in range(T_min, T_max + 1):
                partner = ss - y
                if 0 <= partner < N_max and (partner in A_greedy or partner == y):
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_y = y
        
        if best_y is not None and best_gain > 0:
            A_greedy.add(best_y)
            ind_g[best_y] = 1
    
    # Verify and repair
    counts_g = compute_counts(ind_g)
    if not is_valid(counts_g):
        # Repair by adding more elements
        for _ in range(50):
            if is_valid(counts_g):
                break
            gaps = np.where(counts_g == 0)[0]
            best_y = None
            best_gain = -1
            for y in range(N_max):
                if ind_g[y] == 1:
                    continue
                gain = 0
                for gi in gaps:
                    s = T_min + gi
                    a = s - y
                    if 0 <= a < N_max and (ind_g[a] == 1 or a == y):
                        gain += 1
                if gain > best_gain:
                    best_gain = gain
                    best_y = y
            if best_y is not None:
                A_greedy.add(best_y)
                ind_g[best_y] = 1
                counts_g = compute_counts(ind_g)
            else:
                break
    
    if is_valid(counts_g):
        peak_g = max_count(counts_g)
        print(f"Greedy basis: |A|={len(A_greedy)}, peak={peak_g}")
        
        # Refine with simulated annealing
        remaining = deadline - time.time() - 3.0
        if remaining > 5.0:
            A_refined, peak_refined = simulated_annealing(A_greedy, remaining - 5.0)
            if peak_refined < peak_g:
                A_greedy = A_refined
                peak_g = peak_refined
        
        return list(A_greedy)
    
    # Fallback
    return list(range(64, 129))
