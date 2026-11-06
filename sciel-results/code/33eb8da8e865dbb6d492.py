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
    
    # Precompute indicator vector for fast convolution
    # We'll maintain a numpy array ind of length N_max where ind[x] = 1 if x in A
    
    def compute_counts_from_ind(ind):
        """
        Compute representation counts for all sums in [T_min, T_max].
        ind is a numpy array of length N_max with 0/1 values.
        Uses numpy convolution for speed.
        """
        conv = np.convolve(ind, ind, mode='full')
        counts = np.zeros(T_max + 1, dtype=np.int32)
        for s in range(T_min, T_max + 1):
            if s >= len(conv):
                break
            c_ord = conv[s]
            if s % 2 == 0:
                half = s // 2
                if half < N_max and ind[half] == 1:
                    n_eq = 1
                else:
                    n_eq = 0
            else:
                n_eq = 0
            counts[s] = (c_ord + n_eq) // 2
        return counts
    
    def get_max_count(counts):
        return int(np.max(counts[T_min:T_max+1]))
    
    def is_valid(counts):
        return np.all(counts[T_min:T_max+1] > 0)
    
    best_A = None
    best_max = float('inf')
    
    def try_improve(A_initial, label=""):
        nonlocal best_A, best_max
        A_set = set(A_initial)
        ind = np.zeros(N_max, dtype=np.int32)
        for x in A_set:
            ind[x] = 1
        counts = compute_counts_from_ind(ind)
        
        if not is_valid(counts):
            # Repair: add elements to cover gaps
            for _ in range(50):
                gaps = np.where(counts[T_min:T_max+1] == 0)[0] + T_min
                if len(gaps) == 0:
                    break
                
                best_y = None
                best_gain = -1
                # Sample candidates
                candidates = random.sample(range(0, N_max), min(30, N_max))
                for y in candidates:
                    if ind[y] == 1:
                        continue
                    gain = 0
                    for s in gaps:
                        a = s - y
                        if 0 <= a < N_max and (ind[a] == 1 or a == y):
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
                
                if best_y is not None:
                    A_set.add(best_y)
                    ind[best_y] = 1
                    counts = compute_counts_from_ind(ind)
                else:
                    break
            
            if not is_valid(counts):
                return
        
        current_max = get_max_count(counts)
        if current_max < best_max:
            best_max = current_max
            best_A = set(A_set)
        
        # Local Search
        A_cur = set(A_set)
        ind_cur = ind.copy()
        counts_cur = counts
        max_cur = current_max
        
        no_improve_count = 0
        while time.time() < deadline - 2.0:
            # Pick random swap
            A_list_cur = list(A_cur)
            x = random.choice(A_list_cur)
            y = random.randint(0, N_max - 1)
            if ind_cur[y] == 1:
                continue
            
            # Simulate swap: remove x, add y
            ind_new = ind_cur.copy()
            ind_new[x] = 0
            ind_new[y] = 1
            
            c_new = compute_counts_from_ind(ind_new)
            
            if not is_valid(c_new):
                continue
            
            m_new = get_max_count(c_new)
            
            if m_new < max_cur:
                A_cur = A_cur - {x}
                A_cur.add(y)
                ind_cur = ind_new
                counts_cur = c_new
                max_cur = m_new
                no_improve_count = 0
                if max_cur < best_max:
                    best_max = max_cur
                    best_A = set(A_cur)
            elif m_new == max_cur:
                # Accept with prob 0.1 to escape local minima
                if random.random() < 0.1:
                    A_cur = A_cur - {x}
                    A_cur.add(y)
                    ind_cur = ind_new
                    counts_cur = c_new
            else:
                no_improve_count += 1
                if no_improve_count > 500:
                    break
    
    # 1. Try Seed
    A_seed = list(range(64, 129))
    try_improve(A_seed, "seed")
    
    # 2. Try Random Sizes
    if time.time() < deadline - 10:
        for N in [40, 45, 50, 55, 60]:
            for trial in range(2):
                if time.time() > deadline - 5:
                    break
                random.seed(trial * 100 + N)
                A_rand = random.sample(range(0, N_max), N)
                try_improve(A_rand, f"rand_{N}_{trial}")
                
    if best_A is None:
        return list(range(64, 129))
        
    return list(best_A)
