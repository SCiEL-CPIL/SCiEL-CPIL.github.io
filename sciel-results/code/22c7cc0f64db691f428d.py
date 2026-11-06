import numpy as np
import time

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We use 145s.
    deadline = start_time + 145.0
    
    N = 256
    
    # 1. Precompute all 4-term arithmetic progressions (0-indexed)
    # A 4-AP is (a, a+d, a+2d, a+3d) with 0 <= a, a+3d < N
    # d >= 1
    ap_list = []
    for d in range(1, N // 3 + 1):
        max_a = N - 3 * d
        for a in range(0, max_a):
            ap_list.append((a, a+d, a+2*d, a+3*d))
            
    if not ap_list:
        return [0] * N
        
    aps = np.array(ap_list, dtype=np.int32)
    num_aps = len(aps)
    
    ap_a = aps[:, 0]
    ap_b = aps[:, 1]
    ap_c = aps[:, 2]
    ap_d = aps[:, 3]
    
    # Precompute for each index i, the list of AP indices it belongs to
    # This allows O(degree) delta calculation instead of O(num_aps)
    index_to_aps = [[] for _ in range(N)]
    for j in range(num_aps):
        for idx in (ap_a[j], ap_b[j], ap_c[j], ap_d[j]):
            index_to_aps[idx].append(j)
    
    # Convert to numpy arrays for faster access if needed, 
    # but lists of lists are often faster for small degrees in Python loops 
    # if we avoid overhead. However, we will use a vectorized approach 
    # for the initial score and a loop for SA steps.
    
    # Helper to calculate current score
    def calc_score(c):
        ca = c[ap_a]
        cb = c[ap_b]
        cc = c[ap_c]
        cd = c[ap_d]
        return np.sum((ca == cb) & (cb == cc) & (cc == cd))
    
    # Helper to calculate delta for flipping a specific index i
    # We can vectorize this over the APs containing i
    def calc_delta(c, i):
        # Get all APs containing index i
        ap_indices = index_to_aps[i]
        if not ap_indices:
            return 0
            
        ap_indices_arr = np.array(ap_indices, dtype=np.int32)
        
        # Get the 4 positions for these APs
        a = ap_a[ap_indices_arr]
        b = ap_b[ap_indices_arr]
        c_pos = ap_c[ap_indices_arr]
        d_pos = ap_d[ap_indices_arr]
        
        # Current colors
        ca = c[a]
        cb = c[b]
        cc = c[c_pos]
        cd = c[d_pos]
        
        # Current mono status
        is_mono_now = (ca == cb) & (cb == cc) & (cc == cd)
        
        # New colors if we flip i
        # We need to determine which position i occupies in each AP
        # pos_a: i == a
        # pos_b: i == b
        # etc.
        
        # Instead of branching, we can compute the new color array for the 4 positions
        # and check mono status.
        
        # Create copies of the color arrays
        new_ca = ca.copy()
        new_cb = cb.copy()
        new_cc = cc.copy()
        new_cd = cd.copy()
        
        # Flip the specific position
        mask_a = (a == i)
        mask_b = (b == i)
        mask_c = (c_pos == i)
        mask_d = (d_pos == i)
        
        new_ca[mask_a] = 1 - new_ca[mask_a]
        new_cb[mask_b] = 1 - new_cb[mask_b]
        new_cc[mask_c] = 1 - new_cc[mask_c]
        new_cd[mask_d] = 1 - new_cd[mask_d]
        
        # New mono status
        is_mono_new = (new_ca == new_cb) & (new_cb == new_cc) & (new_cc == new_cd)
        
        # Delta is sum of changes
        return np.sum(is_mono_new) - np.sum(is_mono_now)

    best_score = np.inf
    best_coloring = None
    
    # Simulated Annealing
    # We use a single long run with a cooling schedule.
    # Initial temperature high enough to accept bad moves, cool down to find local min.
    
    rng = np.random.default_rng(42)
    
    # Initialize with random coloring
    c = rng.integers(0, 2, size=N).astype(np.int8)
    current_score = calc_score(c)
    
    if current_score < best_score:
        best_score = current_score
        best_coloring = c.copy()
        
    # SA Parameters
    # We want to run as many steps as possible.
    # Each step involves:
    # 1. Pick random index (O(1))
    # 2. Calc delta (O(degree) ~ O(10-20) operations, vectorized over small array)
    # 3. Accept/Reject (O(1))
    
    # Let's estimate speed. 
    # calc_delta creates small arrays. This might be slow in Python due to overhead.
    # Alternative: Precompute the "effect" of flipping each bit on each AP it touches?
    # No, the effect depends on the current state of the other 3 bits.
    
    # Optimization: 
    # Instead of creating arrays for every delta calc, we can iterate through the 
    # APs containing i and update a counter.
    # Since degree is small (max ~30-40?), a Python loop might be competitive 
    # if we avoid numpy overhead for small arrays.
    
    def calc_delta_fast(c, i):
        delta = 0
        for j in index_to_aps[i]:
            a = ap_a[j]
            b = ap_b[j]
            c_idx = ap_c[j]
            d_idx = ap_d[j]
            
            ca = c[a]
            cb = c[b]
            cc = c[c_idx]
            cd = c[d_idx]
            
            is_mono_now = (ca == cb) and (cb == cc) and (cc == cd)
            
            # Determine new colors
            # We flip i.
            if i == a:
                nca = 1 - ca
                ncb = cb
                ncc = cc
                ncd = cd
            elif i == b:
                nca = ca
                ncb = 1 - cb
                ncc = cc
                ncd = cd
            elif i == c_idx:
                nca = ca
                ncb = cb
                ncc = 1 - cc
                ncd = cd
            else: # i == d_idx
                nca = ca
                ncb = cb
                ncc = cc
                ncd = 1 - cd
                
            is_mono_new = (nca == ncb) and (ncb == ncc) and (ncc == ncd)
            
            if is_mono_new and not is_mono_now:
                delta += 1
            elif not is_mono_new and is_mono_now:
                delta -= 1
            # else delta += 0
            
        return delta

    # Test speed of calc_delta_fast
    # Let's run SA
    
    T_start = 1.0
    T_end = 0.01
    # We don't know the number of steps in advance, so we use time-based cooling
    # Or we can run for a fixed number of steps and restart if time permits.
    # Given the overhead of Python loops, we might only get ~100k-500k steps in 145s.
    # That might be too few.
    
    # Let's try to optimize calc_delta further.
    # We can precompute for each AP j, the 4 indices.
    # And for each index i, the list of (j, pos_in_j).
    
    # Actually, the numpy version `calc_delta` might be faster if the degree is large enough 
    # to amortize the overhead, but for degree ~20, Python loop with simple ints is often faster 
    # than creating numpy arrays.
    
    # Let's use the fast version.
    
    # To improve search quality, we can use multiple restarts with different T_start 
    # or just one long run. One long run with proper cooling is usually better.
    
    # We will run until deadline - 5 seconds.
    
    T = T_start
    # Cooling rate: T = T * alpha + T_min * (1-alpha) ?
    # Or T = T_start * (T_end/T_start)^(t/T_total)?
    # Since we don't know T_total, we use a geometric cooling based on time.
    # Let's assume we can do ~200,000 steps in 140 seconds.
    # This is a guess. We will monitor and adjust if we crash, but we can't adjust 
    # during the run. We will just run as many steps as we can.
    
    step_count = 0
    last_time = time.time()
    
    while time.time() < deadline - 5.0:
        # Pick random index
        i = rng.integers(0, N)
        
        # Calculate delta
        delta = calc_delta_fast(c, i)
        
        # Accept if delta < 0 (improvement)
        # Or with probability exp(-delta/T) if delta > 0
        if delta < 0:
            # Accept
            c[i] = 1 - c[i]
            current_score += delta
            if current_score < best_score:
                best_score = current_score
                best_coloring = c.copy()
        else:
            # Accept with probability
            # Avoid overflow in exp
            if delta == 0:
                prob = 1.0
            else:
                # If T is very small, exp(-delta/T) -> 0
                # delta is integer, T is float
                prob = np.exp(-delta / T)
            
            if rng.random() < prob:
                c[i] = 1 - c[i]
                current_score += delta
                if current_score < best_score:
                    best_score = current_score
                    best_coloring = c.copy()
        
        step_count += 1
        
        # Cool down
        # Simple linear or geometric cooling
        # Let's use geometric: T = T * 0.9999
        # If T gets too small, it becomes greedy.
        T *= 0.9999
        if T < T_end:
            T = T_end
            
    return best_coloring.tolist()
