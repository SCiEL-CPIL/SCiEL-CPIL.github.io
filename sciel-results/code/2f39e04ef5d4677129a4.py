import numpy as np
import time

# EVOLVE-BLOCK-START
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
    
    # 2. Precompute "partner" indices for each position in each AP
    # For each index i, store the list of (other1, other2, other3) for each AP containing i.
    # This allows computing the delta of flipping i in O(degree(i)).
    index_to_aps = [[] for _ in range(N)]
    for ap_idx in range(num_aps):
        a, b, c, d = aps[ap_idx]
        index_to_aps[a].append((b, c, d))
        index_to_aps[b].append((a, c, d))
        index_to_aps[c].append((a, b, d))
        index_to_aps[d].append((a, b, c))
        
    # Convert to numpy arrays for faster access if possible, but lists of tuples are okay for N=256
    # We will compute delta for a specific index i given current coloring c.
    
    def compute_score(c):
        """Compute total number of monochromatic 4-APs."""
        ca = c[aps[:, 0]]
        cb = c[aps[:, 1]]
        cc = c[aps[:, 2]]
        cd = c[aps[:, 3]]
        is_mono = (ca == cb) & (cb == cc) & (cc == cd)
        return np.sum(is_mono)
        
    def get_delta(c, i):
        """
        Compute the change in score if we flip index i.
        Returns negative delta if flipping improves (reduces) score.
        """
        val_i = c[i]
        new_val_i = 1 - val_i
        delta = 0
        
        for p1, p2, p3 in index_to_aps[i]:
            v1, v2, v3 = c[p1], c[p2], c[p3]
            # Mono before: v1==v2==v3==val_i
            # Mono after: v1==v2==v3==new_val_i
            if v1 == v2 and v2 == v3:
                if v1 == val_i:
                    delta -= 1 # Was mono, won't be
                else:
                    delta += 1 # Wasn't mono, will be
        return delta

    def tabu_search(initial_c, max_iters=2000, tabu_size=50):
        """
        Run Tabu Search starting from initial_c.
        Returns best_score, best_coloring.
        """
        c = initial_c.copy()
        current_score = compute_score(c)
        best_score = current_score
        best_c = c.copy()
        
        # Tabu list: circular buffer of indices
        tabu_buffer = [-1] * tabu_size
        tabu_ptr = 0
        
        # Aspiration: if a move is tabu but improves global best, allow it.
        
        no_improve_count = 0
        stuck_threshold = 200 # Restart if stuck for this many iterations
        
        for it in range(max_iters):
            if time.time() > deadline - 2:
                break
            
            # If stuck, perturb
            if no_improve_count > stuck_threshold:
                # Randomly flip 5% of bits
                n_flip = max(1, N // 20)
                flip_indices = np.random.choice(N, n_flip, replace=False)
                for idx in flip_indices:
                    c[idx] = 1 - c[idx]
                current_score = compute_score(c)
                if current_score < best_score:
                    best_score = current_score
                    best_c = c.copy()
                no_improve_count = 0
                # Clear tabu list
                tabu_buffer = [-1] * tabu_size
                tabu_ptr = 0
                continue

            # Find best move
            best_delta = 0 # 0 means no improvement found yet, but we must move
            best_move_idx = -1
            best_move_delta = 0
            found_improvement = False
            
            # We evaluate all moves. N=256, degree ~ O(N), so 256 * 100 ~ 25k ops. Fast.
            # To avoid bias, we could shuffle, but Tabu usually uses fixed order or random.
            # Let's use random order to diversify.
            order = np.random.permutation(N)
            
            for i in order:
                # Check if i is tabu
                # Simple check: is i in tabu_buffer? 
                # For efficiency with small tabu_size, linear search or set is fine.
                # Let's use a set for O(1) lookup, but rebuilding set each iter is O(T).
                # T=50, N=256. 256*50 = 12800 checks. Fine.
                is_tabu = False
                for t_idx in range(tabu_size):
                    if tabu_buffer[t_idx] == i:
                        is_tabu = True
                        break
                
                if is_tabu:
                    # Check aspiration: does this move improve global best?
                    delta = get_delta(c, i)
                    if current_score + delta < best_score:
                        # Allow it
                        if best_move_idx == -1 or delta < best_move_delta:
                            best_move_idx = i
                            best_move_delta = delta
                            found_improvement = (delta < 0)
                    # Else skip
                else:
                    delta = get_delta(c, i)
                    if best_move_idx == -1 or delta < best_move_delta:
                        best_move_idx = i
                        best_move_delta = delta
                        found_improvement = (delta < 0)
            
            if best_move_idx == -1:
                # Should not happen unless N=0
                break
                
            # Execute move
            c[best_move_idx] = 1 - c[best_move_idx]
            current_score += best_move_delta
            
            # Update tabu
            tabu_buffer[tabu_ptr] = best_move_idx
            tabu_ptr = (tabu_ptr + 1) % tabu_size
            
            # Update best
            if current_score < best_score:
                best_score = current_score
                best_c = c.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
        return best_score, best_c

    best_score = np.inf
    best_coloring = None
    
    # Strategy: Multi-start Tabu Search
    # 1. Random starts
    # 2. Structured starts (Thue-Morse, Alternating)
    
    rng = np.random.default_rng()
    
    # Start 1: Thue-Morse sequence
    # Thue-Morse: t_n = parity of number of 1s in binary representation of n
    def thue_morse(n):
        return bin(n).count('1') % 2
        
    c_tm = np.array([thue_morse(i) for i in range(N)], dtype=np.int8)
    score_tm, c_tm_best = tabu_search(c_tm, max_iters=1000)
    if score_tm < best_score:
        best_score = score_tm
        best_coloring = c_tm_best.copy()
        
    # Start 2: Alternating 0,1,0,1...
    c_alt = np.array([i % 2 for i in range(N)], dtype=np.int8)
    score_alt, c_alt_best = tabu_search(c_alt, max_iters=1000)
    if score_alt < best_score:
        best_score = score_alt
        best_coloring = c_alt_best.copy()
        
    # Multi-start random Tabu Search
    # We have time budget. Each tabu_search takes ~1000 iters * 256 moves * 100 partners = 25M ops?
    # Python is slow. 25M ops in Python is ~1-2 seconds? 
    # Let's estimate: get_delta is a loop over ~10-20 partners on average?
    # Max APs per index:
    # For i, number of APs is sum over d such that i is in (a, a+d, a+2d, a+3d).
    # This is roughly O(N) in worst case? No, O(N/d) for each d. Sum 1/d ~ log N.
    # So degree is small, ~10-20.
    # 256 * 20 = 5120 ops per iteration.
    # 1000 iters = 5M ops.
    # 5M Python ops ~ 0.5s - 1s.
    # We have 145s. So we can run ~100-200 restarts.
    
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
            
        c_init = rng.integers(0, 2, size=N, dtype=np.int8)
        
        # Adaptive max_iters: use more time if early, less if late?
        # Just use fixed 500 iters to allow more restarts.
        score, c_best = tabu_search(c_init, max_iters=500, tabu_size=30)
        
        if score < best_score:
            best_score = score
            best_coloring = c_best.copy()
            
    if best_coloring is None:
        return [0] * N
        
    return best_coloring.tolist()
# EVOLVE-BLOCK-END
