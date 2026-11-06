import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 5s safety margin
    
    n = 128
    target_low = 128
    target_high = 256
    num_targets = target_high - target_low + 1  # 129
    
    # Pre-compute indices for target range in convolution result
    # Convolution of two length 129 vectors has length 257
    # Index t in convolution corresponds to sum t
    target_indices = np.arange(target_low, target_high + 1)
    
    def compute_conv(v):
        return np.convolve(v, v)
    
    def calc_metrics(C):
        # C[t] is count of ordered pairs (i,j) with i+j=t
        # We need unordered pairs {a,b} with a<=b
        # N_t = (C[t] + I(t/2 in A)) / 2
        # I(t/2 in A) is 1 if t is even and t/2 < len(v) and v[t//2]==1
        
        N = np.zeros(num_targets, dtype=np.float64)
        for idx, t in enumerate(target_indices):
            c_val = C[t]
            diag = 0.0
            if t % 2 == 0:
                half = t // 2
                if 0 <= half < len(v):
                    diag = v[half]
            N[idx] = (c_val + diag) / 2.0
        
        # Check coverage
        uncovered = np.sum(N < 1.0)
        if uncovered > 0:
            # Heavy penalty for not covering
            return float('inf'), N
        
        peak = np.max(N)
        return peak, N
    
    def objective(C, v):
        peak, N = calc_metrics(C)
        if peak == float('inf'):
            return peak
        # Smooth surrogate: sum of squares + peak
        # Using sum of squares helps balance the load
        score = np.sum(N**2) + 10.0 * peak
        return score
    
    best_score = float('inf')
    best_v = None
    
    # Try different sizes of A
    # Based on problem constraints, k around 30-50 might be optimal
    sizes_to_try = [35, 40, 45, 50, 55]
    
    for k in sizes_to_try:
        if time.time() > deadline:
            break
            
        # Multiple restarts for each size
        max_restarts = 10
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Initialize A
            A = set(random.sample(range(0, n + 1), k))
            v = np.zeros(n + 1, dtype=np.float64)
            for a in A:
                v[a] = 1.0
            
            C = compute_conv(v)
            curr_score = objective(C, v)
            
            # If initial is invalid (inf), try to fix or restart
            if curr_score == float('inf'):
                # Quick local search to fix coverage
                fixed = False
                for _ in range(100):
                    if time.time() > deadline: break
                    # Add a random element or swap
                    # Try adding a random element not in A
                    candidates = [i for i in range(n+1) if i not in A]
                    if not candidates: continue
                    y = random.choice(candidates)
                    # Find x to remove? No, just add? 
                    # Size must be fixed? The problem says "subset A", size is not fixed.
                    # "Minimize the largest representation multiplicity".
                    # Smaller A generally has fewer pairs, so lower peak potential, 
                    # but must cover.
                    # Let's allow size to change slightly or keep fixed.
                    # Keeping fixed is easier for local search.
                    # If invalid, we can try swapping to improve coverage.
                    
                    # Pick random x in A, y not in A
                    x = random.choice(list(A))
                    # Try all y? No, random y.
                    # Actually, if invalid, we should prioritize coverage.
                    # Let's just do a simple hill climb on the score.
                    # If score is inf, any move that reduces inf is good.
                    
                    # Let's just run the local search loop below, it will handle it.
                    pass
            
            # Local Search
            # We use a smooth objective: sum(N^2) + lambda * peak
            # But we must enforce N >= 1.
            
            curr_peak, curr_N = calc_metrics(C)
            
            for step in range(5000):
                if time.time() > deadline:
                    break
                
                # Pick x in A, y not in A
                if not A: continue
                x = random.choice(list(A))
                # Get candidates for y
                # To speed up, we can just pick a random y in [0, 128]
                y = random.randint(0, n)
                
                if x == y:
                    continue
                if y in A:
                    continue
                
                # Update v and C
                # v[x] = 0, v[y] = 1
                # C_new = C + 2*v*(e_y - e_x) + (e_y - e_x)*(e_y - e_x)
                
                # v * e_y is v shifted by y
                # v * e_x is v shifted by x
                # Note: np.convolve handles boundaries.
                # Let's do the shift manually for speed or use np.roll?
                # v has length 129. C has length 257.
                
                # delta_v = -e_x + e_y
                # C_new = C + 2 * np.convolve(v, delta_v) + np.convolve(delta_v, delta_v)
                # This is O(N) if we do it smart, but np.convolve is O(N^2) or FFT.
                # N=129 is small. O(N^2) is ~16k ops. 5000 steps * 16k = 80M ops.
                # Python loop might be too slow. NumPy is fast.
                
                # Let's try to update C directly using slicing.
                # C[s] = sum v[i] v[s-i]
                # Change at x: v[x] goes 1->0.
                # Change at y: v[y] goes 0->1.
                
                # Contribution of x to C:
                # For each i, C[i+x] includes v[i]v[x].
                # So we subtract v[i] from C[i+x] for all i.
                # Contribution of y to C:
                # For each i, C[i+y] includes v[i]v[y].
                # So we add v_new[i] to C[i+y] for all i.
                
                # v_new is v with x=0, y=1.
                
                # 1. Remove x
                # For i in 0..128:
                #   C[i+x] -= v[i]
                # Note: v[x] is currently 1.
                
                # 2. Add y
                # For i in 0..128:
                #   C[i+y] += v_new[i]
                # v_new[i] is v[i] except v_new[x]=0, v_new[y]=1.
                
                # Let's implement this update.
                
                C_new = C.copy()
                
                # Remove x
                # v[i] for i in 0..128
                # C indices: i+x
                # We need to be careful with boundaries of C (len 257)
                # i+x ranges from x to 128+x. Since x<=128, max index 256. OK.
                
                # Subtract v from C shifted by x
                # C_new[x : x + 129] -= v
                
                # Add y
                # We need v_new.
                # v_new is v, but v[x]=0, v[y]=1.
                # Since x!=y, and x in A (v[x]=1), y not in A (v[y]=0).
                # v_new[x] = 0, v_new[y] = 1.
                # Other v_new[i] = v[i].
                
                # C_new[y : y + 129] += v_new
                
                # Let's construct v_new slice efficiently?
                # v_new is just v with two points changed.
                
                # Step 1: Subtract v from C_new[x:x+129]
                C_new[x:x+129] -= v
                
                # Step 2: Add v_new to C_new[y:y+129]
                # v_new = v.copy()
                # v_new[x] = 0
                # v_new[y] = 1
                # But we can do:
                # C_new[y:y+129] += v
                # Then adjust for the difference between v and v_new.
                # v_new[i] - v[i] is non-zero only at x and y.
                # At x: v_new[x]=0, v[x]=1. Diff = -1.
                # At y: v_new[y]=1, v[y]=0. Diff = +1.
                
                # So C_new[y:y+129] += v
                C_new[y:y+129] += v
                
                # Now apply corrections:
                # At index corresponding to i=x in the shift y:
                # The term added was v[x]=1. We wanted v_new[x]=0.
                # So we subtract 1 from C_new[y+x].
                C_new[y+x] -= 1
                
                # At index corresponding to i=y in the shift y:
                # The term added was v[y]=0. We wanted v_new[y]=1.
                # So we add 1 to C_new[y+y].
                C_new[y+y] += 1
                
                # Wait, is this correct?
                # C_new[s] = sum_i v_new[i] v_new[s-i]
                # Old C[s] = sum_i v[i] v[s-i]
                
                # Let's verify the update logic.
                # C_new = (v - e_x + e_y) * (v - e_x + e_y)
                # = v*v - v*e_x - e_x*v + e_x*e_x + v*e_y - v*e_x - e_y*v + e_y*e_y + e_y*e_x + e_x*e_y
                # This is getting complicated. The "remove then add" logic above is:
                # C_temp = C - v*e_x (Remove x contribution from old v)
                # C_new = C_temp + v_new*e_y ? No.
                
                # Let's stick to the simpler, slightly slower but robust NumPy convolution for now,
                # or verify the incremental update carefully.
                
                # Actually, for N=129, np.convolve is very fast.
                # 5000 steps * 129^2 ~ 80M operations.
                # In NumPy, this is likely < 1 second per 1000 steps?
                # Let's benchmark mentally. 129*129 = 16k multiplies/adds.
                # 5000 * 16k = 80M.
                # Modern CPU does ~100M-1000M ops/sec in C/NumPy.
                # So 5000 steps should take < 1s.
                # This is safe.
                
                # Let's revert to simple convolution update for correctness and simplicity.
                # It's likely fast enough.
                
                v_trial = v.copy()
                v_trial[x] = 0.0
                v_trial[y] = 1.0
                
                C_trial = np.convolve(v_trial, v_trial)
                
                peak_trial, N_trial = calc_metrics(C_trial)
                
                if peak_trial == float('inf'):
                    score_trial = float('inf')
                else:
                    score_trial = np.sum(N_trial**2) + 10.0 * peak_trial
                
                # Acceptance criterion
                if score_trial < curr_score:
                    # Accept
                    v = v_trial
                    C = C_trial
                    A.remove(x)
                    A.add(y)
                    curr_score = score_trial
                    curr_peak = peak_trial
                    curr_N = N_trial
                    
                    if curr_score < best_score:
                        best_score = curr_score
                        best_v = v.copy()
                        best_A = set(i for i, val in enumerate(v) if val > 0.5)
                        
                # Optional: Simulated Annealing acceptance for equal/worse scores to escape local minima
                # if score_trial <= curr_score + 1.0 and random.random() < 0.1:
                #     ...
                # Given time constraints, greedy hill climbing with restarts is safer.
    
    if best_v is None:
        # Fallback: return a valid set, e.g., all even numbers or something dense
        # [0, 2, ..., 128] sums to [0, 256] even. Odd sums not covered.
        # [0, 1, ..., 64] sums to [0, 128]. Not enough.
        # Just return range(0, 129) which is valid.
        return list(range(0, 129))
    
    return list(best_A)
