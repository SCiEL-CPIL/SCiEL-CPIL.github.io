import time
import numpy as np
from scipy.optimize import minimize
import itertools

def _score_set(U):
    """Compute the lower bound 1 + log(|U-U|/|U+U|)/log(2 max U + 1)."""
    if not U or len(U) < 2:
        return 0.0
    arr = np.array(U, dtype=np.int64)
    if arr[0] != 0:
        arr = np.concatenate(([0], arr))
    arr = np.unique(arr)
    n = len(arr)
    if n < 2:
        return 0.0
    mx = int(arr.max())
    if mx == 0:
        return 0.0
    
    # Size of sumset U+U: values in [0, 2*mx]
    # Size of diffset U-U: values in [-mx, mx], shift by mx -> [0, 2*mx]
    N = 2 * mx + 1
    
    # Efficient counting via FFT
    sig = np.zeros(N, dtype=np.float64)
    sig[arr] = 1.0
    
    fft_size = 1
    while fft_size < 2 * N:
        fft_size *= 2
        
    F = np.fft.rfft(sig, n=fft_size)
    
    # Sumset
    S_sum = np.fft.irfft(F * F, n=fft_size)
    sumset_size = np.sum(S_sum[:N] > 1e-6) # Sums range 0..2*mx which is 0..N-1
    
    # Diffset
    flip_sig = np.zeros(N, dtype=np.float64)
    flip_sig[:] = sig[::-1]
    
    F_flip = np.fft.rfft(flip_sig, n=fft_size)
    S_diff = np.fft.irfft(F * F_flip, n=fft_size)
    
    diffset_size = np.sum(S_diff > 1e-6)
    
    if sumset_size == 0:
        return 0.0
        
    ratio = diffset_size / sumset_size
    if ratio <= 0:
        return 0.0
        
    denom = np.log(2 * mx + 1)
    if denom == 0:
        return 0.0
        
    return 1.0 + np.log(ratio) / denom

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Strategy:
    # 1. Parameterize U as a subset of {0, 1, ..., L-1} using a continuous vector p in [0, 1]^L.
    # 2. The "active" set is determined by p > 0.5.
    # 3. We optimize p to maximize a smoothed surrogate of the score.
    #    The score is discrete, so we use a penalty-based or direct gradient ascent on a smoothed version.
    #    Actually, direct gradient ascent on a step function is hard.
    #    Instead, we use a "soft" indicator: w_i = sigmoid(k * (p_i - 0.5)).
    #    However, the sumset size depends on the *support* of the set, not the weights.
    #    
    #    Alternative: 
    #    We fix the size of U to be N. We represent U by a sorted list of integers x_0 < x_1 < ... < x_{N-1}.
    #    x_0 = 0.
    #    We optimize the gaps d_i = x_i - x_{i-1}.
    #    To allow differentiability, we don't directly optimize integers, but we can use a 
    #    "relaxed" set and then round? No, rounding destroys the structure.
    
    #    Let's stick to the continuous parameterization of the *indicator vector* on a fixed grid [0, M].
    #    We define a vector v of length M+1, where v[i] is 1 if i is in U, 0 otherwise.
    #    We optimize v to be binary. This is a 0-1 problem.
    #    We can use scipy.optimize.minimize with method 'COBYLA' or 'Nelder-Mead' on the continuous relaxation?
    #    
    #    Better approach: 
    #    Use the fact that for small N (e.g., 20-50 elements), we can do a local search on the integer positions.
    #    But the current program already does local search. The "inspiration" mentions "non-superincreasing sequences".
    #    
    #    Let's try a different structural change: 
    #    Instead of atoms (mixed radix), we treat U as a "Golomb Ruler" variant but optimized for the specific ratio.
    #    We will use a genetic algorithm / hill climbing on the *gaps* between elements.
    #    State: list of gaps g_1, ..., g_{N-1} where x_i = sum_{j=1}^i g_j.
    #    x_0 = 0.
    #    
    #    To make this "structurally different" from the current atom-based approach:
    #    The current approach assumes a product structure (U = D_0 + M D_1 + ...).
    #    This new approach assumes a *flat* list of integers, allowing arbitrary interactions between all elements.
    #    This can find sets that are NOT mixed-radix factorizable.
    
    # We will perform a local search on the gaps.
    # To make it fast, we need a fast score function.
    # FFT is fast, but for small N (e.g., < 100) and moderate Max U, set operations might be faster?
    # Actually, FFT is O(M log M) where M is the max value. If M is large (e.g., 10^5), FFT is good.
    # If we keep M small (e.g., < 1000), set operations are O(N^2) which is very fast for N=50.
    
    # Let's aim for N ~ 30-50 elements, with Max U ~ 500-2000.
    # This allows O(N^2) scoring to be very fast (< 1ms per eval).
    
    def compute_score_from_gaps(gaps):
        # gaps: list of positive integers
        # Construct U
        U = [0]
        for g in gaps:
            U.append(U[-1] + g)
        U = np.array(U)
        n = len(U)
        if n < 2: return 0.0
        mx = U[-1]
        if mx == 0: return 0.0
        
        # Use sets for speed if mx is small
        if mx < 5000:
            sumset = set()
            diffset = set()
            for i in range(n):
                for j in range(n):
                    sumset.add(U[i] + U[j])
                    diffset.add(U[i] - U[j])
            s_size = len(sumset)
            d_size = len(diffset)
            if s_size == 0: return 0.0
            ratio = d_size / s_size
            if ratio <= 1: return 0.0
            return 1.0 + np.log(ratio) / np.log(2 * mx + 1)
        else:
            # Fall back to FFT if mx is large
            return _score_set(U)

    # Initial population
    # We want to start with diverse structures.
    # 1. Arithmetic progression (bad ratio)
    # 2. Geometric progression (good ratio, large mx)
    # 3. Random gaps
    # 4. "Golomb" style: gaps increasing
    
    best_score = 0.0
    best_gaps = []
    
    # Define a few initial candidates
    candidates = []
    
    # Candidate 1: Small geometric
    g = [1]
    for i in range(1, 10):
        g.append(g[-1] * 2)
    candidates.append(g)
    
    # Candidate 2: Random small gaps
    np.random.seed(42)
    for _ in range(5):
        g = np.random.randint(1, 5, size=15).tolist()
        candidates.append(g)
        
    # Candidate 3: Increasing gaps (convex)
    g = list(range(1, 15))
    candidates.append(g)
    
    # Candidate 4: Alternating gaps
    g = [1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
    candidates.append(g)
    
    # Evaluate initial candidates
    for g in candidates:
        s = compute_score_from_gaps(g)
        if s > best_score:
            best_score = s
            best_gaps = g[:]
            
    # Local Search
    # Moves:
    # 1. Change a gap by +/- 1
    # 2. Insert a new gap
    # 3. Remove a gap
    # 4. Swap two gaps
    
    current_gaps = best_gaps[:]
    current_score = best_score
    
    # We will iterate until time limit
    # To speed up, we can batch evaluations? No, sequential is fine if score is fast.
    
    # Heuristic: 
    # If we increase a gap, mx increases (denom increases), but sumset/diffset structure changes.
    # Generally, we want to keep mx as small as possible for a given ratio improvement.
    
    # Let's add a "global" move: restart with a new random seed if stuck.
    
    step = 0
    no_improve_count = 0
    max_no_improve = 200
    
    while time.time() < deadline:
        step += 1
        improved = False
        
        # Choose a move randomly
        move_type = np.random.randint(0, 4)
        
        new_gaps = current_gaps[:]
        
        if move_type == 0: # Modify existing gap
            if len(new_gaps) > 0:
                idx = np.random.randint(0, len(new_gaps))
                delta = np.random.choice([-2, -1, 1, 2, 3, 5])
                new_val = new_gaps[idx] + delta
                if new_val >= 1:
                    new_gaps[idx] = new_val
        
        elif move_type == 1: # Insert gap
            if len(new_gaps) < 30: # Cap size
                idx = np.random.randint(0, len(new_gaps) + 1)
                new_val = np.random.randint(1, 5)
                new_gaps.insert(idx, new_val)
                
        elif move_type == 2: # Remove gap
            if len(new_gaps) > 5:
                idx = np.random.randint(0, len(new_gaps))
                new_gaps.pop(idx)
                
        elif move_type == 3: # Swap gaps
            if len(new_gaps) > 1:
                i, j = np.random.choice(len(new_gaps), 2, replace=False)
                new_gaps[i], new_gaps[j] = new_gaps[j], new_gaps[i]
        
        if new_gaps != current_gaps:
            new_score = compute_score_from_gaps(new_gaps)
            if new_score > current_score:
                current_gaps = new_gaps
                current_score = new_score
                improved = True
                no_improve_count = 0
                if current_score > best_score:
                    best_score = current_score
                    best_gaps = current_gaps[:]
            else:
                # Simulated annealing style acceptance? 
                # For now, just reject.
                no_improve_count += 1
                
        # Random restart if stuck
        if no_improve_count > max_no_improve:
            # Generate a new random candidate
            size = np.random.randint(10, 25)
            new_gaps = np.random.randint(1, 6, size=size).tolist()
            current_gaps = new_gaps
            current_score = compute_score_from_gaps(current_gaps)
            no_improve_count = 0
            
            # Also try a "scaled" version of the best
            if best_gaps:
                scale = np.random.choice([0.5, 1.5, 2.0])
                scaled = [max(1, int(g * scale)) for g in best_gaps]
                s = compute_score_from_gaps(scaled)
                if s > current_score:
                    current_gaps = scaled
                    current_score = s

    # Construct final set from best_gaps
    if not best_gaps:
        return [0, 1]
        
    U = [0]
    for g in best_gaps:
        U.append(U[-1] + g)
        
    return U

if __name__ == "__main__":
    s = construct()
    print(f"Result: {s[:10]}... Size: {len(s)}, Score: {_score_set(s)}")
