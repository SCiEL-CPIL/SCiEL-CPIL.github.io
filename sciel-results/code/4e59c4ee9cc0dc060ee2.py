import numpy as np
import time

N = 50

def compute_ratio(h):
    h = np.asarray(h, dtype=np.float64)
    h = np.maximum(h, 0.0)
    l1 = np.sum(h)
    if l1 < 1e-15:
        return 0.0
    h_norm = h / l1
    h2_sq = np.sum(h_norm**2)
    conv = np.convolve(h_norm, h_norm, mode='full')
    conv_inf = np.max(conv)
    if conv_inf < 1e-15:
        return 0.0
    return h2_sq / conv_inf

def construct():
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin for 150s kill

    best_h = np.zeros(N, dtype=np.float64)
    best_score = -np.inf

    def update_best(h, score):
        nonlocal best_h, best_score
        if score > best_score:
            best_score = score
            best_h = h.copy()

    def time_left():
        return deadline - time.time()

    # --- Phase 1: Generate Structured Candidates ---
    
    # 1. Single spike
    h_spike = np.zeros(N)
    h_spike[N//2] = 1.0
    update_best(h_spike, compute_ratio(h_spike))

    # 2. Two-point distributions
    # We found that [2/3, 1/3] adjacent gives 1.25.
    # Let's search for better 2-point configs.
    for i in range(N):
        for j in range(i+1, N):
            # Search a in [0, 1] for fixed i, j
            # Ratio is (a^2 + (1-a)^2) / max(a^2, (1-a)^2, 2a(1-a)) if positions are distinct enough
            # But positions matter for boundary effects in 'full' conv? 
            # Actually, for 'full' conv, shifting the whole vector shifts the conv.
            # The max of the conv is shift-invariant.
            # So relative position d = j-i matters.
            # Let's just grid search a for a few key pairs.
            # Key pairs: adjacent (d=1), d=2, d=3, etc.
            # To save time, we only do a fine grid for d=1,2,3 and a coarse grid for others.
            d = j - i
            if d <= 3:
                a_vals = np.linspace(0.01, 0.99, 100)
            else:
                a_vals = np.linspace(0.01, 0.99, 20)
            
            for a in a_vals:
                h = np.zeros(N)
                h[i] = a
                h[j] = 1.0 - a
                sc = compute_ratio(h)
                update_best(h, sc)
            if time_left() < 30.0:
                return list(best_h)

    # 3. Three-point distributions
    # Try [a, b, a] symmetric and [a, b, c]
    # Focus on small support sizes.
    # Let's try all triplets (i, j, k) with i<j<k and small gaps.
    # To keep it fast, we'll iterate over a subset.
    # Or just use random triplets + local opt.
    
    # 4. Sidon-like sets (Sparse, distinct sums)
    # A Sidon set minimizes collisions in pair sums, thus minimizing max conv for uniform weights.
    # Let's generate a few Sidon sets greedily.
    
    def find_sidon_set_greedy(n, start_idx=0, step=1):
        A = []
        used_sums = set()
        for x in range(start_idx, N, step):
            valid = True
            new_sums = []
            for a in A:
                s = a + x
                if s in used_sums:
                    valid = False
                    break
                new_sums.append(s)
            if 2 * x in used_sums:
                valid = False
            if valid:
                A.append(x)
                used_sums.update(new_sums)
                used_sums.add(2 * x)
        return A

    for step in [1, 2, 3]:
        for start in range(0, 5, step):
            A = find_sidon_set_greedy(N, start_idx=start, step=step)
            if len(A) >= 2:
                h = np.zeros(N)
                for idx in A:
                    h[idx] = 1.0
                sc = compute_ratio(h)
                update_best(h, sc)
            if time_left() < 30.0:
                return list(best_h)

    # 5. Random Sparse Sets
    rng = np.random.default_rng(42)
    for _ in range(50):
        k = rng.integers(2, 10)
        idx = rng.choice(N, size=k, replace=False)
        h = np.zeros(N)
        h[idx] = 1.0
        sc = compute_ratio(h)
        update_best(h, sc)
        if time_left() < 30.0:
            return list(best_h)

    # 6. Gaussian-like
    x = np.arange(N) - N // 2
    for sigma in np.linspace(0.5, 10.0, 50):
        h = np.exp(-0.5 * (x/sigma)**2)
        sc = compute_ratio(h)
        update_best(h, sc)
        if time_left() < 30.0:
            return list(best_h)

    # --- Phase 2: Local Optimization ---
    # We will refine the best candidates found so far.
    # We also start from random initializations.
    
    def local_optimize(h_init):
        h = h_init.copy()
        h = np.maximum(h, 0.0)
        s = np.sum(h)
        if s < 1e-15: return h, 0.0
        h = h / s
        
        current_score = compute_ratio(h)
        best_local_h = h.copy()
        best_local_score = current_score
        
        # Simple coordinate ascent with line search
        # For each coordinate i, we try to adjust h[i] while keeping sum=1 and h>=0.
        # This is equivalent to moving mass from i to other coordinates.
        # A simpler heuristic: Perturb h, normalize, check score.
        
        for _ in range(200):
            if time_left() < 15.0:
                break
            
            improved = False
            # Try perturbing a random subset of coordinates
            num_perturb = rng.integers(1, 4)
            indices = rng.choice(N, size=num_perturb, replace=False)
            
            h_new = h.copy()
            for idx in indices:
                # Add a small random amount
                h_new[idx] += rng.uniform(-0.1, 0.1)
            
            h_new = np.maximum(h_new, 0.0)
            s_new = np.sum(h_new)
            if s_new < 1e-15:
                continue
            h_new = h_new / s_new
            
            sc_new = compute_ratio(h_new)
            if sc_new > current_score:
                h = h_new
                current_score = sc_new
                improved = True
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local_h = h.copy()
            
            if not improved:
                # If no improvement, try smaller steps or break
                # Let's try a few more with smaller steps
                small_improved = False
                for _ in range(5):
                    h_new = h.copy()
                    idx = rng.choice(N)
                    h_new[idx] += rng.uniform(-0.01, 0.01)
                    h_new = np.maximum(h_new, 0.0)
                    s_new = np.sum(h_new)
                    if s_new < 1e-15: continue
                    h_new = h_new / s_new
                    sc_new = compute_ratio(h_new)
                    if sc_new > current_score:
                        h = h_new
                        current_score = sc_new
                        small_improved = True
                        if current_score > best_local_score:
                            best_local_score = current_score
                            best_local_h = h.copy()
                if not small_improved:
                    break
                    
        return best_local_h, best_local_score

    # Refine the global best
    h_opt, sc_opt = local_optimize(best_h)
    update_best(h_opt, sc_opt)

    # Refine some other good candidates
    # We don't have a list of candidates, so we'll just do a few random starts and refine them.
    for _ in range(20):
        if time_left() < 15.0:
            break
        # Random start: sparse
        k = rng.integers(2, 8)
        idx = rng.choice(N, size=k, replace=False)
        h_rand = np.zeros(N)
        h_rand[idx] = 1.0
        h_opt, sc_opt = local_optimize(h_rand)
        update_best(h_opt, sc_opt)

    # Refine from the specific 2-point [2/3, 1/3] which was analytically good
    h_2pt = np.zeros(N)
    h_2pt[N//2] = 2.0/3.0
    h_2pt[N//2 + 1] = 1.0/3.0
    h_opt, sc_opt = local_optimize(h_2pt)
    update_best(h_opt, sc_opt)

    return list(best_h)
