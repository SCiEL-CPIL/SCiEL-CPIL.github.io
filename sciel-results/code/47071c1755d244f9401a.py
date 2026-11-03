"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def _score(x):
    """Compute max absolute autoconvolution / (sum x)^2. Lower is better."""
    x = np.asarray(x, dtype=np.float64)
    s = float(np.sum(x))
    if s == 0.0:
        return float('inf')
    # linear autoconvolution
    ac = np.correlate(x, x, mode='full')
    # exclude zero-lag (center index N-1)
    center = N - 1
    ac_sidelobes = np.concatenate([ac[:center], ac[center+1:]])
    peak = float(np.max(np.abs(ac_sidelobes)))
    return peak / (s * s)

def _local_search(x, time_budget):
    """Local search: flip elements to reduce peak autoconvolution magnitude."""
    x = x.copy().astype(np.float64)
    t_start = time.time()
    deadline = t_start + time_budget
    
    # Precompute autoconvolution
    ac = np.correlate(x, x, mode='full')
    center = N - 1
    # Sidelobe indices
    sl_idx = np.concatenate([np.arange(0, center), np.arange(center+1, 2*N-1)])
    sl_vals = ac[sl_idx]
    peak_val = float(np.max(np.abs(sl_vals)))
    peak_sl_idx = int(np.argmax(np.abs(sl_vals)))
    
    # Temperature schedule
    T = 1.0
    T_min = 0.01
    T_decay = 0.999
    
    # Track best
    best_x = x.copy()
    best_score = _score(x)
    
    # For efficiency, we'll work with the autoconvolution array
    # Flipping x[i] changes ac[k] for k = i-j terms
    # Specifically, ac[m] = sum_j x[j]*x[j+m-N+1] (depending on convention)
    # np.correlate(x, x, 'full')[m] = sum_j x[j] * x[j + m - (N-1)]
    # So ac[m] involves x[j] and x[j + m - center]
    
    # When we flip x[i] (from +1 to -1 or vice versa), it's equivalent to x[i] -> -x[i]
    # This changes all ac[m] where x[i] is involved:
    # ac[m] contains x[i]*x[i+m-center] and x[i-center+m]*x[i] (if in range)
    # More precisely: ac[m] = sum_j x[j] x[j+m-center]
    # Term j=i: x[i] x[i+m-center]
    # Term j+m-center=i => j = i+m-center: x[i+m-center] x[i]
    # So two terms unless i+m-center == i (m=center, but that's zero-lag which we exclude)
    
    # Delta for flipping x[i]:
    # new_ac[m] = ac[m] - 2*x[i]*x[i+m-center] (if i+m-center in [0,N-1])
    #            - 2*x[i+m-center]*x[i] (if i+m-center in [0,N-1])
    # Wait, let me be more careful.
    # ac[m] = sum_{j: 0<=j<N, 0<=j+m-center<N} x[j] x[j+m-center]
    # If we flip x[i], the terms involving x[i] are:
    # 1. j=i: x[i]*x[i+m-center], valid if 0 <= i+m-center < N
    # 2. j+m-center=i => j=i+m-center: x[i+m-center]*x[i], valid if 0 <= i+m-center < N
    # If i+m-center == i, i.e., m=center, these are the same term (zero-lag).
    # For m != center, if 0 <= i+m-center < N, both terms exist and are equal: 2*x[i]*x[i+m-center]
    # So delta_ac[m] = -4*x[i]*x[i+m-center] if 0 <= i+m-center < N and m != center
    # If i+m-center == i (m=center), delta = -2*x[i]^2 = -2, but we don't track center.
    
    # Let's precompute for each m in sl_idx, the contribution.
    # Actually, let's just recompute efficiently.
    
    # For each flip candidate i, the change in ac for sidelobe index m:
    # Let j2 = i + m - center. If 0 <= j2 < N and m != center:
    #   delta = -4 * x[i] * x[j2]
    # else:
    #   delta = 0 (for that m)
    
    # We want to minimize max |ac_sidelobe|
    # Use simulated annealing or greedy with occasional random restarts
    
    no_improve_count = 0
    max_no_improve = 5000
    
    while time.time() < deadline - 2.0:
        # Find current peak sidelobe
        peak_sl_idx = int(np.argmax(np.abs(sl_vals)))
        peak_abs = float(np.abs(sl_vals[peak_sl_idx]))
        m_peak = sl_idx[peak_sl_idx]
        
        # Try to find a flip that reduces the peak
        best_delta_peak = 0.0
        best_i = -1
        best_new_ac_peak = peak_abs
        
        # Sample a subset of positions for speed
        # Try all positions if fast enough, else random subset
        if N <= 500:
            candidates = range(N)
        else:
            candidates = np.random.choice(N, size=min(N, 200), replace=False)
        
        for i in candidates:
            j2 = i + m_peak - center
            if 0 <= j2 < N and j2 != i:
                # Flipping x[i] changes ac[m_peak] by -4*x[i]*x[j2]
                delta = -4.0 * x[i] * x[j2]
                new_peak_val = peak_abs + delta * (1 if x[i]*x[j2] < 0 else -1)
                # Wait, let me think again.
                # If x[i]*x[j2] = 1, delta = -4, new_ac = peak_val - 4 (if peak_val > 0)
                # If x[i]*x[j2] = -1, delta = 4, new_ac = peak_val + 4
                # We want to reduce |new_ac|
                # If peak_val > 0: we want new_ac to be smaller (could go negative)
                #   If x[i]*x[j2] = 1: new_ac = peak_val - 4, |new_ac| = |peak_val - 4|
                #   If x[i]*x[j2] = -1: new_ac = peak_val + 4, |new_ac| = peak_val + 4 (worse)
                # If peak_val < 0: we want new_ac to be less negative (closer to 0)
                #   If x[i]*x[j2] = 1: new_ac = peak_val - 4, |new_ac| = |peak_val - 4| = 4 - peak_val (worse if peak_val < 0)
                #   If x[i]*x[j2] = -1: new_ac = peak_val + 4, |new_ac| = |peak_val + 4|
                
                # Simpler: compute new absolute value
                new_ac_val = peak_val if m_peak == center else ac[m_peak]
                # Actually peak_val is ac[m_peak]
                # Let me just compute directly
                prod = x[i] * x[j2]
                new_ac_mp = ac[m_peak] - 4.0 * prod
                new_abs = abs(new_ac_mp)
                if new_abs < best_new_ac_peak - 1e-10:
                    best_new_ac_peak = new_abs
                    best_i = i
                    best_delta_peak = -4.0 * prod
        
        if best_i >= 0:
            # Apply flip
            x[best_i] = -x[best_i]
            # Update ac
            # For all m, update ac[m]
            for m in range(2*N-1):
                j2 = best_i + m - center
                if 0 <= j2 < N and j2 != best_i:
                    ac[m] -= 4.0 * x[best_i] * x[j2]
                    # Wait, x[best_i] is already flipped. Let me redo.
                    # Before flip: x_old[best_i] = -x[best_i] (current)
                    # delta = -4 * x_old[best_i] * x_old[j2]
                    # x_old[best_i] = -x[best_i], x_old[j2] = x[j2]
                    # So delta = -4 * (-x[best_i]) * x[j2] = 4 * x[best_i] * x[j2]
                    # But I already applied the flip to x. Let me be careful.
            # This is getting complicated. Let me just recompute ac.
            ac = np.correlate(x, x, mode='full')
            sl_vals = ac[sl_idx]
            
            no_improve_count = 0
        else:
            no_improve_count += 1
            # Try random flips
            if no_improve_count > max_no_improve:
                # Restart with random sequence
                x = np.random.choice([-1.0, 1.0], size=N)
                ac = np.correlate(x, x, mode='full')
                sl_vals = ac[sl_idx]
                no_improve_count = 0
                T = max(T, 0.5)
            else:
                # Random single flip
                i = int(np.random.randint(N))
                x[i] = -x[i]
                ac = np.correlate(x, x, mode='full')
                sl_vals = ac[sl_idx]
        
        T *= T_decay
        if T < T_min:
            T = T_min
        
        # Update best
        cur_score = _score(x)
        if cur_score < best_score:
            best_score = cur_score
            best_x = x.copy()
    
    return best_x, best_score

def _genetic_algorithm(time_budget):
    """Simple GA with local search on elites."""
    t_start = time.time()
    deadline = t_start + time_budget
    
    pop_size = 50
    x0 = np.random.choice([-1.0, 1.0], size=(pop_size, N))
    scores = np.array([_score(x0[i]) for i in range(pop_size)])
    
    best_x = x0[np.argmin(scores)].copy()
    best_score = float(np.min(scores))
    
    gen = 0
    while time.time() < deadline - 5.0:
        # Sort by score
        order = np.argsort(scores)
        x0 = x0[order]
        scores = scores[order]
        
        # Elitism: keep top half
        elite_size = pop_size // 2
        new_pop = np.zeros_like(x0)
        new_pop[:elite_size] = x0[:elite_size]
        
        # Crossover and mutation for rest
        for i in range(elite_size, pop_size):
            if np.random.rand() < 0.7:
                # Crossover
                p1 = int(np.random.randint(0, elite_size))
                p2 = int(np.random.randint(0, elite_size))
                cross = np.random.randint(1, N)
                new_pop[i] = np.concatenate([x0[p1][:cross], x0[p2][cross:]])
            else:
                # Mutation of random individual
                p1 = int(np.random.randint(0, pop_size))
                new_pop[i] = x0[p1].copy()
            
            # Bit flips
            n_flips = np.random.randint(1, max(2, N//10))
            flip_idx = np.random.choice(N, size=n_flips, replace=False)
            new_pop[i][flip_idx] *= -1
        
        # Evaluate
        for i in range(pop_size):
            scores[i] = _score(new_pop[i])
        
        # Update best
        min_idx = int(np.argmin(scores))
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_x = new_pop[min_idx].copy()
        
        x0 = new_pop
        gen += 1
        
        # Apply local search to top few
        if time.time() < deadline - 10.0:
            order = np.argsort(scores)
            top_k = min(3, pop_size)
            for idx in order[:top_k]:
                if time.time() > deadline - 10.0:
                    break
                x_ls, s_ls = _local_search(x0[idx], time_budget=10)
                if s_ls < best_score:
                    best_score = s_ls
                    best_x = x_ls.copy()
                x0[idx] = x_ls
                scores[idx] = s_ls
    
    return best_x, best_score

def construct():
    t_start = time.time()
    total_budget = 120.0  # wall-clock limit per task
    
    # Start with several random initializations and local search
    best_x = None
    best_score = float('inf')
    
    # Try a few different strategies
    # 1. Random + local search
    for trial in range(5):
        if time.time() - t_start > total_budget - 20.0:
            break
        x_rand = np.random.choice([-1.0, 1.0], size=N)
        x_ls, s_ls = _local_search(x_rand, time_budget=15)
        if s_ls < best_score:
            best_score = s_ls
            best_x = x_ls
    
    # 2. GA if time permits
    if time.time() - t_start < total_budget - 30.0:
        x_ga, s_ga = _genetic_algorithm(time_budget=total_budget - (time.time() - t_start) - 5.0)
        if s_ga < best_score:
            best_score = s_ga
            best_x = x_ga
    
    if best_x is None:
        best_x = np.ones(N)
    
    return best_x
# EVOLVE-BLOCK-END
