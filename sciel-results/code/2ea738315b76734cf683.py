"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 400

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 143.0  # 7s margin before 150s kill

    rng = np.random.RandomState(42)

    def compute_correlation(h):
        """Compute aperiodic autocorrelation using FFT."""
        # Pad to power of 2 for FFT
        size = 1 << (2 * N - 1).bit_length()
        H = np.fft.rfft(h, size)
        C = np.fft.irfft(H * np.conj(H), size)[:2 * N - 1]
        # C[0] = c[0], C[1] = c[N-1], ..., C[N-1] = c[0]... 
        # Actually np.fft correlation: C[k] for k=0..N-1 is c[N-1-k]... 
        # Let me use direct: C[0..N-1] corresponds to lags N-1, N-2, ..., 0
        # C[N..2N-2] corresponds to lags 1, 2, ..., N-1
        # So c[k] for k=0..N-1: c[0] = C[N-1], c[k] = C[N-1+k] for k>=1
        c = np.zeros(2 * N - 1)
        c[0] = C[N - 1]  # c[0]
        for k in range(1, N):
            c[k] = C[N - 1 + k]
            c[2 * N - 1 - k] = c[k]  # symmetric
        return c

    def compute_sidelobe_max(h, c):
        """Get max |c[k]| for k != 0."""
        return np.max(np.abs(c[1:]))

    def get_initial_h():
        return rng.choice([-1.0, 1.0], size=N)

    def local_search(h, time_limit, rng, use_sa=True, T_init=50.0, T_min=0.1):
        """Local search with optional simulated annealing."""
        c = compute_correlation(h)
        current_cost = compute_sidelobe_max(h, c)
        best_cost = current_cost
        best_h = h.copy()
        
        T = T_init
        steps = 0
        stall_count = 0
        stall_limit = 2000
        
        while time.time() < time_limit:
            # Pick random position to flip
            j = rng.randint(N)
            old_val = h[j]
            h[j] = -old_val
            
            # Compute new correlation efficiently
            # Update c in-place
            c_new = c.copy()
            for k in range(1, N):
                # delta from h[j]*h[j+k] term (if valid)
                if j + k < N:
                    c_new[k] -= 2.0 * old_val * h[j + k]
                    c_new[2 * N - 1 - k] -= 2.0 * old_val * h[j + k]
                # delta from h[j-k]*h[j] term (if valid)  
                if j - k >= 0:
                    c_new[k] -= 2.0 * h[j - k] * old_val
                    c_new[2 * N - 1 - k] -= 2.0 * h[j - k] * old_val
            # c[0] changes but we don't track it for cost
            c = c_new
            new_cost = compute_sidelobe_max(h, c)
            delta = new_cost - current_cost
            
            if delta < 0:
                current_cost = new_cost
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_h = h.copy()
                stall_count = 0
            else:
                if use_sa and T > T_min:
                    prob = np.exp(-delta / T)
                    if rng.random_sample() < prob:
                        current_cost = new_cost
                    else:
                        h[j] = old_val
                        c = c_new  # Wait, this is wrong. Need to revert c too.
                        # Actually let's just recompute c if we revert
                        c = compute_correlation(h)
                        current_cost = compute_sidelobe_max(h, c)
                else:
                    h[j] = old_val
                    c = compute_correlation(h)
                    current_cost = compute_sidelobe_max(h, c)
                    stall_count += 1
            
            steps += 1
            T *= 0.9995
            
            if stall_count > stall_limit and not use_sa:
                break
            
            # Periodic check
            if steps % 5000 == 0 and time.time() > time_limit:
                break
        
        return best_h, best_cost

    # Phase 1: Multiple random restarts with local search
    best_h = get_initial_h()
    best_cost = np.inf

    num_restarts = 20
    for r in range(num_restarts):
        if time.time() > deadline - 5:
            break
        
        h = get_initial_h()
        time_per_restart = (deadline - 5 - time.time()) / (num_restarts - r)
        time_per_restart = max(1.0, min(time_per_restart, 5.0))
        
        h_local, cost_local = local_search(h, time.time() + time_per_restart, rng, use_sa=True)
        if cost_local < best_cost:
            best_cost = cost_local
            best_h = h_local.copy()
        
        # Print progress
        if (r + 1) % 5 == 0:
            elapsed = time.time() - start_time
            print(f"Restart {r+1}/{num_restarts}, best_cost={best_cost:.4f}, elapsed={elapsed:.1f}s")

    # Phase 2: Final refinement with more aggressive local search on best
    if time.time() < deadline - 2:
        h_refined, cost_refined = local_search(best_h.copy(), deadline - 1, rng, use_sa=False)
        if cost_refined < best_cost:
            best_cost = cost_refined
            best_h = h_refined.copy()

    return best_h
# EVOLVE-BLOCK-END
