"""Erdos minimum overlap: combine theoretical construction with FFT-based coordinate refinement."""
import time
import math
import numpy as np

N = 100

# Theoretical optimum for continuous case: h=a on [0,1], h=b=1-a on [1,2]
# minimize max(a, (1-a)^2) -> a = (3 - sqrt(5))/2
A = (3.0 - math.sqrt(5.0)) / 2.0  # ~0.381966
B = 1.0 - A                        # ~0.618034


def _compute_max_overlap(h):
    """
    Compute the maximum overlap of h with shifted (1-h).
    Overlap at shift k: sum_i h[i] * (1 - h[(i+k) % N]) for k=0,...,N-1
    Returns the maximum value over all shifts.
    """
    # overlap[k] = sum_i h[i] * (1 - h[(i+k)%N])
    #            = sum(h) - sum_i h[i]*h[(i+k)%N]
    # sum_i h[i]*h[(i+k)%N] is the circular autocorrelation at lag k
    H = np.fft.fft(h)
    autocorr = np.fft.ifft(np.abs(H) ** 2).real  # autocorr[m] = sum_i h[i]*h[(i-m)%N]
    # By symmetry of autocorrelation for real signals: autocorr[k] = autocorr[-k]
    # So sum_i h[i]*h[(i+k)%N] = autocorr[k] (since autocorr is even)
    sum_h = np.sum(h)
    overlaps = sum_h - autocorr
    return np.max(overlaps)


def _enforce_sum_constraint(h, target_sum):
    """Clip to [0,1] and adjust to make sum exactly target_sum."""
    h = np.clip(h, 0.0, 1.0)
    diff = target_sum - np.sum(h)
    if abs(diff) < 1e-14:
        return h
    if diff > 0:
        # Need to increase sum
        room = 1.0 - h
        total_room = np.sum(room)
        if total_room > 0:
            h += diff * room / total_room
    else:
        # Need to decrease sum
        room = h
        total_room = np.sum(room)
        if total_room > 0:
            h += diff * room / total_room
    h = np.clip(h, 0.0, 1.0)
    # Final fine adjustment
    diff = target_sum - np.sum(h)
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            remaining = diff - add
            if abs(remaining) > 1e-12:
                for i in range(N):
                    if abs(remaining) < 1e-12:
                        break
                    room = 1.0 - h[i]
                    if room > 0:
                        adj = min(remaining, room)
                        h[i] += adj
                        remaining -= adj
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            remaining = -diff - sub
            if abs(remaining) > 1e-12:
                for i in range(N):
                    if abs(remaining) < 1e-12:
                        break
                    room = h[i]
                    if room > 0:
                        adj = min(remaining, room)
                        h[i] -= adj
                        remaining -= adj
    return np.clip(h, 0.0, 1.0)


def construct():
    """
    Construct step heights h in [0,1] with sum = N/2 that minimize
    the maximum overlap of h with shifted (1-h).
    
    Strategy:
    1. Start with the theoretical two-block construction (A on first half, B on second half)
    2. Apply FFT-based coordinate descent to refine the discrete solution
    3. Use multiple restarts with different block boundaries and random perturbations
    """
    deadline = time.time() + 110.0  # Safety margin before 120s limit
    target_sum = N / 2.0
    
    best_h = None
    best_score = float('inf')
    
    # Phase 1: Theoretical two-block construction
    half = N // 2
    h0 = np.zeros(N)
    h0[:half] = A
    h0[half:] = B
    h0 = _enforce_sum_constraint(h0, target_sum)
    
    score0 = _compute_max_overlap(h0)
    if score0 < best_score:
        best_score = score0
        best_h = h0.copy()
    
    # Phase 2: Coordinate descent refinement
    # Try different step sizes and multiple passes
    step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
    
    for step in step_sizes:
        if time.time() > deadline:
            break
        
        improved = True
        max_passes = 20
        for _pass in range(max_passes):
            if time.time() > deadline:
                break
            if not improved:
                break
            improved = False
            
            for i in range(N):
                if time.time() > deadline:
                    break
                
                old_val = h0[i]
                # Try increasing
                new_val = min(1.0, old_val + step)
                if abs(new_val - old_val) > 1e-10:
                    h0[i] = new_val
                    h0 = _enforce_sum_constraint(h0, target_sum)
                    new_score = _compute_max_overlap(h0)
                    if new_score < best_score - 1e-10:
                        best_score = new_score
                        best_h = h0.copy()
                        improved = True
                    else:
                        h0[i] = old_val
                        h0 = _enforce_sum_constraint(h0, target_sum)
                
                # Try decreasing
                new_val = max(0.0, old_val - step)
                if abs(new_val - old_val) > 1e-10:
                    h0[i] = new_val
                    h0 = _enforce_sum_constraint(h0, target_sum)
                    new_score = _compute_max_overlap(h0)
                    if new_score < best_score - 1e-10:
                        best_score = new_score
                        best_h = h0.copy()
                        improved = True
                    else:
                        h0[i] = old_val
                        h0 = _enforce_sum_constraint(h0, target_sum)
    
    # Phase 3: Try alternative block boundaries
    if time.time() < deadline:
        for boundary in range(N // 4, 3 * N // 4 + 1, 5):
            if time.time() > deadline:
                break
            
            h_alt = np.zeros(N)
            h_alt[:boundary] = A
            h_alt[boundary:] = B
            h_alt = _enforce_sum_constraint(h_alt, target_sum)
            
            alt_score = _compute_max_overlap(h_alt)
            if alt_score < best_score - 1e-10:
                best_score = alt_score
                best_h = h_alt.copy()
                # Refine this solution
                h_work = h_alt.copy()
                for step in [0.05, 0.01, 0.001]:
                    if time.time() > deadline:
                        break
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(N):
                            if time.time() > deadline:
                                break
                            old_val = h_work[i]
                            new_val = min(1.0, old_val + step)
                            if abs(new_val - old_val) > 1e-10:
                                h_work[i] = new_val
                                h_work = _enforce_sum_constraint(h_work, target_sum)
                                new_score = _compute_max_overlap(h_work)
                                if new_score < best_score - 1e-10:
                                    best_score = new_score
                                    best_h = h_work.copy()
                                    improved = True
                                else:
                                    h_work[i] = old_val
                                    h_work = _enforce_sum_constraint(h_work, target_sum)
                            
                            new_val = max(0.0, old_val - step)
                            if abs(new_val - old_val) > 1e-10:
                                h_work[i] = new_val
                                h_work = _enforce_sum_constraint(h_work, target_sum)
                                new_score = _compute_max_overlap(h_work)
                                if new_score < best_score - 1e-10:
                                    best_score = new_score
                                    best_h = h_work.copy()
                                    improved = True
                                else:
                                    h_work[i] = old_val
                                    h_work = _enforce_sum_constraint(h_work, target_sum)
    
    # Phase 4: Random restarts with sinusoidal perturbation
    if time.time() < deadline:
        rng = np.random.RandomState(42)
        for restart in range(5):
            if time.time() > deadline:
                break
            
            h_r = np.zeros(N)
            for i in range(N):
                # Sinusoidal perturbation around the two-block structure
                base = A if i < half else B
                perturbation = 0.1 * math.sin(2 * math.pi * (i + rng.randint(10)) / N * (restart + 1))
                h_r[i] = np.clip(base + perturbation, 0.0, 1.0)
            h_r = _enforce_sum_constraint(h_r, target_sum)
            
            r_score = _compute_max_overlap(h_r)
            if r_score < best_score - 1e-10:
                best_score = r_score
                best_h = h_r.copy()
                # Refine
                h_work = h_r.copy()
                for step in [0.05, 0.01, 0.001]:
                    if time.time() > deadline:
                        break
                    improved = True
                    while improved and time.time() < deadline:
                        improved = False
                        for i in range(N):
                            if time.time() > deadline:
                                break
                            old_val = h_work[i]
                            new_val = min(1.0, old_val + step)
                            if abs(new_val - old_val) > 1e-10:
                                h_work[i] = new_val
                                h_work = _enforce_sum_constraint(h_work, target_sum)
                                new_score = _compute_max_overlap(h_work)
                                if new_score < best_score - 1e-10:
                                    best_score = new_score
                                    best_h = h_work.copy()
                                    improved = True
                                else:
                                    h_work[i] = old_val
                                    h_work = _enforce_sum_constraint(h_work, target_sum)
                            
                            new_val = max(0.0, old_val - step)
                            if abs(new_val - old_val) > 1e-10:
                                h_work[i] = new_val
                                h_work = _enforce_sum_constraint(h_work, target_sum)
                                new_score = _compute_max_overlap(h_work)
                                if new_score < best_score - 1e-10:
                                    best_score = new_score
                                    best_h = h_work.copy()
                                    improved = True
                                else:
                                    h_work[i] = old_val
                                    h_work = _enforce_sum_constraint(h_work, target_sum)
    
    # Final: ensure valid output
    if best_h is None:
        best_h = h0.copy()
    
    best_h = _enforce_sum_constraint(best_h, target_sum)
    
    # Final verification
    assert np.all(best_h >= -1e-10) and np.all(best_h <= 1.0 + 1e-10), "Heights out of bounds"
    assert abs(np.sum(best_h) - target_sum) < 1e-8, f"Sum constraint violated: {np.sum(best_h)} vs {target_sum}"
    
    return best_h
