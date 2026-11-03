"""Erdos minimum overlap: minimize max overlap of h with shifted 1-h.
Approach: Simulated Annealing with local hill-climbing polish.
Uses direct evaluation of the max overlap constraint.
"""
import time
import numpy as np

N = 100
_DEADLINE = time.time() + 115  # Safety margin before 120s kill


def _linear_overlap_values(h):
    """
    Compute the linear overlap values C_k for k=0..N-1.
    C_k = (2/N) * (N/2 - sum_{i=k}^{N-1} h_i h_{i-k})
    Vectorized using FFT for speed.
    """
    n = len(h)
    n_fft = 2 * n
    h_pad = np.zeros(n_fft)
    h_pad[:n] = h
    H = np.fft.rfft(h_pad, n=n_fft)
    corr = np.fft.irfft(H * np.conj(H), n=n_fft)
    r = corr[:n]
    cs = (2.0 / n) * (n / 2.0 - r)
    return cs


def _score(h):
    """
    Compute the maximum linear overlap.
    """
    h = np.asarray(h, dtype=np.float64)
    cs = _linear_overlap_values(h)
    return float(np.max(cs))


def _fix_sum(h):
    """
    Adjust h to have sum exactly N/2, keeping values in [0, 1].
    This is a projection onto the hyperplane sum(h)=N/2 intersected with the box [0,1]^N.
    We use a simple iterative projection (POCS) for robustness.
    """
    h = np.array(h, dtype=np.float64, copy=True)
    target = N / 2.0
    
    # Simple POCS: alternate between box constraint and sum constraint
    for _ in range(20):
        # Project onto box
        h = np.clip(h, 0.0, 1.0)
        # Project onto sum
        diff = target - np.sum(h)
        h += diff / N
        # Project onto box again to ensure feasibility after sum adjust
        h = np.clip(h, 0.0, 1.0)
        
    # Final exact fix if needed (rarely needed with POCS)
    diff = target - np.sum(h)
    if abs(diff) > 1e-12:
        # Distribute error among non-boundary elements
        mask = (h > 1e-9) & (h < 1.0 - 1e-9)
        if np.sum(mask) > 0:
            h[mask] += diff / np.sum(mask)
        else:
            # If all are at bounds, just fix one arbitrarily (should not happen for N=100, target=50)
            h[0] += diff
        h = np.clip(h, 0.0, 1.0)
    return h


def _generate_initial():
    """
    Generate a valid initial state.
    We use a structured pattern that is likely to have low overlap, 
    e.g., alternating high/low, then add noise.
    """
    i = np.arange(N)
    # Start with a square wave: 0.8 and 0.2
    h = np.where(i % 2 == 0, 0.8, 0.2)
    # Add some random perturbation
    h += np.random.uniform(-0.1, 0.1, N)
    h = _fix_sum(h)
    return h


def construct():
    """Construct step heights minimizing maximum overlap with shifted complement."""
    t0 = time.time()
    np.random.seed(42)
    
    h = _generate_initial()
    current_score = _score(h)
    best_h = h.copy()
    best_score = current_score
    
    # Simulated Annealing Parameters
    T_start = 0.1
    T_end = 1e-6
    n_iterations = 200000  # Adjust based on time limit
    
    # Pre-calculate constants for speed
    # We will use a cooling schedule: T_k = T_start * (T_end/T_start)^(k/n_iters)
    cooling_rate = (T_end / T_start) ** (1.0 / n_iterations)
    
    T = T_start
    
    for it in range(n_iterations):
        if time.time() - t0 > _DEADLINE - 5:
            break
            
        # 1. Generate a new candidate state
        # Strategy: Pick a random index, perturb it, and re-normalize.
        # To be more efficient, we can pick a block of indices.
        # Let's try a single index perturbation first, as it's simple and effective with SA.
        
        idx = np.random.randint(N)
        # Perturbation size
        delta = np.random.uniform(-0.1, 0.1)
        
        h_new = h.copy()
        h_new[idx] += delta
        
        # Re-normalize
        h_new = _fix_sum(h_new)
        
        # Calculate new score
        new_score = _score(h_new)
        
        # 2. Metropolis Acceptance
        if new_score < current_score:
            # Accept if better
            h = h_new
            current_score = new_score
            if new_score < best_score:
                best_score = new_score
                best_h = h_new.copy()
        else:
            # Accept if worse with probability exp(-(new_score - current_score) / T)
            delta_E = new_score - current_score
            if delta_E > 0:
                prob = np.exp(-delta_E / T)
                if np.random.rand() < prob:
                    h = h_new
                    current_score = new_score
                    if new_score < best_score:
                        best_score = new_score
                        best_h = h_new.copy()
        
        # 3. Cool down
        T *= cooling_rate
        
    # 4. Local Polish (Hill Climbing)
    # Perform local search to fine-tune the solution
    # We try small adjustments to each index and accept if it improves the score.
    # Repeat a few times to ensure convergence to a local minimum.
    
    improved = True
    while improved and (time.time() - t0 < _DEADLINE - 2):
        improved = False
        for idx in range(N):
            if time.time() - t0 > _DEADLINE - 2:
                break
            # Try small positive and negative steps
            for step in [0.01, 0.005, -0.01, -0.005]:
                h_try = h.copy()
                h_try[idx] += step
                h_try = _fix_sum(h_try)
                
                # Check if valid (sum and bounds)
                if np.any(h_try < -1e-9) or np.any(h_try > 1.0 + 1e-9):
                    continue
                    
                score_try = _score(h_try)
                if score_try < current_score - 1e-10:
                    h = h_try
                    current_score = score_try
                    improved = True
                    if current_score < best_score:
                        best_score = current_score
                        best_h = h.copy()
                    break # Restart loop for this index? Or just continue? 
                    # Breaking inner loop is safer to re-evaluate neighbors
                    # But since we are doing hill climbing, if we move, we should check again.
                    # Let's just update and continue to next idx for simplicity in this pass.
                    # A more thorough hill climb would re-check the same idx.
                    # Given time constraints, one pass per index is often sufficient.
        
    # Final fix and return
    best_h = _fix_sum(best_h)
    return best_h
