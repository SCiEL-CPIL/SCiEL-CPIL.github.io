import time
import math
import numpy as np
from scipy.optimize import minimize

N = 100
TARGET_SUM = N / 2.0
DEADLINE = 120.0  # Safety margin for 150s limit

def _linear_overlap_values(h):
    """
    Compute linear overlap values O[k] for k = 0 to N-1.
    O[k] = sum_{i=0}^{N-1-k} h[i] * (1 - h[i+k])
    """
    # O[k] = sum_{i=0}^{N-1-k} h[i] - sum_{i=0}^{N-1-k} h[i] h[i+k]
    # We can compute this efficiently.
    # sum_h_prefix[i] = sum_{j=0}^i h[j]
    # But the second term is a cross-correlation.
    
    overlaps = np.zeros(N)
    for k in range(N):
        # Valid i: 0 <= i <= N-1-k
        # h[i] and h[i+k]
        if k == 0:
            # sum h[i] - sum h[i]^2
            overlaps[k] = np.sum(h) - np.sum(h**2)
        else:
            # sum_{i=0}^{N-1-k} h[i]
            s1 = np.sum(h[:N-k])
            # sum_{i=0}^{N-1-k} h[i] h[i+k]
            s2 = np.sum(h[:N-k] * h[k:])
            overlaps[k] = s1 - s2
    return overlaps

def _max_overlap(h):
    """Compute the raw score: max overlap."""
    return np.max(_linear_overlap_values(h))

def _smooth_max_overlap(h, tau):
    """Smooth approximation of max overlap."""
    ov = _linear_overlap_values(h)
    # Avoid overflow
    m = np.max(ov)
    if m > 700:
        m = 700
    # log-sum-exp
    lse = m + tau * math.log(np.sum(np.exp((ov - m) / tau)))
    return lse

def _grad_smooth_max_overlap(h, tau):
    """
    Gradient of the smooth max overlap w.r.t. h.
    G = sum_k w_k * grad(O[k])
    w_k = exp(O[k]/tau) / sum(exp(O[m]/tau))
    """
    ov = _linear_overlap_values(h)
    m = np.max(ov)
    if m > 700:
        m = 700
    exp_ov = np.exp((ov - m) / tau)
    w = exp_ov / np.sum(exp_ov)
    
    # Compute gradient of each O[k]
    # grad(O[k])[j] = 1_{j < N-k} - (h[j+k] 1_{j+k < N} + h[j-k] 1_{j >= k})
    
    grad_h = np.zeros(N)
    for k in range(N):
        wk = w[k]
        if wk < 1e-10:
            continue
        
        # Part 1: d/dh[j] (sum_{i=0}^{N-1-k} h[i])
        # This is 1 if j < N-k, else 0.
        # Vectorized: 1 - eye(N-k) essentially, but we just add 1 to first N-k elements.
        # Actually, sum_{i=0}^{N-1-k} h[i] depends on h[0]...h[N-1-k].
        # So grad is 1 for j in [0, N-1-k], 0 otherwise.
        
        # Part 2: d/dh[j] (sum_{i=0}^{N-1-k} h[i]h[i+k])
        # Term h[j]h[j+k] exists if j+k < N => j < N-k. Deriv: h[j+k].
        # Term h[j-k]h[j] exists if j-k >= 0 => j >= k. Deriv: h[j-k].
        
        # Initialize temp grad for this k
        g_k = np.zeros(N)
        
        # Add 1 to indices 0 to N-1-k
        if N-k > 0:
            g_k[:N-k] += 1.0
            
        # Subtract h[j+k] for j < N-k
        if N-k > 0:
            # j ranges 0 to N-1-k
            # j+k ranges k to N-1
            g_k[:N-k] -= h[k:]
            
        # Subtract h[j-k] for j >= k
        if k < N:
            # j ranges k to N-1
            # j-k ranges 0 to N-1-k
            g_k[k:] -= h[:N-k]
            
        grad_h += wk * g_k
        
    return grad_h

def _project_sum(h, target_sum):
    """
    Project h to satisfy sum(h) = target_sum and 0 <= h <= 1.
    Simple iterative water-filling or proportional adjustment.
    """
    h = np.clip(h, 0.0, 1.0)
    current_sum = np.sum(h)
    diff = target_sum - current_sum
    
    if abs(diff) < 1e-10:
        return h
        
    # If we need to add (diff > 0), add to elements with most room.
    # If we need to subtract (diff < 0), subtract from elements with most value.
    # To be robust, we do a few iterations of proportional adjustment.
    for _ in range(5):
        if abs(diff) < 1e-12:
            break
        if diff > 0:
            room = 1.0 - h
            total_room = np.sum(room)
            if total_room < 1e-10:
                break
            h += diff * room / total_room
        else:
            total_val = np.sum(h)
            if total_val < 1e-10:
                break
            h += diff * h / total_val
        h = np.clip(h, 0.0, 1.0)
        diff = target_sum - np.sum(h)
        
    # Final fix: distribute remaining diff to the element with most room/value
    if abs(diff) > 1e-12:
        if diff > 0:
            idx = np.argmax(1.0 - h)
            add = min(diff, 1.0 - h[idx])
            h[idx] += add
            diff -= add
            if diff > 1e-12:
                # Distribute rest
                mask = np.ones(N, dtype=bool)
                mask[idx] = False
                room = 1.0 - h[mask]
                tr = np.sum(room)
                if tr > 0:
                    h[mask] += diff * room / tr
        else:
            idx = np.argmin(h)
            sub = min(-diff, h[idx])
            h[idx] -= sub
            diff += sub
            if diff < -1e-12:
                mask = np.ones(N, dtype=bool)
                mask[idx] = False
                tv = np.sum(h[mask])
                if tv > 0:
                    h[mask] += diff * h[mask] / tv
                    
    h = np.clip(h, 0.0, 1.0)
    return h

def _optimize_start(h0, time_limit):
    """
    Run projected gradient descent from h0 until time_limit or convergence.
    """
    h = h0.copy()
    h = _project_sum(h, TARGET_SUM)
    
    best_h = h.copy()
    best_score = _max_overlap(best_h)
    
    tau = 1.0
    lr = 0.01
    min_tau = 0.05
    min_lr = 0.0001
    
    # Annealing schedule
    # We want to spend most time at low tau for fine tuning
    
    start_time = time.time()
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() > time_limit:
            break
            
        # Compute gradient
        g = _grad_smooth_max_overlap(h, tau)
        
        # Update
        h_new = h - lr * g
        
        # Project
        h_new = _project_sum(h_new, TARGET_SUM)
        
        # Evaluate
        score = _max_overlap(h_new)
        
        if score < best_score:
            best_score = score
            best_h = h_new.copy()
            h = h_new
        else:
            # If score didn't improve, reduce lr and tau slightly
            # This is a simple heuristic; full line search is expensive
            h = h_new # Accept move anyway to escape plateaus? 
            # Or reject? For non-convex, accepting noisy moves helps.
            # Let's accept if it's not significantly worse, or just accept.
            pass
            
        # Anneal
        if it % 50 == 0:
            tau = max(min_tau, tau * 0.95)
            lr = max(min_lr, lr * 0.95)
            
    return best_h, best_score

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5.0 # 5s safety margin
    
    best_h_global = None
    best_score_global = float('inf')
    
    # Define several initial patterns
    starts = []
    
    # 1. Two-block theoretical start (from seed)
    A = (3.0 - math.sqrt(5.0)) / 2.0
    B = 1.0 - A
    h1 = np.zeros(N)
    h1[:N//2] = A
    h1[N//2:] = B
    starts.append(h1)
    
    # 2. Random Binary
    np.random.seed(42)
    h2 = np.random.randint(0, 2, N).astype(float)
    starts.append(h2)
    
    # 3. Random Continuous
    h3 = np.random.rand(N)
    starts.append(h3)
    
    # 4. Alternating
    h4 = np.zeros(N)
    h4[::2] = 0.8
    h4[1::2] = 0.2
    starts.append(h4)
    
    # 5. Sine wave
    h5 = 0.5 + 0.4 * np.sin(np.linspace(0, 4*np.pi, N))
    starts.append(h5)
    
    # Allocate time to each start
    num_starts = len(starts)
    time_per_start = (deadline - start_time) / num_starts
    
    for i, h0 in enumerate(starts):
        current_time = time.time()
        remaining = deadline - current_time
        if remaining < 1.0:
            break
            
        # Cap time per start
        t_limit = min(current_time + time_per_start, deadline)
        
        h_opt, score_opt = _optimize_start(h0, t_limit)
        
        if score_opt < best_score_global:
            best_score_global = score_opt
            best_h_global = h_opt
            
        # Check if we should stop early if we hit a very good score
        # The theoretical lower bound for max overlap is not 0.
        # For N=100, a score around 0.5-0.6 is expected?
        # Seed score ~1.0 (which is the raw value, not the ratio).
        # Wait, the task says "score is the upper bound on C5".
        # And "search score is seed/value".
        # If seed value is V_seed, and my value is V_me, score = V_seed / V_me.
        # Current score 1.0047 means V_me is slightly better than V_seed.
        # So V_seed is the raw overlap value.
        
    # Final check
    if best_h_global is None:
        # Fallback
        best_h_global = starts[0].copy()
        best_h_global = _project_sum(best_h_global, TARGET_SUM)
        
    # Final verification and cleanup
    best_h_global = _project_sum(best_h_global, TARGET_SUM)
    
    return best_h_global
