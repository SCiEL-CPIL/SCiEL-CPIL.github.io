import numpy as np
import time

N = 400

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: all ones
    x = np.ones(N)
    
    # Compute initial autocorrelation (linear)
    # A[k] = sum x[i] * x[i+k]
    # Using FFT is faster for full computation, but incremental is better for search
    # For N=400, direct computation is fine.
    
    def compute_A(x):
        A = np.zeros(2*N - 1)
        # A[k] for k in -(N-1)..(N-1)
        # We only need k != 0 for the objective, but A[0] is needed for updates?
        # No, objective is max_{k!=0} |A[k]|.
        # But updates depend on A[k].
        # Let's store A for lags 0..N-1. Symmetric.
        A_pos = np.zeros(N)
        for k in range(N):
            # A[k] = sum_{i=0}^{N-1-k} x[i] * x[i+k]
            if k == 0:
                A_pos[k] = np.sum(x**2)
            else:
                A_pos[k] = np.sum(x[:N-k] * x[k:])
        return A_pos

    A = compute_A(x)
    current_max_sidelobe = np.max(np.abs(A[1:]))
    current_sum = np.sum(x)
    current_val = current_max_sidelobe / (current_sum**2)
    best_x = x.copy()
    best_val = current_val
    
    # Local Search
    # We want to minimize max_{k!=0} |A[k]| / (sum x)^2
    # Since scaling is invariant, we can fix sum x = 1?
    # No, local search on x directly is fine.
    
    # To speed up, we can normalize x to have sum 1 at each step?
    # If we normalize, the geometry changes.
    # Better to just minimize the ratio directly.
    
    # Coordinate Descent
    # For each i, find delta that minimizes new ratio.
    # This is complex.
    # Simpler: Minimize max_sidelobe while keeping sum roughly constant?
    # Or just minimize max_sidelobe, and if sum drops, ratio might increase.
    
    # Let's try to minimize the numerator (max sidelobe) primarily.
    # If we reduce max sidelobe significantly, ratio improves.
    
    # Update rule:
    # A_new[k] = A[k] + delta * (x[i+k] + x[i-k]) for k != 0
    # Note: indices must be valid.
    
    for iteration in range(2000):
        if time.time() > deadline:
            break
            
        improved = False
        for i in range(N):
            # Calculate u[k] = x[i+k] + x[i-k]
            # Valid k are such that i+k < N and i-k >= 0
            # i+k < N => k < N-i
            # i-k >= 0 => k <= i
            # So k in [0, min(i, N-1-i)]
            # But A is defined for k up to N-1.
            # For k > min(i, N-1-i), the terms are 0.
            
            # We need to check all k in 1..N-1
            # u[k] = 0 if out of bounds.
            
            # To avoid creating a full vector u, we can just evaluate the objective for candidate deltas.
            # But we need the optimal delta.
            
            # Let's try a few candidate deltas:
            # 1. Delta that makes the current max lags smaller.
            # Identify top 3 lags with high |A[k]|.
            
            # This is getting complicated.
            # Let's use a simpler heuristic:
            # Try delta = +/- small step.
            # Or better, use the fact that we can solve for delta analytically if we approximate.
            
            # Alternative: Random perturbations.
            # Pick random i, random delta, if better, accept.
            
            # Given time, let's stick to a robust simple search.
            # We'll do a few passes of "try to reduce the max peak".
            
            # Find the lag k_max with max |A[k]|
            # We want to reduce A[k_max].
            # dA[k_max] = delta * (x[i+k_max] + x[i-k_max])
            # If (x[i+k_max] + x[i-k_max]) has same sign as A[k_max], we should choose delta with opposite sign.
            
            # Let's just do a grid search for delta for each i? Too slow.
            
            # Let's try:
            # For each i, compute the vector u for all k.
            # u = np.zeros(N)
            # u[:min(i, N-1-i)+1] = ...
            
            # Actually, let's just use a simple gradient step on the max function.
            # Subgradient of max(|A[k]|) w.r.t delta is sign(A[k_max]) * u[k_max].
            # We want to move in direction -subgrad.
            
            # Let's pick the k with max |A[k]|.
            k_max = np.argmax(np.abs(A[1:])) + 1
            
            # Compute u[k_max]
            val_k_max = A[k_max]
            
            # u_k = x[i+k] + x[i-k]
            idx1 = i + k_max
            idx2 = i - k_max
            u_val = 0.0
            if 0 <= idx1 < N:
                u_val += x[idx1]
            if 0 <= idx2 < N:
                u_val += x[idx2]
            
            if u_val == 0:
                continue
                
            # We want to reduce |val_k_max + delta * u_val|
            # Optimal delta makes val_k_max + delta * u_val = 0
            # delta_opt = -val_k_max / u_val
            
            # But this might increase other peaks.
            # Let's try delta_opt with a damping factor.
            delta = -val_k_max / u_val * 0.5
            
            # Apply delta
            x[i] += delta
            
            # Update A
            # A[k] += delta * (x_new[i+k] + x_old[i-k]?) 
            # Careful: x changed.
            # A_new[k] = A_old[k] + delta * (x_new[i+k] + x_new[i-k] - x_old[i+k] - x_old[i-k])?
            # No.
            # A(k) = sum x_j x_{j+k}
            # Change x_i by delta.
            # Term j=i: x_i x_{i+k} -> (x_i+delta) x_{i+k} = x_i x_{i+k} + delta x_{i+k}
            # Term j=i-k: x_{i-k} x_i -> x_{i-k} (x_i+delta) = x_{i-k} x_i + delta x_{i-k}
            # So A[k] += delta * (x_new[i+k] + x_new[i-k])?
            # Wait, x_new[i+k] is the same as x_old[i+k] unless k=0.
            # For k != 0, x_{i+k} and x_{i-k} are unchanged.
            # So A[k] += delta * (x_old[i+k] + x_old[i-k])
            
            # We need to update A for all k.
            # This is O(N).
            
            # Let's just recompute A from scratch if N is small?
            # N=400. O(N^2) = 160k ops.
            # 2000 iterations * 400 i * 160k = 1.28 * 10^11. Too slow.
            
            # Incremental update is necessary.
            
            # We need u[k] for all k.
            # u[k] = x[i+k] + x[i-k]
            # We can compute this vector in O(N).
            
            u_vec = np.zeros(N)
            # i+k < N => k < N-i
            # i-k >= 0 => k <= i
            max_k = min(N-1-i, i)
            if max_k >= 0:
                # k from 0 to max_k
                # x[i+k] for k=0..max_k is x[i : i+max_k+1]
                # x[i-k] for k=0..max_k is x[i : i+max_k+1][::-1]
                seg1 = x[i : i+max_k+1]
                seg2 = x[i : i+max_k+1][::-1]
                u_vec[:max_k+1] = seg1 + seg2
                
            # Update A[0..N-1]
            # A[k] += delta * u_vec[k]
            A += delta * u_vec
            
            # Recompute max sidelobe
            new_max = np.max(np.abs(A[1:]))
            new_sum = np.sum(x)
            new_val = new_max / (new_sum**2)
            
            if new_val < current_val:
                current_val = new_val
                current_max_sidelobe = new_max
                current_sum = new_sum
                improved = True
            else:
                # Revert
                x[i] -= delta
                A -= delta * u_vec
                # Restore current_val? It didn't change.
                
        if not improved:
            # Try random restarts or perturbations
            # Add small noise
            x += np.random.normal(0, 0.01, N)
            A = compute_A(x)
            current_max_sidelobe = np.max(np.abs(A[1:]))
            current_sum = np.sum(x)
            current_val = current_max_sidelobe / (current_sum**2)
            if current_val < best_val:
                best_val = current_val
                best_x = x.copy()

    return best_x
