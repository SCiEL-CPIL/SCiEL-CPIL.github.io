"""Rewritten program for Erdos minimum overlap using spectral optimization."""

import time
import numpy as np

N = 100

def _linear_overlap_max(h):
    """Calculate max linear overlap for shifts k=1..N-1."""
    max_ov = 0.0
    # Precompute suffix sums of h to speed up the sum part if needed, 
    # but the product sum is the tricky part.
    # For N=100, direct O(N^2) is 10,000 ops, which is very fast.
    # We use this for validation and final scoring.
    for k in range(1, N):
        # overlap = sum_{i=k}^{N-1} h[i] * (1 - h[i-k])
        ov = np.sum(h[k:] * (1.0 - h[:-k]))
        if ov > max_ov:
            max_ov = ov
    return max_ov

def _analytic_overlap_max(m, A, N):
    """
    Estimate max linear overlap for h_i = 0.5 + A * cos(2*pi*m*i/N).
    Uses closed-form sums for speed during search.
    """
    if A == 0:
        # Constant case
        return (N - 1) / 4.0
        
    theta_step = 2 * np.pi * m / N
    max_ov = 0.0
    
    # Precompute some terms if possible, but N is small.
    for k in range(1, N):
        # Term 1: 0.25 * (N-k)
        t1 = 0.25 * (N - k)
        
        # Term 2: -0.5 * A * sum_{j=0}^{N-1-k} cos(2*pi*m*j/N)
        # Sum of cos is Re( sum (e^{i theta})^j ) = Re( (1 - e^{i theta (N-k)}) / (1 - e^{i theta}) )
        # Let M = N - k
        M = N - k
        if M == 0:
            t2 = 0
        else:
            # Handle theta close to 0 (m=0 or m=N, not in loop) or m=N/2
            z = np.exp(1j * theta_step)
            if np.isclose(z, 1.0):
                sum_cos = M
            else:
                # Geometric series sum
                sum_z = (1 - z**M) / (1 - z)
                sum_cos = np.real(sum_z)
            t2 = -0.5 * A * sum_cos
            
        # Term 3: 0.5 * A * sum_{i=k}^{N-1} cos(2*pi*m*i/N)
        # This is sum_{i=k}^{N-1} cos(theta_step * i)
        # = Re( e^{i theta k} * (1 - e^{i theta (N-k)}) / (1 - e^{i theta}) )
        if np.isclose(z, 1.0):
            sum_cos_2 = M
        else:
            sum_z_2 = (1 - z**M) / (1 - z)
            sum_cos_2 = np.real(z**k * sum_z_2)
        t3 = 0.5 * A * sum_cos_2
        
        # Term 4: -A^2 * sum_{i=k}^{N-1} cos(theta_i) cos(theta_{i-k})
        # = -A^2 * 0.5 * [ (N-k) * cos(2*pi*m*k/N) + sum_{i=k}^{N-1} cos(2*pi*m*(2i-k)/N) ]
        
        # Part 4a: (N-k) * cos(2*pi*m*k/N)
        t4a = M * np.cos(2 * np.pi * m * k / N)
        
        # Part 4b: sum_{i=k}^{N-1} cos(2*pi*m*(2i-k)/N)
        # Let phi_i = 2*pi*m*(2i-k)/N.
        # Step is 4*pi*m/N.
        # Initial angle for i=k: 2*pi*m*(2k-k)/N = 2*pi*m*k/N.
        # Number of terms: M.
        step_2 = 4 * np.pi * m / N
        z_2 = np.exp(1j * step_2)
        if np.isclose(z_2, 1.0):
            sum_cos_4b = M
        else:
            sum_z_4b = (1 - z_2**M) / (1 - z_2)
            # Start angle is theta_start = 2*pi*m*k/N
            sum_cos_4b = np.real(np.exp(1j * 2 * np.pi * m * k / N) * sum_z_4b)
            
        t4 = -0.5 * A * A * (t4a + sum_cos_4b)
        
        ov = t1 + t2 + t3 + t4
        if ov > max_ov:
            max_ov = ov
            
    return max_ov

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    # Baseline: constant 0.5
    best_h = np.full(N, 0.5)
    best_score = _linear_overlap_max(best_h)
    
    # Spectral search: h_i = 0.5 + A * cos(2*pi*m*i/N + phase)
    # We can assume phase=0 without loss of generality for the max overlap profile?
    # Shifting the sequence in space just shifts the k-index of the overlap, 
    # but since we take MAX over all k, the phase doesn't change the max value.
    # So we only need to search m and A.
    
    # Search frequencies
    for m in range(1, N // 2 + 1):
        if time.time() > deadline - 10.0:
            break
            
        # For each m, find optimal A
        # A is in [0, 0.5]
        # We can do a coarse grid search for A first
        best_A_for_m = 0.0
        best_score_for_m = best_score
        
        # Coarse search for A
        A_coarse = np.linspace(0.0, 0.5, 100)
        for A in A_coarse:
            if time.time() > deadline - 5.0:
                break
            score = _analytic_overlap_max(m, A, N)
            if score < best_score_for_m:
                best_score_for_m = score
                best_A_for_m = A
        
        # Fine search around best_A_for_m
        if best_A_for_m > 0:
            A_fine = np.linspace(max(0, best_A_for_m - 0.01), min(0.5, best_A_for_m + 0.01), 50)
            for A in A_fine:
                if time.time() > deadline - 5.0:
                    break
                score = _analytic_overlap_max(m, A, N)
                if score < best_score_for_m:
                    best_score_for_m = score
                    best_A_for_m = A
        
        # If this m is better than global best, update
        if best_score_for_m < best_score:
            best_score = best_score_for_m
            i = np.arange(N)
            best_h = 0.5 + best_A_for_m * np.cos(2 * np.pi * m * i / N)
            # Ensure bounds and sum constraint
            # The analytic formula assumes exact sum N/2.
            # Cosine sum is 0 for integer m not multiple of N.
            # So sum is exactly N/2.
            # Clip to [0,1] just in case of numerical noise, though A<=0.5 ensures [0,1].
            best_h = np.clip(best_h, 0.0, 1.0)
            
            # Refine with local search around this spectral solution
            # This handles any deviations from the pure cosine model
            for trial in range(20):
                if time.time() > deadline - 2.0:
                    break
                h = best_h + np.random.normal(0, 0.02, N)
                h = np.clip(h, 0.0, 1.0)
                current_sum = np.sum(h)
                if current_sum > 0:
                    h = h * (N / 2.0) / current_sum
                    h = np.clip(h, 0.0, 1.0)
                    current_sum = np.sum(h)
                    if current_sum > 0:
                        h = h * (N / 2.0) / current_sum
                score = _linear_overlap_max(h)
                if score < best_score:
                    best_score = score
                    best_h = h.copy()

    return best_h

if __name__ == "__main__":
    construct()
