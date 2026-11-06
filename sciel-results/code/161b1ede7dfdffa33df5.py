import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    # The exhaustive optimum for N=24 (degree 23) plus-minus-one polynomials
    # is known to have sup norm / sqrt(24) ≈ 1.2583758.
    # We use a fast FFT-based local search to find one of the optimal patterns.
    
    N = 24
    M = 4096  # FFT size for evaluation
    
    # Precompute the DFT matrix for evaluation
    # We evaluate P(z) at M points on the unit circle
    theta = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * theta)
    
    # Build the Vandermonde matrix Z where Z[k, j] = z_j^k
    # For efficiency, we use the fact that we can compute P(z_j) = sum c_k z_j^k
    # This is equivalent to evaluating the polynomial at each z_j.
    # We can use np.polynomial.polynomial.polyval or just matrix multiplication.
    # Let's use a direct approach: for each z_j, P(z_j) = sum_{k=0}^{23} c_k * z_j^k
    
    # Precompute powers of z
    Z = np.zeros((N, M), dtype=complex)
    Z[0, :] = 1.0
    for k in range(1, N):
        Z[k, :] = Z[k-1, :] * z
    
    def calc_score(c):
        # c is shape (N,) of +1/-1
        # P = Z^T @ c gives shape (M,)
        P = Z.T @ c
        return np.max(np.abs(P))
    
    # Generate Golay-Rudin-Shapiro sequence as a good starting point
    def gr_sequence(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return np.array(p[:n], dtype=float)
    
    best_coeffs = gr_sequence(N)
    best_score = calc_score(best_coeffs)
    
    # Local search with FFT-based scoring
    def local_search(coeffs, max_iterations=10000):
        coeffs = coeffs.copy()
        score = calc_score(coeffs)
        improved = True
        iteration = 0
        while improved and iteration < max_iterations and time.time() < deadline - 2.0:
            improved = False
            iteration += 1
            # Try flipping each bit
            for i in range(N):
                old_val = coeffs[i]
                coeffs[i] = -old_val
                new_score = calc_score(coeffs)
                if new_score < score - 1e-10:
                    score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
        return coeffs, score
    
    # Start with GR sequence
    c1, s1 = local_search(best_coeffs)
    if s1 < best_score:
        best_score = s1
        best_coeffs = c1
    
    # Try a few random restarts to avoid local minima
    np.random.seed(42)
    for restart in range(20):
        if time.time() > deadline - 15.0:
            break
        rand_coeffs = np.random.choice([-1.0, 1.0], size=N)
        c2, s2 = local_search(rand_coeffs)
        if s2 < best_score:
            best_score = s2
            best_coeffs = c2
    
    # Final high-precision polish: use a finer grid for the best candidate
    if time.time() < deadline - 5.0:
        M_fine = 8192
        theta_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
        z_fine = np.exp(1j * theta_fine)
        Z_fine = np.zeros((N, M_fine), dtype=complex)
        Z_fine[0, :] = 1.0
        for k in range(1, N):
            Z_fine[k, :] = Z_fine[k-1, :] * z_fine
        
        def calc_score_fine(c):
            P = Z_fine.T @ c
            return np.max(np.abs(P))
        
        # One more round of local search with fine scoring
        coeffs = best_coeffs.copy()
        score = calc_score_fine(coeffs)
        improved = True
        while improved and time.time() < deadline - 1.0:
            improved = False
            for i in range(N):
                old_val = coeffs[i]
                coeffs[i] = -old_val
                new_score = calc_score_fine(coeffs)
                if new_score < score - 1e-10:
                    score = new_score
                    improved = True
                else:
                    coeffs[i] = old_val
        if score < best_score:
            best_score = score
            best_coeffs = coeffs
    
    # Return the best coefficients found
    return [int(c) for c in best_coeffs]
