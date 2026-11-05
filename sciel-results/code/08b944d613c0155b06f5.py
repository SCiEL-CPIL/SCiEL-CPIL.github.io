"""Program for finding the flattest 24-coefficient plus-minus-one polynomial."""
import numpy as np
import time
from itertools import combinations

def construct():
    """
    Finds the optimal 24 coefficients in {+1, -1} minimizing the sup norm.
    Uses a direct combinatorial search strategy leveraging the known small search space.
    The search space is 2^24 = 16,777,216 states, but symmetries reduce this significantly.
    We use a combination of:
    1. Fixed first two coefficients (symmetry reduction).
    2. Efficient evaluation via FFT-based sup norm approximation.
    3. Parallel-like batch processing with numpy.
    4. Local search on the best candidates.
    """
    N = 24
    start_time = time.time()
    deadline = start_time + 145  # 145s limit, 5s buffer

    # Precompute powers of z for dense sampling
    M = 4096  # Number of sample points
    theta = 2 * np.pi * np.arange(M) / M
    # z_j = exp(i * theta_j)
    # We can precompute z_j^k for k=0..23
    # But it's better to use FFT or matrix multiplication.
    # Let's use matrix multiplication: P_j = sum c_k z_j^k
    # This is c @ Z where Z is M x N matrix with Z[j,k] = exp(i * k * theta_j)
    
    # Construct Z matrix
    k = np.arange(N)
    # Z[j, k] = exp(i * k * theta_j)
    # theta_j = 2*pi*j/M
    # k * theta_j = 2*pi * j * k / M
    # We can vectorize this
    j_idx = np.arange(M)
    # Z = exp(1j * 2*pi * np.outer(j_idx, k) / M)
    # This is exactly the DFT matrix (unscaled)
    # Actually, numpy's fft uses exp(-2*pi*i*k*n/M) for forward, so ifft uses positive.
    # P(z_j) = sum c_k exp(i * k * theta_j) = sum c_k exp(i * 2*pi * j * k / M)
    # This is related to IFFT.
    # Let's just compute Z directly for clarity and correctness.
    Z = np.exp(1j * 2 * np.pi * np.outer(j_idx, k) / M)
    
    def fast_score(coeffs):
        """
        Compute sup norm score for a single coefficient vector.
        coeffs: numpy array of shape (N,) with values +/-1
        """
        # P_j = Z @ coeffs
        P = Z @ coeffs
        max_val = np.max(np.abs(P))
        # Refinement for top peaks
        # Find top few peaks and refine
        idx = np.argsort(np.abs(P))[-5:]
        max_refined = max_val
        for i in idx:
            # Golden section around theta_i
            w = 2 * np.pi / M
            a = theta[i] - w
            b = theta[i] + w
            gr = (5 ** 0.5 - 1) / 2
            c = b - gr * (b - a)
            d = a + gr * (b - a)
            
            def f(th):
                z_val = np.exp(1j * th)
                p_val = 0.0
                for coeff in reversed(coeffs):
                    p_val = p_val * z_val + coeff
                return np.abs(p_val)
            
            fc = f(c)
            fd = f(d)
            for _ in range(20):
                if fc < fd:
                    a = c; c = d; fc = fd
                    d = a + gr * (b - a); fd = f(d)
                else:
                    b = d; d = c; fd = fc
                    c = b - gr * (b - a); fc = f(c)
            peak_val = max(fc, fd)
            if peak_val > max_refined:
                max_refined = peak_val
        return max_refined / np.sqrt(N)

    # Symmetry: P(z) and P(-z) have same sup norm.
    # Also, flipping all signs gives same sup norm.
    # Fix c_0 = 1, c_1 = 1.
    # Remaining 22 coefficients can be +/-1.
    # Total 2^22 = 4,194,304 candidates.
    
    # Generate all candidates efficiently
    # We'll iterate over integers 0 to 2^22 - 1
    # and use bit representation to determine signs of c_2...c_23
    
    num_candidates = 1 << 22
    best_score = float('inf')
    best_coeffs = None
    
    # Batch processing for speed
    batch_size = 10000
    for start in range(0, num_candidates, batch_size):
        if time.time() > deadline:
            break
        end = min(start + batch_size, num_candidates)
        # Generate batch of indices
        idxs = np.arange(start, end)
        # Extract bits for positions 2..23 (22 bits)
        # Bit k of idx corresponds to c_{k+2}
        # Let's create a matrix of bits
        # bits[b, i] = (idxs[b] >> i) & 1
        # We need bits for i=0..21
        powers = np.arange(22)
        bits = ((idxs[:, None] >> powers[None, :]) & 1).astype(float)
        # Map bits to signs: 0 -> -1, 1 -> 1
        # signs = 1 - 2*bits
        # But we fixed c_0=1, c_1=1
        # So coeffs[0]=1, coeffs[1]=1, coeffs[2:]=1-2*bits
        # Let's construct coeffs matrix of shape (batch_size, N)
        coeffs_batch = np.zeros((end - start, N))
        coeffs_batch[:, 0] = 1.0
        coeffs_batch[:, 1] = 1.0
        coeffs_batch[:, 2:] = 1.0 - 2.0 * bits
        
        # Compute scores for all in batch
        # P = coeffs_batch @ Z.T  (since Z is M x N, coeffs is B x N, result B x M)
        # Wait, Z[j,k] = exp(i * k * theta_j), so P_j = sum_k c_k Z[j,k]
        # coeffs_batch is B x N, Z.T is N x M
        # P_batch = coeffs_batch @ Z.T  -> B x M
        P_batch = coeffs_batch @ Z.T
        max_vals = np.max(np.abs(P_batch), axis=1)
        scores = max_vals / np.sqrt(N)
        
        # Find best in batch
        min_idx = np.argmin(scores)
        if scores[min_idx] < best_score:
            best_score = scores[min_idx]
            best_coeffs = coeffs_batch[min_idx].copy()
            # Refine the best candidate from this batch
            refined_score = fast_score(best_coeffs)
            if refined_score < best_score:
                best_score = refined_score
    
    # Local search on the best candidate found
    if best_coeffs is not None:
        # Simple local search: flip one bit at a time
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(2, N):
                old_val = best_coeffs[i]
                best_coeffs[i] = -old_val
                new_score = fast_score(best_coeffs)
                if new_score < best_score:
                    best_score = new_score
                    improved = True
                else:
                    best_coeffs[i] = old_val
    
    return [int(c) for c in best_coeffs]
