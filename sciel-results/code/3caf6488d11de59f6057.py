import time
import math
import numpy as np
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s limit
    
    N_COEFFS = 24
    SQRT_N = math.sqrt(N_COEFFS)
    
    # Precompute powers of z for evaluation
    k_arr = np.arange(N_COEFFS, dtype=np.float64)
    
    # Precompute Z matrix for dense sampling
    # Use 4096 points for coarse search to balance accuracy and speed
    N_SEARCH = 4096
    thetas_search = 2 * np.pi * np.arange(N_SEARCH) / N_SEARCH
    Z_search = np.exp(1j * np.outer(k_arr, thetas_search))  # shape (N_COEFFS, N_SEARCH)
    
    def evaluate_coarse_batch(candidates):
        """
        Evaluate a list of candidate coefficient vectors.
        Returns a list of scores (max |P| / sqrt(N)).
        Uses dense sampling without refinement.
        """
        if not candidates:
            return []
        
        # candidates: list of lists, each of length N_COEFFS
        # Convert to array (M, N_COEFFS)
        C = np.array(candidates, dtype=np.float64)
        
        # Compute values: V[j, m] = P_m(z_j)
        # Z_search is (N_COEFFS, N_SEARCH)
        # C is (M, N_COEFFS)
        # We need V (N_SEARCH, M)
        # V = Z_search.T @ C.T  -> (N_SEARCH, N_COEFFS) @ (N_COEFFS, M) -> (N_SEARCH, M)
        V = Z_search.T @ C.T
        
        mags = np.abs(V)
        max_mags = np.max(mags, axis=0)
        
        return [m / SQRT_N for m in max_mags]

    def evaluate_precise(coeffs):
        """High-precision evaluation with golden section refinement."""
        coeffs_arr = np.array(coeffs, dtype=np.float64)
        
        # Use 8192 points for higher resolution before refinement
        N = 8192
        thetas = 2 * np.pi * np.arange(N) / N
        z = np.exp(1j * np.outer(k_arr, thetas))
        vals = z.T @ coeffs_arr
        mags = np.abs(vals)
        
        max_idx = np.argmax(mags)
        theta_max = thetas[max_idx]
        
        # Golden section refinement
        gr = (math.sqrt(5) - 1) / 2
        a = theta_max - np.pi / N
        b = theta_max + np.pi / N
        
        c_pt = b - gr * (b - a)
        d_pt = a + gr * (b - a)
        fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c_pt)))
        fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d_pt)))
        
        for _ in range(30):
            if fc < fd:
                a = c_pt
                c_pt = d_pt
                fc = fd
                d_pt = a + gr * (b - a)
                fd = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d_pt)))
            else:
                b = d_pt
                d_pt = c_pt
                fd = fc
                c_pt = b - gr * (b - a)
                fc = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c_pt)))
        
        refined_max = max(fc, fd)
        
        # Check top 3 peaks
        top_indices = np.argsort(mags)[::-1][:3]
        for idx in top_indices[1:]:
            theta_peak = thetas[idx]
            a2 = theta_peak - np.pi / N
            b2 = theta_peak + np.pi / N
            c2 = b2 - gr * (b2 - a2)
            d2 = a2 + gr * (b2 - a2)
            fc2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c2)))
            fd2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d2)))
            for _ in range(30):
                if fc2 < fd2:
                    a2 = c2
                    c2 = d2
                    fc2 = fd2
                    d2 = a2 + gr * (b2 - a2)
                    fd2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * d2)))
                else:
                    b2 = d2
                    d2 = c2
                    fd2 = fc2
                    c2 = b2 - gr * (b2 - a2)
                    fc2 = np.abs(np.sum(coeffs_arr * np.exp(1j * k_arr * c2)))
            refined_peak = max(fc2, fd2)
            refined_max = max(refined_max, refined_peak)
        
        return refined_max / SQRT_N

    # Initial seeds
    known_solutions = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        [1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1],
        [1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1],
        [1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1],
        [1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, -1, -1, 1],
        [-1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, 1, 1, 1],
        [1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, 1, -1, -1, 1, 1, -1],
        [1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1],
        [1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1],
        [1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
        [1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1, 1, 1, 1, -1, -1, -1],
        [1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1],
    ]
    
    # Add Rudin-Shapiro
    def get_rs_poly(n):
        p, q = [1], [1]
        while len(p) < n:
            p, q = p + q, p + [-c for c in q]
        return p[:n]
    rs_coeffs = get_rs_poly(N_COEFFS)
    known_solutions.append(rs_coeffs)
    
    # Evaluate initial seeds
    scores = evaluate_coarse_batch(known_solutions)
    best_idx = np.argmin(scores)
    best_coeffs = known_solutions[best_idx][:]
    best_score = evaluate_precise(best_coeffs) # Get precise score for current best
    
    current_coeffs = best_coeffs[:]
    current_score = best_score
    
    # Search Loop
    while time.time() - start_time < deadline - 5:
        improved = False
        
        # Generate neighbors
        neighbors = []
        
        # Single flips
        for i in range(N_COEFFS):
            new_c = current_coeffs[:]
            new_c[i] = -new_c[i]
            neighbors.append(new_c)
            
        # Double flips (random subset to keep it fast)
        # Generate 50 random double flips
        for _ in range(50):
            i, j = random.sample(range(N_COEFFS), 2)
            new_c = current_coeffs[:]
            new_c[i] = -new_c[i]
            new_c[j] = -new_c[j]
            neighbors.append(new_c)
            
        # Evaluate all neighbors in batch
        neighbor_scores = evaluate_coarse_batch(neighbors)
        
        # Find the best neighbor
        min_score_idx = np.argmin(neighbor_scores)
        min_score = neighbor_scores[min_score_idx]
        
        # If the best neighbor is better than current (with some tolerance for coarse vs precise)
        # We use a small tolerance because coarse score might be slightly lower/higher
        if min_score < current_score - 1e-6:
            current_coeffs = neighbors[min_score_idx][:]
            # Update precise score
            current_score = evaluate_precise(current_coeffs)
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs[:]
            improved = True
            
        if not improved:
            # Try a random kick
            n_kicks = random.choice([1, 2, 3])
            kick_indices = random.sample(range(N_COEFFS), n_kicks)
            for k in kick_indices:
                current_coeffs[k] = -current_coeffs[k]
            current_score = evaluate_precise(current_coeffs)
            if current_score < best_score:
                best_score = current_score
                best_coeffs = current_coeffs[:]

    # Final polish: try all single and double flips on the best solution with precise evaluation
    if time.time() - start_time < deadline - 2:
        # Single flips
        for i in range(N_COEFFS):
            new_c = best_coeffs[:]
            new_c[i] = -new_c[i]
            s = evaluate_precise(new_c)
            if s < best_score:
                best_score = s
                best_coeffs = new_c[:]
        
        # Double flips (all pairs)
        for i in range(N_COEFFS):
            for j in range(i+1, N_COEFFS):
                if time.time() - start_time > deadline - 1:
                    break
                new_c = best_coeffs[:]
                new_c[i] = -new_c[i]
                new_c[j] = -new_c[j]
                s = evaluate_precise(new_c)
                if s < best_score:
                    best_score = s
                    best_coeffs = new_c[:]

    return best_coeffs
# EVOLVE-BLOCK-END
