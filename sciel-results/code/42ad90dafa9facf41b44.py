import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s limit
    
    N = 24
    # Use 4096 points for fast evaluation during search
    M = 4096
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False)
    z = np.exp(1j * angles)
    Z = np.zeros((N, M), dtype=complex)
    Z[0, :] = 1.0
    for k in range(1, N):
        Z[k, :] = Z[k-1, :] * z
    
    def evaluate(coeffs):
        """Compute max |P(z)| on the grid for given coefficients."""
        vals = coeffs @ Z
        return np.max(np.abs(vals))
    
    # Generate Rudin-Shapiro sequence
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    rs = np.array(p[:N], dtype=np.float64)
    
    best_coeffs = rs.copy()
    best_score = evaluate(best_coeffs)
    
    # Phase 1: Local search from RS and random restarts
    def local_search(current, current_score):
        """Greedy local search: flip one bit at a time if it improves."""
        improved = True
        while improved:
            if time.time() > deadline - 10.0:
                break
            improved = False
            for i in range(N):
                current[i] *= -1
                new_score = evaluate(current)
                if new_score < current_score - 1e-15:
                    current_score = new_score
                    improved = True
                else:
                    current[i] *= -1  # flip back
        return current, current_score
    
    # Start from RS
    current = rs.copy()
    current_score = best_score
    current, current_score = local_search(current, current_score)
    if current_score < best_score:
        best_score = current_score
        best_coeffs = current.copy()
    
    # Random restarts
    rng = np.random.default_rng(42)
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline - 10.0:
            break
        current = rng.choice([-1.0, 1.0], size=N)
        current_score = evaluate(current)
        current, current_score = local_search(current, current_score)
        if current_score < best_score:
            best_score = current_score
            best_coeffs = current.copy()
    
    # Phase 2: Exhaustive search over Hamming ball around best solution
    # Try all solutions at Hamming distance 1, 2, 3 from best_coeffs
    # Distance 1: 24 candidates
    # Distance 2: C(24,2) = 276 candidates
    # Distance 3: C(24,3) = 2024 candidates
    # Total: 2324 additional candidates - very fast
    
    if time.time() < deadline - 5.0:
        # Distance 1
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            candidate = best_coeffs.copy()
            candidate[i] *= -1
            score = evaluate(candidate)
            if score < best_score - 1e-15:
                best_score = score
                best_coeffs = candidate.copy()
        
        # Distance 2
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5.0:
                    break
                candidate = best_coeffs.copy()
                candidate[i] *= -1
                candidate[j] *= -1
                score = evaluate(candidate)
                if score < best_score - 1e-15:
                    best_score = score
                    best_coeffs = candidate.copy()
        
        # Distance 3
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            for j in range(i + 1, N):
                if time.time() > deadline - 5.0:
                    break
                for k in range(j + 1, N):
                    if time.time() > deadline - 5.0:
                        break
                    candidate = best_coeffs.copy()
                    candidate[i] *= -1
                    candidate[j] *= -1
                    candidate[k] *= -1
                    score = evaluate(candidate)
                    if score < best_score - 1e-15:
                        best_score = score
                        best_coeffs = candidate.copy()
    
    # Phase 3: Refine with a much finer grid and golden-section search
    # Use 32768 points for accurate peak finding
    M_fine = 32768
    angles_fine = np.linspace(0, 2 * np.pi, M_fine, endpoint=False)
    z_fine = np.exp(1j * angles_fine)
    Z_fine = np.zeros((N, M_fine), dtype=complex)
    Z_fine[0, :] = 1.0
    for k in range(1, N):
        Z_fine[k, :] = Z_fine[k-1, :] * z_fine
    
    vals_fine = best_coeffs @ Z_fine
    abs_vals = np.abs(vals_fine)
    peak_idx = np.argmax(abs_vals)
    fine_score = abs_vals[peak_idx]
    
    # Golden-section refinement around the peak
    angle_center = angles_fine[peak_idx]
    delta_angle = 2 * np.pi / M_fine
    a = angle_center - delta_angle
    c = angle_center + delta_angle
    
    def P_at_angle(angle):
        z_val = np.exp(1j * angle)
        powers = np.ones(N, dtype=complex)
        for k in range(1, N):
            powers[k] = powers[k-1] * z_val
        return np.abs(np.sum(best_coeffs * powers))
    
    gr = (np.sqrt(5) - 1) / 2
    b_pt = a + gr * (c - a)
    d_pt = a + (1 - gr) * (c - a)
    f_b = P_at_angle(b_pt)
    f_d = P_at_angle(d_pt)
    
    for _ in range(60):
        if f_b < f_d:
            c = d_pt
            d_pt = b_pt
            f_d = f_b
            b_pt = a + gr * (c - a)
            f_b = P_at_angle(b_pt)
        else:
            a = b_pt
            b_pt = d_pt
            f_b = f_d
            d_pt = a + (1 - gr) * (c - a)
            f_d = P_at_angle(d_pt)
    
    # Also check if there's a higher peak elsewhere by scanning
    # The golden section found the local max near the grid peak,
    # but let's verify by checking a few other potential peaks
    # Actually, for ±1 polynomials, the sup norm is well-approximated
    # by the fine grid, and golden section refines it accurately.
    
    return [int(c) for c in best_coeffs]
