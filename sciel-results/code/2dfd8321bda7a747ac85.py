"""Broadband antireflection coating design via spectral-similarity greedy + vectorized gradient descent."""
import time
import numpy as np

# EVOLVE-BLOCK-START

def _compute_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, wavelengths):
    """Compute reflectance (fraction, not percent) at given wavelengths.
    Returns array of R values (len = len(wavelengths))."""
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    d = np.array(thicknesses, dtype=float)
    m = len(labels)
    
    if m == 0:
        r = (n_amb - n_sub) / (n_amb + n_sub)
        return np.full(len(wavelengths), np.abs(r) ** 2)
    
    # Vectorized over wavelengths
    # M starts as identity for each wavelength
    M00 = np.ones(len(wavelengths), dtype=complex)
    M01 = np.zeros(len(wavelengths), dtype=complex)
    M10 = np.zeros(len(wavelengths), dtype=complex)
    M11 = np.ones(len(wavelengths), dtype=complex)
    
    for i in range(m):
        delta = 2.0 * np.pi * n_mat[i] * d[i] / wavelengths
        c = np.cos(delta)
        s = np.sin(delta)
        # Mi = [[c, i*s/n], [i*n*s, c]]
        # M_new = M @ Mi
        nmi = n_mat[i]
        Mi00 = c
        Mi01 = 1j * s / nmi
        Mi10 = 1j * nmi * s
        Mi11 = c
        
        new_M00 = M00 * Mi00 + M01 * Mi10
        new_M01 = M00 * Mi01 + M01 * Mi11
        new_M10 = M10 * Mi00 + M11 * Mi10
        new_M11 = M10 * Mi01 + M11 * Mi11
        
        M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11
    
    Y = M10 / M00
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return R


def _mean_R(labels, thicknesses, n_amb, n_sub, n_L, n_H, wavelengths):
    """Mean reflectance in percent."""
    R = _compute_reflectance(labels, thicknesses, n_amb, n_sub, n_L, n_H, wavelengths)
    return np.mean(R) * 100.0


def _grad_descent_thicknesses(labels, thicknesses, n_amb, n_sub, n_L, n_H, 
                               wavelengths, max_total_nm, max_layer_nm,
                               max_iter=300, lr=0.5, deadline=None):
    """Optimize thicknesses using gradient descent with analytic gradient.
    Returns (optimal_thicknesses, score)."""
    n_mat = np.array([n_L if l == 'L' else n_H for l in labels], dtype=float)
    m = len(labels)
    d = np.array(thicknesses, dtype=float).copy()
    d = np.clip(d, 0.0, max_layer_nm)
    
    best_d = d.copy()
    best_score = _mean_R(labels, list(d), n_amb, n_sub, n_L, n_H, wavelengths)
    
    # Use a coarser wavelength grid for gradient computation (speed)
    n_coarse = min(101, len(wavelengths))
    wl_coarse = np.linspace(wavelengths[0], wavelengths[-1], n_coarse)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline - 5.0:
            break
        
        # Compute reflectance and gradient on coarse grid
        # R = |r|^2, r = (n_amb - Y)/(n_amb + Y), Y = M10/M00
        # dR/dd_j = 2*Re(r* * dr/dd_j)
        # dr/dd_j = dr/dY * dY/dd_j
        # dr/dY = -2*n_amb/(n_amb+Y)^2
        # Y = M10/M00, dY/dd_j = (dM10/M00 - M10*dM00/M00^2) = (dM10*M00 - M10*dM00)/M00^2
        
        # We need dM/dd_j for each layer j.
        # M = M_1 @ M_2 @ ... @ M_m
        # dM/dd_j = M_1 @ ... @ (dM_j) @ ... @ M_m
        # where dM_j has only the j-th factor differentiated.
        
        # Precompute prefix and suffix products
        # Prefix: P_0 = I, P_k = M_1 @ ... @ M_k
        # Suffix: S_m = I, S_k = M_{k+1} @ ... @ M_m
        
        # Compute M_k for each layer at each wavelength
        deltas = np.zeros((m, len(wl_coarse)))
        cs = np.zeros((m, len(wl_coarse)))
        ss = np.zeros((m, len(wl_coarse)))
        for i in range(m):
            deltas[i] = 2.0 * np.pi * n_mat[i] * d[i] / wl_coarse
            cs[i] = np.cos(deltas[i])
            ss[i] = np.sin(deltas[i])
        
        # Build layer matrices
        # Mi_00 = c_i, Mi_01 = i*s_i/n_i, Mi_10 = i*n_i*s_i, Mi_11 = c_i
        # dMi_00 = -s_i * 2*pi*n_i*d_i/lam = -s_i * delta_i/d_i * d_i ... 
        # Actually d(delta)/d(d_j) = 2*pi*n_j/lam for j=i, 0 otherwise
        # d(c)/d(d_i) = -s * 2*pi*n_i/lam
        # d(s)/d(d_i) = c * 2*pi*n_i/lam
        
        # For layer i:
        # dMi_00 = -ss[i] * 2*pi*n_mat[i]/wl_coarse
        # dMi_01 = 1j * cs[i] * 2*pi*n_mat[i]/wl_coarse / n_mat[i] = 1j * cs[i] * 2*pi/wl_coarse
        # dMi_10 = 1j * n_mat[i] * cs[i] * 2*pi*n_mat[i]/wl_coarse
        # dMi_11 = -ss[i] * 2*pi*n_mat[i]/wl_coarse
        
        # Prefix products: P[k] = M_1 @ ... @ M_k (as 4-component arrays)
        # P[0] = I
        P00 = [np.ones(len(wl_coarse), dtype=complex)]
        P01 = [np.zeros(len(wl_coarse), dtype=complex)]
        P10 = [np.zeros(len(wl_coarse), dtype=complex)]
        P11 = [np.ones(len(wl_coarse), dtype=complex)]
        
        for k in range(m):
            nmk = n_mat[k]
            Mi00 = cs[k]
            Mi01 = 1j * ss[k] / nmk
            Mi10 = 1j * nmk * ss[k]
            Mi11 = cs[k]
            
            # P[k+1] = P[k] @ Mi
            nP00 = P00[k] * Mi00 + P01[k] * Mi10
            nP01 = P00[k] * Mi01 + P01[k] * Mi11
            nP10 = P10[k] * Mi00 + P11[k] * Mi10
            nP11 = P10[k] * Mi01 + P11[k] * Mi11
            P00.append(nP00)
            P01.append(nP01)
            P10.append(nP10)
            P11.append(nP11)
        
        # Suffix products: S[k] = M_{k+1} @ ... @ M_m
        # S[m] = I
        S00 = [None] * (m + 1)
        S01 = [None] * (m + 1)
        S10 = [None] * (m + 1)
        S11 = [None] * (m + 1)
        S00[m] = np.ones(len(wl_coarse), dtype=complex)
        S01[m] = np.zeros(len(wl_coarse), dtype=complex)
        S10[m] = np.zeros(len(wl_coarse), dtype=complex)
        S11[m] = np.ones(len(wl_coarse), dtype=complex)
        
        for k in range(m - 1, -1, -1):
            nmk = n_mat[k]
            Mi00 = cs[k]
            Mi01 = 1j * ss[k] / nmk
            Mi10 = 1j * nmk * ss[k]
            Mi11 = cs[k]
            
            # S[k] = Mi @ S[k+1]
            nS00 = Mi00 * S00[k+1] + Mi01 * S10[k+1]
            nS01 = Mi00 * S01[k+1] + Mi01 * S11[k+1]
            nS10 = Mi10 * S00[k+1] + Mi11 * S10[k+1]
            nS11 = Mi10 * S01[k+1] + Mi11 * S11[k+1]
            S00[k] = nS00
            S01[k] = nS01
            S10[k] = nS10
            S11[k] = nS11
        
        # Total M = P[m]
        M00_t = P00[m]
        M01_t = P01[m]
        M10_t = P10[m]
        M11_t = P11[m]
        
        Y = M10_t / M00_t
        r = (n_amb - Y) / (n_amb + Y)
        R_coarse = np.abs(r) ** 2
        
        # Gradient of R with respect to each d_j
        # dR/dd_j = 2 * Re(conj(r) * dr/dd_j)
        # dr/dY = -2*n_amb / (n_amb + Y)^2
        # dY/dd_j = (dM10_j * M00_t - M10_t * dM00_j) / M00_t^2
        
        grad = np.zeros(m)
        drdY = -2.0 * n_amb / (n_amb + Y) ** 2
        
        for j in range(m):
            # dM_j:
            # dMi_00 = -ss[j] * 2*pi*n_mat[j]/wl_coarse
            # dMi_01 = 1j * cs[j] * 2*pi/wl_coarse
            # dMi_10 = 1j * n_mat[j] * cs[j] * 2*pi*n_mat[j]/wl_coarse
            # dMi_11 = -ss[j] * 2*pi*n_mat[j]/wl_coarse
            dMi00 = -ss[j] * 2.0 * np.pi * n_mat[j] / wl_coarse
            dMi01 = 1j * cs[j] * 2.0 * np.pi / wl_coarse
            dMi10 = 1j * n_mat[j] * cs[j] * 2.0 * np.pi * n_mat[j] / wl_coarse
            dMi11 = -ss[j] * 2.0 * np.pi * n_mat[j] / wl_coarse
            
            # dM = P[j] @ dMi @ S[j+1]
            # P[j] @ dMi:
            A00 = P00[j] * dMi00 + P01[j] * dMi10
            A01 = P00[j] * dMi01 + P01[j] * dMi11
            A10 = P10[j] * dMi00 + P11[j] * dMi10
            A11 = P10[j] * dMi01 + P11[j] * dMi11
            
            # (P[j] @ dMi) @ S[j+1]:
            dM00 = A00 * S00[j+1] + A01 * S10[j+1]
            dM01 = A00 * S01[j+1] + A01 * S11[j+1]
            dM10 = A10 * S00[j+1] + A11 * S10[j+1]
            dM11 = A10 * S01[j+1] + A11 * S11[j+1]
            
            dY = (dM10 * M00_t - M10_t * dM00) / (M00_t ** 2)
            dr = drdY * dY
            dR = 2.0 * np.real(np.conj(r) * dr)
            grad[j] = np.mean(dR)
        
        # Gradient of mean R (in fraction) w.r.t. d
        # We want to minimize, so update: d -= lr * grad
        # But grad is d(R_fraction)/d(d_nm), so we scale
        # Use adaptive learning rate
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        # Adaptive step: normalize gradient, use fixed step in nm
        step_size = 2.0  # nm
        d_new = d - step_size * grad / (grad_norm + 1e-10) * grad_norm
        
        # Actually, simpler: d_new = d - lr * grad, with lr adaptive
        # Let's use: d_new = d - step_size * grad / (||grad||/max_layer_nm)
        # This gives step of ~step_size nm in the gradient direction
        if grad_norm > 1e-10:
            d_new = d - step_size * grad / grad_norm
        else:
            d_new = d.copy()
        
        # Project to feasible region
        d_new = np.clip(d_new, 0.0, max_layer_nm)
        if np.sum(d_new) > max_total_nm:
            scale = max_total_nm / np.sum(d_new)
            d_new = d_new * scale
        
        # Evaluate on full grid
        score_new = _mean_R(labels, list(d_new), n_amb, n_sub, n_L, n_H, wavelengths)
        
        if score_new < best_score:
            best_score = score_new
            best_d = d_new.copy()
            d = d_new.copy()
        else:
            # Reduce step size
            step_size *= 0.5
            if step_size < 0.01:
                break
    
    return best_d, best_score


def _greedy_build_sequence(n_amb, n_sub, n_L, n_H, wavelengths, max_layers, max_total_nm, max_layer_nm, deadline):
    """Build a good layer sequence using spectral similarity to target.
    Target: zero reflectance (Y = n_amb).
    We greedily add layers that best reduce the mean |r|^2."""
    
    best_labels = []
    best_thicknesses = []
    best_score = _mean_R([], [], n_amb, n_sub, n_L, n_H, wavelengths)
    
    # Precompute some useful values
    # For a single layer on substrate:
    # Y = (n_sub * cos(delta) + i*n_sub*sin(delta)) / (cos(delta) + i*(n_sub/n_layer)*sin(delta))
    # Simplified: for quarter-wave, Y = n_layer^2 / n_sub
    
    centre = 0.5 * (wavelengths[0] + wavelengths[-1])
    
    for step in range(max_layers):
        if time.time() > deadline - 10.0:
            break
        
        best_add = None
        best_add_score = best_score
        
        for mat in ['L', 'H']:
            # Try adding this material at the end (substrate side)
            # Initial thickness: quarter-wave at center
            n_mat = n_L if mat == 'L' else n_H
            d_qw = centre / (4.0 * n_mat)
            
            if d_qw > max_layer_nm:
                d_qw = max_layer_nm
            
            # Try a few thickness values
            for d_try in [d_qw, d_qw * 0.5, d_qw * 1.5, d_qw * 0.8, d_qw * 1.2]:
                d_try = np.clip(d_try, 0.0, max_layer_nm)
                if d_try < 1.0:  # skip too thin
                    continue
                
                new_labels = best_labels + [mat]
                new_thick = best_thicknesses + [d_try]
                
                if sum(new_thick) > max_total_nm:
                    continue
                
                score = _mean_R(new_labels, new_thick, n_amb, n_sub, n_L, n_H, wavelengths)
                if score < best_add_score:
                    best_add_score = score
                    best_add = (mat, d_try)
        
        if best_add is None:
            break
        
        best_labels.append(best_add[0])
        best_thicknesses.append(best_add[1])
        best_score = best_add_score
        
        # Check if we've converged (score very low)
        if best_score < 0.1:
            break
    
    return best_labels, best_thicknesses, best_score


def design(problem):
    """Design a broadband antireflection coating stack."""
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)
    
    best_labels = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Phase 1: Greedy construction to get a good starting sequence
    greedy_labels, greedy_thick, greedy_score = _greedy_build_sequence(
        n_amb, n_sub, n_L, n_H, wavelengths, 
        min(max_layers, 12), max_total_nm, max_layer_nm, deadline
    )
    
    if greedy_labels:
        best_labels = greedy_labels[:]
        best_thicknesses = greedy_thick[:]
        best_score = greedy_score
    
    # Phase 2: Gradient descent to refine thicknesses for the greedy sequence
    if best_labels and time.time() < deadline - 20.0:
        opt_thick, opt_score = _grad_descent_thicknesses(
            best_labels, best_thicknesses, n_amb, n_sub, n_L, n_H,
            wavelengths, max_total_nm, max_layer_nm,
            max_iter=200, deadline=deadline
        )
        if opt_score < best_score:
            best_score = opt_score
            best_thicknesses = opt_thick
    
    # Phase 3: Try alternative sequences (different starting materials, different lengths)
    # Use a faster evaluation with fewer wavelengths for sequence search
    wl_fast = np.linspace(band_nm[0], band_nm[1], 101)
    
    candidate_sequences = []
    
    # Alternating sequences of various lengths
    for n_layers in range(2, min(max_layers + 1, 14)):
        seq_L = ["L" if i % 2 == 0 else "H" for i in range(n_layers)]
        seq_H = ["H" if i % 2 == 0 else "L" for i in range(n_layers)]
        candidate_sequences.append(seq_L)
        if seq_H != seq_L:
            candidate_sequences.append(seq_H)
    
    # Add the greedy sequence
    if best_labels and best_labels not in candidate_sequences:
        candidate_sequences.insert(0, best_labels[:])
    
    # Add some structured non-alternating patterns
    # L L H H L L H H ...
    for n_layers in range(4, min(max_layers + 1, 12), 2):
        seq = ["L" if (i // 2) % 2 == 0 else "H" for i in range(n_layers)]
        if seq not in candidate_sequences:
            candidate_sequences.append(seq)
        seq2 = ["H" if (i // 2) % 2 == 0 else "L" for i in range(n_layers)]
        if seq2 not in candidate_sequences:
            candidate_sequences.append(seq2)
    
    # Evaluate and optimize each candidate
    for labels in candidate_sequences:
        if time.time() > deadline - 15.0:
            break
        
        if len(labels) > max_layers:
            continue
        
        # Quick evaluation with quarter-wave initial guess
        centre = 0.5 * (band_nm[0] + band_nm[1])
        init_thick = [centre / (4.0 * (n_L if l == 'L' else n_H)) for l in labels]
        
        if any(t > max_layer_nm for t in init_thick):
            init_thick = [min(t, max_layer_nm) for t in init_thick]
        
        if sum(init_thick) > max_total_nm:
            # Scale down
            scale = max_total_nm / sum(init_thick)
            init_thick = [t * scale for t in init_thick]
        
        # Fast evaluation
        quick_score = _mean_R(labels, init_thick, n_amb, n_sub, n_L, n_H, wl_fast)
        
        # Only do full optimization if quick score is promising
        if quick_score < best_score + 1.0:  # within 1% of best
            opt_thick, opt_score = _grad_descent_thicknesses(
                labels, init_thick, n_amb, n_sub, n_L, n_H,
                wavelengths, max_total_nm, max_layer_nm,
                max_iter=150, deadline=deadline
            )
            if opt_score < best_score:
                if all(0 <= t <= max_layer_nm for t in opt_thick) and sum(opt_thick) <= max_total_nm:
                    best_score = opt_score
                    best_labels = labels[:]
                    best_thicknesses = opt_thick
    
    # Phase 4: Final polish on the best sequence with more iterations
    if best_labels and time.time() < deadline - 10.0:
        opt_thick, opt_score = _grad_descent_thicknesses(
            best_labels, best_thicknesses, n_amb, n_sub, n_L, n_H,
            wavelengths, max_total_nm, max_layer_nm,
            max_iter=300, deadline=deadline
        )
        if opt_score < best_score:
            best_score = opt_score
            best_thicknesses = opt_thick
    
    # Fallback
    if best_labels is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        best_labels = ["L"]
        best_thicknesses = [centre / (4.0 * n_L)]
    
    # Final constraint enforcement
    best_thicknesses = [max(0.0, min(t, max_layer_nm)) for t in best_thicknesses]
    if sum(best_thicknesses) > max_total_nm:
        scale = max_total_nm / sum(best_thicknesses)
        best_thicknesses = [t * scale for t in best_thicknesses]
    
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
