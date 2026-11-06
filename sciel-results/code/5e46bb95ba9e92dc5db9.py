"""Antireflection coating design with gradient-based optimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _reflectance_spectrum(materials, thicknesses, problem):
    """Compute reflectance (fraction) at all wavelengths in band, vectorized."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    n_layers = len(materials)
    
    # Phase for each layer at each wavelength: (n_layers, n_wls)
    phase = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wls[None, :]
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    n_inv = 1.0 / n_mat
    
    # Build 2x2 matrices for each layer and wavelength: (n_layers, n_wls, 2, 2)
    # B[i, w] = [[cos, i*sin/n], [i*n*sin, cos]]
    # We need product of all B matrices for each wavelength
    # M = B[0] @ B[1] @ ... @ B[n-1]
    # Start with identity
    M0 = np.ones((len(wls), 2, 2), dtype=complex)
    M1 = np.ones((len(wls), 2, 2), dtype=complex)
    for w in range(len(wls)):
        M0[w] = np.eye(2, dtype=complex)
        M1[w] = np.eye(2, dtype=complex)
    
    # Multiply layer by layer
    for i in range(n_layers):
        for w in range(len(wls)):
            c = cos_p[i, w]
            s = sin_p[i, w]
            n_i = n_mat[i]
            B = np.array([[c, 1j * s / n_i],
                          [1j * n_i * s, c]], dtype=complex)
            M0[w] = M0[w] @ B
    
    # Reflectance
    Y = M0[:, 0, 1] / M0[:, 1, 1]
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    return R

def _mean_reflectance(materials, thicknesses, problem):
    """Compute mean reflectance (percent) over the band."""
    R = _reflectance_spectrum(materials, thicknesses, problem)
    return float(np.mean(R) * 100.0)

def _reflectance_and_grad(materials, thicknesses, problem):
    """Compute mean reflectance and gradient via finite differences or analytical."""
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    wls = np.linspace(band[0], band[1], 401)
    
    n_mat = np.array([n_L if m == 'L' else n_H for m in materials])
    d = np.array(thicknesses, dtype=float)
    n_layers = len(materials)
    n_wls = len(wls)
    
    # Compute forward pass and store intermediate matrices
    phase = 2.0 * np.pi * n_mat[:, None] * d[:, None] / wls[None, :]
    cos_p = np.cos(phase)
    sin_p = np.sin(phase)
    
    # Forward: M[w] = product of B_i[w]
    M = np.ones((n_wls, 2, 2), dtype=complex)
    for w in range(n_wls):
        M[w] = np.eye(2, dtype=complex)
    
    # Store all B matrices
    B_list = []
    for i in range(n_layers):
        B_i = np.ones((n_wls, 2, 2), dtype=complex)
        for w in range(n_wls):
            c = cos_p[i, w]
            s = sin_p[i, w]
            n_i = n_mat[i]
            B_i[w] = np.array([[c, 1j * s / n_i],
                               [1j * n_i * s, c]], dtype=complex)
        B_list.append(B_i)
        # Update M: M = M @ B_i
        for w in range(n_wls):
            M[w] = M[w] @ B_i[w]
    
    # Compute reflectance
    Y = M[:, 0, 1] / M[:, 1, 1]
    r = (n_amb - Y) / (n_amb + Y)
    R = np.abs(r) ** 2
    mean_R = np.mean(R)
    
    # Backward pass for gradient
    # dR/dY = 2 * Re(conj(r) * dr/dY)
    # r = (n_amb - Y)/(n_amb + Y)
    # dr/dY = -(n_amb + Y) - (n_amb - Y) / (n_amb + Y)^2 = -2*n_amb / (n_amb + Y)^2
    dr_dY = -2.0 * n_amb / (n_amb + Y)**2
    
    # R = |r|^2 = r * conj(r)
    # dR/dY = conj(r) * dr/dY + r * conj(dr/dY)
    dR_dY = np.conj(r) * dr_dY + r * np.conj(dr_dY)
    
    # Y = M01 / M11
    M01 = M[:, 0, 1]
    M11 = M[:, 1, 1]
    dY_dM01 = 1.0 / M11
    dY_dM11 = -M01 / M11**2
    
    # dR/dM01 and dR/dM11
    dR_dM01 = dR_dY * dY_dM01
    dR_dM11 = dR_dY * dY_dM11
    
    # Backprop through matrix product
    # M = I @ B_0 @ B_1 @ ... @ B_{n-1}
    # We need dM_final/dB_i for each i
    # Let P_i = I @ B_0 @ ... @ B_{i-1} (prefix product up to i-1)
    # Let S_i = B_{i+1} @ ... @ B_{n-1} (suffix product from i+1)
    # Then M = P_i @ B_i @ S_i
    # dM/dB_i = P_i @ (dB_i) @ S_i
    
    # Compute prefix and suffix products
    P = np.ones((n_layers + 1, n_wls, 2, 2), dtype=complex)
    P[0] = np.ones((n_wls, 2, 2), dtype=complex)
    for w in range(n_wls):
        P[0][w] = np.eye(2, dtype=complex)
    
    for i in range(n_layers):
        for w in range(n_wls):
            P[i+1][w] = P[i][w] @ B_list[i][w]
    
    S = np.ones((n_layers + 1, n_wls, 2, 2), dtype=complex)
    S[n_layers] = np.ones((n_wls, 2, 2), dtype=complex)
    for w in range(n_wls):
        S[n_layers][w] = np.eye(2, dtype=complex)
    
    for i in range(n_layers - 1, -1, -1):
        for w in range(n_wls):
            S[i][w] = B_list[i][w] @ S[i+1][w]
    
    # Gradient of mean_R with respect to each thickness d_j
    grad = np.zeros(n_layers, dtype=float)
    
    for i in range(n_layers):
        # dM/dB_i = P_i @ dB_i @ S_i
        # We need d(mean_R)/dB_i = sum over w of (dR_dM01[w] * dM01/dB_i[w] + dR_dM11[w] * dM11/dB_i[w])
        # 
        # dB_i/dd_j: B_i depends on d_i only (since n_i is fixed)
        # phase_i = 2*pi*n_i*d_i/w
        # dB_i/dd_i:
        # dcos_p/dd = -sin_p * 2*pi*n_i/w
        # dsin_p/dd = cos_p * 2*pi*n_i/w
        # dB/dd = [[-sin_p * dp, i * cos_p * dp / n_i], [i * n_i * cos_p * dp, -sin_p * dp]]
        # where dp = 2*pi*n_i/w
        
        for w in range(n_wls):
            dp = 2.0 * np.pi * n_mat[i] / wls[w]
            c = cos_p[i, w]
            s = sin_p[i, w]
            n_i = n_mat[i]
            
            # dB_i/dd_i
            dB_dd = np.array([[-s * dp, 1j * c * dp / n_i],
                              [1j * n_i * c * dp, -s * dp]], dtype=complex)
            
            # dM01/dB_i = (P_i @ dB_dd @ S_i)[0,1]
            # dM11/dB_i = (P_i @ dB_dd @ S_i)[1,1]
            temp = P[i][w] @ dB_dd @ S[i+1][w]
            dM01_dBi = temp[0, 1]
            dM11_dBi = temp[1, 1]
            
            # Contribution to gradient
            contrib = dR_dM01[w] * dM01_dBi + dR_dM11[w] * dM11_dBi
            # dR/d(d_i) = Re(contrib) since R is real
            grad[i] += np.real(contrib)
    
    # Average over wavelengths
    grad /= n_wls
    # Convert to percent
    grad *= 100.0
    
    return mean_R * 100.0, grad

def _optimize_for_structure(materials, init_thick, problem, deadline):
    """Optimize thicknesses for a fixed material sequence using L-BFGS-B."""
    n_layers = len(materials)
    max_layer_nm = problem["max_layer_nm"]
    max_total_nm = problem["max_total_nm"]
    
    x0 = np.array(init_thick, dtype=float)
    bounds = [(0.0, max_layer_nm)] * n_layers
    
    def obj_and_grad(x):
        if time.time() > deadline:
            return 1000.0, np.zeros(n_layers)
        # Check total thickness constraint
        if np.sum(x) > max_total_nm + 1e-6:
            # Penalize heavily
            penalty = 10.0 * (np.sum(x) - max_total_nm)
            return 1000.0 + penalty, np.full(n_layers, 10.0)
        
        f, g = _reflectance_and_grad(materials, list(x), problem)
        return f, g
    
    try:
        res = minimize(obj_and_grad, x0, method='L-BFGS-B', bounds=bounds,
                      options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8,
                               'maxfun': 500})
        if res.success or res.fun < 1000.0:
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) <= max_total_nm:
                return x_opt, float(res.fun)
    except Exception:
        pass
    
    # Fallback to Nelder-Mead if L-BFGS-B fails
    try:
        def obj(x):
            if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                return 1000.0
            return _mean_reflectance(materials, list(x), problem)
        
        res = minimize(obj, x0, method='Nelder-Mead',
                      options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-8})
        if res.success and res.fun < 1000.0:
            x_opt = np.clip(res.x, 0, max_layer_nm)
            if np.sum(x_opt) <= max_total_nm:
                return x_opt, float(res.fun)
    except Exception:
        pass
    
    return x0, _mean_reflectance(materials, list(x0), problem)

def design(problem):
    """Design broadband AR coating using gradient-based optimization."""
    t0 = time.time()
    deadline = t0 + 135.0  # Safety margin before 150s kill
    
    n_amb = problem["ambient_index"]
    n_sub = problem["substrate_index"]
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    centre = 0.5 * (band[0] + band[1])
    
    # Best solution found
    best_mats = ["L"]
    best_thick = [centre / (4.0 * n_L)]
    best_score = _mean_reflectance(best_mats, best_thick, problem)
    
    # Try different numbers of layers with various initializations
    for n_layers in range(1, min(max_layers + 1, 21)):
        if time.time() > deadline - 20:
            break
        
        # Try alternating starting from L and H
        for start in ['L', 'H']:
            if time.time() > deadline - 20:
                break
            
            mats = []
            for i in range(n_layers):
                mats.append(start if i % 2 == 0 else ('H' if start == 'L' else 'L'))
            
            # Initial thicknesses: quarter-wave
            init_thick = []
            for m in mats:
                n_m = n_L if m == 'L' else n_H
                init_thick.append(centre / (4.0 * n_m))
            
            if sum(init_thick) > max_total_nm or any(t > max_layer_nm for t in init_thick):
                continue
            
            x_opt, score = _optimize_for_structure(mats, init_thick, problem, deadline)
            if score < best_score:
                best_score = score
                best_mats = mats[:]
                best_thick = list(x_opt)
            
            # Try eighth-wave initialization
            if n_layers <= 10:
                init_thick8 = []
                for m in mats:
                    n_m = n_L if m == 'L' else n_H
                    init_thick8.append(centre / (8.0 * n_m))
                
                if sum(init_thick8) <= max_total_nm and all(t <= max_layer_nm for t in init_thick8):
                    x_opt8, score8 = _optimize_for_structure(mats, init_thick8, problem, deadline)
                    if score8 < best_score:
                        best_score = score8
                        best_mats = mats[:]
                        best_thick = list(x_opt8)
    
    # Try some non-alternating patterns for smaller stacks
    if time.time() < deadline - 30:
        patterns = [
            "LLHH", "HHLL", "LHLH", "HLHL", "LLH", "HHL",
            "LLHHLL", "HHLLHH", "LHLHLH", "HLHLHL",
        ]
        for pat in patterns:
            if time.time() > deadline - 20:
                break
            if len(pat) > max_layers:
                continue
            mats = list(pat)
            init_thick = []
            for m in mats:
                n_m = n_L if m == 'L' else n_H
                init_thick.append(centre / (4.0 * n_m))
            
            if sum(init_thick) > max_total_nm or any(t > max_layer_nm for t in init_thick):
                continue
            
            x_opt, score = _optimize_for_structure(mats, init_thick, problem, deadline)
            if score < best_score:
                best_score = score
                best_mats = mats[:]
                best_thick = list(x_opt)
    
    # Final polish with Nelder-Mead on best solution
    if time.time() < deadline - 10 and len(best_mats) >= 2:
        try:
            def obj(x):
                if np.any(x < 0) or np.any(x > max_layer_nm) or np.sum(x) > max_total_nm:
                    return 1000.0
                return _mean_reflectance(best_mats, list(x), problem)
            
            res = minimize(obj, np.array(best_thick), method='Nelder-Mead',
                          options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-9})
            if res.success and res.fun < best_score:
                x_opt = np.clip(res.x, 0, max_layer_nm)
                if np.sum(x_opt) <= max_total_nm:
                    best_score = float(res.fun)
                    best_thick = list(x_opt)
        except Exception:
            pass
    
    # Ensure constraints are satisfied
    best_thick = [max(0.0, min(t, max_layer_nm)) for t in best_thick]
    if sum(best_thick) > max_total_nm:
        scale = max_total_nm / sum(best_thick)
        best_thick = [t * scale for t in best_thick]
    
    return best_mats, best_thick
# EVOLVE-BLOCK-END
