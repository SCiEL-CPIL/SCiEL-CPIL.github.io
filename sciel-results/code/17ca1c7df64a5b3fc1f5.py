import time
import math
import numpy as np
from scipy.optimize import minimize

def _reflectance_array(materials, thicknesses, n_ambient, n_substrate, wls, n_mat):
    """
    Compute reflectance array for given materials and thicknesses using
    the thin-film characteristic-matrix method.
    Returns array of reflectance values (0 to 1).
    """
    if len(materials) == 0:
        r = (n_ambient - n_substrate) / (n_ambient + n_substrate)
        return np.full(len(wls), r * r)

    n_layers = len(materials)
    n_wls = len(wls)
    
    # Precompute beta = 2*pi*n*d/wl for all layers and wavelengths
    # beta[i, j] corresponds to layer i, wavelength j
    beta = np.zeros((n_layers, n_wls))
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        if d != 0:
            beta[i, :] = 2.0 * math.pi * n * d / wls

    cosb = np.cos(beta)
    sinb = np.sin(beta)

    # Initialize total characteristic matrix
    # M = [[M11, M12], [M21, M22]]
    M11 = np.ones(n_wls, dtype=complex)
    M12 = np.zeros(n_wls, dtype=complex)
    M21 = np.zeros(n_wls, dtype=complex)
    M22 = np.ones(n_wls, dtype=complex)

    for i, m in enumerate(materials):
        n = n_mat[m]
        c = cosb[i]
        s = sinb[i]
        
        # Layer matrix elements
        A = c
        B = 1j * s / n
        C = 1j * n * s
        D = c

        # Matrix multiplication: M_total = M_total * M_layer
        nM11 = M11 * A + M12 * C
        nM12 = M11 * B + M12 * D
        nM21 = M21 * A + M22 * C
        nM22 = M21 * B + M22 * D
        
        M11, M12, M21, M22 = nM11, nM12, nM21, nM22

    # Admittance Y
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    
    # Reflection coefficient r
    r = (n_ambient - Y) / (n_ambient + Y)
    
    # Reflectance R
    R = np.abs(r) ** 2
    return R

def _mean_reflectance_and_grad(materials, thicknesses, n_ambient, n_substrate, wls, n_mat, max_total):
    """
    Compute mean reflectance and its gradient with respect to thicknesses.
    Includes a penalty for exceeding max_total thickness.
    """
    n_layers = len(materials)
    n_wls = len(wls)
    
    # 1. Compute Reflectance and derivatives
    # We need dR/dtheta_i where theta_i = 2*pi*n_i*d_i/wl
    # dR/dd_i = (dR/dtheta_i) * (2*pi*n_i/wl)
    
    # Recalculate matrices to get derivatives
    # This is efficient enough for n_layers <= 20 and n_wls = 401
    
    R = np.zeros(n_wls)
    grad_R = np.zeros((n_layers, n_wls)) # dR/dtheta_i
    
    # We will propagate the matrix and its derivatives
    # M = M1 * M2 * ... * M_L
    # dM/dtheta_k = M1 * ... * (dM_k/dtheta_k) * ... * ML
    
    # For efficiency, we can compute the total matrix and then use the chain rule
    # Or simply use complex step differentiation? No, analytical is better.
    
    # Let's use the standard chain rule for matrix products.
    # This requires storing intermediate matrices.
    
    # M_total = M_1 M_2 ... M_L
    # d(M_total)/d(theta_k) = M_1 ... M_{k-1} (dM_k/dtheta_k) M_{k+1} ... M_L
    
    # To avoid O(N^2) matrix multiplications for each k, we can use:
    # Let P_k = M_1 ... M_{k-1}
    # Let S_k = M_{k+1} ... M_L
    # Then dM/dtheta_k = P_k * (dM_k/dtheta_k) * S_k
    
    # We can precompute P_k and S_k.
    # P_1 = I
    # P_{k+1} = P_k * M_k
    # S_L = I
    # S_{k-1} = M_k * S_k
    
    M_list = []
    dM_list = [] # dM_k/dtheta_k
    
    for i, m in enumerate(materials):
        n = n_mat[m]
        d = thicknesses[i]
        theta = 2.0 * math.pi * n * d / wls
        
        c = np.cos(theta)
        s = np.sin(theta)
        
        # M_k = [[c, 1j*s/n], [1j*n*s, c]]
        M_k = np.array([
            [c, 1j * s / n],
            [1j * n * s, c]
        ], dtype=complex) # Shape (2, 2, n_wls) -> actually (2, 2) per wavelength
        
        # dM_k/dtheta
        # d(c)/dtheta = -s
        # d(s)/dtheta = c
        dM_k = np.array([
            [-s, 1j * c / n],
            [1j * n * c, -s]
        ], dtype=complex)
        
        M_list.append(M_k)
        dM_list.append(dM_k)
        
    # Compute P and S
    P_list = [None] * (n_layers + 1)
    S_list = [None] * (n_layers + 1)
    
    P_list[0] = np.eye(2) # Identity
    for k in range(n_layers):
        if k == 0:
            P_list[k+1] = M_list[0]
        else:
            # Matrix multiplication element-wise over wavelengths
            # P_{k+1} = P_k * M_k
            P_prev = P_list[k]
            M_k = M_list[k]
            P_new = np.zeros((2, 2, n_wls), dtype=complex)
            for i in range(2):
                for j in range(2):
                    P_new[i, j, :] = P_prev[i, 0, :] * M_k[0, j, :] + P_prev[i, 1, :] * M_k[1, j, :]
            P_list[k+1] = P_new
            
    S_list[n_layers] = np.eye(2)
    for k in range(n_layers - 1, -1, -1):
        M_k = M_list[k]
        S_next = S_list[k+1]
        S_new = np.zeros((2, 2, n_wls), dtype=complex)
        for i in range(2):
            for j in range(2):
                S_new[i, j, :] = M_k[i, 0, :] * S_next[0, j, :] + M_k[i, 1, :] * S_next[1, j, :]
        S_list[k] = S_new
        
    # Compute M_total
    M_total = P_list[n_layers]
    
    # Compute Y and R
    M11 = M_total[0, 0, :]
    M12 = M_total[0, 1, :]
    M21 = M_total[1, 0, :]
    M22 = M_total[1, 1, :]
    
    Y = (M21 + n_substrate * M22) / (M11 + n_substrate * M12)
    r = (n_ambient - Y) / (n_ambient + Y)
    R = np.abs(r) ** 2
    
    # Compute gradients dR/dtheta_k
    # R = r * conj(r)
    # dR/dtheta = conj(r) * dr/dtheta + r * conj(dr/dtheta)
    # dr/dtheta = (dY/dtheta * (n_ambient + Y) - (n_ambient - Y) * dY/dtheta) / (n_ambient + Y)^2
    #           = 2 * n_ambient * dY/dtheta / (n_ambient + Y)^2
    
    # dY/dtheta = (dM21 + n_sub dM22)(M11 + n_sub M12) - (M21 + n_sub M22)(dM11 + n_sub dM12) / (M11 + n_sub M12)^2
    #           = (A' B - A B') / B^2 where A = M21 + n_sub M22, B = M11 + n_sub M12
    
    denom_Y = (M11 + n_substrate * M12) ** 2
    A = M21 + n_substrate * M22
    B = M11 + n_substrate * M12
    
    for k in range(n_layers):
        # Compute dM_total/dtheta_k = P_k * dM_k * S_k
        P_k = P_list[k]
        dM_k = dM_list[k]
        S_k = S_list[k+1] # S_{k+1} is M_{k+1}...M_L
        
        dM_total = np.zeros((2, 2, n_wls), dtype=complex)
        for i in range(2):
            for j in range(2):
                dM_total[i, j, :] = (
                    P_k[i, 0, :] * dM_k[0, 0, :] * S_k[0, j, :] + 
                    P_k[i, 0, :] * dM_k[0, 1, :] * S_k[1, j, :] + 
                    P_k[i, 1, :] * dM_k[1, 0, :] * S_k[0, j, :] + 
                    P_k[i, 1, :] * dM_k[1, 1, :] * S_k[1, j, :]
                )
                
        dM11 = dM_total[0, 0, :]
        dM12 = dM_total[0, 1, :]
        dM21 = dM_total[1, 0, :]
        dM22 = dM_total[1, 1, :]
        
        dA = dM21 + n_substrate * dM22
        dB = dM11 + n_substrate * dM12
        
        dY = (dA * B - A * dB) / denom_Y
        
        # dr/dtheta
        denom_r = (n_ambient + Y) ** 2
        dr = 2.0 * n_ambient * dY / denom_r
        
        # dR/dtheta
        dR_dtheta = np.conj(r) * dr + r * np.conj(dr)
        dR_dtheta = np.real(dR_dtheta) # Reflectance is real
        
        grad_R[k, :] = dR_dtheta
        
    mean_R = np.mean(R)
    grad_mean_R = np.mean(grad_R, axis=1)
    
    # Convert dR/dtheta to dR/dd
    # theta = 2*pi*n*d/wl
    # dtheta/dd = 2*pi*n/wl
    grad_mean_d = np.zeros(n_layers)
    for k in range(n_layers):
        n = n_mat[materials[k]]
        # grad_mean_R[k] is d(mean R)/d(theta_k) averaged over wavelengths?
        # No, grad_R[k, j] is dR_j/dtheta_k.
        # mean R = (1/N) sum R_j
        # d(mean R)/dtheta_k = (1/N) sum dR_j/dtheta_k = mean(grad_R[k, :])
        # d(mean R)/dd_k = d(mean R)/dtheta_k * dtheta_k/dd_k
        # But dtheta_k/dd_k depends on wavelength!
        # So we must compute the derivative for each wavelength and then average.
        
        # Let's redo the gradient calculation correctly.
        # d(mean R)/dd_k = (1/N) sum_j (dR_j/dd_k)
        # dR_j/dd_k = dR_j/dtheta_k * (2*pi*n_k/wl_j)
        
        # We already have grad_R[k, j] = dR_j/dtheta_k
        factor = 2.0 * math.pi * n / wls
        dR_dd_k = grad_R[k, :] * factor
        grad_mean_d[k] = np.mean(dR_dd_k)
        
    # Add penalty for total thickness
    total_th = sum(thicknesses)
    penalty = 0.0
    grad_penalty = np.zeros(n_layers)
    if total_th > max_total:
        # Quadratic penalty to keep it smooth
        excess = total_th - max_total
        penalty = 100.0 * excess * excess
        grad_penalty = 200.0 * excess * np.ones(n_layers)
        
    total_obj = mean_R * 100.0 + penalty
    total_grad = grad_mean_d * 100.0 + grad_penalty
    
    return total_obj, total_grad

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0
    
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    max_layers = problem["max_layers"]
    max_total = problem["max_total_nm"]
    max_layer = problem["max_layer_nm"]
    
    band_start, band_end = problem["band_nm"]
    wls = np.linspace(band_start, band_end, 401)
    n_mat = problem["materials"]
    
    best_val = float('inf')
    best_mats = None
    best_ths = None
    
    # Candidate layer counts
    # We try a range of layer counts. 
    # Fewer layers are easier to optimize and often sufficient for broadband.
    # More layers allow for better flatness but are harder to optimize and constrained by total thickness.
    layer_counts = list(range(2, max_layers + 1, 2)) # Even counts are standard for symmetric-ish AR
    if max_layers % 2 == 1:
        layer_counts.append(max_layers)
        
    # We will prioritize lower layer counts first as they are less constrained by total thickness
    for n_layers in layer_counts:
        if time.time() > deadline - 10.0:
            break
            
        # Alternating sequence starting with L (Air -> L -> H -> ... -> Substrate)
        # This is the standard configuration for AR coatings (n_air < n_L < n_sub < n_H)
        mats = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        
        # Initial guesses
        # Quarter wave stack at various center wavelengths
        center_wls = [450, 500, 550, 600, 650]
        
        for cw in center_wls:
            if time.time() > deadline - 2.0:
                break
                
            x0 = []
            for m in mats:
                n = n_mat[m]
                d = cw / (4.0 * n)
                # Ensure within bounds
                if d > max_layer:
                    d = max_layer
                x0.append(d)
            
            # Check initial sum
            if sum(x0) > max_total:
                # Scale down
                factor = max_total / sum(x0)
                x0 = [d * factor for d in x0]
                
            # Define objective for L-BFGS-B
            def objective(x):
                # x is a numpy array of thicknesses
                # Clip to bounds to be safe, though L-BFGS-B handles bounds
                x_clipped = np.clip(x, 0.0, max_layer)
                obj, grad = _mean_reflectance_and_grad(mats, list(x_clipped), n_ambient, n_substrate, wls, n_mat, max_total)
                return obj, grad
            
            bounds = [(0.0, max_layer) for _ in range(n_layers)]
            
            try:
                res = minimize(
                    objective, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds, 
                    options={'maxiter': 100, 'ftol': 1e-8, 'gtol': 1e-5}
                )
                
                if res.success:
                    x_opt = res.x
                    # Final constraint check
                    x_opt = np.clip(x_opt, 0.0, max_layer)
                    if np.sum(x_opt) > max_total:
                        x_opt = x_opt * (max_total / np.sum(x_opt))
                        x_opt = np.clip(x_opt, 0.0, max_layer)
                    
                    val, _ = _mean_reflectance_and_grad(mats, list(x_opt), n_ambient, n_substrate, wls, n_mat, max_total)
                    
                    if val < best_val:
                        best_val = val
                        best_mats = mats
                        best_ths = list(x_opt)
                        
            except Exception as e:
                # If optimization fails, skip
                pass

    # If no valid solution found, return a simple 1-layer AR
    if best_mats is None:
        best_mats = ['L']
        best_ths = [600.0 / (4.0 * n_L)]
        if best_ths[0] > max_layer:
            best_ths[0] = max_layer
            
    # Final safety check
    best_ths = [min(max(t, 0.0), max_layer) for t in best_ths]
    if sum(best_ths) > max_total:
        f = max_total / sum(best_ths)
        best_ths = [t * f for t in best_ths]
        
    return list(best_mats), list(best_ths)
