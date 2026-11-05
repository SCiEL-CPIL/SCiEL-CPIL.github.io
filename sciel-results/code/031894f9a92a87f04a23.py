"""Broadband antireflection coating design using analytical gradient-based optimization."""
import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def design(problem):
    """
    Design a broadband antireflection coating.
    Uses analytical gradients to optimize thicknesses for various layer structures.
    """
    n_L = problem["materials"]["L"]
    n_H = problem["materials"]["H"]
    n_ambient = problem["ambient_index"]
    n_substrate = problem["substrate_index"]
    band_nm = problem["band_nm"]
    wl_min, wl_max = band_nm[0], band_nm[1]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    # Wavelengths for evaluation (401 equally spaced)
    wls = np.linspace(wl_min, wl_max, 401)
    N_wl = len(wls)
    
    # Precompute constants for speed
    # beta_j = 2*pi*n_j*d_j/wl
    # We need d(beta)/d(d_j) = 2*pi*n_j/wl
    
    # Structure generation
    structures = []
    # Generate alternating patterns of odd lengths 1 to 19
    for length in range(1, 20, 2):
        # Start with H
        mats = ["H" if i % 2 == 0 else "L" for i in range(length)]
        structures.append(mats)
        # Start with L
        mats = ["L" if i % 2 == 0 else "H" for i in range(length)]
        structures.append(mats)
    
    # Add some even length structures if space permits (though odd is usually better for AR)
    for length in [4, 6, 8, 10]:
        mats = ["H" if i % 2 == 0 else "L" for i in range(length)]
        structures.append(mats)
        mats = ["L" if i % 2 == 0 else "H" for i in range(length)]
        structures.append(mats)
        
    # Shuffle structures to ensure diverse attempts
    import random
    random.seed(42)
    random.shuffle(structures)
    
    best_materials = None
    best_thicknesses = None
    best_score = float('inf')
    
    # Try to import scipy
    try:
        from scipy.optimize import minimize
    except ImportError:
        minimize = None
        # Fallback if no scipy? The prompt says SciPy is available. But just in case.
        # If no scipy, we can't do gradient descent easily without implementing it.
        # Let's assume it's there as per contract.
        
    def _compute_objective_and_grad(x, materials):
        """
        Compute mean reflectance and its gradient with respect to thicknesses x.
        x: array of thicknesses (nm)
        materials: list of 'L' or 'H'
        """
        n_layers = len(materials)
        if n_layers == 0:
            return 4.26, np.zeros(0) # Approximation for bare glass? No, calculate exactly.
        
        n_vals = np.array([problem["materials"][m] for m in materials])
        
        # Precompute per-wavelength factors
        # beta = 2*pi*n*d/wl
        # d_beta_dd = 2*pi*n/wl
        # cos(beta), sin(beta)
        
        # We will iterate over wavelengths to compute sum of R(wl) and sum of dR/dd_j(wl)
        # R = |r|^2 = (n_amb - Y)/(n_amb + Y) -> |...|^2
        # Let r = (n_amb - Y)/(n_amb + Y)
        # dr/dY = -2*n_amb / (n_amb + Y)^2
        # R = r * r*
        # dR/dY = r* * dr/dY + r * (dr/dY)*
        
        # Y = (M11*n_sub + M12) / (M21*n_sub + M22)
        # Let A = M11*n_sub + M12, B = M21*n_sub + M22
        # Y = A/B
        # dY/dM = (dA*B - A*dB) / B^2
        
        # M = product of layer matrices.
        # dM/dx_k = sum over layers j of (M_1..M_{j-1} @ dM_j/dx_k @ M_{j+1}..M_N)
        
        # This is computationally expensive to do analytically for every wavelength and every layer.
        # However, N_wl=401, N_layers<=20.
        # 401 * 20 * 20 matrix multiplications (2x2 complex) per gradient evaluation.
        # 401 * 400 = 160,400 2x2 complex mat mults.
        # This is very fast in numpy if vectorized, but loops are okay too.
        
        # Let's try a semi-vectorized approach or just efficient loops.
        
        sum_R = 0.0
        sum_grad = np.zeros(n_layers)
        
        for i, wl in enumerate(wls):
            # Compute M and its derivatives
            # M is 2x2
            # We need M and dM/dx_k for all k
            
            # Forward pass to get M and partials
            # Let M_total = I
            # dM_total/dx_k = 0
            # For each layer j:
            #   M_total = M_total @ L_j
            #   dM_total/dx_k = dM_total/dx_k @ L_j + M_total @ dL_j/dx_k (if k==j)
            
            M_total = np.eye(2, dtype=complex)
            dM_total = np.zeros((n_layers, 2, 2), dtype=complex)
            
            for j in range(n_layers):
                n = n_vals[j]
                d = x[j]
                
                beta = 2.0 * np.pi * n * d / wl
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                
                # L_j matrix
                # [ cos_b, -1j*sin_b/n ]
                # [ 1j*n*sin_b, cos_b ]
                L = np.array([
                    [cos_b, -1j * sin_b / n],
                    [1j * n * sin_b, cos_b]
                ], dtype=complex)
                
                # dL_j/dx_j
                # d(beta)/dx_j = 2*pi*n/wl
                dbeta = 2.0 * np.pi * n / wl
                d_cos = -sin_b * dbeta
                d_sin = cos_b * dbeta
                
                dL = np.array([
                    [d_cos, -1j * d_sin / n],
                    [1j * n * d_sin, d_cos]
                ], dtype=complex)
                
                # Update M_total
                M_total = M_total @ L
                
                # Update dM_total
                for k in range(n_layers):
                    if k == j:
                        # d(M_prev @ L) = dM_prev @ L + M_prev @ dL
                        # But we are accumulating.
                        # Actually, dM_total/dx_k after step j:
                        # If k < j: dM_k @ L_j
                        # If k == j: dM_k @ L_j + M_prev @ dL_j
                        # If k > j: dM_k (unchanged from previous step, which is 0 initially? No.)
                        
                        # Let's use the chain rule properly.
                        # M_j = M_{j-1} @ L_j
                        # dM_j/dx_k = dM_{j-1}/dx_k @ L_j + M_{j-1} @ dL_j/dx_k (if k==j else 0)
                        
                        # We need M_{j-1}
                        # We can store M_prev
                        pass
                
                # This loop inside loop is O(N^2) per wavelength.
                # Let's optimize.
                # We can compute the derivative of the total matrix product using the identity:
                # d(M_1...M_N)/dx_k = M_1...M_{k-1} dM_k/dx_k M_{k+1}...M_N
                
                # Let's precompute prefix and suffix products?
                # That would be O(N) matrices per wavelength.
                # Total O(N^2) matrices per wavelength.
                # 401 * 400 = 160k 2x2 matrices. This is fine.
                
                # Let's restart the inner logic for efficiency.
                break # Break out of simple loop to use prefix/suffix method
                
            # Prefix products: P[j] = M_1 ... M_j
            # Suffix products: S[j] = M_{j+1} ... M_N
            # M_total = P[N-1] = P[N-1] @ S[N-1] (S is identity for N)
            
            P = [np.eye(2, dtype=complex)] * (n_layers + 1)
            P[0] = np.eye(2, dtype=complex)
            for j in range(n_layers):
                n = n_vals[j]
                d = x[j]
                beta = 2.0 * np.pi * n * d / wl
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                L = np.array([
                    [cos_b, -1j * sin_b / n],
                    [1j * n * sin_b, cos_b]
                ], dtype=complex)
                P[j+1] = P[j] @ L
                
            S = [np.eye(2, dtype=complex)] * (n_layers + 1)
            S[n_layers] = np.eye(2, dtype=complex)
            for j in range(n_layers - 1, -1, -1):
                n = n_vals[j]
                d = x[j]
                beta = 2.0 * np.pi * n * d / wl
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                L = np.array([
                    [cos_b, -1j * sin_b / n],
                    [1j * n * sin_b, cos_b]
                ], dtype=complex)
                S[j] = L @ S[j+1]
                
            M_total = P[n_layers]
            
            # Now compute derivatives dM_total/dx_k
            # dM_total/dx_k = P[k] @ dL_k/dx_k @ S[k+1]
            
            dM_dxs = np.zeros((n_layers, 2, 2), dtype=complex)
            for k in range(n_layers):
                n = n_vals[k]
                d = x[k]
                beta = 2.0 * np.pi * n * d / wl
                cos_b = np.cos(beta)
                sin_b = np.sin(beta)
                dbeta = 2.0 * np.pi * n / wl
                d_cos = -sin_b * dbeta
                d_sin = cos_b * dbeta
                dL = np.array([
                    [d_cos, -1j * d_sin / n],
                    [1j * n * d_sin, d_cos]
                ], dtype=complex)
                
                dM_dxs[k] = P[k] @ dL @ S[k+1]
            
            # Compute Y and dY/dM
            M11, M12 = M_total[0,0], M_total[0,1]
            M21, M22 = M_total[1,0], M_total[1,1]
            
            A = M11 * n_substrate + M12
            B = M21 * n_substrate + M22
            Y = A / B
            
            # dY/dM
            # Y = A/B
            # dY = (dA*B - A*dB)/B^2
            # dA = dM11*n_sub + dM12
            # dB = dM21*n_sub + dM22
            
            dY_dM = np.zeros((2,2), dtype=complex)
            # dY/dM11 = (n_sub * B) / B^2 = n_sub / B
            # dY/dM12 = 1 / B
            # dY/dM21 = (-A * n_sub) / B^2
            # dY/dM22 = -A / B^2
            
            dY_dM[0,0] = n_substrate / B
            dY_dM[0,1] = 1.0 / B
            dY_dM[1,0] = -A * n_substrate / (B*B)
            dY_dM[1,1] = -A / (B*B)
            
            # Compute r and dr/dY
            r = (n_ambient - Y) / (n_ambient + Y)
            R = abs(r)**2
            
            # dr/dY = -2*n_amb / (n_amb + Y)^2
            dr_dY = -2.0 * n_ambient / (n_ambient + Y)**2
            
            # dR/dY = r* * dr_dY + r * conj(dr_dY)
            dR_dY = np.conj(r) * dr_dY + r * np.conj(dr_dY)
            
            sum_R += R
            
            # Gradient with respect to x_k
            # dR/dx_k = dR/dY * dY/dM : dM/dx_k
            # dY/dM is a 2x2 matrix of partials.
            # dM/dx_k is a 2x2 matrix.
            # The contraction is sum_{i,j} (dY/dM)_{ij} * (dM/dx_k)_{ji} ?
            # Wait. Y = f(M). dY = sum (dY/dM)_{ij} dM_{ij}.
            # dR/dx_k = dR/dY * sum_{i,j} (dY/dM)_{ij} (dM/dx_k)_{ij}
            
            for k in range(n_layers):
                # Trace( dY_dM^T @ dM_dxs[k] )?
                # No, it's the Frobenius inner product.
                # sum_{i,j} dY_dM[i,j] * dM_dxs[k][i,j]
                grad_val = np.sum(dY_dM * dM_dxs[k])
                sum_grad[k] += dR_dY * grad_val
                
        mean_R = sum_R / N_wl
        grad = sum_grad / N_wl
        
        return mean_R * 100.0, grad * 100.0 # Scale to percent for consistency with objective
        
    # Optimization Loop
    for mats in structures:
        if time.time() > deadline - 5.0:
            break
            
        n_layers = len(mats)
        
        # Initialize thicknesses
        # Quarter wave at center wavelength
        wl_center = 0.5 * (wl_min + wl_max)
        x0 = []
        for m in mats:
            n = problem["materials"][m]
            x0.append(wl_center / (4.0 * n))
        
        # Check validity
        if sum(x0) > max_total_nm:
            continue
        if any(t > max_layer_nm for t in x0):
            continue
            
        if minimize is None:
            # Fallback: simple random search or fixed
            # Just use x0
            obj, _ = _compute_objective_and_grad(np.array(x0), mats)
            if obj < best_score:
                best_score = obj
                best_materials = mats[:]
                best_thicknesses = x0[:]
            continue
            
        # Define objective for scipy
        def obj_func(x):
            val, _ = _compute_objective_and_grad(x, mats)
            return val
            
        def grad_func(x):
            _, g = _compute_objective_and_grad(x, mats)
            return g
            
        # Combine for speed?
        def obj_grad_func(x):
            v, g = _compute_objective_and_grad(x, mats)
            return v, g
            
        bounds = [(0.0, max_layer_nm)] * n_layers
        
        # Use L-BFGS-B with analytical gradient
        result = minimize(
            obj_grad_func,
            np.array(x0),
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.fun < best_score:
            # Check validity of result
            new_ths = [max(0.0, min(max_layer_nm, t)) for t in result.x]
            if sum(new_ths) <= max_total_nm:
                best_score = result.fun
                best_materials = mats[:]
                best_thicknesses = new_ths
                
    # Fallback if no valid structure found
    if best_materials is None:
        wl_center = 0.5 * (wl_min + wl_max)
        return ["L"], [wl_center / (4.0 * n_L)]
        
    return best_materials, best_thicknesses

# EVOLVE-BLOCK-END
