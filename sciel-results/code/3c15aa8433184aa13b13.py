import numpy as np
import scipy.optimize
import time

def _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict):
    """
    Calculate mean reflectance using the characteristic matrix method.
    materials: list of 'L' or 'H'
    thicknesses: list of float thicknesses in nm
    wls: array of wavelengths in nm
    """
    n_layers = len(materials)
    if n_layers == 0:
        # Bare glass reflectance
        r = (ambient_index - substrate_index) / (ambient_index + substrate_index)
        return (r * r) * 100.0
    
    # Pre-compute indices for each layer
    n_layers_arr = np.array([materials_dict[m] for m in materials], dtype=float)
    d_layers_arr = np.array(thicknesses, dtype=float)
    
    # Wavelengths in vacuum
    k = 2 * np.pi / wls
    
    # Characteristic matrix for each layer
    # M_j = [[cos(delta_j), (i/sin(delta_j))/n_j], [i*n_j*sin(delta_j), cos(delta_j)]]
    # We compute the product of all matrices.
    
    # To avoid numerical instability with large matrices or complex numbers, 
    # we can compute the admittance recursively or use the full matrix product.
    # Given max 20 layers, full matrix product is fine.
    
    # Initialize identity matrix for the stack
    # We work in the complex domain
    # M_total = M_1 @ M_2 @ ... @ M_N
    
    # Efficient vectorized computation is tricky due to the matrix multiplication order.
    # We can loop over layers, which is fast enough for 20 layers and 401 wavelengths.
    
    # Let's compute M for each layer at all wavelengths simultaneously.
    # delta_j = k * n_j * d_j
    # n_j is shape (N,), d_j is shape (N,), k is shape (W,)
    # delta is shape (N, W)
    
    delta = k[None, :] * n_layers_arr[:, None] * d_layers_arr[:, None]
    
    cos_delta = np.cos(delta)
    sin_delta = np.sin(delta)
    
    # Avoid division by zero in sin_delta
    sin_delta_safe = np.where(np.abs(sin_delta) < 1e-9, 1e-9, sin_delta)
    
    # Matrix elements for each layer j and wavelength w
    # M_j = [[cos, i/(n_j sin)], [i n_j sin, cos]]
    
    # We need to multiply these matrices.
    # Let's do it layer by layer to keep memory usage low and code simple.
    # Start with M_stack = I
    m_stack = np.eye(2, dtype=complex)
    
    for j in range(n_layers):
        c = cos_delta[j]
        s = sin_delta[j]
        n_j = n_layers_arr[j]
        
        # M_j = [[c, 1j/(n_j*s)], [1j*n_j*s, c]]
        m_j = np.array([
            [c, 1j / (n_j * s)],
            [1j * n_j * s, c]
        ], dtype=complex)
        
        # M_stack = M_stack @ M_j
        # m_stack is (2, 2), m_j is (2, 2) but broadcasted over wavelengths?
        # No, c and s are arrays of shape (W,).
        # So m_j is effectively a stack of matrices.
        # We need to perform matrix multiplication for each wavelength.
        
        # m_stack is currently (2, 2) of complex scalars? No, it needs to be (2, 2, W) or we loop over W.
        # Let's maintain m_stack as a 2x2 matrix of arrays of shape (W,).
        
        # Initialize m_stack properly if it's the first layer
        if j == 0:
            m_stack = m_j
        else:
            # Matrix multiplication: (A @ B)
            # A is 2x2 of arrays, B is 2x2 of arrays
            new_m_stack = np.empty((2, 2), dtype=complex)
            new_m_stack[0, 0] = m_stack[0, 0] * m_j[0, 0] + m_stack[0, 1] * m_j[1, 0]
            new_m_stack[0, 1] = m_stack[0, 0] * m_j[0, 1] + m_stack[0, 1] * m_j[1, 1]
            new_m_stack[1, 0] = m_stack[1, 0] * m_j[0, 0] + m_stack[1, 1] * m_j[1, 0]
            new_m_stack[1, 1] = m_stack[1, 0] * m_j[0, 1] + m_stack[1, 1] * m_j[1, 1]
            m_stack = new_m_stack
            
    # Admittance Y = (ambient_index * m_stack[0,0] + 1j * ambient_index * m_stack[0,1]) / (m_stack[1,0] + m_stack[1,1]) ?
    # Formula: Y = (m11 + m12 * Y_s) / (m21 + m22 * Y_s) where Y_s is substrate admittance.
    # Wait, the standard formula for admittance seen from air:
    # Y_in = (m11 * Y_s + m12 * Y_s^2) / (m21 * Y_s + m22) ? No.
    # Let's use the reflection coefficient formula directly.
    # r = (Y_air - Y_in) / (Y_air + Y_in)
    # Y_in = (m11 + m12 * Y_sub) / (m21 + m22 * Y_sub) is incorrect.
    
    # Correct relation:
    # [E]   [m11 m12] [E_s]
    # [H] = [m21 m22] [H_s]
    # H_s = Y_s * E_s
    # So [E]   [m11 m12 Y_s] [E_s]
    #    [H] = [m21 m22 Y_s] [H_s]
    # Y_in = H/E = (m21 * E_s + m22 * Y_s * E_s) / (m11 * E_s + m12 * Y_s * E_s)
    # Y_in = (m21 + m22 * Y_s) / (m11 + m12 * Y_s)
    
    Y_sub = substrate_index
    Y_air = ambient_index
    
    Y_in = (m_stack[1, 0] + m_stack[1, 1] * Y_sub) / (m_stack[0, 0] + m_stack[0, 1] * Y_sub)
    
    r = (Y_air - Y_in) / (Y_air + Y_in)
    reflectance = np.abs(r) ** 2
    return np.mean(reflectance) * 100.0

def _is_valid(materials, thicknesses, problem):
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    if len(materials) > max_layers:
        return False
    if len(materials) != len(thicknesses):
        return False
    for t in thicknesses:
        if t < -1e-9 or t > max_layer_nm + 1e-9:
            return False
    if sum(thicknesses) > max_total_nm + 1e-9:
        return False
    return True

def _clamp_thick(t, max_layer_nm):
    if t < 0:
        return 0.0
    if t > max_layer_nm:
        return max_layer_nm
    return t

def _optimize_with_lbfgsb(materials, thicknesses, problem, wls, materials_dict,
                          ambient_index, substrate_index, deadline):
    n_layers = len(materials)
    if n_layers == 0:
        return materials, thicknesses
    
    max_layer_nm = problem.get("max_layer_nm", 400)
    max_total_nm = problem.get("max_total_nm", 1000)
    
    # Initial guess
    x0 = np.array(thicknesses, dtype=float)
    
    # Bounds
    bounds = [(0.0, max_layer_nm) for _ in range(n_layers)]
    
    def objective(x):
        # Check total thickness constraint
        if np.sum(x) > max_total_nm + 1e-9:
            return 1e6 # Penalty
        thicks = x.tolist()
        return _evaluate(materials, thicks, wls, ambient_index, substrate_index, materials_dict)
    
    try:
        result = scipy.optimize.minimize(objective, x0, method='L-BFGS-B', bounds=bounds, 
                                         options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 10000})
        if result.success or result.fun < objective(x0):
            best_thicks = result.x.tolist()
            # Clamp to be safe
            best_thicks = [_clamp_thick(t, max_layer_nm) for t in best_thicks]
            if _is_valid(materials, best_thicks, problem):
                return materials, best_thicks
    except Exception:
        pass
    
    return materials, thicknesses

def design(problem):
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s buffer for safety
    
    materials_dict = problem.get("materials", {'L': 1.38, 'H': 2.35})
    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    band_nm = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)
    
    wls = np.linspace(band_nm[0], band_nm[1], 401)
    
    # Strategy: Start with a 10-layer quarter-wave stack.
    # Central wavelength lambda_0 = 600 nm
    # Optical thickness = lambda_0 / 4 = 150 nm
    # Physical thickness = 150 / n
    
    n_layers = 10
    # Pattern: L H L H L H L H L H (Start with L to match air better? Or H? 
    # Usually, for AR on glass (n=1.52), starting with L (1.38) is good as it's closer to air (1.0) than H (2.35).
    # Let's try L H L H ...
    materials = []
    thicknesses = []
    
    lambda_0 = 600.0
    for i in range(n_layers):
        if i % 2 == 0:
            mat = 'L'
        else:
            mat = 'H'
        materials.append(mat)
        n_mat = materials_dict[mat]
        d = lambda_0 / (4.0 * n_mat)
        thicknesses.append(d)
        
    # Check validity of initial guess
    if not _is_valid(materials, thicknesses, problem):
        # Fallback to simpler 2-layer or 4-layer if 10 is too thick?
        # 10 layers * 150nm optical / avg 1.8 index ~ 830nm total. Should be under 1000.
        # Let's check sum
        total = sum(thicknesses)
        if total > max_total_nm:
            # Scale down
            scale = max_total_nm / total
            thicknesses = [t * scale for t in thicknesses]
    
    best_materials = materials
    best_thicknesses = thicknesses
    best_score = _evaluate(materials, thicknesses, wls, ambient_index, substrate_index, materials_dict)
    
    # Run optimization
    if time.time() < deadline:
        opt_materials, opt_thicknesses = _optimize_with_lbfgsb(materials, thicknesses, problem, wls, materials_dict,
                                                               ambient_index, substrate_index, deadline)
        if _is_valid(opt_materials, opt_thicknesses, problem):
            score = _evaluate(opt_materials, opt_thicknesses, wls, ambient_index, substrate_index, materials_dict)
            if score < best_score:
                best_materials = opt_materials
                best_thicknesses = opt_thicknesses
                best_score = score

    # Try a few other configurations if time permits?
    # The directive says "KEEP this design and investigate improvements within its family".
    # The "family" is the optimization of the thicknesses of a fixed layer sequence.
    # We can try different sequences if time allows, but L-BFGS-B on 10 variables is fast.
    # Let's try a 5-layer stack as well (L H L H L)
    if time.time() < deadline - 20:
        materials_5 = ['L', 'H', 'L', 'H', 'L']
        thicknesses_5 = []
        for mat in materials_5:
            n_mat = materials_dict[mat]
            d = lambda_0 / (4.0 * n_mat)
            thicknesses_5.append(d)
        
        if _is_valid(materials_5, thicknesses_5, problem):
            opt_materials_5, opt_thicknesses_5 = _optimize_with_lbfgsb(materials_5, thicknesses_5, problem, wls, materials_dict,
                                                                       ambient_index, substrate_index, deadline)
            if _is_valid(opt_materials_5, opt_thicknesses_5, problem):
                score_5 = _evaluate(opt_materials_5, opt_thicknesses_5, wls, ambient_index, substrate_index, materials_dict)
                if score_5 < best_score:
                    best_materials = opt_materials_5
                    best_thicknesses = opt_thicknesses_5
                    best_score = score_5

    return best_materials, best_thicknesses
