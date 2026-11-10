"""Broadband antireflection coating design using Differential Evolution."""
import numpy as np
import time
import sys
from scipy.optimize import differential_evolution

# EVOLVE-BLOCK-START

def _characteristic_matrix_reflectance(n0, ns, n_layers, thicknesses_nm, lambdas_nm):
    """
    Compute normal-incidence reflectance using the characteristic matrix method.
    
    Parameters:
    -----------
    n0 : float
        Ambient index (air)
    ns : float
        Substrate index (glass)
    n_layers : array-like
        Refractive indices of each layer
    thicknesses_nm : array-like
        Physical thicknesses of each layer in nm
    lambdas_nm : array-like
        Wavelengths in nm
    
    Returns:
    --------
    R : ndarray
        Reflectance values (0 to 1) at each wavelength
    """
    N = len(n_layers)
    n_layers = np.asarray(n_layers, dtype=float)
    thicknesses_nm = np.asarray(thicknesses_nm, dtype=float)
    lambdas_nm = np.asarray(lambdas_nm, dtype=float)
    
    # Phase factor: delta_j = 2*pi*n_j*d_j / lambda
    # For each layer j and wavelength k:
    # delta[j, k] = 2*pi*n_layers[j]*thicknesses_nm[j]/lambdas_nm[k]
    
    # Build the characteristic matrix for each wavelength
    # M_total = M_1 * M_2 * ... * M_N
    # where M_j = [[cos(delta_j), i*sin(delta_j)/n_j],
    #              [i*n_j*sin(delta_j), cos(delta_j)]]
    
    # Initialize total matrix as identity for all wavelengths
    M11 = np.ones((len(lambdas_nm),), dtype=complex)
    M12 = np.zeros((len(lambdas_nm),), dtype=complex)
    M21 = np.zeros((len(lambdas_nm),), dtype=complex)
    M22 = np.ones((len(lambdas_nm),), dtype=complex)
    
    for j in range(N):
        nj = n_layers[j]
        dj = thicknesses_nm[j]
        
        # delta = 2*pi*nj*dj/lambda
        delta = 2.0 * np.pi * nj * dj / lambdas_nm
        
        c = np.cos(delta)
        s = np.sin(delta)
        
        # M_j = [[c, i*s/nj], [i*nj*s, c]]
        # M_total_new = M_total * M_j
        # 
        # [M11, M12]   [c,     i*s/nj]
        # [M21, M22] * [i*nj*s, c     ]
        #
        # new_M11 = M11*c + M12*(i*nj*s)
        # new_M12 = M11*(i*s/nj) + M12*c
        # new_M21 = M21*c + M22*(i*nj*s)
        # new_M22 = M21*(i*s/nj) + M22*c
        
        new_M11 = M11 * c + M12 * (1j * nj * s)
        new_M12 = M11 * (1j * s / nj) + M12 * c
        new_M21 = M21 * c + M22 * (1j * nj * s)
        new_M22 = M21 * (1j * s / nj) + M22 * c
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22
    
    # Admittance of the stack:
    # Y = (M21 + M22 * ns) / (M11 + M12 * ns)
    # Wait, let me be careful with the convention.
    
    # The characteristic matrix relates the fields at the front and back:
    # [E_f]   [M11 M12] [E_b]
    # [H_f] = [M21 M22] [H_b]
    
    # At the back (substrate side), for a plane wave going into substrate:
    # H_b = (ns/n0) * E_b  (normalized admittance)
    # Actually, the standard result is:
    # Y_stack = (M21 + M22 * Y_s) / (M11 + M12 * Y_s)
    # where Y_s = ns (admittance of substrate, normalized to... let me think)
    
    # The reflectance is:
    # r = (Y_stack - n0) / (Y_stack + n0)
    # R = |r|^2
    
    # Let me use the standard formula. The characteristic matrix M satisfies:
    # [E]     = M [E']
    # [H/n0]      [H'/n0]  -- no, this gets confusing.
    
    # Standard approach:
    # The total characteristic matrix is the product of individual layer matrices.
    # For a stack on substrate with refractive index ns, the input admittance is:
    # Y_in = (M21 + M22 * ns) / (M11 + M12 * ns)
    # where the matrices are as defined above with the cos/sin form.
    # Then the reflection coefficient is:
    # r = (Y_in - n0) / (Y_in + n0)
    # R = |r|^2
    
    Y_in = (M21 + M22 * ns) / (M11 + M12 * ns)
    r = (Y_in - n0) / (Y_in + n0)
    R = np.abs(r) ** 2
    
    return R


def _mean_reflectance_percent(n0, ns, n_layers, thicknesses_nm, band_nm):
    """Compute mean reflectance in percent over the band."""
    lambdas = np.linspace(band_nm[0], band_nm[1], 401)
    R = _characteristic_matrix_reflectance(n0, ns, n_layers, thicknesses_nm, lambdas)
    return 100.0 * np.mean(R)


def _is_valid(n_layers, thicknesses_nm, max_layers, max_total_nm, max_layer_nm):
    """Check if the design is valid."""
    if len(n_layers) > max_layers:
        return False
    if len(n_layers) != len(thicknesses_nm):
        return False
    if any(t < 0 or t > max_layer_nm for t in thicknesses_nm):
        return False
    if sum(thicknesses_nm) > max_total_nm:
        return False
    return True


def design(problem):
    """
    Design a broadband antireflection coating using Differential Evolution.
    """
    start_time = time.time()
    deadline = 130.0  # Safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    nL = materials["L"]
    nH = materials["H"]
    
    # We'll search over a fixed number of layers (up to max_layers)
    # Each layer has a type (L or H) and a thickness.
    # To use differential evolution, we encode:
    # - For each layer: a continuous value in [0, 2) where <1 means L, >=1 means H
    # - Thickness in [0, max_layer_nm]
    # But this is tricky because the number of layers should be variable.
    
    # Alternative: Use a fixed number of layers (say, 10) and allow zero thickness.
    # A layer with zero thickness is effectively absent.
    # We also need to handle the material type.
    
    # Let's use a different encoding:
    # For each of the max_layers possible layers, we have:
    # - thickness in [0, max_layer_nm]
    # - material selector in [0, 1] where <0.5 means L, >=0.5 means H
    # But this doubles the dimensionality.
    
    # Better approach: Use a fixed number of layers N (e.g., 8-12) and optimize
    # thicknesses and material assignments.
    
    # Let's try with N = 8 layers first, then try more if time permits.
    
    def objective(x):
        """Objective function for differential evolution."""
        # x is a vector of length 2*N: [t1, s1, t2, s2, ..., tN, sN]
        # where tj is thickness, sj is material selector
        N = len(x) // 2
        thicknesses = x[0::2]
        selectors = x[1::2]
        
        # Filter out layers with zero thickness
        n_layers = []
        thicknesses_nm = []
        for i in range(N):
            t = thicknesses[i]
            if t < 1e-6:  # Essentially zero
                continue
            s = selectors[i]
            if s < 0.5:
                n_layers.append(nL)
            else:
                n_layers.append(nH)
            thicknesses_nm.append(t)
        
        if len(n_layers) == 0:
            return 100.0  # No coating, high reflectance
        
        if len(n_layers) > max_layers:
            return 100.0
        
        if sum(thicknesses_nm) > max_total_nm:
            return 100.0
        
        if any(t > max_layer_nm for t in thicknesses_nm):
            return 100.0
        
        return _mean_reflectance_percent(n0, ns, n_layers, thicknesses_nm, band_nm)
    
    # Try different numbers of layers
    best_result = None
    best_score = float('inf')
    
    for N in [4, 6, 8, 10, 12]:
        if time.time() > deadline - 30:
            break
        
        # Bounds for differential evolution
        # thickness: [0, max_layer_nm]
        # selector: [0, 1]
        bounds = [(0, max_layer_nm), (0, 1)] * N
        
        # Use differential evolution
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=100,
                popsize=15,
                tol=1e-4,
                mutation=(0.3, 1.0),
                recombination=0.7,
                seed=42,
                polish=True,
                init='latinhypercube'
            )
            
            if result.success or result.fun < best_score:
                x = result.x
                N_actual = len(x) // 2
                thicknesses = x[0::2]
                selectors = x[1::2]
                
                n_layers = []
                thicknesses_nm = []
                for i in range(N_actual):
                    t = thicknesses[i]
                    if t < 1e-6:
                        continue
                    s = selectors[i]
                    if s < 0.5:
                        n_layers.append(nL)
                    else:
                        n_layers.append(nH)
                    thicknesses_nm.append(t)
                
                if len(n_layers) > 0 and len(n_layers) <= max_layers:
                    if sum(thicknesses_nm) <= max_total_nm:
                        if all(t <= max_layer_nm for t in thicknesses_nm):
                            score = result.fun
                            if score < best_score:
                                best_score = score
                                best_result = (n_layers, thicknesses_nm)
        except Exception as e:
            pass
        
        if time.time() > deadline - 20:
            break
    
    # If no valid result found, return a simple quarter-wave coating
    if best_result is None:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        thickness = centre / (4.0 * nL)
        return ["L"], [thickness]
    
    # Convert to labels and thicknesses
    n_layers, thicknesses_nm = best_result
    labels = []
    for n in n_layers:
        if abs(n - nL) < 1e-9:
            labels.append("L")
        else:
            labels.append("H")
    
    return labels, thicknesses_nm

# EVOLVE-BLOCK-END
