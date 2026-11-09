"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution

def _reflectance_spectrum(thicknesses, materials, ambient_index, substrate_index,
                          band_nm, n_points=401):
    """Compute mean normal-incidence reflectance (%) using characteristic-matrix method."""
    L = materials['L']
    H = materials['H']
    # Determine which index each layer uses based on order
    # We need to know the labels. Let's reconstruct from the design function context.
    # Actually, the helper should take labels too or we pass indices directly.
    # Let's modify the internal helper to take indices list.
    pass

def _mean_reflectance_pct(labels, thicknesses, materials, ambient_index, substrate_index, band_nm):
    """
    Compute mean reflectance in percent over the band.
    labels: list of 'L' or 'H'
    thicknesses: list of floats in nm
    """
    n_points = 401
    wavelengths = np.linspace(band_nm[0], band_nm[1], n_points)
    k = 2 * np.pi / wavelengths  # wavenumber in 1/nm
    
    # Indices
    n0 = ambient_index
    ns = substrate_index
    n_layers = len(labels)
    
    # Pre-compute layer indices
    n_list = []
    for lab in labels:
        n_list.append(materials[lab])
    
    # Characteristic matrix for each layer
    # M_j = [[cos(d_j), i*sin(d_j)/n_j], [i*n_j*sin(d_j), cos(d_j)]]
    # For normal incidence, admittance is real.
    # We can use complex arithmetic.
    
    # To be efficient, we can vectorize over wavelengths.
    # But n_points is 401 and max_layers is 20, so 8020 matrix multiplications.
    # Doing this in a loop over wavelengths might be slow in pure Python/NumPy if not careful.
    # Let's try vectorizing the matrix multiplication.
    
    # Matrix M for layer j at wavelength lambda:
    # delta_j = pi * n_j * d_j / lambda
    # cos(delta), sin(delta)
    
    # Let's compute for all wavelengths at once for each layer.
    # M_j[0,0] = cos(delta_j)
    # M_j[0,1] = 1j * sin(delta_j) / n_j
    # M_j[1,0] = 1j * n_j * sin(delta_j)
    # M_j[1,1] = cos(delta_j)
    
    # Total matrix M = M_1 * M_2 * ... * M_N
    # Then Y = (n0 * M[1,0] + M[1,1]) / (n0 * M[0,0] + M[0,1]) ? 
    # Wait, the standard formula for admittance Y at the front:
    # Y = (M11 + M12 * ns) / (M21 + M22 * ns)  ... wait, indices are 0-based in code, 1-based in theory.
    # Let M = [[M11, M12], [M21, M22]]
    # Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    # R = | (n0 - Y) / (n0 + Y) |^2
    
    # Let's verify the matrix multiplication order.
    # Light travels from left to right.
    # M_total = M_1 * M_2 * ... * M_N
    # Where M_j is the matrix for layer j.
    # The vector [E, H] at the left of layer j is related to the right by M_j.
    # So [E, H]_left = M_total * [E, H]_substrate_interface.
    # At the substrate interface, H/E = ns (assuming forward wave only in substrate? No, there is reflection).
    # Actually, the boundary condition at the substrate is that the admittance seen from the coating side is ns.
    # So Y_N = ns.
    # Then we propagate back to the front? Or forward?
    # The characteristic matrix method usually computes the admittance at the front surface given the substrate admittance.
    # Y_0 = (M11 + M12 * ns) / (M21 + M22 * ns)
    # R = | (n0 - Y_0) / (n0 + Y_0) |^2
    
    # Let's implement this vectorized.
    
    M = np.ones((2, 2, n_points), dtype=complex)
    
    for j in range(n_layers):
        n_j = n_list[j]
        d_j = thicknesses[j]
        
        delta = np.pi * n_j * d_j / wavelengths  # shape (n_points,)
        c = np.cos(delta)
        s = np.sin(delta)
        
        M_j = np.ones((2, 2, n_points), dtype=complex)
        M_j[0, 0, :] = c
        M_j[0, 1, :] = 1j * s / n_j
        M_j[1, 0, :] = 1j * n_j * s
        M_j[1, 1, :] = c
        
        # M = M * M_j
        # M[i, k, :] = sum_l M[i, l, :] * M_j[l, k, :]
        # Since it's small 2x2, we can do it explicitly or use einsum.
        # M_new[0,0] = M[0,0]*M_j[0,0] + M[0,1]*M_j[1,0]
        # M_new[0,1] = M[0,0]*M_j[0,1] + M[0,1]*M_j[1,1]
        # M_new[1,0] = M[1,0]*M_j[0,0] + M[1,1]*M_j[1,0]
        # M_new[1,1] = M[1,0]*M_j[0,1] + M[1,1]*M_j[1,1]
        
        M00 = M[0, 0, :] * M_j[0, 0, :] + M[0, 1, :] * M_j[1, 0, :]
        M01 = M[0, 0, :] * M_j[0, 1, :] + M[0, 1, :] * M_j[1, 1, :]
        M10 = M[1, 0, :] * M_j[0, 0, :] + M[1, 1, :] * M_j[1, 0, :]
        M11 = M[1, 0, :] * M_j[0, 1, :] + M[1, 1, :] * M_j[1, 1, :]
        
        M[0, 0, :] = M00
        M[0, 1, :] = M01
        M[1, 0, :] = M10
        M[1, 1, :] = M11
    
    # Compute Admittance Y
    # Y = (M11 + M12 * ns) / (M21 + M22 * ns)
    # Using 0-based indexing: M[0,0] is M11, M[0,1] is M12, etc.
    Y = (M[0, 0, :] + M[0, 1, :] * ns) / (M[1, 0, :] + M[1, 1, :] * ns)
    
    # Compute Reflectance R
    R = np.abs((n0 - Y) / (n0 + Y)) ** 2
    
    # Mean Reflectance in Percent
    mean_R = np.mean(R) * 100.0
    return mean_R

def design(problem):
    """
    Design a stack of alternating L and H layers with a linear gradient in thickness.
    Optimizes base thickness and gradient factor using differential evolution.
    """
    ambient_index = problem["ambient_index"]
    substrate_index = problem["substrate_index"]
    materials = problem["materials"]
    band_nm = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Strategy:
    # We will fix the number of layers to a reasonable number, say 10 or 12, to allow for optimization.
    # The directive says "Represent the coating stack as a periodic sequence of alternating L and H layers... parameterized by a single base thickness and a linear gradient factor".
    # This implies a fixed structure pattern (alternating L, H, L, H...) but varying thicknesses.
    # Let's try a few different layer counts and pick the best.
    # However, differential evolution is expensive. We should limit the search space.
    
    # Let's define a function that takes (base_thickness, gradient_factor, num_layers) and returns the mean reflectance.
    # But num_layers is an integer. We can try a fixed set of num_layers, say [4, 6, 8, 10, 12, 14, 16].
    # For each, we optimize base_thickness and gradient_factor.
    
    # Structure:
    # Layer 0: L
    # Layer 1: H
    # Layer 2: L
    # ...
    # Thickness of layer i: t_i = base * (1 + gradient * i)
    # Or perhaps a symmetric gradient? The directive says "smooth transition from low to high optical density".
    # Optical density is proportional to n * d.
    # If we alternate L and H, the optical density alternates.
    # "Linear gradient factor to enforce a smooth transition" might mean the thicknesses change linearly with layer index.
    # Let's assume t_i = base * (1 + g * i) where i is 0-based index.
    # We need to ensure t_i is in [0, max_layer_nm] and sum(t_i) <= max_total_nm.
    
    def objective(params, num_layers):
        base, g = params
        # Generate thicknesses
        thicknesses = []
        labels = []
        for i in range(num_layers):
            t = base * (1.0 + g * i)
            if t < 0 or t > max_layer_nm:
                return 100.0  # Penalty
            thicknesses.append(t)
            labels.append('L' if i % 2 == 0 else 'H')
        
        total = sum(thicknesses)
        if total > max_total_nm:
            return 100.0  # Penalty
        
        # Compute reflectance
        try:
            return _mean_reflectance_pct(labels, thicknesses, materials, ambient_index, substrate_index, band_nm)
        except:
            return 100.0
    
    best_score = 100.0
    best_labels = []
    best_thicknesses = []
    
    # Try different numbers of layers
    # The maximum is 20. Let's try even numbers up to 20, and maybe some odd ones if starting with L and ending with H or L.
    # Starting with L is standard for AR on glass (n_glass > n_L).
    # Let's try num_layers in [4, 6, 8, 10, 12, 14, 16, 18, 20]
    
    # To save time, we'll use a coarse DE first or limit iterations.
    # 60s budget. DE with popsize=15, maxiter=50 for 9 layer counts might be too slow.
    # Let's reduce the search space.
    # base_thickness range: [10, 200] nm (typical quarter wave at 600nm is ~100-150nm)
    # gradient range: [-0.5, 0.5]
    
    # We can also fix the number of layers to 10 or 12, which is a good compromise.
    # Let's try 10 and 12 layers.
    
    for num_layers in [10, 12]:
        bounds = [(10, 200), (-0.5, 0.5)]
        
        # Use differential evolution
        # To speed up, use a smaller population and fewer iterations if needed.
        # But we want good quality.
        # Let's try popsize=20, maxiter=30
        # This is 600 evaluations per num_layers.
        # Each evaluation involves 401 wavelengths * num_layers matrix mults.
        # 12 layers * 401 points = 4812 ops. Very fast in NumPy.
        # 600 evals * 0.001s = 0.6s. So 9 layer counts * 0.6s = 5.4s. This is fine.
        
        result = differential_evolution(
            objective,
            bounds,
            args=(num_layers,),
            maxiter=50,
            popsize=20,
            tol=0.01,
            seed=42,
            polish=True
        )
        
        if result.fun < best_score:
            best_score = result.fun
            base, g = result.x
            best_labels = []
            best_thicknesses = []
            for i in range(num_layers):
                t = base * (1.0 + g * i)
                best_thicknesses.append(t)
                best_labels.append('L' if i % 2 == 0 else 'H')
    
    # If no valid solution found (unlikely), fall back to seed
    if not best_labels:
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * materials["L"])]
        
    # Final check on constraints
    if sum(best_thicknesses) > max_total_nm or any(t < 0 or t > max_layer_nm for t in best_thicknesses):
        centre = 0.5 * (band_nm[0] + band_nm[1])
        return ["L"], [centre / (4.0 * materials["L"])]
        
    return best_labels, best_thicknesses

# EVOLVE-BLOCK-END
