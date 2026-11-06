"""Broadband antireflection coating design via simulated annealing over layer sequences."""
import time
import random
import numpy as np

def design(problem):
    """Design a low/high-index stack minimizing mean reflectance 400-800 nm."""
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    ambient_index = problem.get("ambient_index", 1.0)
    substrate_index = problem.get("substrate_index", 1.52)
    materials = problem.get("materials", {"L": 1.38, "H": 2.35})
    band_nm = problem.get("band_nm", [400, 800])
    max_layers = problem.get("max_layers", 20)
    max_total_nm = problem.get("max_total_nm", 1000)
    max_layer_nm = problem.get("max_layer_nm", 400)

    n_low = materials["L"]
    n_high = materials["H"]
    n_amb = ambient_index
    n_sub = substrate_index

    # Wavelength grid: 401 equally spaced points from 400 to 800 nm
    wavelengths = np.linspace(band_nm[0], band_nm[1], 401)

    def compute_reflectance(materials_list, thicknesses_list):
        """Compute mean reflectance (%) using characteristic matrix method."""
        n_layers = len(materials_list)
        if n_layers == 0:
            # Bare substrate
            r = (n_amb - n_sub) / (n_amb + n_sub)
            return 100.0 * r * r

        total_t = sum(thicknesses_list)
        if total_t > max_total_nm + 1e-9:
            return 1e10  # penalty

        for t in thicknesses_list:
            if t < -1e-9 or t > max_layer_nm + 1e-9:
                return 1e10

        # Build the total characteristic matrix
        # M_total = product of individual layer matrices
        # Each layer matrix: [[cos(delta), i*sin(delta)/n], [i*n*sin(delta), cos(delta)]]
        # where delta = 2*pi*n*t/lambda

        # For efficiency, precompute for all wavelengths
        n_wl = len(wavelengths)
        
        # Initialize total matrix as identity for each wavelength
        # M[i][0,0]=1, M[i][0,1]=0, M[i][1,0]=0, M[i][1,1]=1
        M00 = np.ones(n_wl, dtype=complex)
        M01 = np.zeros(n_wl, dtype=complex)
        M10 = np.zeros(n_wl, dtype=complex)
        M11 = np.ones(n_wl, dtype=complex)

        # Multiply from air side to substrate
        for i in range(n_layers):
            n_i = n_low if materials_list[i] == 'L' else n_high
            t_i = thicknesses_list[i]
            
            # delta = 2*pi*n*t/lambda
            delta = 2.0 * np.pi * n_i * t_i / wavelengths
            
            cos_d = np.cos(delta)
            sin_d = np.sin(delta)
            
            # Layer matrix elements
            L00 = cos_d
            L01 = 1j * sin_d / n_i
            L10 = 1j * n_i * sin_d
            L11 = cos_d
            
            # New M = M * L (matrix multiplication)
            # New M00 = M00*L00 + M01*L10
            # New M01 = M00*L01 + M01*L11
            # New M10 = M10*L00 + M11*L10
            # New M11 = M10*L01 + M11*L11
            
            new_M00 = M00 * L00 + M01 * L10
            new_M01 = M00 * L01 + M01 * L11
            new_M10 = M10 * L00 + M11 * L10
            new_M11 = M10 * L01 + M11 * L11
            
            M00, M01, M10, M11 = new_M00, new_M01, new_M10, new_M11

        # Admittance: Y = n_sub * M11 + M10 / (n_sub)  ... wait, let me re-derive
        # The characteristic matrix relates:
        # [B, C]^T = M * [E_s, H_s]^T
        # where E_s and H_s are the fields at the substrate interface
        # For normal incidence, the admittance Y = B/C
        # B = M00 * E_s + M01 * H_s
        # C = M10 * E_s + M11 * H_s
        # At the substrate, H_s = E_s / n_sub (for normal incidence, H = E/n)
        # So B = E_s * (M00 + M01 / n_sub)
        #    C = E_s * (M10 + M11 / n_sub)
        # Y = B/C = (M00 + M01/n_sub) / (M10 + M11/n_sub)
        # Reflectance: R = |(1 - Y/n_amb) / (1 + Y/n_amb)|^2  ... 
        # Actually, the reflection coefficient is r = (Y_amb - Y) / (Y_amb + Y)
        # where Y_amb = n_amb (for air)
        
        Y = (M00 + M01 / n_sub) / (M10 + M11 / n_sub)
        
        r = (n_amb - Y) / (n_amb + Y)
        R = np.abs(r) ** 2
        return 100.0 * np.mean(R)

    # Initial design: a few quarter-wave layers
    centre_wl = 0.5 * (band_nm[0] + band_nm[1])
    
    def make_initial_design():
        """Create a reasonable starting design."""
        # Start with 5 layers: L H L H L pattern
        n_init = 5
        labels = ['L', 'H', 'L', 'H', 'L']
        thicknesses = []
        for lab in labels:
            n = n_low if lab == 'L' else n_high
            t = centre_wl / (4.0 * n)
            thicknesses.append(min(t, max_layer_nm))
        return labels, thicknesses

    best_labels, best_thick = make_initial_design()
    best_score = compute_reflectance(best_labels, best_thick)
    
    # Simulated annealing
    current_labels, current_thick = best_labels[:], best_thick[:]
    current_score = best_score
    
    T = 5.0  # initial temperature
    T_min = 0.01
    cooling_rate = 0.9995
    
    # Number of iterations based on time budget
    # Each iteration: propose move, evaluate
    # Let's estimate: ~10000 evaluations should be feasible
    
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Check time periodically
        if rng.random() < 0.01:
            if time.time() > deadline:
                break
        
        # Propose a move
        move_type = rng.random()
        
        new_labels = current_labels[:]
        new_thick = current_thick[:]
        
        if move_type < 0.3:
            # Change a layer's material
            if len(new_labels) > 0:
                idx = rng.randint(0, len(new_labels) - 1)
                old_mat = new_labels[idx]
                new_mat = 'H' if old_mat == 'L' else 'L'
                new_labels[idx] = new_mat
                # Adjust thickness to maintain quarter-wave at center
                n_new = n_low if new_mat == 'L' else n_high
                new_thick[idx] = min(centre_wl / (4.0 * n_new), max_layer_nm)
                
        elif move_type < 0.5:
            # Change a layer's thickness
            if len(new_thick) > 0:
                idx = rng.randint(0, len(new_thick) - 1)
                # Gaussian perturbation
                delta_t = rng.gauss(0, 20)
                new_thick[idx] = max(0, min(max_layer_nm, new_thick[idx] + delta_t))
                
        elif move_type < 0.7:
            # Add a layer
            if len(new_labels) < max_layers:
                pos = rng.randint(0, len(new_labels))
                new_mat = rng.choice(['L', 'H'])
                new_labels.insert(pos, new_mat)
                n_new = n_low if new_mat == 'L' else n_high
                t_new = min(centre_wl / (4.0 * n_new), max_layer_nm)
                new_thick.insert(pos, t_new)
                
        elif move_type < 0.9:
            # Remove a layer
            if len(new_labels) > 1:
                pos = rng.randint(0, len(new_labels) - 1)
                new_labels.pop(pos)
                new_thick.pop(pos)
                
        else:
            # Swap two layers
            if len(new_labels) >= 2:
                i, j = rng.sample(range(len(new_labels)), 2)
                new_labels[i], new_labels[j] = new_labels[j], new_labels[i]
                new_thick[i], new_thick[j] = new_thick[j], new_thick[i]
        
        # Evaluate
        new_score = compute_reflectance(new_labels, new_thick)
        
        # Acceptance criterion
        if new_score < current_score:
            accept = True
        else:
            delta = new_score - current_score
            if T > T_min:
                accept = rng.random() < np.exp(-delta / T)
            else:
                accept = False
        
        if accept:
            current_labels, current_thick = new_labels, new_thick
            current_score = new_score
            
            if current_score < best_score:
                best_score = current_score
                best_labels = current_labels[:]
                best_thick = current_thick[:]
        
        # Cool
        T *= cooling_rate
        
        # Safety check
        if time.time() > deadline:
            break
    
    # Final validation
    if sum(best_thick) > max_total_nm:
        # Scale down
        scale = max_total_nm / sum(best_thick)
        best_thick = [t * scale for t in best_thick]
    
    # Clamp thicknesses
    best_thick = [max(0, min(max_layer_nm, t)) for t in best_thick]
    
    return best_labels, best_thick
