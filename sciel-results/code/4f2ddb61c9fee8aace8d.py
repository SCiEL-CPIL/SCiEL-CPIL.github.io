import numpy as np

def design(problem):
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band_min, band_max = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    wavelengths = np.linspace(band_min, band_max, 401)
    num_wl = len(wavelengths)
    
    num_layers = 12
    layer_indices = np.array([nL if i % 2 == 0 else nH for i in range(num_layers)])
    layer_labels = ['L' if i % 2 == 0 else 'H' for i in range(num_layers)]
    
    def calculate_reflectance(thicknesses):
        # thicknesses: array of shape (num_layers,)
        # Check constraints
        if np.sum(thicknesses) > max_total_nm:
            return 100.0
        if np.any(thicknesses < 0) or np.any(thicknesses > max_layer_nm):
            return 100.0
            
        # deltas: (num_layers, num_wl)
        deltas = 2.0 * np.pi * np.outer(layer_indices, thicknesses) / wavelengths[np.newaxis, :]
        c = np.cos(deltas)
        s = np.sin(deltas)
        
        # Initialize M11, M12, M21, M22 as arrays of shape (num_wl,)
        M11 = np.ones(num_wl, dtype=np.complex128)
        M12 = np.zeros(num_wl, dtype=np.complex128)
        M21 = np.zeros(num_wl, dtype=np.complex128)
        M22 = np.ones(num_wl, dtype=np.complex128)
        
        for i in range(num_layers):
            c_i = c[i]
            s_i = s[i]
            n_i = layer_indices[i]
            
            m11 = c_i
            m12 = 1j * s_i / n_i
            m21 = 1j * n_i * s_i
            m22 = c_i
            
            new_M11 = M11 * m11 + M12 * m21
            new_M12 = M11 * m12 + M12 * m22
            new_M21 = M21 * m11 + M22 * m21
            new_M22 = M21 * m12 + M22 * m22
            
            M11 = new_M11
            M12 = new_M12
            M21 = new_M21
            M22 = new_M22
            
        Y = (M21 + M22 * ns) / (M11 + M12 * ns)
        R = np.abs((Y - n0) / (Y + n0)) ** 2
        
        return np.mean(R) * 100.0

    center_wl = (band_min + band_max) / 2.0
    initial_thicknesses = np.array([center_wl / (4.0 * n) for n in layer_indices])
    
    T = 10.0
    T_min = 0.1
    alpha = 0.99
    current_thicknesses = initial_thicknesses.copy()
    current_R = calculate_reflectance(current_thicknesses)
    best_thicknesses = current_thicknesses.copy()
    best_R = current_R
    
    import random
    random.seed(42)
    
    max_iterations = 3000
    for i in range(max_iterations):
        new_thicknesses = current_thicknesses.copy()
        num_perturb = random.randint(1, 3)
        for _ in range(num_perturb):
            idx = random.randint(0, num_layers - 1)
            noise = random.gauss(0, 20.0)
            new_thicknesses[idx] += noise
            new_thicknesses[idx] = np.clip(new_thicknesses[idx], 0, max_layer_nm)
        
        if np.sum(new_thicknesses) > max_total_nm:
            scale = max_total_nm / np.sum(new_thicknesses)
            new_thicknesses *= scale
            new_thicknesses = np.clip(new_thicknesses, 0, max_layer_nm)
            if np.sum(new_thicknesses) > max_total_nm:
                continue
            
        new_R = calculate_reflectance(new_thicknesses)
        
        if new_R < current_R:
            current_thicknesses = new_thicknesses
            current_R = new_R
            if current_R < best_R:
                best_R = current_R
                best_thicknesses = current_thicknesses.copy()
        else:
            delta_E = new_R - current_R
            if delta_E < 0:
                current_thicknesses = new_thicknesses
                current_R = new_R
            else:
                prob = np.exp(-delta_E / T)
                if random.random() < prob:
                    current_thicknesses = new_thicknesses
                    current_R = new_R
        
        T *= alpha
        if T < T_min:
            break
            
    if np.sum(best_thicknesses) > max_total_nm:
        best_thicknesses = initial_thicknesses
        
    return layer_labels, list(best_thicknesses)
