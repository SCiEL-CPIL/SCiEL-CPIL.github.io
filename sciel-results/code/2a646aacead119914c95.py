"""Broadband AR coating design: Discrete multi-start optimization with gradient refinement."""
import numpy as np
import time
import random

# EVOLVE-BLOCK-START
def design(problem):
    """Design a broadband AR coating using discrete multi-start optimization with gradient refinement."""
    start_time = time.time()
    DEADLINE = 145.0  # safety margin before 150s kill
    
    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    nL = problem["materials"]["L"]
    nH = problem["materials"]["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation (401 points from 400 to 800 nm)
    lambdas = np.linspace(band[0], band[1], 401)
    lambda_c = 0.5 * (band[0] + band[1])
    n_wl = len(lambdas)
    
    def mean_reflectance(materials, thicknesses):
        """Compute mean reflectance in percent over the band using vectorized characteristic matrix method."""
        n_layers = np.array([nL if m == 'L' else nH for m in materials], dtype=float)
        n = len(materials)
        if n == 0:
            r = (n0 - ns) / (n0 + ns)
            return (np.abs(r) ** 2) * 100.0
        
        # Vectorized computation over all wavelengths
        m_total = np.zeros((n_wl, 2, 2), dtype=complex)
        m_total[:, 0, 0] = 1.0
        m_total[:, 1, 1] = 1.0
        
        for i in range(n):
            ni = n_layers[i]
            di = thicknesses[i]
            delta = 2.0 * np.pi * ni * di / lambdas
            c = np.cos(delta)
            s = np.sin(delta)
            
            m_layer = np.zeros((n_wl, 2, 2), dtype=complex)
            m_layer[:, 0, 0] = c
            m_layer[:, 0, 1] = 1j * s / ni
            m_layer[:, 1, 0] = 1j * ni * s
            m_layer[:, 1, 1] = c
            
            m_total = np.einsum('wij,wjk->wik', m_total, m_layer)
        
        A = m_total[:, 0, 0]
        B = m_total[:, 0, 1]
        C = m_total[:, 1, 0]
        D = m_total[:, 1, 1]
        
        y_in = (C + D * ns) / (A + B * ns)
        r = (n0 - y_in) / (n0 + y_in)
        R = np.abs(r) ** 2
        
        return float(np.mean(R) * 100.0)
    
    def is_valid(materials, thicknesses):
        """Check constraints."""
        if len(materials) != len(thicknesses):
            return False
        if len(materials) > max_layers:
            return False
        if any(t < -1e-9 or t > max_layer_nm + 1e-9 for t in thicknesses):
            return False
        if sum(thicknesses) > max_total_nm + 1e-9:
            return False
        return True
    
    def clip_thicknesses(materials, thicknesses):
        """Clip thicknesses to valid range and ensure total constraint."""
        t = np.array(thicknesses, dtype=float)
        t = np.clip(t, 0.0, max_layer_nm)
        total = np.sum(t)
        if total > max_total_nm:
            scale = max_total_nm / total
            t = t * scale
        return [float(x) for x in t]
    
    def gradient_descent(materials, x0, max_iter=3000, lr_init=5.0, lr_decay=0.999):
        """Deterministic multi-start gradient descent with numerical gradients."""
        n = len(x0)
        x = np.array(x0, dtype=float)
        f = mean_reflectance(materials, x)
        best_x = list(x)
        best_f = f
        
        lr = lr_init
        eps = 1e-4
        
        for it in range(max_iter):
            if (time.time() - start_time) > DEADLINE:
                break
            
            # Compute numerical gradient
            grad = np.zeros(n)
            for i in range(n):
                x_plus = list(x)
                x_plus[i] += eps
                x_plus = clip_thicknesses(materials, x_plus)
                f_plus = mean_reflectance(materials, x_plus)
                
                x_minus = list(x)
                x_minus[i] -= eps
                x_minus = clip_thicknesses(materials, x_minus)
                f_minus = mean_reflectance(materials, x_minus)
                
                grad[i] = (f_plus - f_minus) / (2 * eps)
            
            # Check convergence
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-6:
                break
            
            # Update
            x = x - lr * grad
            x = clip_thicknesses(materials, x)
            
            new_f = mean_reflectance(materials, x)
            
            # Adaptive learning rate
            if new_f > f:
                lr *= 0.5
                x = list(best_x)
                f = best_f
            else:
                if new_f < best_f:
                    best_f = new_f
                    best_x = list(x)
                f = new_f
                lr = min(lr * 1.01, 20.0)
        
        return best_x, best_f
    
    # --- Generate diverse candidate structures ---
    candidates = []
    
    qw_L = lambda_c / (4.0 * nL)
    qw_H = lambda_c / (4.0 * nH)
    
    # Standard alternating structures with various layer counts
    for n_layers in range(1, 17):
        # LHLH...
        mat = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        thick = [qw_L if i % 2 == 0 else qw_H for i in range(n_layers)]
        candidates.append((mat, thick))
        
        # HLHL...
        mat = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thick = [qw_H if i % 2 == 0 else qw_L for i in range(n_layers)]
        candidates.append((mat, thick))
    
    # Geometric progression thicknesses for broader spectral coverage
    for n_layers in [5, 7, 9, 11]:
        # LHLH... with geometric progression
        mat = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        # Start with thinner layers, increase geometrically
        base_L = qw_L * 0.6
        base_H = qw_H * 0.6
        ratio = 1.15
        thick = []
        for i in range(n_layers):
            if i % 2 == 0:
                thick.append(base_L * (ratio ** (i // 2)))
            else:
                thick.append(base_H * (ratio ** (i // 2)))
        thick = clip_thicknesses(mat, thick)
        candidates.append((mat, thick))
        
        # HLHL... with geometric progression
        mat = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        thick = []
        for i in range(n_layers):
            if i % 2 == 0:
                thick.append(base_H * (ratio ** (i // 2)))
            else:
                thick.append(base_L * (ratio ** (i // 2)))
        thick = clip_thicknesses(mat, thick)
        candidates.append((mat, thick))
    
    # Random structures for diversity
    random.seed(42)
    for _ in range(30):
        n_layers = random.randint(3, 15)
        mat = [random.choice(['L', 'H']) for _ in range(n_layers)]
        thick = [random.uniform(qw_L*0.3, qw_L*1.7) if m=='L' else random.uniform(qw_H*0.3, qw_H*1.7) for m in mat]
        thick = clip_thicknesses(mat, thick)
        candidates.append((mat, thick))
    
    # Evaluate all candidates
    best_score = float('inf')
    best_materials = None
    best_thicknesses = None
    scored_candidates = []
    
    for mat, thick in candidates:
        if not is_valid(mat, thick):
            continue
        score = mean_reflectance(mat, thick)
        scored_candidates.append((score, mat, thick))
        if score < best_score:
            best_score = score
            best_materials = mat
            best_thicknesses = thick
    
    if best_materials is None:
        return ["L"], [qw_L]
    
    scored_candidates.sort(key=lambda x: x[0])
    
    # --- Refine top candidates using gradient descent ---
    top_k = min(8, len(scored_candidates))
    for score, materials, thick in scored_candidates[:top_k]:
        if (time.time() - start_time) > DEADLINE:
            break
        
        refined_thick, refined_f = gradient_descent(materials, thick)
        
        if refined_f < best_score:
            best_score = refined_f
            best_materials = materials
            best_thicknesses = refined_thick
    
    # --- Try additional structures with gradient descent ---
    additional_structures = []
    
    # Mixed patterns
    for n_layers in [6, 8, 10, 12]:
        # LHLHHL pattern
        mat = ['L' if i % 2 == 0 else 'H' for i in range(n_layers)]
        # Insert double H at middle
        mid = n_layers // 2
        if mid < n_layers - 1:
            mat.insert(mid, 'H')
            thick = [qw_L if m == 'L' else qw_H for m in mat]
            thick = clip_thicknesses(mat, thick)
            additional_structures.append((mat, thick))
        
        # HLHLHL pattern with double L
        mat = ['H' if i % 2 == 0 else 'L' for i in range(n_layers)]
        mid = n_layers // 2
        if mid < n_layers - 1:
            mat.insert(mid, 'L')
            thick = [qw_H if m == 'H' else qw_L for m in mat]
            thick = clip_thicknesses(mat, thick)
            additional_structures.append((mat, thick))
    
    # Evaluate and refine additional structures
    for mat, thick in additional_structures:
        if (time.time() - start_time) > DEADLINE:
            break
        if not is_valid(mat, thick):
            continue
        score = mean_reflectance(mat, thick)
        if score < best_score * 1.2:  # Only refine if close to best
            refined_thick, refined_f = gradient_descent(mat, thick)
            if refined_f < best_score:
                best_score = refined_f
                best_materials = mat
                best_thicknesses = refined_thick
    
    # Final validation
    best_thicknesses = clip_thicknesses(best_materials, best_thicknesses)
    
    if not is_valid(best_materials, best_thicknesses):
        for mat, thick in scored_candidates:
            if is_valid(mat, thick):
                return mat, thick
        return ["L"], [qw_L]
    
    return best_materials, best_thicknesses
# EVOLVE-BLOCK-END
