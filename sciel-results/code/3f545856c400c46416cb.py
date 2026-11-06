import numpy as np
import time
from itertools import product

def _compute_reflectance(wavelengths, n0, ns, n_layers, t_layers):
    """
    Compute mean reflectance for given wavelengths and layer stack.
    wavelengths: array of wavelengths in nm
    n0: ambient index
    ns: substrate index
    n_layers: array of layer indices
    t_layers: array of layer thicknesses in nm
    """
    if len(n_layers) == 0:
        # Bare substrate
        r = (n0 - ns) / (n0 + ns)
        return np.mean(r**2) * 100.0

    # Precompute
    N = len(n_layers)
    # M is 2x2 complex matrix
    # Initialize M as identity
    M11 = np.ones_like(wavelengths, dtype=complex)
    M12 = np.zeros_like(wavelengths, dtype=complex)
    M21 = np.zeros_like(wavelengths, dtype=complex)
    M22 = np.ones_like(wavelengths, dtype=complex)

    for i in range(N):
        ni = n_layers[i]
        ti = t_layers[i]
        # delta = 2 * pi * ni * ti / lambda
        # Use 1e-7 conversion? No, nm to m? 
        # Formula is 2*pi*n*t/lambda. If t and lambda are in same units, it's fine.
        delta = 2.0 * np.pi * ni * ti / wavelengths
        
        cos_d = np.cos(delta)
        sin_d = np.sin(delta)
        
        # Layer matrix
        # m11 = cos_d
        # m12 = 1j * sin_d / ni
        # m21 = 1j * ni * sin_d
        # m22 = cos_d
        
        # New M = M * Layer
        # M11_new = M11*m11 + M12*m21
        # M12_new = M11*m12 + M12*m22
        # M21_new = M21*m11 + M22*m21
        # M22_new = M21*m12 + M22*m22
        
        m11 = cos_d
        m12 = 1j * sin_d / ni
        m21 = 1j * ni * sin_d
        m22 = cos_d
        
        new_M11 = M11 * m11 + M12 * m21
        new_M12 = M11 * m12 + M12 * m22
        new_M21 = M21 * m11 + M22 * m21
        new_M22 = M21 * m12 + M22 * m22
        
        M11 = new_M11
        M12 = new_M12
        M21 = new_M21
        M22 = new_M22

    # Admittance Y
    # Y = (ns * M11 + 1j * M12) / (ns * M21 + 1j * M22)
    numerator = ns * M11 + 1j * M12
    denominator = ns * M21 + 1j * M22
    
    # Avoid division by zero
    Y = numerator / denominator
    
    # Reflectance
    # R = | (Y - n0) / (Y + n0) |^2
    r = (Y - n0) / (Y + n0)
    R = np.abs(r)**2
    
    return np.mean(R) * 100.0

def design(problem):
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin for 150s limit

    n0 = problem["ambient_index"]
    ns = problem["substrate_index"]
    materials = problem["materials"]
    nL = materials["L"]
    nH = materials["H"]
    band = problem["band_nm"]
    max_layers = problem["max_layers"]
    max_total_nm = problem["max_total_nm"]
    max_layer_nm = problem["max_layer_nm"]
    
    # Wavelengths for evaluation
    # 401 equally spaced points
    wavelengths = np.linspace(band[0], band[1], 401)
    
    best_score = float('inf')
    best_materials = []
    best_thicknesses = []
    
    def _valid_solution(materials_list, thicknesses_list):
        if len(materials_list) > max_layers:
            return False
        if len(materials_list) != len(thicknesses_list):
            return False
        if any(t < 0 for t in thicknesses_list):
            return False
        if any(t > max_layer_nm for t in thicknesses_list):
            return False
        if sum(thicknesses_list) > max_total_nm:
            return False
        for m in materials_list:
            if m not in ["L", "H"]:
                return False
        return True

    def _update_best(materials_list, thicknesses_list, score):
        nonlocal best_score, best_materials, best_thicknesses
        if score < best_score:
            best_score = score
            best_materials = list(materials_list)
            best_thicknesses = list(thicknesses_list)

    def _optimize_thicknesses(materials_list, initial_thicknesses=None):
        """
        Optimize thicknesses for a fixed material sequence.
        Uses a multi-start local optimization (gradient-free).
        """
        N = len(materials_list)
        if N == 0:
            return None, float('inf')
        
        n_seq = [nL if m == "L" else nH for m in materials_list]
        
        def objective(thicknesses):
            # Penalties for violating constraints
            if any(t < 0 for t in thicknesses):
                return 1000.0
            if any(t > max_layer_nm for t in thicknesses):
                return 1000.0
            if sum(thicknesses) > max_total_nm:
                return 1000.0
            return _compute_reflectance(wavelengths, n0, ns, n_seq, thicknesses)

        best_t = None
        best_val = float('inf')
        
        # Multi-start strategy
        # 1. Quarter wave stack centered on 600nm
        lam0 = 600.0
        t_qw = [lam0 / (4.0 * n) for n in n_seq]
        starts = [t_qw]
        
        # 2. Quarter wave stack centered on 500nm
        lam0 = 500.0
        t_qw = [lam0 / (4.0 * n) for n in n_seq]
        starts.append(t_qw)
        
        # 3. Quarter wave stack centered on 700nm
        lam0 = 700.0
        t_qw = [lam0 / (4.0 * n) for n in n_seq]
        starts.append(t_qw)
        
        # 4. Random starts
        import random
        random.seed(42)
        for _ in range(5):
            t_rand = [random.uniform(0.1, max_layer_nm) for _ in range(N)]
            # Scale to fit total thickness constraint if necessary
            total = sum(t_rand)
            if total > max_total_nm:
                t_rand = [t * (max_total_nm / total) for t in t_rand]
            starts.append(t_rand)

        if initial_thicknesses:
            starts.append(initial_thicknesses)

        # Simple coordinate descent
        step_size = 5.0
        shrink = 0.9
        max_iter = 50
        
        for start_t in starts:
            if time.time() > deadline - 10.0:
                break
                
            t_current = np.array(start_t, dtype=float)
            val_current = objective(t_current)
            
            if val_current < best_val:
                best_val = val_current
                best_t = t_current.copy()
            
            # Coordinate descent
            step = step_size
            for _ in range(max_iter):
                improved = False
                for i in range(N):
                    # Try positive step
                    t_trial = t_current.copy()
                    t_trial[i] += step
                    val_trial = objective(t_trial)
                    if val_trial < val_current:
                        t_current = t_trial
                        val_current = val_trial
                        improved = True
                        continue
                    # Try negative step
                    t_trial = t_current.copy()
                    t_trial[i] -= step
                    val_trial = objective(t_trial)
                    if val_trial < val_current:
                        t_current = t_trial
                        val_current = val_trial
                        improved = True
                        continue
                    # Try half step positive
                    t_trial = t_current.copy()
                    t_trial[i] += step * 0.5
                    val_trial = objective(t_trial)
                    if val_trial < val_current:
                        t_current = t_trial
                        val_current = val_trial
                        improved = True
                        continue
                    # Try half step negative
                    t_trial = t_current.copy()
                    t_trial[i] -= step * 0.5
                    val_trial = objective(t_trial)
                    if val_trial < val_current:
                        t_current = t_trial
                        val_current = val_trial
                        improved = True
                        continue
                    
                if not improved:
                    step *= shrink
                    if step < 0.1:
                        break
                
                if val_current < best_val:
                    best_val = val_current
                    best_t = t_current.copy()

        return best_t, best_val

    # Generate candidate sequences
    # Focus on alternating and near-alternating patterns
    # Max layers to consider for optimization is limited by time
    # Let's try up to 10 layers
    max_layers_search = 10
    
    # Generate sequences
    sequences = []
    
    # 1. Pure alternating starting with L
    for N in range(1, max_layers_search + 1):
        seq = ["L" if i % 2 == 0 else "H" for i in range(N)]
        sequences.append(seq)
        
    # 2. Pure alternating starting with H
    for N in range(1, max_layers_search + 1):
        seq = ["H" if i % 2 == 0 else "L" for i in range(N)]
        sequences.append(seq)
        
    # 3. Some specific known good patterns
    # L H L H L H L H L H
    # H L H L H L H L H L
    # Maybe some defects?
    # For now, stick to alternating as they are robust
    # Add some random sequences for diversity
    import random
    random.seed(123)
    for N in range(3, 8):
        for _ in range(2):
            seq = [random.choice(["L", "H"]) for _ in range(N)]
            # Avoid identical neighbors if possible? Not strictly necessary
            sequences.append(seq)

    # Deduplicate sequences
    unique_seqs = []
    seen = set()
    for seq in sequences:
        key = tuple(seq)
        if key not in seen:
            seen.add(key)
            unique_seqs.append(seq)

    # Sort by length to start with simpler cases? Or maybe longer are better?
    # Let's just iterate
    for seq in unique_seqs:
        if time.time() > deadline - 10.0:
            break
        
        # Optimize thicknesses for this sequence
        t_opt, val_opt = _optimize_thicknesses(seq)
        
        if t_opt is not None:
            if val_opt < best_score:
                # Verify constraints one last time
                if _valid_solution(seq, t_opt):
                    best_score = val_opt
                    best_materials = list(seq)
                    best_thicknesses = [float(t) for t in t_opt]

    # If no solution found, use a simple fallback
    if not best_materials:
        # Single L layer at 600nm
        lam0 = 600.0
        t0 = lam0 / (4.0 * nL)
        if t0 <= max_layer_nm and t0 <= max_total_nm:
            best_materials = ["L"]
            best_thicknesses = [t0]
            best_score = _compute_reflectance(wavelengths, n0, ns, [nL], [t0])
        else:
            best_materials = ["L"]
            best_thicknesses = [1.0]
            best_score = _compute_reflectance(wavelengths, n0, ns, [nL], [1.0])
            
    return best_materials, best_thicknesses
