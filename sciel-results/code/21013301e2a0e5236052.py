import numpy as np
import time
import math
import random

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 195.0  # Leave 5s margin for safety against 200s limit

    # 1. Setup Fast Band Solver
    # We use 15x15 = 225 plane waves for the internal search loop.
    half_pw = 7 
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs)
    nG = len(G_vecs)
    
    # Precompute k-points.
    # The evaluator uses 30 points. We use a representative subset for speed, 
    # prioritizing the high-symmetry lines and the Gamma point.
    k_points = []
    n_k = 10
    # Line Gamma-X (0,0) to (pi, 0)
    for i in range(n_k + 1):
        s = i / n_k
        k_points.append((s * np.pi, 0))
    # Line X-M (pi,0) to (pi,pi)
    for i in range(1, n_k + 1):
        s = i / n_k
        k_points.append((np.pi, s * np.pi))
    # Line Gamma-M (0,0) to (pi/2, pi/2)
    for i in range(1, n_k + 1):
        s = i / n_k
        k_points.append((s * np.pi/2, s * np.pi/2))
    k_points = np.array(k_points)
    
    # Precompute kG for all k-points to avoid repeated addition
    # kG shape: (n_k, nG, 2)
    kG_all = np.zeros((len(k_points), nG, 2))
    for idx, (kx, ky) in enumerate(k_points):
        kG_all[idx, :, 0] = kx + G_vecs[:, 0] * (2.0 * np.pi)
        kG_all[idx, :, 1] = ky + G_vecs[:, 1] * (2.0 * np.pi)
    kG2_all = np.sum(kG_all**2, axis=2) # (n_k, nG)

    def compute_score(cell):
        """
        Compute the 4-5 TM band gap score for a given cell (32x32 array of 0/1).
        """
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        
        # FFT of epsilon
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for idx in range(len(k_points)):
            kG2 = kG2_all[idx]
            
            def matvec(v):
                w = kG2 * v
                # The operator is (k+G)^2 * [1/eps] * (k+G)^2.
                # [1/eps] acts as a convolution in G-space.
                u = np.fft.fft2(w.reshape(15, 15))
                z = inv_eps_F * u
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-4)
                vals = np.sort(vals.real)
                
                if len(vals) < 5:
                    return -1.0
                
                omega4 = np.sqrt(max(0, vals[3]))
                omega5 = np.sqrt(max(0, vals[4]))
                
                if omega5 < min_omega5:
                    min_omega5 = omega5
                if omega4 > max_omega4:
                    max_omega4 = omega4
                    
            except Exception:
                return -1.0
        
        if min_omega5 == np.inf or max_omega4 == 0.0:
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    def enforce_full_symmetry(c):
        """
        Enforce C4v symmetry on 32x32 array.
        Fundamental domain: 0 <= i <= 15, 0 <= j <= 15, i <= j.
        """
        c_sym = np.zeros_like(c)
        for i in range(16):
            for j in range(i, 16):
                val = c[i, j]
                # 1. (i, j)
                c_sym[i, j] = val
                # 2. (j, i)
                c_sym[j, i] = val
                # 3. (31-i, 31-j)
                c_sym[31-i, 31-j] = val
                # 4. (31-j, 31-i)
                c_sym[31-j, 31-i] = val
                # 5. (i, 31-j)
                c_sym[i, 31-j] = val
                # 6. (31-j, i)
                c_sym[31-j, i] = val
                # 7. (31-i, j)
                c_sym[31-i, j] = val
                # 8. (j, 31-i)
                c_sym[j, 31-i] = val
        return c_sym

    def generate_moire_cell(f1, f2, angle1, angle2, phase1, phase2, threshold):
        """
        Generate cell using interference of two counter-rotating square-wave gratings.
        f1, f2: spatial frequencies (cycles per unit cell)
        angle1, angle2: rotation angles (radians)
        phase1, phase2: phase shifts
        threshold: threshold for binarization
        """
        x, y = np.meshgrid(np.arange(size), np.arange(size))
        x = x / size
        y = y / size
        
        # Rotate coordinates
        cos_a1 = np.cos(angle1)
        sin_a1 = np.sin(angle1)
        x_rot1 = x * cos_a1 - y * sin_a1
        y_rot1 = x * sin_a1 + y * cos_a1
        
        cos_a2 = np.cos(angle2)
        sin_a2 = np.sin(angle2)
        x_rot2 = x * cos_a2 - y * sin_a2
        y_rot2 = x * sin_a2 + y * cos_a2
        
        # Square wave approximation using sign(sin)
        # We use a sum of sine waves to approximate a square wave more accurately
        # or just use sign(sin) for simplicity and speed.
        # sign(sin) creates a 50% duty cycle square wave.
        
        val1 = np.sign(np.sin(2 * np.pi * f1 * (x_rot1 + y_rot1) + phase1))
        val2 = np.sign(np.sin(2 * np.pi * f2 * (x_rot2 + y_rot2) + phase2))
        
        val = val1 + val2
        
        # Normalize val to [-2, 2] and threshold
        # If val > threshold, it's dielectric (1), else air (0)
        # We want to control the fill factor. 
        # A threshold of 0 gives 50% fill factor on average if phases are random.
        # We can adjust threshold to change fill factor.
        
        cell = (val > threshold).astype(int)
        return enforce_full_symmetry(cell)

    def evaluate_params(params):
        """
        Evaluate the score for a given set of Moire parameters.
        params: (f1, f2, angle1, angle2, phase1, phase2, threshold)
        """
        f1, f2, angle1, angle2, phase1, phase2, threshold = params
        cell = generate_moire_cell(f1, f2, angle1, angle2, phase1, phase2, threshold)
        return compute_score(cell)

    # 2. Search Strategy
    # We will perform a coarse grid search followed by a local refinement.
    # The parameter space is 7-dimensional.
    
    best_score = -np.inf
    best_params = None
    
    # Coarse Grid Search
    # f1, f2: [1.0, 4.0]
    # angle1, angle2: [0, pi/2] (symmetry allows restricting)
    # phase1, phase2: [0, pi]
    # threshold: [-2, 2]
    
    # To save time, we fix some parameters or use a smaller grid.
    # Let's fix angle1 = 0 and angle2 = pi/4 (counter-rotating)
    # This reduces dimensions to 5: f1, f2, phase1, phase2, threshold
    
    f1_range = np.linspace(1.0, 4.0, 5)
    f2_range = np.linspace(1.0, 4.0, 5)
    phase1_range = np.linspace(0, np.pi, 3)
    phase2_range = np.linspace(0, np.pi, 3)
    threshold_range = np.linspace(-1.0, 1.0, 5)
    
    count = 0
    total = len(f1_range) * len(f2_range) * len(phase1_range) * len(phase2_range) * len(threshold_range)
    
    for f1 in f1_range:
        for f2 in f2_range:
            for p1 in phase1_range:
                for p2 in phase2_range:
                    for th in threshold_range:
                        count += 1
                        if time.time() > deadline - 30.0:
                            break
                        
                        params = (f1, f2, 0.0, np.pi/4, p1, p2, th)
                        score = evaluate_params(params)
                        
                        if score > best_score:
                            best_score = score
                            best_params = params
                        
                        if count % 20 == 0 and time.time() > deadline - 20.0:
                            break
                    if time.time() > deadline - 20.0:
                        break
                if time.time() > deadline - 20.0:
                    break
            if time.time() > deadline - 20.0:
                break
        if time.time() > deadline - 20.0:
            break
    
    if best_params is None:
        # Fallback if grid search failed or timed out immediately
        best_params = (2.0, 2.5, 0.0, np.pi/4, 0.0, 0.0, 0.0)
        best_score = evaluate_params(best_params)
    
    # Local Refinement (Nelder-Mead style or simple coordinate descent)
    # We'll use a simple coordinate descent with adaptive step size.
    
    if time.time() < deadline - 10.0 and best_score > -0.5:
        current_params = list(best_params)
        current_score = best_score
        step_sizes = [0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1]
        
        for _ in range(100): # Max iterations
            if time.time() > deadline - 10.0:
                break
            
            improved = False
            for i in range(7):
                for direction in [-1, 1]:
                    new_params = current_params.copy()
                    new_params[i] += direction * step_sizes[i]
                    
                    # Bounds checking
                    if i in [0, 1]: # Frequencies
                        if new_params[i] < 0.5 or new_params[i] > 5.0:
                            continue
                    elif i in [2, 3]: # Angles
                        if new_params[i] < 0 or new_params[i] > np.pi/2:
                            continue
                    elif i in [4, 5]: # Phases
                        new_params[i] = new_params[i] % np.pi
                    elif i == 6: # Threshold
                        if new_params[i] < -2.0 or new_params[i] > 2.0:
                            continue
                    
                    new_score = evaluate_params(new_params)
                    
                    if new_score > current_score:
                        current_params = new_params
                        current_score = new_score
                        if current_score > best_score:
                            best_score = current_score
                            best_params = current_params.copy()
                        improved = True
                        break # Restart coordinate search from new point
                    else:
                        # Reduce step size if no improvement
                        step_sizes[i] *= 0.9
                        
            if not improved:
                # If no coordinate improved, reduce all steps
                for i in range(7):
                    step_sizes[i] *= 0.8
                if max(step_sizes) < 0.01:
                    break

    # 3. Final Pixel Refinement
    # Use the best Moire cell as a starting point for a short, targeted pixel hill-climb.
    # This allows correcting for any symmetry-breaking artifacts or fine-tuning edges.
    
    final_cell = generate_moire_cell(*best_params)
    
    if time.time() < deadline - 5.0:
        T = 0.05
        T_min = 0.001
        cooling = 0.999
        steps = 0
        max_steps = 2000 # Shorter than before, as we start from a good point
        
        current_score = best_score
        cell = final_cell.copy()
        
        while time.time() < deadline - 2.0 and steps < max_steps:
            steps += 1
            
            # Pick a random pixel
            i = random.randint(0, size - 1)
            j = random.randint(0, size - 1)
            
            # Flip it
            new_cell = cell.copy()
            new_cell[i, j] = 1 - new_cell[i, j]
            
            # Enforce symmetry
            new_cell = enforce_full_symmetry(new_cell)
            
            # If no change, skip
            if np.array_equal(new_cell, cell):
                continue
                
            # Calculate score
            new_score = compute_score(new_cell)
            
            # Acceptance criterion
            if new_score > best_score:
                best_score = new_score
                best_params = None # Invalidate params, we are in pixel space now
                final_cell = new_cell.copy()
                cell = new_cell
                current_score = new_score
            else:
                delta = new_score - current_score
                if delta < 0:
                    p = math.exp(delta / T)
                    if random.random() < p:
                        cell = new_cell
                        current_score = new_score
                        
            T = max(T_min, T * cooling)
            
            # Check time frequently
            if steps % 50 == 0 and time.time() > deadline - 5.0:
                break

    # Ensure the returned cell is a list of lists of ints
    return final_cell.tolist()
