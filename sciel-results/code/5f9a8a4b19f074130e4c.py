import numpy as np
import time

def design(problem):
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    start_time = time.time()
    deadline = start_time + 115.0  # Leave margin for evaluator overhead
    
    # 1. Build a fast, accurate band solver
    # We use the same TM solver formulation as before, which is efficient.
    # Operator: A = Eps^{-1} D
    # Eps is circulant (convolution with eps). D is diagonal (k+G)^2.
    # A v = F^H diag(1/eps_F) F D v
    
    half_pw = 8
    G_vecs = []
    for m in range(-half_pw, half_pw + 1):
        for n in range(-half_pw, half_pw + 1):
            G_vecs.append((m, n))
    G_vecs = np.array(G_vecs) * (2.0 * np.pi)
    nG = len(G_vecs)
    
    # K-points: 30 points covering the irreducible BZ (1/8 wedge).
    # We use a dense grid to approximate the exact 30 points or ensure we capture the min/max.
    # The evaluator uses 30 points. A 15x15 grid in the triangle gives 120 points, which is a superset.
    # This is safer for finding the true min omega_5 and max omega_4.
    k_points = []
    n_k = 15
    for i in range(n_k + 1):
        for j in range(n_k + 1):
            if i + j <= n_k:
                s = i / n_k
                t = j / n_k
                # k = (s*pi, t*pi) covers the triangle (0,0)-(pi,0)-(pi,pi)
                k_points.append((s * np.pi, t * np.pi))
    k_points = np.array(k_points)

    def compute_score(cell):
        """Compute the 4-5 TM band gap score for a given cell."""
        eps_grid = eps_low + (eps_high - eps_low) * np.array(cell, dtype=float)
        eps_F = np.fft.fft2(eps_grid) / (size * size)
        
        # Handle 1/eps_F. 
        # If eps_F is zero, this is a problem. But eps is between 1 and 12.
        inv_eps_F = 1.0 / eps_F
        
        min_omega5 = np.inf
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            kG = np.stack([kx + G_vecs[:, 0], ky + G_vecs[:, 1]], axis=1)
            kG2 = kG[:, 0]**2 + kG[:, 1]**2
            
            def matvec(v):
                # 1. w = D v
                w = kG2 * v
                # 2. u = F w
                u = np.fft.fft2(w.reshape(17, 17))
                # 3. z = (1/eps_F) * u
                z = inv_eps_F * u
                # 4. y = F^H z
                y = np.fft.ifft2(z)
                return y.flatten().real
            
            try:
                from scipy.sparse.linalg import eigsh
                from scipy.sparse import LinearOperator
                A_op = LinearOperator((nG, nG), matvec=matvec, dtype=complex)
                # We want the 6 smallest eigenvalues.
                vals, vecs = eigsh(A_op, k=6, which='SA', tol=1e-6)
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

    # 2. Parametric Search: "Split-Ring" Design
    # Strategy: A central dielectric superellipse surrounded by an air gap (annulus).
    # This structure is known to enhance higher-order harmonics.
    # Parameters:
    # - r_in: inner radius of the air gap (distance from center to start of dielectric)
    # - r_out: outer radius of the dielectric
    # - p: superellipse exponent (2=circle, 4=squircle, high=square)
    # - a, b: semi-axes of the outer superellipse (if p!=2, we can stretch it)
    # Actually, let's keep it simple:
    # - r_core: radius of central dielectric (if r_core > 0)
    # - r_gap: radius of air gap
    # - r_outer: outer radius of dielectric ring
    # - p: exponent
    
    # Let's define a more flexible shape:
    # A superellipse ring.
    # |x/a|^p + |y/b|^p is between R_in and R_out.
    # Due to symmetry, we define in first quadrant.
    
    def generate_cell(params):
        # params: [r_in, r_out, p, stretch]
        # r_in, r_out in [0, 0.5]
        # p in [1, 10]
        # stretch in [0.5, 1.5] (b = a * stretch)
        
        r_in, r_out, p, stretch = params
        
        cell = np.zeros((size, size), dtype=int)
        
        # Center is (16, 16) in pixel indices (0-indexed: 15.5, 15.5)
        # We iterate over pixels in the first quadrant (i, j) where i, j in [0, 15]
        # and replicate to other quadrants.
        
        # To ensure symmetry, we define the condition based on distance from center (16,16)
        # But superellipse is not radial.
        # Let's use coordinates relative to center (16, 16).
        # x = j - 15.5, y = i - 15.5
        
        # We need to be careful with the definition of the superellipse.
        # Standard: |x/a|^p + |y/b|^p <= 1
        # Here we want: r_in <= |x/a|^p + |y/b|^p <= r_out ? No, that's not right.
        # We want a ring.
        # Let's define the "radius" in superellipse metric: rho = (|x/a|^p + |y/b|^p)^(1/p)
        # We want: R_in <= rho <= R_out.
        
        # To make it efficient, we can precompute the grid coordinates.
        # But since we call this many times, let's optimize.
        
        # Let's use a vectorized approach for the first quadrant.
        # i in 0..15, j in 0..15
        # x = j - 15.5, y = i - 15.5
        
        # Actually, simple loop is fine for 32x32 = 1024 pixels.
        
        for i in range(size):
            for j in range(size):
                x = (j + 0.5) - size/2.0
                y = (i + 0.5) - size/2.0
                
                # a and b
                # Let's normalize a=1, b=stretch? Or a=r_out, b=r_out*stretch?
                # Let's use a = 1.0, b = stretch.
                # Then rho = (|x|^p + |y/stretch|^p)^(1/p)
                # We want R_in <= rho <= R_out.
                
                # However, if we scale x and y, the "radius" units change.
                # Let's just use:
                # val = (abs(x)**p + abs(y/stretch)**p)
                # We want: val_in <= val <= val_out
                # where val_in = R_in**p, val_out = R_out**p.
                
                # But R_in and R_out are in "pixel units" roughly.
                # Let's map r_in, r_out to pixel distances.
                # If p=2, rho = sqrt(x^2 + y^2/stretch^2).
                # If stretch=1, it's a circle.
                
                # Let's define:
                # a = 1.0
                # b = stretch
                # val = abs(x/a)**p + abs(y/b)**p
                # We want: r_in**p <= val <= r_out**p ?
                # This implies that for p=2, stretch=1, it's a circle of radius r.
                # val = x^2 + y^2. r_in^2 <= x^2+y^2 <= r_out^2.
                # This works.
                
                val = abs(x)**p + abs(y/stretch)**p
                
                if val >= r_in**p and val <= r_out**p:
                    cell[i, j] = 1
        
        return cell.tolist()

    # Optimization: Deterministic Nested Coordinate Descent
    # We start from a good initial guess and refine.
    
    # Initial guess:
    # A ring is good for 4-5 gap?
    # The 4-rod design is best for 4-5 gap usually?
    # Wait, the prompt says "inspiration 1... four-rods... score 5.78".
    # The "split-ring" is a different family.
    # Let's try to optimize a ring.
    
    # Parameters: r_in, r_out, p, stretch
    # r_in in [0.0, 0.4]
    # r_out in [0.1, 0.5]
    # p in [1.0, 8.0]
    # stretch in [0.5, 1.5]
    
    # Initial:
    # Let's try a thin ring.
    init_params = [0.2, 0.3, 2.0, 1.0]
    
    best_score = -1.0
    best_cell = None
    best_params = init_params
    
    def evaluate(params):
        nonlocal best_score, best_cell, best_params
        cell = generate_cell(params)
        score = compute_score(cell)
        if score > best_score:
            best_score = score
            best_cell = cell
            best_params = params[:]
        return score

    # Phase 1: Coarse Grid Search
    # r_in: 0.0, 0.1, 0.2, 0.3
    # r_out: 0.2, 0.3, 0.4, 0.5
    # p: 1.0, 2.0, 3.0, 4.0, 6.0
    # stretch: 0.8, 1.0, 1.2
    
    r_in_vals = [0.0, 0.1, 0.2, 0.3]
    r_out_vals = [0.2, 0.3, 0.4, 0.5]
    p_vals = [1.0, 2.0, 3.0, 4.0, 6.0]
    stretch_vals = [0.8, 1.0, 1.2]
    
    for r_in in r_in_vals:
        for r_out in r_out_vals:
            if r_out <= r_in: continue
            for p in p_vals:
                for stretch in stretch_vals:
                    if time.time() > deadline - 30:
                        break
                    params = [r_in, r_out, p, stretch]
                    evaluate(params)
        
        if time.time() > deadline - 30: break

    if time.time() > deadline - 30:
        if best_cell is None:
             best_cell = generate_cell([0.2, 0.3, 2.0, 1.0])
        return best_cell

    # Phase 2: Local Refinement (Coordinate Descent)
    # Start from best_params
    current_params = best_params[:]
    
    # Define steps
    steps = [0.05, 0.02, 0.01]
    
    for step in steps:
        for _ in range(10): # Iterate a few times per step size
            if time.time() > deadline - 10:
                break
            
            improved = False
            # Try changing each parameter
            for idx in range(4):
                # Try + and -
                for delta in [step, -step]:
                    new_params = current_params[:]
                    new_params[idx] += delta
                    
                    # Clamp
                    new_params[0] = np.clip(new_params[0], 0.0, 0.4) # r_in
                    new_params[1] = np.clip(new_params[1], 0.1, 0.5) # r_out
                    new_params[2] = np.clip(new_params[2], 1.0, 8.0) # p
                    new_params[3] = np.clip(new_params[3], 0.5, 1.5) # stretch
                    
                    # Ensure r_out > r_in
                    if new_params[1] <= new_params[0]:
                        continue
                        
                    score = evaluate(new_params)
                    if score > best_score:
                        current_params = new_params
                        improved = True
            
            if not improved:
                break

    # Final check
    if best_cell is None:
        best_cell = generate_cell([0.2, 0.3, 2.0, 1.0])
        
    return best_cell
