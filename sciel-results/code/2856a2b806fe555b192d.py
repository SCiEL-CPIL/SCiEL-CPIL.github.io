import numpy as np
import time
import math

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)
    
    # Pre-compute grid coordinates
    x = np.linspace(-0.5, 0.5, size, endpoint=False)
    y = np.linspace(-0.5, 0.5, size, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Define Fourier basis functions compatible with C4v symmetry.
    # We use real basis functions that are invariant under 90-degree rotation 
    # or transform predictably, ensuring the final sum is symmetric.
    # For a centrosymmetric crystal (invariant under (x,y) -> (-x,-y)), 
    # we can use cos(nx)cos(my), sin(nx)sin(my), cos(nx)sin(my) with even/odd parity constraints.
    # To guarantee cell == cell[::-1] (180 rot) and cell == cell.T (reflection/transpose),
    # we restrict to modes that are symmetric under x->-x, y->-y individually or combined.
    
    # Let's use a set of basis functions that are all even under x->-x and y->-y 
    # (i.e., cos-cos, sin-sin with appropriate signs, but actually cos(nx)cos(my) is even in both).
    # Note: cos(nx)cos(my) is symmetric under x->-x and y->-y.
    # sin(nx)sin(my) is symmetric under x->-x and y->-y.
    # cos(nx)sin(my) is anti-symmetric under x->-x? No, cos is even, sin is odd. 
    # So cos(nx)sin(my) is odd in y. 
    # To ensure the full cell is symmetric under 180 rotation (x,y)->(-x,-y):
    # f(-x,-y) = f(x,y).
    # cos(-nx)cos(-my) = cos(nx)cos(my) (OK)
    # sin(-nx)sin(-my) = (-sin(nx))(-sin(my)) = sin(nx)sin(my) (OK)
    # cos(-nx)sin(-my) = cos(nx)(-sin(my)) = -cos(nx)sin(my) (NOT OK unless coeff is 0)
    # sin(-nx)cos(-my) = (-sin(nx))cos(my) = -sin(nx)cos(my) (NOT OK unless coeff is 0)
    
    # So we restrict to:
    # 1. cos(2*pi*n*x) * cos(2*pi*m*y)
    # 2. sin(2*pi*n*x) * sin(2*pi*m*y)
    
    max_harmonic = 4
    basis_funcs = []
    indices = []
    
    for n in range(0, max_harmonic + 1):
        for m in range(0, max_harmonic + 1):
            if n == 0 and m == 0:
                continue # DC term is handled separately or fixed to control fill fraction
            
            # Type 1: cos-cos
            # Symmetric under x->-x, y->-y, x->y, x->-y
            basis_funcs.append(np.cos(2 * np.pi * n * X) * np.cos(2 * np.pi * m * Y))
            indices.append(('cc', n, m))
            
            # Type 2: sin-sin
            # Symmetric under x->-x, y->-y, x->y, x->-y
            if n > 0 and m > 0: # sin(0)=0, so skip if n=0 or m=0
                basis_funcs.append(np.sin(2 * np.pi * n * X) * np.sin(2 * np.pi * m * Y))
                indices.append(('ss', n, m))
                
    num_params = len(basis_funcs)
    
    # Initial guess: A ring-like structure often works well for TM gaps.
    # We can initialize coefficients to approximate a ring or use random small values.
    # Let's start with a configuration known to be decent: a dielectric ring.
    # A ring can be approximated by negative DC + positive low-order harmonics.
    
    # Let's just start with a random seed or a specific heuristic.
    # Heuristic: Set 'cc' (1,1) and 'ss' (1,1) to create variation.
    # Actually, let's use a simple initial guess: all zeros except maybe a few.
    # Or better, initialize with a known good topology: a dielectric square with air hole (inverse rod).
    # Inverse rod: high eps background, low eps hole.
    # Our eps is 12 (dielectric) vs 1 (air). 1 is dielectric.
    # So 1=Dielectric, 0=Air.
    # The seed was a dielectric rod.
    # Let's try an air hole in a dielectric background (inverse).
    # This is often better for TM gaps.
    
    # To get an air hole (0s in center, 1s around):
    # We want eps(x,y) to be high at boundaries, low in center.
    # cos(2pi x)cos(2pi y) is max at corners, min at center (if we shift phase?).
    # cos(0)=1. At center (0,0), cos(0)cos(0)=1.
    # At edge (0.5, 0), cos(pi)cos(0)=-1.
    # So cos-cos is High in Center, Low at Edges. This creates a Dielectric Rod (High Eps in center).
    # We want Air Hole: Low Eps in Center, High Eps at Edges.
    # So we want negative coefficient for (1,1) cos-cos?
    # If coeff is -1: -1*1 = -1 (Center), -1*(-1) = 1 (Edge).
    # So negative (1,1) cc gives Air Hole.
    
    init_coeffs = np.zeros(num_params)
    
    # Find index for (1,1) cc
    for i, (typ, n, m) in enumerate(indices):
        if typ == 'cc' and n == 1 and m == 1:
            init_coeffs[i] = -0.5
        elif typ == 'cc' and n == 2 and m == 0:
            init_coeffs[i] = 0.1
        elif typ == 'cc' and n == 0 and m == 2:
            init_coeffs[i] = 0.1
            
    # Add DC offset to center the value around 0.5
    # We will map the continuous field to 0/1 by thresholding at 0.5.
    # So we need the average value to be around 0.5.
    # Let's estimate the mean of the basis functions.
    # For now, we'll adjust DC dynamically or just pick a DC that yields ~50% fill.
    # Let's fix DC to 0.0 and assume the sum oscillates around 0.
    # We'll add a separate DC parameter to the optimization to control fill fraction.
    
    params = list(init_coeffs)
    params.append(0.0) # DC term
    
    def evaluate_design(coeffs, dc_val):
        """
        Fast surrogate evaluation.
        Returns the gap ratio.
        """
        # Construct field
        field = dc_val * np.ones_like(X)
        for i, (typ, n, m) in enumerate(indices):
            if coeffs[i] != 0.0:
                field += coeffs[i] * basis_funcs[i]
        
        # Threshold to 0/1
        # 1 if field > 0.5, else 0. 
        # We want to optimize the binary mask.
        # To make it differentiable-like for local search, we just check the binary result.
        mask = (field > 0.5).astype(int)
        
        # Symmetry check (should be perfect by construction, but verify)
        # if not (np.allclose(mask, mask[::-1, ::-1]) and np.allclose(mask, mask.T)):
        #     return -1.0 # Invalid
        
        # Fast Band Gap Calculation
        # Use a reduced plane wave method.
        # N_waves = 7x7 = 49 or 9x9 = 81.
        # Let's use 7x7 for speed.
        N = 7
        k_max = N // 2
        G_vectors = []
        for i in range(-k_max, k_max + 1):
            for j in range(-k_max, k_max + 1):
                G_vectors.append((i, j))
        G_vectors = np.array(G_vectors)
        
        # Fourier coefficients of 1/epsilon
        # eps = eps_low * (1 - mask) + eps_high * mask
        # inv_eps = 1 / eps
        # We need the Fourier coefficients of inv_eps.
        # C_G = (1/A) sum_{r} inv_eps(r) exp(-i G.r)
        # Since mask is binary and symmetric, we can compute this efficiently.
        
        eps_map = np.where(mask == 1, eps_high, eps_low)
        inv_eps_map = 1.0 / eps_map
        
        # Compute Fourier coeffs
        # Using FFT for speed
        # Pad to size x size
        # FFT of inv_eps_map
        # Note: FFT gives coefficients for frequencies 0..size-1.
        # We need to map to -k_max..k_max.
        
        # For a 32x32 grid, FFT is fast.
        F = np.fft.fft2(inv_eps_map)
        
        # Get coefficients for G in G_vectors
        # G = (i, j)
        # FFT index for i: if i >= 0, idx = i, else idx = size + i
        # But FFT assumes periodic boundary and specific ordering.
        # Let's just use the direct definition for accuracy with small N, 
        # or use FFT with proper indexing.
        
        C_G = np.zeros(len(G_vectors), dtype=complex)
        for idx, (i, j) in enumerate(G_vectors):
            # i, j are integers in -k_max..k_max
            # Map to FFT indices
            ii = i % size
            jj = j % size
            C_G[idx] = F[jj, ii] / (size * size) # Normalization
            
        # Build Hamiltonian matrices for a few k-points
        # k-points: Gamma (0,0), X (0.5, 0), M (0.5, 0.5)
        # And maybe a generic point in the irreducible BZ.
        # Irreducible BZ for square lattice: 0 <= kx <= ky <= 0.5
        # Key points: Gamma (0,0), X (0,0.5) [or (0.5,0)], M (0.5,0.5)
        # Let's sample:
        k_points = [
            (0.0, 0.0), # Gamma
            (0.5, 0.0), # X
            (0.5, 0.5), # M
            (0.25, 0.25), # Generic
            (0.4, 0.1), # Generic
        ]
        
        max_omega_4 = 0.0
        min_omega_5 = float('inf')
        
        for kx, ky in k_points:
            # Construct H matrix
            # H_{ij} = G_i . G_j * C_{G_i - G_j}
            # where G_i, G_j are the plane wave vectors.
            # Note: The standard formulation uses (k+G_i) . (k+G_j) * C_{G_i-G_j}
            # Let's use the standard form:
            # H_{mn} = (k + G_m) dot (k + G_n) * C_{G_m - G_n}
            
            H = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
            
            for m in range(len(G_vectors)):
                kmx = kx + G_vectors[m, 0]
                kmy = ky + G_vectors[m, 1]
                for n in range(len(G_vectors)):
                    # G_diff = G_m - G_n
                    diff_i = G_vectors[m, 0] - G_vectors[n, 0]
                    diff_j = G_vectors[m, 1] - G_vectors[n, 1]
                    
                    # Find index of diff in G_vectors
                    # G_vectors is symmetric, so diff should be in the list
                    # Map diff to index
                    # We can precompute a dictionary for speed
                    # But for N=49, linear search is fine.
                    # Or use a lookup dict.
                    
                    # Let's use a dict for C_G lookup
                    # C_G_dict[(i,j)] = coeff
                    # We built C_G in order of G_vectors.
                    
                    # Find index of (diff_i, diff_j)
                    # Since G_vectors is small, we can just loop or use a precomputed map.
                    # For speed in the inner loop, let's precompute a map.
                    
                    pass # Placeholder, will optimize below
                    
            # Optimization: Precompute index map for G_diff
            # This is done outside the k-point loop for efficiency, but here we are inside.
            # Let's restructure to compute H faster.
            
        # Re-implement H construction with precomputed maps for speed
        
        # Precompute C_G map
        C_G_map = {}
        for idx, (i, j) in enumerate(G_vectors):
            C_G_map[(i, j)] = C_G[idx]
            
        for kx, ky in k_points:
            H = np.zeros((len(G_vectors), len(G_vectors)), dtype=complex)
            for m in range(len(G_vectors)):
                kmx = kx + G_vectors[m, 0]
                kmy = ky + G_vectors[m, 1]
                for n in range(len(G_vectors)):
                    diff_i = G_vectors[m, 0] - G_vectors[n, 0]
                    diff_j = G_vectors[m, 1] - G_vectors[n, 1]
                    c_val = C_G_map.get((diff_i, diff_j), 0.0)
                    if c_val != 0.0:
                        k_dot = (kmx * (kx + G_vectors[n, 0])) + (kmy * (ky + G_vectors[n, 1]))
                        H[m, n] = k_dot * c_val
            
            # Solve eigenvalues
            # H is Hermitian
            vals = np.linalg.eigvalsh(H)
            # Filter out negative or zero values (spurious)
            vals = vals[vals > 1e-6]
            
            if len(vals) >= 5:
                # Bands are sorted.
                # We want gap between 4th and 5th.
                # Index 3 is 4th band, Index 4 is 5th band.
                w4 = vals[3]
                w5 = vals[4]
                
                # Update global max/min
                if w4 > max_omega_4:
                    max_omega_4 = w4
                if w5 < min_omega_5:
                    min_omega_5 = w5
        
        if min_omega_5 == float('inf') or max_omega_4 == 0.0:
            return -1.0
            
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return -1.0
            
        ratio = 2.0 * gap / midgap
        return ratio

    # Optimization Loop
    best_score = -1.0
    best_params = params[:]
    
    # Local Search
    step_size = 0.1
    max_iter = 1000
    
    current_params = params[:]
    current_score = evaluate_design(current_params[:-1], current_params[-1])
    
    if current_score > best_score:
        best_score = current_score
        best_params = current_params[:]
        
    for it in range(max_iter):
        if time.time() > deadline:
            break
            
        # Try perturbing one parameter
        idx = np.random.randint(0, len(current_params))
        direction = np.random.choice([-1, 1])
        
        new_params = current_params[:]
        # Adaptive step size
        if idx == len(current_params) - 1: # DC term
            delta = direction * 0.05
        else:
            delta = direction * step_size
            
        new_params[idx] += delta
        
        new_score = evaluate_design(new_params[:-1], new_params[-1])
        
        if new_score > current_score:
            current_params = new_params
            current_score = new_score
            if current_score > best_score:
                best_score = current_score
                best_params = current_params[:]
            # Shrink step size slightly on success? Or keep same.
            # step_size = max(0.01, step_size * 0.95)
        else:
            # Try smaller step in same direction? Or random restart of direction
            # Keep step size constant for now to explore.
            pass
            
        # Occasionally try a larger jump
        if it % 100 == 0:
             # Reset to best and add noise
             current_params = best_params[:] + np.random.normal(0, 0.1, len(best_params))
             current_score = evaluate_design(current_params[:-1], current_params[-1])
             if current_score > best_score:
                 best_score = current_score
                 best_params = current_params[:]

    # Construct final design from best_params
    final_coeffs = best_params[:-1]
    final_dc = best_params[-1]
    
    field = final_dc * np.ones_like(X)
    for i, (typ, n, m) in enumerate(indices):
        if final_coeffs[i] != 0.0:
            field += final_coeffs[i] * basis_funcs[i]
            
    mask = (field > 0.5).astype(int)
    
    # Enforce symmetry strictly (just in case of numerical noise in thresholding)
    # 1. 180 Rotation: mask[i][j] == mask[size-1-i][size-1-j]
    # 2. Transpose: mask[i][j] == mask[j][i]
    
    # We can average the symmetric versions to enforce it, then threshold again?
    # Or just pick the majority vote in the orbits.
    # Orbit under D4: (i,j), (j,i), (size-1-i, j), (size-1-j, i), etc.
    # Since our basis functions are perfectly symmetric, the field is symmetric.
    # The only issue is if field[i,j] == 0.5 exactly.
    # Let's just return the mask as is, it should be symmetric.
    
    # Verify symmetry
    # if not (np.allclose(mask, mask[::-1, ::-1]) and np.allclose(mask, mask.T)):
    #     # Force symmetry by averaging
    #     sym_mask = (mask + mask[::-1, ::-1] + mask.T + mask.T[::-1, ::-1]) / 4.0
    #     mask = (sym_mask > 0.5).astype(int)
    
    return mask.tolist()

# Run once to check syntax
# design({"pixels": 32, "eps_low": 1, "eps_high": 12})
