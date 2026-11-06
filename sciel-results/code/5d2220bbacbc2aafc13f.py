"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import math
import time
import numpy as np

def design(problem):
    """
    Design a 32x32 unit cell to maximize the 4-5 TM band gap.
    Uses a parametric shape (square with rounded corners / dimples) optimized
    via local search using a fast PWE solver as a surrogate.
    """
    start_time = time.time()
    deadline = start_time + 210.0  # Safety margin before 230s kill

    size = problem.get("pixels", 32)
    eps_low = problem.get("eps_low", 1.0)
    eps_high = problem.get("eps_high", 12.0)

    # Precompute 1/epsilon grid for the two materials
    inv_eps_air = 1.0 / eps_low
    inv_eps_die = 1.0 / eps_high

    def make_cell(params):
        """
        Generate a 32x32 cell from parameters.
        params: (radius, corner_cut, dimple_depth)
        We use a square rod centered at (0.25, 0.25) with:
        - radius: half-width of the square
        - corner_cut: how much to cut the corners (0=sharp, high=round)
        - dimple_depth: depth of dimples on the sides (0=flat)
        
        Actually, let's use a more flexible parametric shape:
        A central square with rounded corners and side dimples.
        
        Parameters:
        r: half-size of the square base
        c: corner rounding factor (0 to 1)
        d: dimple depth (0 to r)
        """
        r, c, d = params
        cell = np.zeros((size, size), dtype=int)
        
        # Pixel centers
        # i corresponds to y, j to x
        # y in [-0.5, 0.5], x in [-0.5, 0.5]
        # Pixel (i,j) center: y_i = (i+0.5)/size - 0.5, x_j = (j+0.5)/size - 0.5
        
        # We place 4 identical shapes at (±0.25, ±0.25)
        # Due to symmetry, we can just define one in the (0.25, 0.25) quadrant
        # and replicate. But for speed, let's just iterate all pixels and check
        # distance to the 4 centers.
        
        centers = [
            (-0.25, -0.25),
            ( 0.25, -0.25),
            (-0.25,  0.25),
            ( 0.25,  0.25)
        ]
        
        # For each pixel, check if it's inside any of the 4 shapes
        # Shape: square with rounded corners and dimples
        # Let's define the shape function f(dx, dy) where dx, dy are coords relative to center
        # |dx| < r and |dy| < r is the square.
        # Rounded corners: cut off the square corners.
        # Dimples: indent the sides.
        
        # A simpler parametric shape that allows for "dimpled" sides:
        # Use a superellipse-like equation or just modify the boundary.
        
        # Let's try:
        # Inside if |dx| < r - d * sin^2(pi*dy/r) and |dy| < r - d * sin^2(pi*dx/r)
        # This creates dimples on the sides.
        
        # And rounded corners:
        # If |dx| > r - c*r and |dy| > r - c*r, check distance to corner center.
        
        # This might be too complex. Let's stick to a simpler "rounded square" first.
        # f(dx, dy) = (|dx|/r)^(2+c) + (|dy|/r)^(2+c) <= 1
        # c=0 is diamond (L1), c->inf is square (Linf).
        # Wait, superellipse x^(2/n) + y^(2/n) = 1.
        # n=2 is circle, n=4 is squircle.
        
        # Let's use: (|dx|/r)^p + (|dy|/r)^p <= 1
        # p=2: circle. p=4: squircle. p=10: very square.
        # We want a square-ish shape, so p > 4.
        # And we want dimples? Superellipses don't have dimples.
        
        # Let's use a different parametric form for dimples.
        # r(theta) = r0 * (1 - d * cos(4*theta))
        # This creates 4 dimples (at theta=0, pi/2, etc) or 4 bulges.
        # If d > 0, it's dimpled at the axes.
        
        # Polar coordinates relative to center (cx, cy):
        # dx = x - cx, dy = y - cy
        # theta = atan2(dy, dx)
        # r_pixel = sqrt(dx^2 + dy^2)
        # r_shape(theta) = r0 * (1 - d * cos(4*theta))
        # Inside if r_pixel < r_shape(theta)
        
        # This creates a 4-fold symmetric shape.
        # r0 controls size.
        # d controls dimple depth (d>0 is dimples at 0,90,180,270 deg).
        
        # Let's try this.
        # params: (r0, d)
        # r0 in [0.1, 0.3]
        # d in [0, 0.5]
        
        # But we also want to adjust the "squareness".
        # Maybe combine superellipse and dimples?
        # Let's just use the polar dimpled shape for now. It's simple and fast.
        
        r0, d = params
        
        # Precompute for speed
        # We can vectorize this with NumPy.
        
        xs = np.arange(size)
        ys = np.arange(size)
        X, Y = np.meshgrid(xs, ys)
        
        # Pixel centers
        px = (X + 0.5) / size - 0.5
        py = (Y + 0.5) / size - 0.5
        
        # For each center, compute if inside
        inside = np.zeros((size, size), dtype=bool)
        
        for cx, cy in centers:
            dx = px - cx
            dy = py - cy
            r = np.sqrt(dx**2 + dy**2)
            theta = np.arctan2(dy, dx)
            r_shape = r0 * (1 - d * np.cos(4 * theta))
            inside |= (r < r_shape)
        
        cell = inside.astype(int)
        
        # Enforce symmetry (should be symmetric by construction, but just in case)
        # The polar shape is symmetric under 90 deg rotation and reflection.
        # The centers are symmetric.
        # So it should be fine.
        
        return cell

    def compute_fourier(cell):
        """
        Compute Fourier coefficients of 1/epsilon.
        Returns dict: (nx, ny) -> coeff
        """
        eps_grid = np.where(cell == 1, eps_high, eps_low)
        inv_eps = 1.0 / eps_grid
        
        # 289 plane waves: |nx|, |ny| <= 11 (since 2*11+1 = 23, 23^2 = 529 > 289? No)
        # 289 = 17^2. So |nx|, |ny| <= 8? 2*8+1 = 17. 17*17 = 289.
        # Yes, G vectors with |nx| <= 8, |ny| <= 8.
        
        N = size
        g_max = 8
        fourier = {}
        
        # Use FFT for speed
        # F[nx, ny] = sum_{i,j} f(i,j) exp(-i (nx*2pi*i/N + ny*2pi*j/N))
        # This is the DFT.
        # F = fft2(f)
        # But we need the coefficients for specific (nx, ny) in the range [-8, 8].
        
        F = np.fft.fft2(inv_eps)
        
        # F[u, v] corresponds to frequency (u, v) in [0, N-1].
        # Negative frequencies are at N-u.
        
        for nx in range(-g_max, g_max + 1):
            for ny in range(-g_max, g_max + 1):
                # Map to FFT index
                u = nx % N
                v = ny % N
                fourier[(nx, ny)] = F[u, v] / (N * N)
                
        return fourier

    def compute_band_gap(cell):
        """
        Compute the 4-5 TM band gap ratio.
        Uses PWE method.
        """
        fourier = compute_fourier(cell)
        
        # G vectors
        g_max = 8
        G = []
        for nx in range(-g_max, g_max + 1):
            for ny in range(-g_max, g_max + 1):
                G.append((nx, ny))
        G = np.array(G)
        num_G = len(G)
        
        # K points in irreducible BZ
        # Square lattice BZ: [-pi, pi] x [-pi, pi] (if lattice constant a=1, k in units of 2pi/a)
        # Irreducible zone: 0 <= kx <= ky <= pi (or similar)
        # Let's use 30 points as specified in the task?
        # "30 wave vectors covering the boundary and interior"
        
        # We'll generate a grid of K points in the irreducible triangle.
        # kx in [0, pi], ky in [kx, pi]
        
        # Let's pick 30 points.
        # Maybe a 6x5 grid?
        kx_vals = np.linspace(0, math.pi, 6)
        ky_vals = np.linspace(0, math.pi, 6)
        
        k_points = []
        for kx in kx_vals:
            for ky in kx_vals:
                if ky >= kx:
                    k_points.append((kx, ky))
        
        # This gives 21 points. Let's add some boundary points.
        # Or just use a denser grid.
        # Let's use 30 points.
        # 5x5 grid in the triangle?
        
        # Actually, the task says "30 wave vectors".
        # Let's just use a fixed set of 30 points.
        # I'll generate them on the fly.
        
        # For speed, I'll use fewer points in the surrogate.
        # Let's use 10 points.
        kx_vals = np.linspace(0, math.pi, 4)
        ky_vals = np.linspace(0, math.pi, 4)
        k_points = []
        for kx in kx_vals:
            for ky in kx_vals:
                if ky >= kx:
                    k_points.append((kx, ky))
        
        # 10 points.
        
        min_omega_5 = float('inf')
        max_omega_4 = 0.0
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build matrix A
            # A_ij = (k + G_i) dot (k + G_j) * fourier[G_i - G_j]
            # Note: fourier index is (G_i - G_j)
            
            # k + G_i
            kG = k + G # shape (num_G, 2)
            
            # Dot products
            # D_ij = (k+G_i) dot (k+G_j)
            # D = kG @ kG.T
            
            D = np.einsum('ij,ij->i', kG, kG) # No, this is norm squared.
            # We need outer product.
            D = kG @ kG.T
            
            # Four matrix
            # F_ij = fourier[G_i - G_j]
            F_mat = np.zeros((num_G, num_G), dtype=complex)
            
            for i in range(num_G):
                for j in range(num_G):
                    diff = (G[i][0] - G[j][0], G[i][1] - G[j][1])
                    F_mat[i, j] = fourier.get(diff, 0.0)
            
            # A = D * F_mat
            A = D * F_mat
            
            # Solve eigenvalue problem
            # A v = omega^2 v
            # Use eigh for Hermitian matrix
            # A is Hermitian?
            # D is real symmetric.
            # F is Hermitian (F_ij = conj(F_ji)).
            # So A is Hermitian.
            
            try:
                eigvals, eigvecs = np.linalg.eigh(A)
            except np.linalg.LinAlgError:
                continue
            
            # eigvals are omega^2
            # We want bands 4 and 5 (1-indexed)
            # So indices 3 and 4 (0-indexed)
            
            if len(eigvals) < 5:
                continue
            
            omega_4 = math.sqrt(eigvals[3])
            omega_5 = math.sqrt(eigvals[4])
            
            max_omega_4 = max(max_omega_4, omega_4)
            min_omega_5 = min(min_omega_5, omega_5)
        
        if min_omega_5 == float('inf'):
            return 0.0
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap == 0:
            return 0.0
        
        return gap / midgap

    # Optimization
    # Start with a good initial guess
    # r0 = 0.2, d = 0.2
    
    best_params = (0.2, 0.2)
    best_score = compute_band_gap(make_cell(best_params))
    
    # Local search
    # Gradient-free: Nelder-Mead or simple coordinate descent
    
    # Let's use a simple random restart + local refinement
    
    num_restarts = 5
    steps_per_restart = 20
    
    for restart in range(num_restarts):
        if time.time() > deadline - 30:
            break
            
        # Random initial point
        r0 = 0.15 + 0.1 * np.random.rand()
        d = 0.1 + 0.3 * np.random.rand()
        params = (r0, d)
        
        current_score = compute_band_gap(make_cell(params))
        
        # Local search
        step_size_r = 0.02
        step_size_d = 0.05
        
        for step in range(steps_per_restart):
            if time.time() > deadline - 10:
                break
                
            improved = False
            
            # Try 8 neighbors
            neighbors = [
                (params[0] + step_size_r, params[1]),
                (params[0] - step_size_r, params[1]),
                (params[0], params[1] + step_size_d),
                (params[0], params[1] - step_size_d),
                (params[0] + step_size_r, params[1] + step_size_d),
                (params[0] - step_size_r, params[1] - step_size_d),
                (params[0] + step_size_r, params[1] - step_size_d),
                (params[0] - step_size_r, params[1] + step_size_d),
            ]
            
            best_neighbor = None
            best_neighbor_score = current_score
            
            for n_params in neighbors:
                # Clip params
                nr0 = max(0.1, min(0.3, n_params[0]))
                nd = max(0.0, min(0.5, n_params[1]))
                n_params = (nr0, nd)
                
                s = compute_band_gap(make_cell(n_params))
                if s > best_neighbor_score:
                    best_neighbor_score = s
                    best_neighbor = n_params
            
            if best_neighbor is not None:
                params = best_neighbor
                current_score = best_neighbor_score
                improved = True
            else:
                # Shrink step size
                step_size_r *= 0.7
                step_size_d *= 0.7
                if step_size_r < 0.001:
                    break
        
        if current_score > best_score:
            best_score = current_score
            best_params = params
            
        # Update step sizes for next restart
        # step_size_r *= 0.8
        # step_size_d *= 0.8

    # Final check
    final_cell = make_cell(best_params)
    
    # Ensure it's a list of lists of ints
    final_cell = final_cell.tolist()
    
    return final_cell

# EVOLVE-BLOCK-END
