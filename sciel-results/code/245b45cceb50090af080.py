import numpy as np
import time
from scipy.linalg import eigh

def design(problem):
    """
    Design a symmetric 32x32 photonic crystal unit cell.
    Uses a local search with a fast plane-wave solver to optimize
    the 4th-5th TM band gap.
    """
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"]
    eps_high = problem["eps_high"]
    lower_band = problem["lower_band"]
    
    # Generate coordinates centered at 0
    coords = np.arange(size)
    x = (coords + 0.5) / size - 0.5
    y = (coords + 0.5) / size - 0.5
    X, Y = np.meshgrid(x, y)
    
    # Initialize with 4 dielectric rods in air background
    dx = 0.25
    dy = 0.25
    R_rod = 0.12
    
    cell = np.zeros((size, size), dtype=int)
    cell[np.sqrt((X - dx)**2 + (Y - dy)**2) <= R_rod] = 1
    cell[np.sqrt((X + dx)**2 + (Y - dy)**2) <= R_rod] = 1
    cell[np.sqrt((X - dx)**2 + (Y + dy)**2) <= R_rod] = 1
    cell[np.sqrt((X + dx)**2 + (Y + dy)**2) <= R_rod] = 1
    
    # Precompute plane-wave vectors (289 waves = 17x17)
    # Using a cutoff of 8.5 (half-integer) for 17x17
    N = 8
    g_vecs = []
    for m in range(-N, N + 1):
        for n in range(-N, N + 1):
            g_vecs.append((m, n))
    g_vecs = np.array(g_vecs, dtype=float)
    Npw = len(g_vecs)
    
    # Precompute k-points in the irreducible Brillouin zone
    # The IBZ for a square lattice is: 0 <= kx <= ky, kx + ky <= 1 (in units of 2π/a)
    # We'll use 30 points covering the boundary and interior
    k_points = []
    # Along kx=0 (Gamma to M)
    for i in range(10):
        t = i / 9.0
        k_points.append((0.0, t))
    # Along ky=kx (X to M)
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((t, t))
    # Along kx+ky=1 (M to X)
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((t, 1.0 - t))
    
    k_points = np.array(k_points, dtype=float)
    Nk = len(k_points)
    
    def compute_epsilon_fourier(cell):
        """Compute the Fourier coefficients of epsilon for the given cell."""
        # epsilon = eps_low + (eps_high - eps_low) * cell
        eps = 1.0 + (eps_high - 1.0) * cell.astype(float)
        # Compute Fourier coefficients using FFT
        # The Fourier coefficient G_{mn} = (1/N^2) * sum_{i,j} eps(i,j) * exp(-2πi*(m*x_i + n*y_j))
        # Using FFT: G_{mn} = (1/N^2) * FFT(eps)[m mod N, n mod N]
        eps_fft = np.fft.fft2(eps)
        eps_fft /= (size * size)
        return eps_fft
    
    def compute_bands(cell, k_points, eps_fft):
        """Compute the band structure using the plane-wave method."""
        bands = np.zeros((Nk, Npw))
        
        for ik, (kx, ky) in enumerate(k_points):
            # Build the matrix for this k-point
            # H_{G,G'} = (k+G)·(k+G') * epsilon_{G-G'}
            # For TM polarization: H_{G,G'} = (k+G)_x * (k+G')_x * eps_{G-G'} + (k+G)_y * (k+G')_y * eps_{G-G'}
            
            kG = k_points[ik][:1] + g_vecs  # (Npw, 2)
            kxG = kG[:, 0]
            kyG = kG[:, 1]
            
            # Build the matrix
            H = np.zeros((Npw, Npw), dtype=complex)
            
            for i in range(Npw):
                for j in range(Npw):
                    # G_i - G_j
                    dG = g_vecs[i] - g_vecs[j]
                    dm, dn = int(dG[0]), int(dG[1])
                    
                    # Get epsilon coefficient
                    # eps_fft is indexed by [m mod size, n mod size]
                    em = dm % size
                    en = dn % size
                    eps_coeff = eps_fft[em, en]
                    
                    H[i, j] = (kxG[i] * kxG[j] + kyG[i] * kyG[j]) * eps_coeff
            
            # Solve the eigenvalue problem
            # H * c = omega^2 * c
            vals, vecs = eigh(H)
            # Take the first Npw eigenvalues
            bands[ik, :] = np.sqrt(np.maximum(vals, 0))
        
        return bands
    
    def compute_gap(bands):
        """Compute the gap-midgap ratio between band 4 and 5."""
        # bands has shape (Nk, Npw), sorted by omega
        # We need min(omega_5) and max(omega_4) over all k-points
        omega_4 = bands[:, 3]  # 0-indexed, so band 4 is index 3
        omega_5 = bands[:, 4]  # band 5 is index 4
        
        max_omega_4 = np.max(omega_4)
        min_omega_5 = np.min(omega_5)
        
        gap = min_omega_5 - max_omega_4
        midgap = (min_omega_5 + max_omega_4) / 2.0
        
        if midgap <= 0:
            return -1.0
        
        return gap / midgap
    
    # Initial evaluation
    eps_fft = compute_epsilon_fourier(cell)
    bands = compute_bands(cell, k_points, eps_fft)
    best_gap = compute_gap(bands)
    best_cell = cell.copy()
    
    # Local search
    # We'll try flipping individual pixels and keep the improvement
    n_iterations = 0
    max_iterations = 2000
    
    while time.time() < deadline and n_iterations < max_iterations:
        n_iterations += 1
        
        # Pick a random pixel to flip
        i = np.random.randint(size)
        j = np.random.randint(size)
        
        # Flip the pixel
        cell[i, j] = 1 - cell[i, j]
        
        # Recompute epsilon Fourier coefficients
        eps_fft = compute_epsilon_fourier(cell)
        
        # Compute bands
        bands = compute_bands(cell, k_points, eps_fft)
        
        # Compute gap
        gap = compute_gap(bands)
        
        # Keep if improved, otherwise revert
        if gap > best_gap:
            best_gap = gap
            best_cell = cell.copy()
        else:
            cell[i, j] = 1 - cell[i, j]  # Revert
    
    # Use the best cell found
    return best_cell.tolist()
