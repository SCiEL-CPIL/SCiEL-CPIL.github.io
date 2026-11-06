"""Parameterized Fourier-harmonic design for photonic crystal TM band gap."""
import numpy as np
from scipy.optimize import minimize

def design(problem):
    """Design a 32x32 unit cell with a large TM band gap between bands 4 and 5.
    
    Uses a parameterized Fourier-harmonic approach with (2,0), (0,2), and (2,2) modes.
    The dielectric distribution is constructed as:
        eps(x,y) = eps_bg + A20*cos(2*2*pi*x + phi20) + A02*cos(2*2*pi*y + phi02) 
                  + A22*cos(2*2*pi*(x+y) + phi22)
    
    We optimize the amplitudes and phases to maximize the gap-midgap ratio.
    """
    size = problem["pixels"]
    eps_low = problem["eps_low"]
    eps_high = problem["eps_high"]
    
    # Create coordinate grid
    x = np.arange(size) / size
    y = np.arange(size) / size
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    def build_cell(params):
        """Build a 32x32 cell from Fourier parameters.
        params: [A20, phi20, A02, phi02, A22, phi22, eps_bg, scale]
        """
        A20, phi20, A02, phi02, A22, phi22, eps_bg, scale = params
        
        # Fourier components
        f20 = A20 * np.cos(2 * 2 * np.pi * X + phi20)
        f02 = A02 * np.cos(2 * 2 * np.pi * Y + phi02)
        f22 = A22 * np.cos(2 * 2 * np.pi * (X + Y) + phi22)
        
        # Combine
        field = eps_bg + scale * (f20 + f02 + f22)
        
        # Map to 0/1: threshold at 0.5
        cell = (field > 0.5).astype(int)
        
        return cell
    
    def compute_gap(cell):
        """Compute the gap-midgap ratio between TM bands 4 and 5 using plane-wave method.
        This is a simplified version for optimization; the actual evaluator uses 289 PWs.
        """
        N = cell.shape[0]
        eps = np.where(cell == 1, eps_high, eps_low).astype(float)
        
        # Use a smaller number of plane waves for speed during optimization
        # We'll use a moderate number that's fast but captures the physics
        Npw = 17  # 17x17 = 289 plane waves, matching the evaluator
        
        # Generate plane wave vectors in the first BZ
        # For square lattice, reciprocal lattice vectors are 2*pi in x and y
        # We need plane waves G such that |k+G| is minimized
        # Irreducible BZ: 0 <= ky <= kx, 0 <= kx <= pi
        
        # Generate G vectors: G = (2*pi*nx, 2*pi*ny) for nx, ny in range
        # We want the smallest |k+G| for k in the BZ
        # For a 17x17 grid, we use nx, ny from -8 to 8
        
        # Actually, let's use a simpler approach: solve the eigenvalue problem
        # for the TM polarization
        
        # For TM polarization, the eigenvalue equation is:
        # sum_G' epsilon^{-1}(G-G') (k+G)^2 c(G') = omega^2 c(G)
        
        # First, compute epsilon^{-1} Fourier coefficients
        inv_eps = 1.0 / eps
        
        # Compute Fourier coefficients of inv_eps
        # inv_eps(G) = (1/N^2) sum_{i,j} inv_eps(i,j) exp(-i G . r_{i,j})
        # where r_{i,j} = (i/N, j/N)
        
        G_vectors = []
        G_indices = []
        for nx in range(-Npw//2, Npw//2 + 1):
            for ny in range(-Npw//2, Npw//2 + 1):
                G_vectors.append((2*np.pi*nx, 2*np.pi*ny))
                G_indices.append((nx, ny))
        
        G_vectors = np.array(G_vectors)
        n_G = len(G_vectors)
        
        # Compute inv_eps Fourier coefficients
        inv_eps_fourier = np.zeros(n_G, dtype=complex)
        for idx, (nx, ny) in enumerate(G_indices):
            Gx, Gy = 2*np.pi*nx, 2*np.pi*ny
            # inv_eps_fourier[idx] = (1/N^2) sum_{i,j} inv_eps(i,j) exp(-i(Gx*i/N + Gy*j/N))
            phase = np.exp(-1j * (Gx * X + Gy * Y))
            inv_eps_fourier[idx] = np.sum(inv_eps * phase) / (N*N)
        
        # Now solve for several k-points in the irreducible BZ
        # Irreducible BZ: 0 <= ky <= kx <= pi
        # We'll sample a few key points
        
        k_points = []
        # Gamma
        k_points.append((0, 0))
        # X = (pi, 0)
        k_points.append((np.pi, 0))
        # M = (pi, pi)
        k_points.append((np.pi, np.pi))
        # Along X-M: (pi, t*pi) for t in [0,1]
        for t in [0.25, 0.5, 0.75]:
            k_points.append((np.pi, t*np.pi))
        # Along Gamma-X: (t*pi, 0)
        for t in [0.25, 0.5, 0.75]:
            k_points.append((t*np.pi, 0))
        # Along Gamma-M: (t*pi, t*pi)
        for t in [0.25, 0.5, 0.75]:
            k_points.append((t*np.pi, t*np.pi))
        
        # For each k-point, solve the eigenvalue problem
        omega_4_max = 0.0
        omega_5_min = np.inf
        
        for kx, ky in k_points:
            k = np.array([kx, ky])
            
            # Build the matrix
            # H(G, G') = epsilon^{-1}(G-G') * |k+G|^2
            # where G-G' means the difference vector
            
            # We need to find the index of G-G' in our list
            # Create a dictionary for fast lookup
            G_dict = {}
            for idx, (nx, ny) in enumerate(G_indices):
                G_dict[(nx, ny)] = idx
            
            H = np.zeros((n_G, n_G), dtype=complex)
            for i, (nxi, nyi) in enumerate(G_indices):
                Gi = np.array([2*np.pi*nxi, 2*np.pi*nyi])
                ki_Gi = k + Gi
                ki_Gi_sq = np.dot(ki_Gi, ki_Gi)
                
                for j, (nxj, nyj) in enumerate(G_indices):
                    Gj = np.array([2*np.pi*nxj, 2*np.pi*nyj])
                    
                    # G_i - G_j = (2*pi*(nxi-nxj), 2*pi*(nyi-nyj))
                    dnx = nxi - nxj
                    dny = nyi - nyj
                    
                    if (dnx, dny) in G_dict:
                        d_idx = G_dict[(dnx, dny)]
                        H[i, j] = inv_eps_fourier[d_idx] * ki_Gi_sq
                    # If not in our truncated set, we ignore it (truncation error)
            
            # Solve eigenvalue problem
            # H is Hermitian if inv_eps is real and symmetric
            # Take real part for numerical stability
            H_real = np.real(H)
            
            # Symmetrize
            H_real = (H_real + H_real.T) / 2
            
            # Solve
            try:
                eigenvalues = np.linalg.eigvalsh(H_real)
                eigenvalues = np.sort(eigenvalues)
                
                # omega^2 = eigenvalues, so omega = sqrt(eigenvalues)
                # We want bands 4 and 5 (0-indexed: 3 and 4)
                if len(eigenvalues) >= 5:
                    omega4 = np.sqrt(max(eigenvalues[3], 0))
                    omega5 = np.sqrt(max(eigenvalues[4], 0))
                    
                    omega_4_max = max(omega_4_max, omega4)
                    omega_5_min = min(omega_5_min, omega5)
            except:
                pass
        
        if omega_5_min == np.inf or omega_4_max == 0:
            return -1.0
        
        gap = omega_5_min - omega_4_max
        midgap = (omega_5_min + omega_4_max) / 2
        
        if midgap == 0:
            return -1.0
        
        return 2 * gap / midgap
    
    def objective(params):
        """Negative gap-midgap ratio for minimization."""
        cell = build_cell(params)
        gap_ratio = compute_gap(cell)
        return -gap_ratio
    
    # Initial guess
    # Start with a reasonable symmetric configuration
    x0 = np.array([0.3, 0.0, 0.3, 0.0, 0.2, 0.0, 0.5, 1.0])
    
    # Bounds
    bounds = [
        (0, 1),    # A20
        (0, 2*np.pi),  # phi20
        (0, 1),    # A02
        (0, 2*np.pi),  # phi02
        (0, 1),    # A22
        (0, 2*np.pi),  # phi22
        (0, 1),    # eps_bg
        (0.5, 2.0) # scale
    ]
    
    # Optimize
    best_cell = None
    best_score = -np.inf
    
    # Try a few random starts
    np.random.seed(42)
    n_trials = 5
    
    for trial in range(n_trials):
        if trial == 0:
            params0 = x0
        else:
            params0 = np.array([
                np.random.uniform(0, 0.5),
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, 0.5),
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0, 0.5),
                np.random.uniform(0, 2*np.pi),
                np.random.uniform(0.3, 0.7),
                np.random.uniform(0.8, 1.5)
            ])
        
        result = minimize(objective, params0, method='Nelder-Mead', 
                         bounds=bounds, 
                         options={'maxiter': 50, 'xatol': 0.05, 'fatol': 0.05})
        
        if result.success or -result.fun > best_score:
            best_score = -result.fun
            best_cell = build_cell(result.x)
    
    # If optimization didn't work well, fall back to a known good design
    if best_score < 0.01:
        # Use a checkerboard-like pattern with dielectric rods
        cell = []
        for i in range(size):
            row = []
            for j in range(size):
                xi = (i + 0.5) / size - 0.5
                yj = (j + 0.5) / size - 0.5
                # Two rods at (0.25, 0.25) and (0.75, 0.75) with radius 0.15
                r1 = np.sqrt((xi - 0.25)**2 + (yj - 0.25)**2)
                r2 = np.sqrt((xi - 0.75)**2 + (yj - 0.75)**2)
                r3 = np.sqrt((xi + 0.25)**2 + (yj + 0.25)**2)
                r4 = np.sqrt((xi + 0.75)**2 + (yj + 0.75)**2)
                if min(r1, r2, r3, r4) < 0.15:
                    row.append(1)
                else:
                    row.append(0)
            cell.append(row)
        best_cell = cell
    
    # Ensure symmetry: the cell should be invariant under 90-degree rotation and reflection
    # Convert to numpy for easy manipulation
    cell_arr = np.array(best_cell, dtype=int)
    
    # Enforce 4-fold rotational symmetry and reflection symmetry
    # Average over the symmetry group
    sym_cell = np.zeros_like(cell_arr, dtype=float)
    count = np.zeros_like(cell_arr, dtype=float)
    
    # Identity
    sym_cell += cell_arr
    count += 1
    
    # 90-degree rotation
    rot90 = np.rot90(cell_arr, 1)
    sym_cell += rot90
    count += 1
    
    # 180-degree rotation
    rot180 = np.rot90(cell_arr, 2)
    sym_cell += rot180
    count += 1
    
    # 270-degree rotation
    rot270 = np.rot90(cell_arr, 3)
    sym_cell += rot270
    count += 1
    
    # Reflection about vertical axis
    refl_v = np.fliplr(cell_arr)
    sym_cell += refl_v
    count += 1
    
    # Reflection about horizontal axis
    refl_h = np.flipud(cell_arr)
    sym_cell += refl_h
    count += 1
    
    # Diagonal reflection
    refl_d = np.transpose(cell_arr)
    sym_cell += refl_d
    count += 1
    
    # Anti-diagonal reflection
    refl_ad = np.transpose(np.fliplr(cell_arr))
    sym_cell += refl_ad
    count += 1
    
    # Average
    sym_cell = sym_cell / count
    
    # Threshold
    final_cell = (sym_cell >= 0.5).astype(int)
    
    # Convert to nested list
    return final_cell.tolist()
