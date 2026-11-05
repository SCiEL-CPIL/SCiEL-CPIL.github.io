"""Photonic crystal design for large TM band gap between bands 4 and 5."""
import time
import math
import numpy as np

def _compute_tm_gap(cell, num_pw=17, num_kpts=20):
    """
    Compute TM band gap ratio using a simplified plane-wave method.
    cell: 32x32 binary list
    Returns gap-midgap ratio (can be negative).
    """
    N = len(cell)
    # Create epsilon map
    eps = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            eps[i, j] = 12.0 if cell[i][j] == 1 else 1.0
    
    # For TM mode: -div(1/eps * grad H) = (omega/c)^2 H
    # We need 1/eps
    inv_eps = np.zeros((N, N), dtype=float)
    for i in range(N):
        for j in range(N):
            inv_eps[i, j] = 1.0 / eps[i, j]
    
    # Compute Fourier coefficients of inv_eps for the first several G vectors
    # Use only G vectors with |Gx|, |Gy| <= num_pw
    G_list = []
    for gx in range(-num_pw, num_pw + 1):
        for gy in range(-num_pw, num_pw + 1):
            G_list.append((gx, gy))
    
    # Compute Fourier coefficients
    # f(G) = (1/N^2) sum_{i,j} f(x_i, y_j) exp(-i 2 pi (Gx x_i + Gy y_j))
    x_coords = np.arange(N) / N  # 0 to (N-1)/N
    y_coords = np.arange(N) / N
    
    inv_eps_F = {}
    for gx, gy in G_list:
        # Compute the sum
        phase_x = 2 * np.pi * gx * x_coords
        phase_y = 2 * np.pi * gy * y_coords
        # Use broadcasting
        phase = np.outer(phase_x, np.ones(N)) + np.outer(np.ones(N), phase_y)
        # Actually, phase[i,j] = 2*pi*(gx*i/N + gy*j/N)
        # f(G) = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-i * phase[i,j])
        coeff = np.sum(inv_eps * np.exp(-1j * phase)) / (N * N)
        inv_eps_F[(gx, gy)] = coeff
    
    # Define k-points in the irreducible BZ
    # For square lattice, irreducible BZ is the triangle from (0,0) to (pi,0) to (pi,pi)
    # We'll sample points along the edges and interior
    k_points = []
    
    # High symmetry points and along X-M-G and M-X paths
    # Let's use a grid of k-points in the first quadrant of the BZ
    # BZ: -pi to pi in both directions, but irreducible is 0<=kx<=ky<=pi (for square)
    # Actually for square lattice, the irreducible BZ is a 1/8 circle sector
    # Let's just use a set of representative k-points
    
    # k-points: (0,0), (pi,0), (pi,pi), (pi/2,0), (pi/2,pi/2), (pi,pi/2)
    # Plus some interior points
    a = 1.0  # lattice constant
    
    k_points = [
        (0.0, 0.0),           # Gamma
        (np.pi, 0.0),         # X
        (np.pi, np.pi),       # M
        (np.pi/2, 0.0),       # Mid X-Gamma
        (np.pi/2, np.pi/2),   # Center
        (np.pi, np.pi/2),     # Mid M-X
        (np.pi/4, 0.0),
        (np.pi/4, np.pi/4),
        (np.pi/2, np.pi/4),
        (np.pi/4, np.pi/2),
    ]
    
    # Add a few more
    for kx_frac in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        for ky_frac in [0.0, 0.2, 0.3, 0.4, 0.5]:
            kx = kx_frac * np.pi
            ky = ky_frac * np.pi
            if kx <= ky + 1e-9:
                k_points.append((kx, ky))
    
    # For each k-point, solve the eigenvalue problem
    # The eigenvalue equation in plane-wave basis:
    # sum_G' [ (k+G)^2 * delta_{G,G'} + (k+G)^2 * (1/eps)(G-G') ] H_{G'} = (omega/a)^2 H_G
    # Wait, for TM: -div(1/eps grad H) = (omega/c)^2 H
    # In plane waves: H = sum_G H_G exp(i(k+G).r)
    # The matrix element is: (k+G)^2 * delta_{G,G'} + (k+G)^2 * (1/eps)(G-G') ... 
    # Actually it's: sum_{G'} (k+G)^2 * delta_{G,G'} * H_{G'} + (k+G)^2 * sum_{G'} (1/eps)(G-G') H_{G'}
    # No, let me think again.
    
    # The Hamiltonian matrix H_{G,G'} = (k+G)^2 delta_{G,G'} + (k+G)^2 f(G-G')
    # where f is the Fourier transform of 1/eps
    # Wait, that's not right either. Let me derive it properly.
    
    # -div(1/eps grad H) = (omega/c)^2 H
    # Let H = sum_{G'} H_{G'} exp(i(k+G').r)
    # grad H = sum_{G'} i(k+G') H_{G'} exp(i(k+G').r)
    # 1/eps * grad H = sum_{G'} (1/eps) i(k+G') H_{G'} exp(i(k+G').r)
    # 1/eps = sum_{G''} f(G'') exp(i G''.r)
    # So 1/eps * grad H = sum_{G'} sum_{G''} f(G'') i(k+G') H_{G'} exp(i(k+G'+G'').r)
    # Let G = G' + G'', so G'' = G - G'
    # 1/eps * grad H = sum_G [ sum_{G'} f(G-G') i(k+G') H_{G'} ] exp(i(k+G).r)
    # -div(1/eps grad H) = - sum_G [ sum_{G'} f(G-G') i(k+G') H_{G'} ] i(k+G) exp(i(k+G).r)
    # = sum_G [ sum_{G'} f(G-G') (k+G').dot(k+G) H_{G'} ] exp(i(k+G).r)
    # So the matrix element is: H_{G,G'} = (k+G').dot(k+G) * f(G-G')
    # And we need H H = (omega/c)^2 H
    
    # So the eigenvalue problem is:
    # sum_{G'} (k+G').dot(k+G) * f(G-G') * H_{G'} = (omega/c)^2 * H_G
    
    max_freq = 0.0
    min_freq = 1e10
    
    for kx, ky in k_points:
        k_vec = np.array([kx, ky])
        
        # Build the matrix
        G_array = np.array(G_list, dtype=float)
        # G_array has shape (num_G, 2)
        
        # k+G for each G
        kG = G_array + k_vec  # shape (num_G, 2)
        
        # Build matrix M where M[i,j] = (k+G_j).dot(k+G_i) * f(G_i - G_j)
        num_G = len(G_list)
        M = np.zeros((num_G, num_G), dtype=complex)
        
        for i in range(num_G):
            Gi = G_array[i]
            for j in range(num_G):
                Gj = G_array[j]
                G_diff = Gi - Gj
                f_val = inv_eps_F.get((int(G_diff[0]), int(G_diff[1])), 0.0)
                dot_prod = np.dot(kG[j], kG[i])
                M[i, j] = dot_prod * f_val
        
        # Solve eigenvalue problem
        # The matrix should be Hermitian if f is properly computed
        # Use np.linalg.eigh for Hermitian matrices
        try:
            eigenvalues = np.linalg.eigvalsh(M)
        except:
            try:
                eigenvalues = np.linalg.eigvals(M)
                eigenvalues = np.sort(np.real(eigenvalues))
            except:
                continue
        
        # The eigenvalues are (omega/c)^2 * a^2 ... wait, need to be careful with units
        # omega^2 / c^2 = eigenvalue, so omega = c * sqrt(eigenvalue)
        # But we're working in units where a=1, so eigenvalues are (omega*a/c)^2
        # For comparing gaps, we just need relative values
        
        # Sort eigenvalues
        eigenvalues = np.sort(np.real(eigenvalues))
        
        # Take the first 5+ bands
        if len(eigenvalues) >= 5:
            omega_4 = np.sqrt(max(eigenvalues[3], 0.0))
            omega_5 = np.sqrt(max(eigenvalues[4], 0.0))
            
            if omega_4 > 0:
                max_freq = max(max_freq, omega_4)
                min_freq = min(min_freq, omega_5)
    
    if max_freq == 0.0 or min_freq == 1e10:
        return -1.0
    
    gap = min_freq - max_freq
    midgap = (min_freq + max_freq) / 2.0
    
    if midgap == 0:
        return -1.0
    
    return 2.0 * gap / midgap


def design(problem):
    """
    Design a symmetric 32x32 unit cell to maximize TM band gap between bands 4 and 5.
    Uses a parametric family of symmetric shapes with geometric primitives.
    """
    size = problem["pixels"]
    start_time = time.time()
    deadline = start_time + 180  # Safety margin before 230s limit
    
    # Start with a good initial guess: super-ellipse with moderate size
    # Try several candidate shapes and pick the best one
    
    candidates = []
    
    # Candidate 1: Super-ellipse with n=4, radius 0.35 (current)
    # Candidate 2: Super-ellipse with n=6, radius 0.33
    # Candidate 3: Square with rounded corners
    # Candidate 4: Cross shape
    # Candidate 5: Ring shape
    
    def make_superellipse(n_exp, radius):
        cell = []
        for i in range(size):
            row = []
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                val = abs(x / radius) ** n_exp + abs(y / radius) ** n_exp
                row.append(1 if val < 1.0 else 0)
            cell.append(row)
        return cell
    
    def make_square_ring(outer_r, inner_r, corner_round=0.05):
        cell = []
        for i in range(size):
            row = []
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                # Square with rounded corners
                abs_x = abs(x)
                abs_y = abs(y)
                # If in corner region, use circular criterion
                if abs_x > outer_r - corner_round and abs_y > outer_r - corner_round:
                    # Corner region
                    cx = abs_x - (outer_r - corner_round)
                    cy = abs_y - (outer_r - corner_round)
                    outer_val = (cx ** 2 + cy ** 2) / (corner_round ** 2) if corner_round > 0 else 0
                else:
                    outer_val = 0
                in_outer = (abs_x < outer_r and abs_y < outer_r) or (outer_val < 1.0)
                
                in_inner = (abs_x < inner_r and abs_y < inner_r)
                
                row.append(1 if in_outer and not in_inner else 0)
            cell.append(row)
        return cell
    
    def make_cross(arm_width, arm_length):
        cell = []
        for i in range(size):
            row = []
            x = (i + 0.5) / size - 0.5
            for j in range(size):
                y = (j + 0.5) / size - 0.5
                # Cross: horizontal bar and vertical bar
                in_h = abs(y) < arm_width and abs(x) < arm_length
                in_v = abs(x) < arm_width and abs(y) < arm_length
                row.append(1 if in_h or in_v else 0)
            cell.append(row)
        return cell
    
    # Generate candidate shapes
    # Try different super-ellipse parameters
    for n_exp in [3.0, 4.0, 5.0, 6.0, 8.0]:
        for radius in [0.25, 0.30, 0.33, 0.35, 0.38, 0.40]:
            candidates.append(make_superellipse(n_exp, radius))
    
    # Try square rings
    for outer_r in [0.40, 0.45]:
        for inner_r in [0.15, 0.20, 0.25]:
            candidates.append(make_square_ring(outer_r, inner_r))
    
    # Try crosses
    for arm_width in [0.10, 0.12, 0.15]:
        for arm_length in [0.45, 0.48]:
            candidates.append(make_cross(arm_width, arm_length))
    
    # Evaluate candidates
    best_cell = candidates[0]
    best_score = -1e10
    
    for idx, cell in enumerate(candidates):
        if time.time() > deadline - 30:
            break
        score = _compute_tm_gap(cell, num_pw=10, num_kpts=10)
        if score > best_score:
            best_score = score
            best_cell = cell
    
    return best_cell
