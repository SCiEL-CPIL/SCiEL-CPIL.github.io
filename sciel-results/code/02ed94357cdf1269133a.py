<<<<<<< SEARCH
import numpy as np
import time

# Precompute G vectors
# 289 plane waves => (2*N+1)^2 = 289 => 2N+1=17 => N=8
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
# Note: These are in units of reciprocal lattice vector b = 2*pi.
# So actual G = 2*pi * G_vectors.
# (k+G)^2 in the eigenvalue eq: k is in 2*pi units?
# Usually k is in rad/m or similar.
# Let's use dimensionless units where lattice constant a=1.
# Reciprocal lattice vector b = 2*pi.
# k-points are in the BZ.
# (k+G)^2 = (2*pi*(k_dimless + G_int))^2.
# The 2*pi^2 factor cancels out in the ratio?
# Eigenvalues are omega^2.
# H_{GG'} = mu(G-G') * (k+G')^2.
# If we use dimensionless k (in units of pi/a?) and G (integers),
# (k+G)^2 needs care.
# Let k = (kx, ky) in units of 2*pi/a? No, standard BZ is -pi..pi.
# Let's define k in units of 1/a. G = 2*pi * n.
# (k + 2*pi*n)^2.
# Let's stick to standard SI-like but a=1.
# k_x, k_y in [0, pi].
# G_x = 2*pi*n.
# (k_x + 2*pi*n)^2.

# Precompute (k+G)^2 for all G for a given k? No, k changes.
# We compute it inside the loop.

def generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    # Irreducible zone: 0 <= ky <= kx <= pi
    # We'll generate a grid and filter, then select 30.
    # To be robust, let's use specific distributions.
    # 10 points on Gamma-X (y=0)
    # 10 points on X-M (x=pi)
    # 10 points on M-Gamma (y=x)
    # This covers boundary. Need interior?
    # "covering the boundary and interior"
    
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma: kx = ky, k in [0, pi/sqrt(2)]? No, M is (pi, pi).
    # Line from M(pi,pi) to Gamma(0,0) is kx=ky.
    # Parameterize by t in [0, pi]: kx=t, ky=t.
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    """
    size = len(cell)
    # 1 = dielectric (eps=12), 0 = air (eps=1)
    # mu = 1/eps
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    # Compute Fourier coefficients of mu
    # mu(G) = 1/N^2 sum mu(x,y) exp(-i G.r)
    # Using FFT
    mu_fft = np.fft.fft2(mu)
    # fft2 gives indices 0..31. We need -16..15 centered at 0.
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    # Extract -8..8
    # Index 16 is 0. So 8..24 is -8..8?
    # Size 32. Center is 16.
    # Range -8 to 8 corresponds to indices 16-8=8 to 16+8=24.
    mu_coeffs = mu_fft_shifted[8:25, 8:25] / (size * size)
    
    # Map G vectors to indices
    # G_vectors are (nx, ny) integers from -8 to 8.
    # mu_coeffs index: (nx+8, ny+8)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for k in K_POINTS:
        kx, ky = k
        # Build Matrix H
        # H[i, j] = mu(G_i - G_j) * (k + G_j)^2
        # Note: G_j is the vector for the column index.
        
        # Precompute (k + G)^2 for all G
        # G_vectors is 289x2.
        # k is 2x1.
        kg = k[None, :] + G_vectors * (2 * np.pi)
        kg_sq = np.sum(kg**2, axis=1)
        
        # H is 289x289
        # H[i, j] = mu(G_i - G_j) * kg_sq[j]
        
        # Construct G_diff = G_i - G_j
        # This is a 289x289x2 array. Memory heavy? 289^2*2*8 bytes = 1.3MB. OK.
        # But we can do it smarter.
        # mu(G_i - G_j) depends on difference.
        
        # Let's build the matrix row by row or using broadcasting.
        # G_i is G_vectors[i]. G_j is G_vectors[j].
        # Diff = G_i - G_j.
        
        # Create difference matrix
        # Diff_x[i, j] = Gx[i] - Gx[j]
        Gx_flat = G_vectors[:, 0]
        Gy_flat = G_vectors[:, 1]
        
        Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
        Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
        
        # Index into mu_coeffs
        # mu_coeffs is 17x17. Index range -8..8 -> 0..16.
        # Diff is -16..16.
        # If |Diff| > 8, mu is effectively 0 in our truncated basis?
        # No, mu(G) is the FFT of the pixel map.
        # The pixel map has finite support. Its FFT is defined for all G.
        # But we only have mu_coeffs for -8..8.
        # If G_i - G_j is outside -8..8, we treat mu as 0?
        # This is the "Folded Spectrum Method" or just truncation error.
        # Standard PW method truncates the basis. The matrix elements are mu(G-G') 
        # calculated from the *full* Fourier series.
        # Since we only have 32x32 pixels, the FFT gives us mu(G) for G in -16..15.
        # Our basis G is -8..8.
        # Differences G-G' are -16..16.
        # So we need mu(G) for G in -16..16.
        # Our mu_fft_shifted covers -16..15.
        # We need to be careful with index wrapping or just extract the full range.
        
        # Let's re-extract mu_coeffs for the full difference range -16..16.
        # mu_fft_shifted is 32x32.
        # Index 16 is 0.
        # We need indices corresponding to -16..16.
        # -16 is index 0. 16 is index 32 (which wraps to 0? No, 32 is 0 in mod 32).
        # Actually, FFT is periodic. mu(16) = mu(-16).
        # Let's just build a lookup table for mu_diff.
        
        # mu_full is 32x32.
        # We need mu(n, m) for n,m in -16..16.
        
        # Get the specific values
        # diff_x is in -16..16.
        # Map diff to index: (diff + 16) % 32.
        
        idx_x = (Diff_x + 16) % 32
        idx_y = (Diff_y + 16) % 32
        
        mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
        
        # Now H[i, j] = mu_diff[i, j] * kg_sq[j]
        H = mu_diff * kg_sq[None, :]
        
        # Solve eigenvalue problem
        # H is Hermitian?
        # mu_diff[i, j] = mu(G_i - G_j). mu(-G) = mu*(G). Since mu is real, mu is symmetric.
        # mu_diff[j, i] = mu(G_j - G_i) = mu(-(G_i - G_j)) = mu(G_i - G_j) = mu_diff[i, j].
        # kg_sq[j] is real.
        # H[j, i] = mu_diff[j, i] * kg_sq[i] = mu_diff[i, j] * kg_sq[i].
        # H[i, j] = mu_diff[i, j] * kg_sq[j].
        # H is not symmetric unless kg_sq[i] = kg_sq[j].
        # Wait, the equation is:
        # sum_j mu(G_i - G_j) (k+G_j)^2 E_j = omega^2 E_i
        # This matrix is NOT symmetric in the standard sense.
        # However, it is similar to a symmetric matrix.
        # Or we can solve the generalized eigenvalue problem.
        # A E = lambda B E
        # A[i, j] = mu(G_i - G_j)
        # B[i, j] = (k+G_i)^2 delta_ij
        # This is better. A is symmetric (real). B is diagonal.
        
        A = mu_diff
        B_diag = kg_sq
        
        # Solve A E = lambda B E  =>  A B^{-1} E = lambda E
        # B is diagonal, so B^{-1} is diag(1/B_diag).
        # M = A * (1/B_diag)[None, :]
        M = A * (1.0 / B_diag)[None, :]
        
        # M is not symmetric either. A[i,j] / B[j].
        # But it has real eigenvalues.
        # We can use eigvals (not eigvalsh) or symmetrize.
        # To symmetrize: S = B^{-1/2} A B^{-1/2}.
        # B^{-1/2} is diag(1/sqrt(B_diag)).
        B_inv_sqrt = 1.0 / np.sqrt(B_diag)
        S = A * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # S is real symmetric.
        evals = np.linalg.eigvalsh(S)
        
        # evals are omega^2.
        # Sort (eigvalsh returns sorted)
        # We need 4th (index 3) and 5th (index 4).
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap
=======
import numpy as np
import time
from scipy.sparse.linalg import eigsh
from scipy.sparse import diags

# Precompute G vectors
N_PW = 8
G_range = np.arange(-N_PW, N_PW + 1)
Gx, Gy = np.meshgrid(G_range, G_range, indexing='ij')
G_vectors = np.column_stack((Gx.ravel(), Gy.ravel()))
Gx_flat = G_vectors[:, 0]
Gy_flat = G_vectors[:, 1]

# Precompute difference indices for mu lookup
# Diff_x[i, j] = Gx[i] - Gx[j]
Diff_x = Gx_flat[:, None] - Gx_flat[None, :]
Diff_y = Gy_flat[:, None] - Gy_flat[None, :]
idx_x = (Diff_x + 16) % 32
idx_y = (Diff_y + 16) % 32

def generate_k_points(num_points=30):
    """Generate 30 k-points in the irreducible BZ of a square lattice."""
    points = []
    # Gamma-X: kx in [0, pi], ky = 0
    for i in range(10):
        kx = (i + 0.5) * (np.pi / 10)
        points.append((kx, 0))
        
    # X-M: kx = pi, ky in [0, pi]
    for i in range(10):
        ky = (i + 0.5) * (np.pi / 10)
        points.append((np.pi, ky))
        
    # M-Gamma: kx = ky
    for i in range(10):
        t = (i + 0.5) * (np.pi / 10)
        points.append((t, t))
        
    return np.array(points)

K_POINTS = generate_k_points()

def evaluate_cell(cell, verbose=False):
    """
    Evaluate the gap-midgap ratio for a 32x32 cell.
    cell: list of lists of 0/1.
    """
    size = len(cell)
    eps = np.array([[12 if c == 1 else 1 for c in row] for row in cell], dtype=float)
    mu = 1.0 / eps
    
    mu_fft = np.fft.fft2(mu)
    mu_fft_shifted = np.fft.fftshift(mu_fft)
    
    min_omega_5 = float('inf')
    max_omega_4 = float('-inf')
    
    for k in K_POINTS:
        kx, ky = k
        kg = k[None, :] + G_vectors * (2 * np.pi)
        kg_sq = np.sum(kg**2, axis=1)
        
        # Get mu_diff
        mu_diff = mu_fft_shifted[idx_x, idx_y] / (size * size)
        
        # A = mu_diff (symmetric)
        # B = diag(kg_sq)
        # Solve A v = lambda B v
        # Symmetrized: S = B^{-1/2} A B^{-1/2}
        B_inv_sqrt = 1.0 / np.sqrt(kg_sq)
        S = mu_diff * B_inv_sqrt[:, None] * B_inv_sqrt[None, :]
        
        # We need eigenvalues 4 and 5 (indices 3 and 4).
        # eigsh finds the smallest eigenvalues by default (which='SM').
        # Since we need the first 5, and the matrix is positive definite (mostly),
        # 'SM' might be slow if the smallest eigenvalues are very small (Gamma point).
        # However, for k=0, the first eigenvalue is 0.
        # It is safer to find the 5 smallest.
        # n=5, k=5, which='SA' (smallest algebraic) or 'SM'.
        # 'SA' is usually more robust for smallest.
        try:
            evals, _ = eigsh(S, k=5, which='SA', return_eigenvectors=False)
        except:
            # Fallback to dense if sparse fails (shouldn't happen for 289x289)
            evals = np.linalg.eigvalsh(S)[:5]
            
        # Sort just in case
        evals = np.sort(evals)
        w4 = np.sqrt(evals[3])
        w5 = np.sqrt(evals[4])
        
        max_omega_4 = max(max_omega_4, w4)
        min_omega_5 = min(min_omega_5, w5)
        
    gap = min_omega_5 - max_omega_4
    midgap = (min_omega_5 + max_omega_4) / 2.0
    if midgap == 0:
        return 0.0
    return gap / midgap
>>>>>>> REPLACE

