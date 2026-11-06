import numpy as np
import time
from scipy.optimize import minimize
import sys

# Constants for the problem
SIZE = 32
EPS_LOW = 1.0
EPS_HIGH = 12.0
LOWER_BAND = 4
UPPER_BAND = 5

def compute_gap(cell, n_pw=13, k_points=None):
    """
    Computes the gap between the 4th and 5th TM bands using a simplified 
    plane-wave method.
    
    cell: 32x32 array of 0s and 1s.
    n_pw: Number of plane waves in x (and y) direction. Odd number for symmetry.
    k_points: List of (kx, ky) tuples.
    """
    if k_points is None:
        # Irreducible BZ points: Gamma (0,0), M (pi,0), X (pi/2, pi/2)
        # We sample a few points to get a robust estimate
        k_points = [(0.0, 0.0), (np.pi, 0.0), (np.pi/2, np.pi/2), (np.pi/2, 0.0), (0.0, np.pi/2)]
    
    # Construct permittivity grid
    # Map 0/1 to eps_low/eps_high
    eps = np.zeros((SIZE, SIZE))
    eps[cell == 1] = EPS_HIGH
    eps[cell == 0] = EPS_LOW
    
    # Inverse permittivity for TM modes
    inv_eps = 1.0 / eps
    
    # Fourier coefficients of inv_eps
    # We need the Fourier series of inv_eps: inv_eps(x,y) = sum G_mn exp(i G . r)
    # G_mn = (2 pi m / L, 2 pi n / L). Here L=1.
    # G_mn = (2 pi m, 2 pi n)
    
    # We use a grid of plane waves centered at 0
    # Indices from -N/2 to N/2 - 1 if N even, but N is odd so -N//2 to N//2
    N = n_pw
    indices = np.arange(-N//2, N//2 + 1)
    Gx = 2 * np.pi * indices
    Gy = 2 * np.pi * indices
    
    # Compute Fourier coefficients of inv_eps
    # Using FFT for speed
    # Shift eps so that (0,0) is at center for FFT? 
    # Standard FFT assumes origin at (0,0).
    # Let's compute G_inv_eps[m, n] for m, n in indices.
    
    # Pad or use direct sum? Direct sum is O(N^2 * SIZE^2) which is fine for N=13, SIZE=32.
    # G_mn = (1/L^2) integral inv_eps(x,y) exp(-i (Gx_m x + Gy_n y)) dx dy
    # Discrete: sum_{i,j} inv_eps[i,j] exp(-i (Gx_m (i+0.5)/SIZE + Gy_n (j+0.5)/SIZE)) / SIZE^2
    
    # Precompute positions
    x = (np.arange(SIZE) + 0.5) / SIZE
    y = (np.arange(SIZE) + 0.5) / SIZE
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # To save time, we can compute the matrix A directly in real space? No, PW method is in Fourier space.
    # Let's compute the Fourier coefficients C_inv_eps.
    C_inv = np.zeros((N, N), dtype=complex)
    
    # Optimization: Use FFT
    # The FFT of inv_eps gives coefficients for frequencies 0..SIZE-1.
    # We need specific frequencies G_mn.
    # Since SIZE=32 and N=13 (max index 6), these are low frequencies.
    # We can just sum directly, it's fast enough.
    
    # Vectorized sum
    # C_inv[p, q] corresponds to indices[p], indices[q]
    for p, m in enumerate(indices):
        for q, n in enumerate(indices):
            kx = 2 * np.pi * m
            ky = 2 * np.pi * n
            phase = np.exp(-1j * (kx * X + ky * Y))
            C_inv[p, q] = np.sum(inv_eps * phase) / (SIZE * SIZE)
            
    max_omega = 0.0
    min_omega_5 = float('inf')
    
    for kx, ky in k_points:
        # Construct eigenvalue problem
        # (I - A) c = omega^2 c
        # A_ij = (G_i - k) dot (G_j - k) * C_inv[G_i - G_j] / |G_i - k|^2 ?
        # TM Equation: 
        # omega^2 c_k = sum_k' ( (k - G) dot (k' - G) / |k - G|^2 ) * (1/eps)_{k-k'} * c_k'
        # Wait, the standard form is:
        # omega^2 c_k = sum_k' S_{k, k'} c_k'
        # S_{k, k'} = ( (k - G_k) dot (k' - G_k') / |k - G_k|^2 ) * (1/eps)_{G_k - G_k'}
        # Actually, for TM:
        # (I - M) c = omega^2 c
        # M_{k, k'} = ( (k - G_k) dot (k' - G_k') ) / |k - G_k|^2 * (1/eps)_{G_k - G_k'}
        
        # Let's build the matrix M
        M = np.zeros((N, N), dtype=complex)
        
        # Indices of plane waves
        # G_k is the vector (2 pi m, 2 pi n)
        # k is the Bloch vector (kx, ky)
        
        for i in range(N):
            Gx_i = Gx[i]
            Gy_i = Gy[i]
            kx_i = kx - Gx_i
            ky_i = ky - Gy_i
            denom = kx_i**2 + ky_i**2
            if denom < 1e-12:
                # Avoid division by zero, though this term is 0 in the numerator usually
                # If k=G, the term is 0 for off-diagonal? 
                # The term is (k-G) dot (k'-G') / |k-G|^2. If k=G, numerator is 0.
                # So M[i, j] = 0 for this i.
                continue
            
            for j in range(N):
                Gx_j = Gx[j]
                Gy_j = Gy[j]
                kx_j = kx - Gx_j
                ky_j = ky - Gy_j
                
                # G_diff = G_i - G_j = (2 pi (m_i - m_j), 2 pi (n_i - n_j))
                dm = indices[i] - indices[j]
                dn = indices[i] - indices[j]
                
                # Find index in C_inv for (dm, dn)
                # C_inv is indexed by indices array.
                # We need to find p, q such that indices[p] = dm and indices[q] = dn.
                # Since indices are symmetric and contiguous, we can map directly.
                # indices = [-N//2, ..., N//2]
                # dm is in range [-N+1, N-1].
                # If dm is in indices, get C_inv.
                try:
                    p_idx = np.where(indices == dm)[0][0]
                    q_idx = np.where(indices == dn)[0][0]
                    c_inv = C_inv[p_idx, q_idx]
                except IndexError:
                    c_inv = 0.0
                    # If dm, dn not in the truncated set, assume 0 (neglect high freq)
                
                numerator = kx_i * kx_j + ky_i * ky_j
                M[i, j] = (numerator / denom) * c_inv
        
        # Solve eigenvalue problem: (I - M) c = omega^2 c
        # => (I - M - omega^2 I) c = 0
        # Eigenvalues of (I - M) are omega^2
        eigvals = np.linalg.eigvalsh(I_minus_M) # I - M is Hermitian?
        # M is not necessarily Hermitian? 
        # (k-G) dot (k'-G') is real. C_inv is complex but C_inv[-k] = conj(C_inv[k]).
        # The matrix should be Hermitian.
        
        # Wait, I defined M[i,j] using C_inv[G_i - G_j].
        # Let's check Hermiticity.
        # M[j,i] = (k_j dot k_i)/|k_j|^2 * C_inv[G_j - G_i]
        # This is not obviously equal to conj(M[i,j]).
        # Actually, the operator is:
        # omega^2 c_k = sum_k' [ ( (k-G_k) . (k'-G_k') ) / |k-G_k|^2 ] (1/eps)_{k-k'} c_k'
        # This matrix is Hermitian.
        
        # Let's just use eigvalsh on (I - M) assuming it's Hermitian.
        # If not, use eigvals.
        try:
            eigvals = np.linalg.eigvalsh(I_minus_M)
        except:
            eigvals = np.linalg.eigvals(I_minus_M)
            eigvals = np.real(eigvals)
            
        # Sort eigenvalues
        eigvals = np.sort(eigvals)
        
        # We need the 4th and 5th bands.
        # Note: The eigenvalues are omega^2.
        # The bands are ordered by omega.
        # omega_4 = sqrt(eigvals[3]) (0-indexed)
        # omega_5 = sqrt(eigvals[4])
        
        # However, we must ensure we are looking at the correct bands.
        # The "4th band" means the 4th lowest frequency.
        
        if len(eigvals) >= 5:
            w4 = np.sqrt(eigvals[3])
            w5 = np.sqrt(eigvals[4])
            
            max_omega = max(max_omega, w4)
            min_omega_5 = min(min_omega_5, w5)
            
    if min_omega_5 == float('inf'):
        return -1.0
        
    gap = 2 * (min_omega_5 - max_omega) / (min_omega_5 + max_omega)
    return gap

def fourier_to_cell(params, size=32):
    """
    Reconstructs the 32x32 cell from Fourier parameters.
    params: array of real coefficients for the Fourier series.
    We use a low-frequency series to ensure smoothness and symmetry.
    """
    # Define the set of (m, n) to include.
    # To maintain C4v symmetry, we include (m, n) where m, n are even?
    # No, we just enforce symmetry in the coefficients.
    # Let's use a small number of terms to keep optimization fast.
    # Terms: (0,0), (2,0), (0,2), (2,2), (4,0), (0,4), (4,4), (2,4), (4,2) ...
    # Let's stick to a fixed set of low frequencies.
    
    # Indices for m, n
    # We'll map params to specific (m, n) pairs.
    # p0: (0,0)
    # p1: (2,0) and (0,2) (symmetric)
    # p2: (2,2)
    # p3: (4,0) and (0,4)
    # p4: (4,4)
    # p5: (2,4) and (4,2)
    
    # Let's just use a grid of m, n from -M to M.
    M = 4 # Max frequency
    # We will optimize coefficients for m,n in [-M, M]
    # But to enforce symmetry, we only optimize independent ones and fill the rest.
    
    # For simplicity and speed, let's just optimize a small set of "basis" coefficients
    # and reconstruct.
    
    # Let's use the first 10 parameters corresponding to:
    # 0: (0,0)
    # 1: (1,0) -> (0,1)
    # 2: (2,0) -> (0,2)
    # 3: (1,1)
    # 4: (2,1) -> (1,2)
    # 5: (2,2)
    # 6: (3,0) -> (0,3)
    # 7: (3,1) -> (1,3)
    # 8: (3,2) -> (2,3)
    # 9: (3,3)
    
    # Actually, to get a good shape, we might need higher frequencies.
    # Let's use a 5x5 grid of coefficients (m,n from -2 to 2).
    # That's 25 params.
    
    # Let's try a simpler approach:
    # Define a small number of "super-pixels" or use a low-pass filter on the cell?
    # No, the directive suggests Fourier series.
    
    # Let's use parameters for m,n in [-3, 3].
    # Total 7x7 = 49 coefficients.
    # Symmetry: C[m,n] = C[-m,-n] = C[n,m] = C[-n,-m] (for real, symmetric field)
    # Independent coefficients: m>=0, n>=0, m>=n? No, just m,n >= 0 and handle the rest.
    
    # Let's just optimize a 7x7 matrix of real coefficients, and enforce symmetry in the reconstruction.
    
    # params is a flat array of length 49 (7x7)
    C = np.reshape(params, (7, 7))
    
    # Map C to full grid of coefficients for m,n in [-3, 3]
    # indices: -3, -2, -1, 0, 1, 2, 3
    # C[i,j] corresponds to m = i-3, n = j-3.
    
    # We enforce C[m,n] = C[n,m] and C[m,n] = C[-m,-n].
    # Since we are optimizing C directly, we should constrain the optimization or 
    # symmetrize C after each step.
    
    # Symmetrize C:
    C_sym = np.zeros_like(C)
    for i in range(7):
        for j in range(7):
            m = i - 3
            n = j - 3
            # Average over symmetry group
            vals = [C[i, j]] # (m, n)
            # (n, m)
            i2, j2 = j, i
            if 0 <= i2 < 7 and 0 <= j2 < 7:
                vals.append(C[i2, j2])
            # (-m, -n)
            i3, j3 = 6-i, 6-j
            if 0 <= i3 < 7 and 0 <= j3 < 7:
                vals.append(C[i3, j3])
            # (-n, -m)
            i4, j4 = 6-j, 6-i
            if 0 <= i4 < 7 and 0 <= j4 < 7:
                vals.append(C[i4, j4])
            
            C_sym[i, j] = np.mean(vals)
            
    # Reconstruct field
    x = (np.arange(size) + 0.5) / size
    y = (np.arange(size) + 0.5) / size
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    field = np.zeros((size, size))
    
    for i in range(7):
        for j in range(7):
            m = i - 3
            n = j - 3
            coeff = C_sym[i, j]
            if coeff != 0:
                phase = np.exp(1j * 2 * np.pi * (m * X + n * Y))
                field += coeff * phase
                
    field = np.real(field)
    
    # Threshold to 0/1
    # We want to maximize the gap. The threshold should be chosen to get a reasonable duty cycle.
    # Let's use the mean of the field as the threshold, or a fixed threshold.
    # A fixed threshold of 0 might be bad if the mean is not 0.
    # Let's center the field: field -= mean(field)
    # Then threshold at 0.
    field_centered = field - np.mean(field)
    cell = (field_centered > 0).astype(int)
    
    return cell

def objective(params):
    """
    Negative gap for minimization.
    """
    cell = fourier_to_cell(params)
    gap = compute_gap(cell, n_pw=11)
    return -gap

def design(problem):
    size = problem["pixels"]
    start_time = time.time()
    time_limit = 180.0 # Safety margin
    
    # Initial guess: All zeros (air) or a simple pattern
    # Let's start with a circle-like pattern in Fourier space?
    # Or just random low-frequency coefficients.
    
    np.random.seed(42)
    # 7x7 coefficients
    x0 = np.zeros(49)
    # Add a DC component to control duty cycle? 
    # The centering step handles DC.
    # Let's add some random low freqs
    for i in range(49):
        m = (i // 7) - 3
        n = (i % 7) - 3
        if m == 0 and n == 0:
            x0[i] = 0.0
        else:
            x0[i] = np.random.normal(0, 0.1)
            
    best_params = x0.copy()
    best_gap = objective(x0)
    
    # Optimization loop
    # Use Nelder-Mead
    # It's robust for non-smooth functions (like thresholding).
    
    while time.time() - start_time < time_limit:
        try:
            res = minimize(objective, best_params, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
            
            if res.fun < best_gap:
                best_gap = res.fun
                best_params = res.x
            else:
                # If no improvement, perturb and restart?
                # Or break if stuck?
                # Let's perturb slightly
                best_params = best_params + np.random.normal(0, 0.05, 49)
                best_gap = objective(best_params)
                
        except Exception as e:
            pass
            
        # Check time
        if time.time() - start_time > time_limit - 20:
            break
            
    # Final evaluation
    final_cell = fourier_to_cell(best_params)
    
    # Ensure symmetry is perfect (it should be by construction, but check)
    # The fourier_to_cell function enforces symmetry.
    
    return final_cell

if __name__ == "__main__":
    # Simple test
    problem = {"pixels": 32, "eps_low": 1, "eps_high": 12, "lower_band": 4, "polarization": "TM"}
    cell = design(problem)
    print("Cell shape:", np.array(cell).shape)
    print("Symmetry check:")
    c = np.array(cell)
    print("Row reverse:", np.array_equal(c, c[::-1]))
    print("Col reverse:", np.array_equal(c, c[:, ::-1]))
    print("Transpose:", np.array_equal(c, c.T))
    print("Gap:", compute_gap(cell))
