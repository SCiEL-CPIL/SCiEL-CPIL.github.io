"""Photonic crystal unit cell design for large TM band gap between bands 4 and 5."""
import time
import math
import random
import numpy as np

# EVOLVE-BLOCK-START
def design(problem):
    """Design a 32x32 symmetric unit cell maximizing TM band gap between bands 4 and 5."""
    size = problem["pixels"]  # 32
    eps_high = problem.get("eps_high", 12)
    eps_low = problem.get("eps_low", 1)
    
    start_time = time.time()
    deadline = 210.0  # safety margin before 230s hard kill
    
    def elapsed():
        return time.time() - start_time
    
    def time_ok():
        return elapsed() < deadline
    
    n = size
    eps_contrast = eps_high - eps_low
    
    # Precompute k-points in irreducible BZ of square lattice
    # Irreducible BZ: 0 <= kx <= ky, kx^2 + ky^2 <= 1 (in units of 2π/a)
    # Use 30 points: boundary + interior
    k_points = []
    # High-symmetry points and lines
    # Gamma (0,0), M (0.5, 0), X (0.5, 0.5)
    # Lines: Gamma-X, Gamma-M, M-X
    # Interior points for better sampling
    k_points.append((0.0, 0.0))       # Gamma
    k_points.append((0.5, 0.0))       # M
    k_points.append((0.5, 0.5))       # X
    
    # Along Gamma-X: kx=ky from 0 to 0.5
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((t, t))
    
    # Along Gamma-M: kx=0, ky from 0 to 0.5
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((0.0, t))
    
    # Along M-X: kx=0.5, ky from 0 to 0.5
    for t in [0.1, 0.2, 0.3, 0.4]:
        k_points.append((0.5, t))
    
    # Interior points
    interior = [
        (0.1, 0.2), (0.2, 0.1), (0.1, 0.3), (0.3, 0.1),
        (0.2, 0.3), (0.3, 0.2), (0.15, 0.25), (0.25, 0.15),
        (0.3, 0.3), (0.2, 0.4), (0.4, 0.2), (0.35, 0.35),
        (0.1, 0.4), (0.4, 0.1), (0.2, 0.35), (0.35, 0.2),
        (0.15, 0.35), (0.35, 0.15), (0.25, 0.35), (0.35, 0.25),
        (0.12, 0.22), (0.22, 0.12), (0.18, 0.28), (0.28, 0.18),
        (0.22, 0.32), (0.32, 0.22), (0.15, 0.40), (0.40, 0.15),
        (0.25, 0.40), (0.40, 0.25), (0.30, 0.40), (0.40, 0.30),
    ]
    for kp in interior:
        if len(k_points) >= 30:
            break
        # Check if in irreducible BZ: kx <= ky, kx^2+ky^2 <= 1
        if kp[0] <= kp[1] and kp[0]**2 + kp[1]**2 <= 1.0:
            k_points.append(kp)
    
    k_points = k_points[:30]
    
    # Precompute G vectors: 289 total, |mx|<=13, |my|<=13
    g_vectors = []
    for mx in range(-13, 14):
        for my in range(-13, 14):
            g_vectors.append((mx, my))
    n_g = len(g_vectors)  # 289
    
    # Precompute G vector squared magnitudes
    g_sq = np.array([mx*mx + my*my for mx, my in g_vectors], dtype=float)
    
    # Precompute k+G vectors for each k-point
    # For each k-point, compute (kx+mx)^2 + (ky+my)^2 for all G
    kg_sq = np.zeros((len(k_points), n_g), dtype=float)
    for ki, (kx, ky) in enumerate(k_points):
        for gi, (mx, my) in enumerate(g_vectors):
            kg_sq[ki, gi] = (kx + mx)**2 + (ky + my)**2
    
    def compute_fourier_coefficients(cell):
        """Compute Fourier coefficients of the dielectric distribution.
        Returns array of shape (n_g,) with complex coefficients.
        eps(x,y) = eps_low + eps_contrast * sum_{G} c_G * exp(i*G.x)
        where c_G is the Fourier coefficient of the binary pixel map.
        """
        # For a pixelated map, the Fourier coefficient is:
        # c_G = (1/N^2) * sum_{i,j} pixel[i][j] * exp(-2*pi*i*(mx*(i+0.5)/N + my*(j+0.5)/N))
        # We can use FFT for speed, but need to be careful about the pixel center offset.
        
        # Actually, for the plane-wave method with pixelated dielectric:
        # The dielectric function is piecewise constant over each pixel.
        # The Fourier coefficient is the integral over the unit cell:
        # c_G = (1/N^2) * sum_{i,j} pixel[i][j] * integral over pixel of exp(-i*G.x) dA
        # For pixel (i,j) centered at ((i+0.5)/N, (j+0.5)/N):
        # integral = exp(-i*G.x_center) * sinc(Gx/(2N)) * sinc(Gy/(2N)) * (1/N^2)
        # where sinc(x) = sin(x)/x
        
        # For simplicity and speed, use the point-sample approximation:
        # c_G ≈ (1/N^2) * sum_{i,j} pixel[i][j] * exp(-2*pi*i*(mx*(i+0.5)/N + my*(j+0.5)/N))
        
        cell_arr = np.array(cell, dtype=float)
        
        # Use FFT approach:
        # First, shift the pixel map so that pixel centers are at integer positions
        # The pixel at (i,j) has center at (i+0.5, j+0.5) in pixel coordinates
        # In unit cell coordinates: ((i+0.5)/N, (j+0.5)/N)
        
        # c_G = (1/N^2) * sum_{i,j} pixel[i][j] * exp(-2*pi*i*(mx*(i+0.5)/N + my*(j+0.5)/N))
        #      = (1/N^2) * exp(-2*pi*i*(mx*0.5/N + my*0.5/N)) * sum_{i,j} pixel[i][j] * exp(-2*pi*i*(mx*i/N + my*j/N))
        #      = (1/N^2) * exp(-pi*i*(mx+my)/N) * FFT(pixel)[-mx, -my]  (with proper indexing)
        
        # Let's just compute directly for accuracy, but use numpy for speed
        coeffs = np.zeros(n_g, dtype=complex)
        
        for gi, (mx, my) in enumerate(g_vectors):
            if mx == 0 and my == 0:
                coeffs[gi] = cell_arr.sum() / (n * n)
                continue
            
            # Compute the sum
            # angle[i,j] = 2*pi*(mx*(i+0.5)/n + my*(j+0.5)/n)
            i_idx = np.arange(n)
            j_idx = np.arange(n)
            
            # For efficiency, separate the phase
            phase_i = 2.0 * math.pi * mx * (i_idx + 0.5) / n
            phase_j = 2.0 * math.pi * my * (j_idx + 0.5) / n
            
            # sum = sum_i exp(-i*phase_i) * sum_j pixel[i,j] * exp(-i*phase_j)
            # This is a rank-1 update, so we can compute it as:
            # col_sum[i] = sum_j pixel[i,j] * exp(-i*phase_j)
            # coeff = (1/N^2) * sum_i exp(-i*phase_i) * col_sum[i]
            
            exp_phase_j = np.exp(-1j * phase_j)  # shape (n,)
            # col_sum[i] = sum_j pixel[i,j] * exp_phase_j[j]
            col_sums = cell_arr @ exp_phase_j  # shape (n,)
            
            exp_phase_i = np.exp(-1j * phase_i)  # shape (n,)
            coeffs[gi] = np.sum(exp_phase_i * col_sums) / (n * n)
        
        return coeffs
    
    def solve_tm_bands(coeffs, k_idx):
        """Solve the TM band structure at k-point k_idx.
        Returns sorted array of omega^2 values (in units where a=1, c=1).
        
        For TM polarization:
        H_z satisfies: (k+G)^2 * H_{k+G} = (1/eps_G) * sum_{G'} H_{k+G'}
        where 1/eps_G is the Fourier transform of 1/eps(x,y).
        
        Actually, for TM in 2D:
        The eigenvalue equation is:
        (k+G)^2 * A_G = sum_{G'} (1/eps_{G-G'}) * A_{G'}
        
        where 1/eps_{G} are the Fourier coefficients of 1/eps(x,y).
        
        For a two-material system with eps_low and eps_high:
        1/eps(x,y) = 1/eps_low for air, 1/eps_high for dielectric
        1/eps(x,y) = (1/eps_low - 1/eps_high) * f(x,y) + 1/eps_high
        where f(x,y) is the binary pixel map (1 for dielectric).
        
        So: 1/eps_G = (1/eps_low - 1/eps_high) * c_G + 1/eps_high  for G != 0
              1/eps_0 = (1/eps_low - 1/eps_high) * c_0 + 1/eps_high
        
        Wait, that's not quite right. Let me think more carefully.
        
        eps(x,y) = eps_low + (eps_high - eps_low) * f(x,y)
        1/eps(x,y) is NOT simply related to f(x,y) linearly.
        
        For the plane-wave method, we need the Fourier coefficients of 1/eps(x,y).
        For a binary dielectric:
        1/eps(x,y) = 1/eps_low * (1-f(x,y)) + 1/eps_high * f(x,y)
                   = 1/eps_low + (1/eps_high - 1/eps_low) * f(x,y)
        
        So the Fourier coefficients of 1/eps are:
        inv_eps_G = (1/eps_high - 1/eps_low) * c_G  for G != 0
        inv_eps_0 = 1/eps_low + (1/eps_high - 1/eps_low) * c_0
        
        Wait, that's still not right because 1/eps is not a linear function of f when eps varies.
        
        Actually, for a piecewise-constant dielectric:
        1/eps(x,y) = 1/eps_low if f(x,y)=0, 1/eps_high if f(x,y)=1
        = 1/eps_low + (1/eps_high - 1/eps_low) * f(x,y)
        
        Yes! This IS linear in f. So:
        inv_eps_G = (1/eps_high - 1/eps_low) * c_G  for all G (including G=0)
        Wait no:
        inv_eps_0 = (1/eps_low) * (1-c_0) + (1/eps_high) * c_0
                  = 1/eps_low + (1/eps_high - 1/eps_low) * c_0
        
        And for G != 0:
        inv_eps_G = (1/eps_high - 1/eps_low) * c_G
        
        So the matrix element M_{G,G'} = (k+G)^2 * delta_{G,G'} - inv_eps_{G-G'}
        
        Actually, the standard PW formulation for TM is:
        sum_{G'} [ (k+G)^2 * delta_{G,G'} - 1/eps_{G-G'} ] * A_{G'} = omega^2 * A_G
        
        Or equivalently, we solve the generalized eigenvalue problem.
        
        Let me use the standard form:
        H_z = sum_G A_G exp(i(k+G).r)
        The wave equation gives:
        (k+G)^2 A_G = sum_{G'} (1/eps_{G-G'}) A_{G'}
        
        So the matrix is:
        M_{G,G'} = (k+G)^2 * delta_{G,G'} - inv_eps_{G-G'}
        And we solve M * A = omega^2 * A
        
        Wait, that would give (k+G)^2 A_G - sum_{G'} inv_eps_{G-G'} A_{G'} = 0
        Which means (k+G)^2 A_G = sum_{G'} inv_eps_{G-G'} A_{G'}
        
        But this is not an eigenvalue problem in the standard form. Let me reconsider.
        
        The correct formulation:
        For TM polarization in 2D photonic crystals:
        The magnetic field H_z satisfies:
        (1/eps) * ∇×∇× H = (omega/c)^2 H
        
        In the PW basis, this becomes:
        sum_{G'} [ (k+G)^2 * delta_{G,G'} - (1/eps)_{G-G'} ] A_{G'} = (omega/c)^2 A_G
        
        Hmm, I need to be more careful. Let me use the standard formulation from Joannopoulos et al.
        
        For TM:
        (k+G)^2 A_G = sum_{G'} (1/eps_{G-G'}) A_{G'}
        
        This can be written as:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - 1/eps_{G-G'} ] A_{G'} = 0
        
        This is a homogeneous system, which only has non-trivial solutions when det = 0.
        But we want eigenvalues, so let me rewrite:
        
        Multiply both sides by 1/(k+G)^2:
        A_G = sum_{G'} [ 1/eps_{G-G'} / (k+G)^2 ] A_{G'}
        
        So the eigenvalue problem is:
        sum_{G'} [ 1/eps_{G-G'} / (k+G)^2 ] A_{G'} = lambda * A_G
        where lambda = 1/(omega/c)^2... 
        
        Actually, let me just use the standard matrix formulation:
        
        The eigenvalue equation for TM is:
        (k+G)^2 A_G = sum_{G'} (1/eps_{G-G'}) A_{G'}
        
        Rearranging:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - 1/eps_{G-G'} ] A_{G'} = 0
        
        This is not a standard eigenvalue problem. The standard approach is to write it as:
        
        D * A = omega^2 * A
        
        where D_{G,G'} = (k+G)^2 delta_{G,G'} * (1/eps_{G-G'})^{-1} ... 
        
        No, let me look at this differently. The standard PW method for TM in 2D:
        
        The eigenvalue equation is:
        (k+G)^2 A_G = sum_{G'} S_{G,G'} A_{G'}
        
        where S_{G,G'} = (1/eps)_{G-G'}
        
        This is a generalized eigenvalue problem:
        A * A = omega^2 * B * A
        
        where A_{G,G'} = (k+G)^2 delta_{G,G'} and B_{G,G'} = (1/eps)_{G-G'}
        
        So: A^{-1} B A = omega^2 A
        
        Or equivalently, we can solve:
        B A = omega^2 A^{-1} A
        
        The standard approach is to solve the matrix equation:
        sum_{G'} (1/eps_{G-G'}) A_{G'} = (k+G)^2 A_G
        
        Which can be written as a matrix eigenvalue problem by defining:
        M_{G,G'} = (1/eps_{G-G'}) / (k+G)^2
        
        Then: sum_{G'} M_{G,G'} A_{G'} = A_G
        
        No, that gives eigenvalue 1.
        
        Let me just use the direct approach:
        Define the matrix H where H_{G,G'} = (1/eps_{G-G'})
        Then the eigenvalue equation is:
        H A = (k+G)^2 * A  (component-wise)
        
        This is a generalized eigenvalue problem: H A = D A
        where D is diagonal with D_{G,G} = (k+G)^2
        
        So: D^{-1} H A = A
        Or: H D^{-1} A = D A  ... this is getting confusing.
        
        Let me use the standard formulation from the literature:
        
        For TM polarization:
        (k+G)^2 A_G = sum_{G'} (1/eps_{G-G'}) A_{G'}
        
        Let's define B_{G,G'} = (1/eps_{G-G'})
        Then: (k+G)^2 A_G = sum_{G'} B_{G,G'} A_{G'}
        
        This is: D A = B A  where D is diagonal with D_{G,G} = (k+G)^2
        
        So: (D - B) A = 0
        
        For non-trivial solutions, we need det(D - B) = 0, but that's not an eigenvalue problem.
        
        The correct way: multiply both sides by D^{-1}:
        A = D^{-1} B A
        
        So the eigenvalue problem is: D^{-1} B A = A, which means all eigenvalues are 1.
        That's not right either.
        
        OK let me think about this more carefully. The dispersion relation is:
        (k+G)^2 = (1/A_G) * sum_{G'} (1/eps_{G-G'}) A_{G'}
        
        The eigenvalue omega^2 is what we're looking for. In the PW method:
        omega^2 = (k+G)^2 for each component, but coupled through the dielectric.
        
        The standard formulation is:
        sum_{G'} [ (k+G)^2 delta_{G,G'} - 1/eps_{G-G'} ] A_{G'} = omega^2 * 0
        
        No, that's not right. Let me look at this from first principles.
        
        Maxwell's equations for TM (H_z only):
        ∇^2 H_z + (omega/c)^2 * eps(x,y) * H_z = 0
        
        Wait, that's for TE. For TM:
        ∇×(∇×H) = (omega/c)^2 * eps * H
        
        In 2D with H = (0,0,H_z):
        ∇×H = (-dH_z/dy, dH_z/dx, 0)
        ∇×(∇×H) = (-d²H_z/dx², -d²H_z/dy², 0) ... 
        
        Actually, the correct equation for TM is:
        (1/eps) ∇×(∇×H) = (omega/c)^2 H
        
        For H = (0,0,H_z):
        ∇×H = (-∂H_z/∂y, ∂H_z/∂x, 0)
        ∇×(∇×H) = (-∂²H_z/∂x², -∂²H_z/∂y², 0)
        
        So: (1/eps) * (-∇²H_z) = (omega/c)^2 * H_z (in x,y components)
        
        This means: -∇²H_z = (omega/c)^2 * eps * H_z
        
        In the PW basis: H_z = sum_G A_G exp(i(k+G).r)
        -∇²H_z = sum_G (k+G)^2 A_G exp(i(k+G).r)
        
        And eps * H_z = sum_{G,G'} eps_{G-G'} A_{G'} exp(i(k+G).r)
        
        So: sum_G (k+G)^2 A_G exp(i(k+G).r) = (omega/c)^2 * sum_{G,G'} eps_{G-G'} A_{G'} exp(i(k+G).r)
        
        Wait, that's not right either. Let me be more careful.
        
        -∇²H_z = sum_G (k+G)^2 A_G exp(i(k+G).r)
        
        eps(x,y) * H_z = sum_{G'} A_{G'} exp(ik.r) * sum_{G''} eps_{G''} exp(iG''.r)
        = sum_{G',G''} eps_{G''} A_{G'} exp(i(k+G'+G'').r)
        
        Let G = G' + G'', so G' = G - G'':
        = sum_{G,G''} eps_{G''} A_{G-G''} exp(i(k+G).r)
        
        So the equation becomes:
        sum_G (k+G)^2 A_G exp(i(k+G).r) = (omega/c)^2 * sum_{G,G''} eps_{G''} A_{G-G''} exp(i(k+G).r)
        
        Therefore:
        (k+G)^2 A_G = (omega/c)^2 * sum_{G''} eps_{G''} A_{G-G''}
        
        Or:
        (k+G)^2 A_G = omega^2 * sum_{G'} eps_{G-G'} A_{G'}
        
        where omega^2 = (omega/c)^2 * c^2 (in appropriate units).
        
        So the eigenvalue equation is:
        (k+G)^2 A_G = omega^2 * sum_{G'} eps_{G-G'} A_{G'}
        
        This is a generalized eigenvalue problem:
        D A = omega^2 * E A
        
        where D_{G,G'} = (k+G)^2 delta_{G,G'} and E_{G,G'} = eps_{G-G'}
        
        To solve this, we can compute D^{-1} E and find its eigenvalues:
        D^{-1} E A = (1/omega^2) A
        
        So the eigenvalues of D^{-1} E are 1/omega^2, and omega^2 = 1/eigenvalue.
        
        Let me implement this.
        
        Actually, for numerical stability, it's better to solve the symmetric generalized eigenvalue problem.
        But for simplicity, let's use D^{-1} E.
        
        Note: D is diagonal with positive entries (k+G)^2 > 0, so D^{-1} exists.
        
        Let me implement this properly.
        """
        # Compute 1/eps Fourier coefficients
        # inv_eps_G = (1/eps_high - 1/eps_low) * c_G for G != 0
        # inv_eps_0 = 1/eps_low + (1/eps_high - 1/eps_low) * c_0
        # 
        # Wait, I need eps_{G} not 1/eps_G for the TM equation above.
        # The equation is: (k+G)^2 A_G = omega^2 * sum_{G'} eps_{G-G'} A_{G'}
        # So I need the Fourier coefficients of eps(x,y), not 1/eps(x,y).
        
        # eps(x,y) = eps_low + (eps_high - eps_low) * f(x,y)
        # eps_G = (eps_high - eps_low) * c_G for G != 0
        # eps_0 = eps_low + (eps_high - eps_low) * c_0
        
        eps_coeff = np.zeros(n_g, dtype=float)
        for gi in range(n_g):
            if g_vectors[gi] == (0, 0):
                eps_coeff[gi] = eps_low + eps_contrast * coeffs[gi].real
            else:
                eps_coeff[gi] = eps_contrast * coeffs[gi].real
        
        # Build the matrix M = D^{-1} E
        # D_{G,G} = (k+G)^2
        # E_{G,G'} = eps_{G-G'}
        
        kx, ky = k_points[k_idx]
        d_diag = kg_sq[k_idx]  # (k+G)^2 for each G
        
        # M_{G,G'} = eps_{G-G'} / (k+G)^2
        # We need to build this matrix
        
        # For efficiency, use the fact that E is a circulant-like matrix (depends on G-G')
        # M = D^{-1} E where D is diagonal
        
        # Build E matrix: E_{G,G'} = eps_coeff[G - G']
        # We need to map G-G' to the correct index in g_vectors
        
        # Create a lookup: (mx, my) -> index
        g_to_idx = {}
        for gi, (mx, my) in enumerate(g_vectors):
            g_to_idx[(mx, my)] = gi
        
        E = np.zeros((n_g, n_g), dtype=float)
        for gi in range(n_g):
            for gj in range(n_g):
                dmx = g_vectors[gi][0] - g_vectors[gj][0]
                dmy = g_vectors[gi][1] - g_vectors[gj][1]
                if (dmx, dmy) in g_to_idx:
                    E[gi, gj] = eps_coeff[g_to_idx[(dmx, dmy)]]
        
        # M = D^{-1} E
        M = E / d_diag[:, np.newaxis]
        
        # Solve eigenvalue problem
        try:
            eigvals = np.linalg.eigvalsh(M)
        except:
            eigvals = np.linalg.eigvals(M).real
        
        # omega^2 = 1/eigenvalue
        # Filter out negative or zero eigenvalues
        valid = eigvals > 1e-10
        if not np.any(valid):
            return np.zeros(5)
        
        omega2 = 1.0 / eigvals[valid]
        omega2 = np.sort(omega2)[::-1]  # descending order
        
        # We need bands 4 and 5 (0-indexed: 3 and 4)
        # Return the top 5 bands
        if len(omega2) >= 5:
            return omega2[:5]
        else:
            # Pad with zeros
            result = np.zeros(5)
            result[:len(omega2)] = omega2
            return result
    
    def compute_gap(cell):
        """Compute the 4-5 TM band gap for a given cell."""
        coeffs = compute_fourier_coefficients(cell)
        
        max_omega4 = 0.0
        min_omega5 = float('inf')
        
        for ki in range(len(k_points)):
            if not time_ok():
                return 0.0
            omega2 = solve_tm_bands(coeffs, ki)
            if len(omega2) >= 5:
                omega4 = math.sqrt(omega2[3])  # 4th band (0-indexed: 3)
                omega5 = math.sqrt(omega2[4])  # 5th band (0-indexed: 4)
                max_omega4 = max(max_omega4, omega4)
                min_omega5 = min(min_omega5, omega5)
        
        if min_omega5 == float('inf'):
            return 0.0
        
        gap = 2.0 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)
        return gap
    
    def enforce_symmetry(cell):
        """Enforce full d4 symmetry."""
        n = len(cell)
        sym_cell = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                vals = [
                    cell[i][j],
                    cell[n - 1 - i][n - 1 - j],
                    cell[j][i],
                    cell[n - 1 - j][n - 1 - i],
                ]
                sym_cell[i][j] = 1 if sum(vals) >= 2 else 0
        return sym_cell
    
    def make_cell_from_pattern(pattern_func):
        """Create a cell from a pattern function."""
        cell = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                x = (i + 0.5) / n - 0.5
                y = (j + 0.5) / n - 0.5
                cell[i][j] = 1 if pattern_func(x, y) else 0
        return enforce_symmetry(cell)
    
    def four_rod_pattern(radius, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    if dx * dx + dy * dy < radius * radius:
                        return True
            return False
        return f
    
    def ring_pattern(inner_r, outer_r):
        def f(x, y):
            r2 = x * x + y * y
            return inner_r * inner_r < r2 < outer_r * outer_r
        return f
    
    def cross_pattern(arm_width, arm_length):
        def f(x, y):
            if abs(y) < arm_width / 2 and abs(x) < arm_length / 2:
                return True
            if abs(x) < arm_width / 2 and abs(y) < arm_length / 2:
                return True
            return False
        return f
    
    def four_rings_pattern(inner_r, outer_r, offset):
        def f(x, y):
            for sx in [1, -1]:
                for sy in [1, -1]:
                    dx = x - sx * offset
                    dy = y - sy * offset
                    r2 = dx * dx + dy * dy
                    if inner_r * inner_r < r2 < outer_r * outer_r:
                        return True
            return False
        return f
    
    def rod_pattern(radius):
        def f(x, y):
            return x * x + y * y < radius * radius
        return f
    
    # Generate initial candidates
    candidates = []
    
    # Four-rod patterns
    for r in [0.08, 0.10, 0.12, 0.14, 0.16]:
        for off in [0.20, 0.22, 0.25, 0.28, 0.30]:
            candidates.append(four_rod_pattern(r, off))
    
    # Ring patterns
    for ir, orr in [(0.10, 0.20), (0.12, 0.22), (0.15, 0.25), (0.08, 0.18)]:
        candidates.append(ring_pattern(ir, orr))
    
    # Cross patterns
    for aw in [0.10, 0.15, 0.20]:
        for al in [0.35, 0.40, 0.45]:
            candidates.append(cross_pattern(aw, al))
    
    # Four rings
    for ir, orr in [(0.04, 0.10), (0.05, 0.12), (0.06, 0.14)]:
        for off in [0.22, 0.25, 0.28]:
            candidates.append(four_rings_pattern(ir, orr, off))
    
    # Single rods
    for r in [0.15, 0.18, 0.20, 0.22, 0.25]:
        candidates.append(rod_pattern(r))
    
    # Evaluate initial candidates
    best_cell = None
    best_gap = -1.0
    
    for idx, pattern_func in enumerate(candidates):
        if not time_ok():
            break
        
        cell = make_cell_from_pattern(pattern_func)
        gap = compute_gap(cell)
        
        if gap > best_gap:
            best_gap = gap
            best_cell = cell
    
    if best_cell is None:
        best_cell = make_cell_from_pattern(four_rod_pattern(0.12, 0.25))
        best_gap = compute_gap(best_cell)
    
    # Local search: flip pixel orbits to improve the gap
    if time_ok():
        current_cell = best_cell
        current_gap = best_gap
        n_local = n
        
        # Precompute all unique orbits under d4 symmetry
        orbits = []
        visited = set()
        for i in range(n_local // 2 + 1):
            for j in range(n_local // 2 + 1):
                orbit = set()
                orbit.add((i, j))
                orbit.add((n_local - 1 - i, n_local - 1 - j))
                orbit.add((j, i))
                orbit.add((n_local - 1 - j, n_local - 1 - i))
                # Remove duplicates
                orbit = tuple(sorted(orbit))
                if orbit not in visited:
                    visited.add(orbit)
                    orbits.append(orbit)
        
        random.seed(12345)
        
        for restart in range(5):
            if not time_ok():
                break
            
            # Local search from current best
            for iteration in range(200):
                if not time_ok():
                    break
                
                # Pick a random orbit to flip
                orbit_idx = random.randint(0, len(orbits) - 1)
                orbit = orbits[orbit_idx]
                
                new_cell = [row[:] for row in current_cell]
                for (oi, oj) in orbit:
                    new_cell[oi][oj] = 1 - new_cell[oi][oj]
                
                new_gap = compute_gap(new_cell)
                
                if new_gap > current_gap:
                    current_cell = new_cell
                    current_gap = new_gap
                    if current_gap > best_gap:
                        best_gap = current_gap
                        best_cell = current_cell
                else:
                    # Try a few more random orbits
                    for _ in range(3):
                        if not time_ok():
                            break
                        orbit_idx2 = random.randint(0, len(orbits) - 1)
                        orbit2 = orbits[orbit_idx2]
                        
                        new_cell2 = [row[:] for row in current_cell]
                        for (oi, oj) in orbit2:
                            new_cell2[oi][oj] = 1 - new_cell2[oi][oj]
                        
                        new_gap2 = compute_gap(new_cell2)
                        if new_gap2 > current_gap:
                            current_cell = new_cell2
                            current_gap = new_gap2
                            if current_gap > best_gap:
                                best_gap = current_gap
                                best_cell = current_cell
                            break
            
            # Restart from a different initial point
            if restart < len(candidates) - 1 and time_ok():
                start_idx = (restart + 1) % len(candidates)
                current_cell = make_cell_from_pattern(candidates[start_idx])
                current_gap = compute_gap(current_cell)
                if current_gap > best_gap:
                    best_gap = current_gap
                    best_cell = current_cell
    
    return best_cell
# EVOLVE-BLOCK-END
