import time
import math
import numpy as np

# EVOLVE-BLOCK-START

def design(problem):
    """
    Design a symmetric 32x32-pixel unit cell maximizing the 4-5 TM band gap.
    Strategy: Direct stochastic hill-climbing on the binary pixel representation,
    exploiting the 4-fold symmetry by optimizing only the 16x16 top-left quadrant.
    """
    size = problem["pixels"]
    n = size // 2  # 16
    n_pw = 17
    half = n_pw // 2
    
    # 1. Setup Plane Wave Basis
    # G vectors are (2*pi*mx, 2*pi*my) for mx, my in [-8, 8]
    G_indices = [(mx, my) for mx in range(-half, half + 1) for my in range(-half, half + 1)]
    G_vectors = np.array([(2.0 * math.pi * mx, 2.0 * math.pi * my) for mx, my in G_indices])
    
    # Precompute G-G' indices (mx_diff, my_diff) for the coefficient lookup
    # We need to map (mx_diff, my_diff) to an index in a lookup table
    # Max diff is 16, min is -16.
    # We will create a dictionary or a large array for 1/eps coefficients.
    # Since size is fixed at 32, the coefficients depend on the pixel map.
    
    # 2. Setup k-points (Irreducible BZ)
    # Triangle: (0,0) -> (pi,0) -> (pi,pi) -> (0,0)
    # We need 30 points.
    k_points = []
    # Gamma-X: 10 points
    for i in range(10):
        t = i / 9.0
        k_points.append((math.pi * t, 0.0))
    # X-M: 10 points
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((math.pi, math.pi * t))
    # M-Gamma: 10 points
    for i in range(1, 10):
        t = i / 9.0
        k_points.append((math.pi * (1 - t), math.pi * (1 - t)))
        
    k_points = np.array(k_points)
    
    # 3. Precomputation for Fast Evaluation
    # We need to compute the matrix M for each k.
    # M_{ij} = (k+G_i) dot (k+G_j) * f_{G_i-G_j}
    # f_G is the Fourier coefficient of 1/eps.
    
    # To speed up, we can precompute the "structure" of the matrix multiplication
    # or at least the dot products part? No, k changes.
    # However, we can precompute the list of (i, j, mx_diff, my_diff) pairs.
    # And precompute (k+G_i) dot (k+G_j) is not possible for all k easily without memory blowup,
    # but we can compute it on the fly.
    
    # Let's optimize the coefficient calculation.
    # f_{mx,my} = (1/1024) * sum_{x,y} (1/eps(x,y)) * exp(-i 2pi (mx*x + my*y)/32)
    # 1/eps is 1 for air (0) and 1/12 for dielectric (1).
    # Let's define a matrix E of 1/eps values.
    # We can use FFT? No, the grid is 32x32, and we need specific Fourier components up to 16.
    # A 32x32 FFT gives components 0..31 (or -16..15).
    # Our required range is -8..8. This fits within the FFT range.
    # Using FFT on the 32x32 grid will give us the coefficients for all 32x32 modes.
    # We just need to extract the ones where |mx|<=8 and |my|<=8.
    # This is MUCH faster than summing 1024 pixels for 289 coefficients.
    
    # 4. Define the fast evaluator
    def evaluate_quarter(q):
        """
        q: 16x16 numpy array of 0s and 1s.
        Returns the gap ratio.
        """
        # Construct full 32x32 cell
        # q is top-left.
        # cell[0:16, 0:16] = q
        # cell[0:16, 16:32] = q[:, ::-1]
        # cell[16:32, 0:16] = q[::-1, :]
        # cell[16:32, 16:32] = q[::-1, ::-1]
        
        # Create 1/eps map
        # 1/eps = 1 - (11/12) * cell
        # Let's work with the 'dielectric' map (1 where dielectric)
        # inv_eps = 1 - (11/12) * dielectric_map
        
        # Use FFT to get Fourier coefficients of inv_eps
        # Note: numpy.fft.fft2 expects input such that index (0,0) is DC.
        # The physical coordinate x = (i+0.5)/32.
        # Standard FFT assumes x = i/32.
        # The shift by 0.5 introduces a phase factor exp(-i 2pi (mx*0.5 + my*0.5)/32)
        # = exp(-i pi (mx+my)/32).
        # Let's verify this.
        # f_{mx,my} = sum_{i,j} v_{i,j} exp(-i 2pi (mx (i+0.5) + my (j+0.5))/32)
        # = exp(-i 2pi (mx+my)/64) * sum_{i,j} v_{i,j} exp(-i 2pi (mx i + my j)/32)
        # The sum part is exactly the DFT/FFT (unnormalized).
        # Numpy FFT returns sum_{i} v_i exp(-i 2pi k i / N).
        # So we can compute FFT, then multiply by the phase factor.
        
        # Construct full dielectric map
        # q is 16x16
        # top_left = q
        # top_right = q[:, ::-1]
        # bot_left = q[::-1, :]
        # bot_right = q[::-1, ::-1]
        
        # We can construct the full 32x32 array
        # But doing this for every evaluation might be slow if we allocate.
        # Let's try to do it efficiently.
        
        # Actually, we can compute the FFT of the full 32x32 map directly.
        # Or, we can exploit symmetry in the FFT domain?
        # Simpler: just build the 32x32 array. 1024 floats is small.
        
        # Build 32x32 dielectric map
        # d is 0 or 1
        # We can use np.empty and fill
        # Or just use slicing if we have a pre-allocated buffer?
        # Let's just build it.
        
        # Optimization: Pre-compute the phase factors for the FFT shift
        # phase_{mx,my} = exp(-i * pi * (mx+my) / 32)
        
        # 1. Build 32x32 dielectric map from 16x16 quarter
        # We'll use a fixed buffer to avoid allocation overhead if possible, 
        # but numpy is fast enough for 32x32.
        
        # Let's create the full map
        # q is the input
        # full_map = np.zeros((32, 32))
        # full_map[:16, :16] = q
        # full_map[:16, 16:] = q[:, ::-1]
        # full_map[16:, :16] = q[::-1, :]
        # full_map[16:, 16:] = q[::-1, ::-1]
        
        # To save time, let's just do the FFT on the constructed array.
        
        # Alternative: Compute the coefficients directly from the 16x16 quarter using the symmetry?
        # The full map is symmetric.
        # f_{mx,my} = sum_{i,j=0}^{31} v_{i,j} e^{-i 2pi (mx i + my j)/32}
        # Due to symmetry v_{i,j} = v_{31-i, 31-j} = v_{i, 31-j} = v_{31-i, j}
        # This implies f_{mx,my} is real?
        # v(x,y) = v(-x,-y) = v(-x,y) = v(x,-y) (with appropriate centering)
        # If the function is even, the Fourier coefficients are real.
        # Let's check:
        # v(i,j) = v(31-i, 31-j).
        # f_{mx,my} = sum v(i,j) e^{-i 2pi (mx i + my j)/32}
        # Let i' = 31-i, j' = 31-j.
        # sum v(i',j') e^{-i 2pi (mx (31-i') + my (31-j'))/32}
        # = e^{-i 2pi (31(mx+my))/32} sum v(i',j') e^{i 2pi (mx i' + my j')/32}
        # = e^{-i 2pi (31(mx+my))/32} f_{-mx, -my}^*
        # Since v is real, f_{-mx, -my} = f_{mx, my}^*.
        # So f_{mx,my} = e^{-i 2pi (31(mx+my))/32} f_{mx,my}
        # This implies f_{mx,my} is 0 unless the phase is 1?
        # e^{-i 2pi (31(mx+my))/32} = 1 => 31(mx+my) is a multiple of 32.
        # Since 31 and 32 are coprime, mx+my must be a multiple of 32.
        # For small mx,my, this means mx+my = 0.
        # So only coefficients where mx+my=0 are non-zero?
        # Wait, the symmetry is about the center of the cell.
        # The pixel centers are at (i+0.5)/32.
        # The center of the cell is at 0.5.
        # Symmetry: v(i,j) = v(31-i, 31-j).
        # This is inversion symmetry about (15.5, 15.5) in pixel indices?
        # No, 31-i is the mirror of i about 15.5.
        # So the function is symmetric about the center of the cell.
        # This usually means only even harmonics? Or specific combinations?
        
        # Let's stick to the FFT method, it's robust.
        
        # Construct 32x32
        # We can avoid building the full 32x32 array if we compute the sum directly?
        # No, FFT is O(N^2 log N) vs O(N^4) for direct sum.
        # 32^2 * log(32^2) ~ 1000 * 11 ~ 11000 ops.
        # Direct sum for 289 coeffs: 289 * 1024 ~ 300,000 ops.
        # FFT is faster.
        
        # Let's build the 32x32 array.
        # To minimize overhead, we can pre-allocate a buffer.
        
        # 1/eps map
        # inv_eps = 1 - (11/12) * dielectric
        
        # We'll compute the FFT of the dielectric map, then convert to inv_eps coefficients.
        # Linearity: F[1 - (11/12)D] = F[1] - (11/12)F[D]
        # F[1] is 1 at (0,0) and 0 elsewhere.
        # So we only need F[D].
        
        # Build D (32x32)
        # Using the quarter q (16x16)
        
        # We can use a pre-allocated buffer for D
        # But let's just create it.
        
        # Optimization: 
        # The evaluator calls this many times.
        # Let's try to make this as fast as possible.
        
        # 1. Build 32x32 D
        # We can use numpy slicing on a pre-allocated array if we pass it in, 
        # but let's just do it simply first.
        
        # Actually, we can compute the coefficients of D directly from the 16x16 quarter?
        # D(i,j) for i,j in 0..31.
        # D(i,j) = q[i,j] if i<16, j<16
        # D(i,j) = q[i, 31-j] if i<16, j>=16
        # D(i,j) = q[31-i, j] if i>=16, j<16
        # D(i,j) = q[31-i, 31-j] if i>=16, j>=16
        
        # F[D]_{mx,my} = sum_{i,j} D(i,j) e^{-i 2pi (mx i + my j)/32}
        # = sum_{i=0}^{15} sum_{j=0}^{15} q(i,j) [ e^{-i 2pi (mx i + my j)/32} + e^{-i 2pi (mx i + my (31-j))/32} + e^{-i 2pi (mx (31-i) + my j)/32} + e^{-i 2pi (mx (31-i) + my (31-j))/32} ]
        
        # Let's simplify the terms:
        # T1 = e^{-i 2pi (mx i + my j)/32}
        # T2 = e^{-i 2pi (mx i + my 31)/32} e^{i 2pi my j/32} = e^{-i 2pi my 31/32} e^{i 2pi my j/32}
        # T3 = e^{-i 2pi mx 31/32} e^{i 2pi mx i/32} e^{-i 2pi my j/32}
        # T4 = e^{-i 2pi (mx+my) 31/32} e^{i 2pi (mx i + my j)/32}
        
        # This allows us to compute the sum over the 16x16 quarter only!
        # This is 4x faster than building 32x32 and FFTing.
        # And we can precompute the phase factors.
        
        # Let's precompute the phases for all mx, my in [-8, 8]
        # We need to sum over i,j in 0..15 of q(i,j) * (T1+T2+T3+T4)
        
        # Let's precompute a 3D array or just compute on the fly?
        # There are 17x17 = 289 (mx,my) pairs.
        # And 16x16 = 256 (i,j) pairs.
        # Total ops: 289 * 256 * 4 ~ 300,000.
        # This is comparable to the direct 32x32 sum.
        # But we can use matrix multiplication!
        # Sum_{i,j} q(i,j) * A_{mx,my,i,j}
        # This is a tensor contraction.
        
        # Let's stick to the 32x32 FFT for now, it's easier to implement correctly and likely fast enough.
        # 32x32 FFT is very fast in numpy.
        
        # Construct 32x32
        # We'll use a pre-allocated buffer to avoid allocation.
        
        # Let's define the buffer outside the function? No, closure.
        
        # For now, just build it.
        
        # 1. Build 32x32 dielectric map
        # We can do this with a single np.empty and 4 assignments.
        
        # 2. FFT
        # 3. Extract coefficients
        # 4. Apply phase shift for pixel centers
        # 5. Convert to 1/eps coefficients
        # 6. Build matrix M for each k
        # 7. Solve eigenvalues
        # 8. Compute gap
        
        # Let's implement this.
        
        # Pre-allocate buffer for 32x32 map
        # We can't easily pre-allocate in the closure without passing it.
        # Let's just create it.
        
        # 1. Build 32x32
        # q is 16x16
        # d32 = np.zeros((32, 32))
        # d32[:16, :16] = q
        # d32[:16, 16:] = q[:, ::-1]
        # d32[16:, :16] = q[::-1, :]
        # d32[16:, 16:] = q[::-1, ::-1]
        
        # 2. FFT
        # F_d = np.fft.fft2(d32)
        
        # 3. Extract and phase shift
        # We need coefficients for mx,my in [-8, 8]
        # fftfreq(32) gives [0, 1, ..., 15, -16, ..., -1]
        # We need to map our desired mx,my to the FFT indices.
        # fft index k corresponds to frequency k if k < 16, else k-32.
        
        # Let's create a lookup for the FFT indices
        # fft_idx = np.fft.fftfreq(32) * 32
        # This gives [0, 1, ..., 15, -16, -15, ..., -1]
        
        # We want mx in [-8, 8].
        # For mx=0, idx=0
        # For mx=1, idx=1
        # ...
        # For mx=8, idx=8
        # For mx=-1, idx=31
        # For mx=-8, idx=24
        
        # Let's create a mapping array
        # mx_range = np.arange(-8, 9)
        # fft_indices_x = np.zeros(17, dtype=int)
        # for i, mx in enumerate(mx_range):
        #     if mx >= 0:
        #         fft_indices_x[i] = mx
        #     else:
        #         fft_indices_x[i] = mx + 32
        
        # Same for y.
        
        # 4. Phase shift
        # f_{mx,my} = F_d[idx_x, idx_y] * exp(-i pi (mx+my)/32) / 1024
        # Wait, numpy FFT is unnormalized.
        # sum_{i} v_i e^{-i ...} = F_d
        # The coefficient in the Fourier series is (1/N) F_d.
        # So we divide by 1024.
        
        # 5. Convert to 1/eps
        # f_inv_eps_{mx,my} = delta_{mx,my} - (11/12) f_d_{mx,my}
        
        # 6. Build Matrix
        # M_{ij} = (k+G_i) dot (k+G_j) * f_inv_eps_{G_i-G_j}
        
        # Let's precompute the G_diff indices for the 289x289 matrix.
        # This is a fixed structure.
        
        # Let's do this setup once.
        
        # Setup code (outside evaluate_quarter)
        
        # 7. Optimization:
        # The matrix M is symmetric and real (if f_inv_eps is real).
        # Is f_inv_eps real?
        # d(x,y) is real and even (symmetric about center).
        # F_d is real?
        # For a real even function, the Fourier transform is real and even.
        # So f_d is real.
        # So f_inv_eps is real.
        # So M is real and symmetric.
        # We can use np.linalg.eigvalsh.
        
        # Let's write the full code.
        
        pass # Placeholder

    # --- Precomputation ---
    
    # FFT indices for mx, my in [-8, 8]
    mx_range = np.arange(-8, 9)
    fft_indices_x = np.zeros(17, dtype=int)
    for i, mx in enumerate(mx_range):
        if mx >= 0:
            fft_indices_x[i] = mx
        else:
            fft_indices_x[i] = mx + 32
            
    fft_indices_y = fft_indices_x.copy()
    
    # Phase factors for pixel center shift
    # phase_{mx,my} = exp(-i pi (mx+my)/32)
    # Since we established f is real, the imaginary part should be 0 or negligible.
    # But let's keep it complex just in case, or take real part.
    # Actually, if the symmetry is perfect, the imaginary part is 0.
    # Let's precompute the real and imaginary parts of the phase.
    # But wait, the phase factor is applied to the FFT result.
    # F_d is real (for even real input).
    # So f_d = F_d * phase / 1024.
    # If phase is complex, f_d becomes complex.
    # But the true coefficient is real.
    # This implies that the imaginary part of F_d * phase must be 0.
    # Let's just use the real part of the final coefficient to be safe, 
    # or assume it's real and take .real.
    
    # Let's precompute the phase factors for the 17x17 grid
    phase_real = np.zeros((17, 17))
    phase_imag = np.zeros((17, 17))
    for i, mx in enumerate(mx_range):
        for j, my in enumerate(mx_range):
            angle = -math.pi * (mx + my) / 32.0
            phase_real[i, j] = math.cos(angle)
            phase_imag[i, j] = math.sin(angle)
            
    # Precompute G_diff indices for the matrix
    # G_vectors is 289x2
    # We need to map (G_i - G_j) to (mx_diff, my_diff)
    # and then to the index in the coefficient grid (0..16, 0..16)
    
    # coeff_indices[i, j] will be the linear index into the flattened 17x17 coeff array
    # Or we can just store (mx_diff, my_diff) and look up in a dict?
    # Dict lookup is slow.
    # Let's create a 289x289 array of indices into the 17x17 coeff array.
    
    # coeff_grid[i, j] corresponds to mx_range[i], my_range[j]
    # We need to map (mx_diff, my_diff) to (i, j)
    # mx_diff is in [-16, 16].
    # But our coeff grid only covers [-8, 8].
    # Wait, the plane waves are up to 8.
    # G_i is up to 8. G_j is up to 8.
    # G_i - G_j is up to 16.
    # But the dielectric map has non-zero Fourier coefficients only up to 16?
    # No, the map is band-limited to 16 pixels, so max frequency is 16.
    # But the plane wave expansion only includes G up to 8.
    # The matrix element involves f_{G_i - G_j}.
    # If G_i - G_j is outside the range of the FFT (which is 32, so -16..15),
    # we need to be careful.
    # Actually, the FFT of a 32-pixel map gives frequencies -16..15.
    # Our required G_diff is -16..16.
    # 16 is equivalent to -16 in the DFT?
    # e^{-i 2pi 16 x / 32} = e^{-i pi x} = e^{-i 2pi (-16) x / 32}.
    # So yes, 16 and -16 are the same.
    # So we can map any diff to [-16, 15].
    # But our coeff grid is only [-8, 8].
    # Do we have coefficients for 9..16?
    # Yes, the FFT gives them.
    # But the plane wave basis only has G up to 8.
    # So we only need f_{G_i - G_j} where G_i, G_j are in [-8, 8].
    # So G_i - G_j is in [-16, 16].
    # We need to extract these coefficients from the FFT.
    # The FFT gives all 32x32 coefficients.
    # We can just extract the needed ones.
    
    # Let's create a lookup table for the FFT indices for any diff in [-16, 16]
    # fft_idx_for_diff[d] = index in 0..31
    # d in -16..16
    # if d >= 0: idx = d
    # else: idx = d + 32
    
    # We will store the coefficients in a 32x32 array (or just the needed ones).
    # Let's store the full 32x32 coeff array for 1/eps.
    # It's only 1024 complex numbers.
    
    # In evaluate_quarter:
    # 1. Build 32x32 D
    # 2. FFT D -> F_d (32x32)
    # 3. Compute F_inv_eps (32x32)
    #    F_inv_eps[idx_x, idx_y] = (1 if idx_x==0 and idx_y==0 else 0) - (11/12) * F_d[idx_x, idx_y] * phase[idx_x, idx_y] / 1024
    #    Wait, the phase factor depends on the physical mx, my.
    #    The FFT index k corresponds to physical frequency k (if k<16) or k-32 (if k>=16).
    #    So we need to apply the phase factor based on the physical frequency.
    
    # Let's precompute the phase factor for all 32x32 FFT indices.
    phase_full_real = np.zeros((32, 32))
    phase_full_imag = np.zeros((32, 32))
    for ix in range(32):
        for iy in range(32):
            mx = ix if ix < 16 else ix - 32
            my = iy if iy < 16 else iy - 32
            angle = -math.pi * (mx + my) / 32.0
            phase_full_real[ix, iy] = math.cos(angle)
            phase_full_imag[ix, iy] = math.sin(angle)
            
    # Now, in evaluate_quarter:
    # F_d = fft2(D)
    # F_inv_eps = - (11/12) * F_d * (phase_full_real + i phase_full_imag) / 1024
    # Add DC term: F_inv_eps[0,0] += 1.0
    
    # Then we need to access F_inv_eps[mx_diff, my_diff] for the matrix.
    # We can just index it directly using the fft indices.
    
    # Precompute the fft indices for G_diff
    # G_diff_x = G_i_x - G_j_x
    # fft_idx_x = G_diff_x if G_diff_x >= 0 else G_diff_x + 32
    
    # Let's precompute a 289x289 array of (fft_idx_x, fft_idx_y)
    # Or just a linear index into the 32x32 array.
    
    # G_vectors is 289x2
    # G_diff = G_i - G_j
    # We can compute this for all pairs.
    
    # Let's do this precomputation now.
    
    # G_vectors is a list of tuples, let's make it an array
    # G_vectors is already an array
    # G_diff[i, j] = G_vectors[i] - G_vectors[j]
    # This is 289x289x2.
    
    # We can compute this efficiently.
    # G_diff_x = G_vectors[:, 0][:, np.newaxis] - G_vectors[:, 0][np.newaxis, :]
    # G_diff_y = G_vectors[:, 1][:, np.newaxis] - G_vectors[:, 1][np.newaxis, :]
    
    G_diff_x = G_vectors[:, 0][:, np.newaxis] - G_vectors[:, 0][np.newaxis, :]
    G_diff_y = G_vectors[:, 1][:, np.newaxis] - G_vectors[:, 1][np.newaxis, :]
    
    # Map to fft indices
    fft_idx_x = G_diff_x.copy()
    fft_idx_x[G_diff_x < 0] += 32
    
    fft_idx_y = G_diff_y.copy()
    fft_idx_y[G_diff_y < 0] += 32
    
    # Linear index into 32x32 array
    # idx = fft_idx_x * 32 + fft_idx_y
    coeff_indices = fft_idx_x * 32 + fft_idx_y
    
    # Precompute the dot products part?
    # M_{ij} = (k+G_i) dot (k+G_j) * f_{G_i-G_j}
    # (k+G_i) dot (k+G_j) = (kx+G_ix)(kx+G_jx) + (ky+G_iy)(ky+G_jy)
    # = kx^2 + kx(G_ix+G_jx) + G_ix G_jx + ky^2 + ky(G_iy+G_jy) + G_iy G_jy
    
    # We can precompute:
    # A_{ij} = G_ix G_jx + G_iy G_jy
    # B_{ij} = G_ix + G_jx
    # C_{ij} = G_iy + G_jy
    
    # Then (k+G_i) dot (k+G_j) = kx^2 + ky^2 + kx B_{ij} + ky C_{ij} + A_{ij}
    
    A = G_vectors[:, 0][:, np.newaxis] * G_vectors[:, 0][np.newaxis, :] + \
        G_vectors[:, 1][:, np.newaxis] * G_vectors[:, 1][np.newaxis, :]
    B = G_vectors[:, 0][:, np.newaxis] + G_vectors[:, 0][np.newaxis, :]
    C = G_vectors[:, 1][:, np.newaxis] + G_vectors[:, 1][np.newaxis, :]
    
    # Now the evaluator is ready.
    
    def evaluate_quarter(q):
        # 1. Build 32x32 D
        # Use a pre-allocated buffer?
        # Let's just create it.
        d32 = np.empty((32, 32), dtype=np.float64)
        d32[:16, :16] = q
        d32[:16, 16:] = q[:, ::-1]
        d32[16:, :16] = q[::-1, :]
        d32[16:, 16:] = q[::-1, ::-1]
        
        # 2. FFT
        F_d = np.fft.fft2(d32)
        
        # 3. Compute F_inv_eps
        # F_inv_eps = - (11/12) * F_d * (phase_real + i phase_imag) / 1024
        # phase is precomputed as phase_full_real, phase_full_imag
        # We can combine them into a complex array once.
        # phase_complex = phase_full_real + 1j * phase_full_imag
        # F_inv_eps = - (11/12) * F_d * phase_complex / 1024
        
        # Precompute phase_complex
        # Let's do it here to save memory? No, it's small.
        # Let's precompute it outside.
        
        # We'll use the precomputed phase_complex
        F_inv_eps = - (11.0/12.0) * F_d * phase_complex / 1024.0
        
        # Add DC term
        F_inv_eps[0, 0] += 1.0
        
        # 4. Build Matrix and Solve
        # We need to compute M for each k.
        # M = (kx^2 + ky^2) * I + kx * B + ky * C + A  (element-wise multiplied by f)
        # Wait, the term f_{G_i-G_j} multiplies the whole dot product.
        # M_{ij} = (kx^2 + ky^2 + kx B_{ij} + ky C_{ij} + A_{ij}) * f_{ij}
        
        # Let's precompute the base matrix parts
        # Base = kx * B + ky * C + A
        # M = (kx^2 + ky^2 + Base) * F_inv_eps_flattened
        
        # F_inv_eps is 32x32. We need to flatten it and index by coeff_indices.
        # coeff_indices is 289x289.
        # F_inv_eps_flat = F_inv_eps.flatten()
        # f_vals = F_inv_eps_flat[coeff_indices]  # 289x289
        
        # This is a large array (289x289 = 83k elements).
        # Doing this for each k-point (30 points) is 30 * 83k = 2.5M lookups.
        # This might be slow in Python if done naively.
        # But numpy indexing is fast.
        
        # Let's try it.
        
        max_omega_lower = 0.0
        min_omega_upper = float('inf')
        
        F_inv_eps_flat = F_inv_eps.flatten()
        
        for kx, ky in k_points:
            # Compute Base = kx * B + ky * C + A
            Base = kx * B + ky * C + A
            # Add k^2
            Base += (kx*kx + ky*ky)
            
            # Get f values
            f_vals = F_inv_eps_flat[coeff_indices]
            
            # M = Base * f_vals
            M = Base * f_vals
            
            # Ensure symmetry (numerical noise)
            # M = (M + M.T) / 2.0
            
            # Solve
            eigenvalues = np.linalg.eigvalsh(M)
            omegas = np.sqrt(np.maximum(0, eigenvalues))
            
            # Extract bands 4 and 5 (indices 3 and 4)
            if len(omegas) > 4:
                max_omega_lower = max(max_omega_lower, omegas[3])
                min_omega_upper = min(min_omega_upper, omegas[4])
                
        if min_omega_upper == float('inf'):
            return -1.0
            
        gap = min_omega_upper - max_omega_lower
        midgap = (min_omega_upper + max_omega_lower) / 2.0
        if midgap == 0:
            return 0.0
        return gap / midgap

    # Precompute phase_complex
    phase_complex = phase_full_real + 1j * phase_full_imag
    
    # 5. Optimization Loop
    start_time = time.time()
    deadline = start_time + 220.0  # 10s margin
    
    # Initial guess: 4 rods
    q = np.zeros((16, 16), dtype=np.float64)
    # Rods at (4,4), (4,12), (12,4), (12,12) in 16x16?
    # In 32x32, rods at (8,8), (8,24), etc.
    # In 16x16 quarter, we only have (0..15, 0..15).
    # The rod at (8,8) in 32x32 is at (4,4) in 16x16?
    # 32x32 index 8 -> 16x16 index 4.
    # Let's place a rod at (4,4) with radius 3.
    for i in range(16):
        for j in range(16):
            dx = i - 4
            dy = j - 4
            if dx*dx + dy*dy < 9:
                q[i, j] = 1.0
                
    best_score = evaluate_quarter(q)
    best_q = q.copy()
    
    # Hill climbing
    while time.time() < deadline:
        # Pick a random pixel to flip
        i = np.random.randint(16)
        j = np.random.randint(16)
        
        # Flip
        q[i, j] = 1.0 - q[i, j]
        
        # Evaluate
        score = evaluate_quarter(q)
        
        # Accept if better
        if score > best_score:
            best_score = score
            best_q = q.copy()
        else:
            # Revert
            q[i, j] = 1.0 - q[i, j]
            
    # Construct final cell
    cell = np.zeros((32, 32), dtype=int)
    cell[:16, :16] = best_q
    cell[:16, 16:] = best_q[:, ::-1]
    cell[16:, :16] = best_q[::-1, :]
    cell[16:, 16:] = best_q[::-1, ::-1]
    
    # Convert to nested list
    return cell.tolist()

# EVOLVE-BLOCK-END
