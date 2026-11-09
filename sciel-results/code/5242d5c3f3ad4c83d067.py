import numpy as np
import time
import sys

def design(problem):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    size = problem["pixels"] # 32
    eps_low = problem["eps_low"] # 1
    eps_high = problem["eps_high"] # 12
    
    # Initialize 16x16 quadrant
    # Try a square rod first
    q = np.zeros((16, 16), dtype=int)
    # Center is at 7.5, 7.5 in the 16x16 grid?
    # The full cell is 32x32. Center is 16,16.
    # Quadrant covers [0,16) x [0,16) of the full cell?
    # Symmetry: cell[i][j] == cell[31-i][31-j] == cell[j][i] ...
    # If we define the top-left 16x16, we can mirror it.
    # Let's define q[i][j] for i,j in 0..15.
    # Full cell C[i][j]:
    # C[i][j] = q[i % 16][j % 16] ? No, that's periodic.
    # Symmetry is about the center.
    # C[i][j] = q[i][j] for i<16, j<16.
    # C[i][j] = q[31-i][j] for i>=16, j<16.
    # C[i][j] = q[i][31-j] for i<16, j>=16.
    # C[i][j] = q[31-i][31-j] for i>=16, j>=16.
    # This ensures C is symmetric under reflection across center.
    
    def make_full(q):
        C = np.zeros((32, 32), dtype=int)
        C[:16, :16] = q
        C[16:, :16] = q[::-1, :]
        C[:16, 16:] = q[:, ::-1]
        C[16:, 16:] = q[::-1, ::-1]
        return C

    # Initial guess: Square rod
    # Radius ~ 0.2 * 32 = 6.4 pixels?
    # In quadrant, center is effectively at 8,8?
    # Let's just use a square of size 8x8 in the corner?
    # A centered rod in 32x32 corresponds to a shape in 16x16 that is "centered" at 8,8?
    # If we mirror 16x16 to 32x32, the "center" of the 32x32 is the corner (15,15) of the quadrant?
    # No.
    # C[0:16, 0:16] = q.
    # C[16:32, 0:16] = q[::-1].
    # So C[15,15] = q[15,15] and C[16,15] = q[15,15]. They are adjacent.
    # The center of the 32x32 cell is at (15.5, 15.5).
    # So q[15,15] is the pixel closest to the center.
    # A centered rod should have q[15,15]=1 and extend outwards.
    
    # Let's initialize with a circle centered at 15.5, 15.5 in the 32x32 grid.
    # In q coords, this is centered at 15.5, 15.5? No, q is 0..15.
    # Distance from (15.5, 15.5).
    for i in range(16):
        for j in range(16):
            # Map q[i][j] to full cell coord
            # The pixel q[i][j] represents the block at [i, i+1] x [j, j+1] in the top-left quadrant?
            # Actually, let's just compute the center of the pixel in the 32x32 grid.
            # Pixel (i,j) in q corresponds to (i,j) in C.
            # Center of C[i][j] is (i+0.5, j+0.5).
            # We want distance from (15.5, 15.5).
            dx = (i + 0.5) - 15.5
            dy = (j + 0.5) - 15.5
            if dx*dx + dy*dy < 6.0**2: # Radius 6
                q[i][j] = 1
    # This creates a rod centered in the full cell.

    best_score = -1.0
    best_q = q.copy()
    
    # Pre-compute G vectors and phases for speed?
    # We need \tilde{\epsilon}(G) for G in cutoff.
    # Cutoff N=11 -> G_x, G_y in {-5..5}.
    # G = (m, n) * 2pi / L? L=1 (unit cell).
    # G_x = m, G_y = n.
    
    def compute_eps_fourier(C):
        # C is 32x32 int array
        # We need \tilde{\epsilon}(G) = 1/32^2 sum C[i][j] eps(C[i][j]) exp(-i G . r_ij)
        # This is exactly the DFT if we scale correctly.
        # np.fft.fft2(C) gives sum C[i][j] exp(-2pi i (m i + n j)/32).
        # We need exp(-i (m i + n j)) where m,n are integers (since L=1, G=2pi m, but wait).
        # Plane waves are exp(i (k+G) . r).
        # G = 2pi m / L. If L=1, G = 2pi m.
        # The exponent in Fourier coeff is exp(-i G . r) = exp(-i 2pi m (x/L)).
        # Here x = i/32 * L? No, x is in units of L.
        # Pixel i covers x in [i/32, (i+1)/32].
        # Integral approx: (1/32) sum C[i] exp(-i 2pi m i/32).
        # This is just fft2(C) / 32?
        # Let's use fft2.
        F = np.fft.fft2(C.astype(float))
        # F[m][n] corresponds to frequency (m, n).
        # We need to shift to center zero frequency.
        F = np.fft.fftshift(F)
        return F

    def get_eps(Gx, Gy, F):
        # Gx, Gy are integers m, n in the shifted FFT grid.
        # Shifted grid index for m: if m >= 0, idx = m + 16. If m < 0, idx = m + 32?
        # np.fft.fftshift moves the zero freq to the center.
        # Original indices 0..31. Shifted: 16..31, 0..15.
        # So index = (m + 16) % 32.
        idx_x = (Gx + 16) % 32
        idx_y = (Gy + 16) % 32
        return F[idx_x, idx_y] / (32*32) # Normalization

    # We need to solve the eigenvalue problem.
    # N_pws = 11 -> Gx, Gy in -5..5.
    # G_list = [ (m, n) for m in -5..5 for n in -5..5 ]
    
    G_list = [(m, n) for m in range(-5, 6) for n in range(-5, 6)]
    N_pw = len(G_list)
    
    # Pre-compute G dot G and G dot G'
    G_vecs = np.array(G_list)
    G2 = np.sum(G_vecs**2, axis=1)
    G_dot_G = G_vecs @ G_vecs.T
    
    # K points
    k_points = [
        (0.0, 0.0),
        (0.5, 0.0),
        (0.5, 0.5),
        (0.25, 0.0),
        (0.0, 0.25)
    ]
    # Note: k is in units of reciprocal lattice vectors?
    # Standard: k_x = 2pi * k_norm_x.
    # In the matrix, (k-G) appears.
    # If k is given as (0.5, 0) meaning k_x = 0.5 * 2pi, then (k-G) = 2pi(0.5 - m).
    # The factor 2pi cancels out in the ratio or scales omega.
    # Let's work in units where 2pi=1 for simplicity, or keep 2pi.
    # Matrix A_{ij} = (k-G_i) . (k-G_j) ? No.
    # Equation: sum (k-G_i)^2 E_i = omega^2 sum eps(G_i-G_j) E_j.
    # So A is diagonal with (k-G_i)^2.
    # B is eps(G_i-G_j).
    
    def evaluate_gap(C):
        F = compute_eps_fourier(C)
        
        min_omega5 = float('inf')
        max_omega4 = 0.0
        
        for kx, ky in k_points:
            # k in units of 1 (so G=1 is 2pi)
            # k_vec = (kx, ky)
            # G_vecs are integers.
            # k - G = (kx - m, ky - n)
            kg = G_vecs - np.array([kx, ky]) # Shape (N, 2)
            kg2 = np.sum(kg**2, axis=1) # Shape (N,)
            
            # Matrix A (Diagonal)
            # Matrix B (Dense)
            B = np.zeros((N_pw, N_pw), dtype=complex)
            for i in range(N_pw):
                for j in range(N_pw):
                    G_diff = G_vecs[i] - G_vecs[j]
                    B[i, j] = get_eps(G_diff[0], G_diff[1], F)
            
            # Solve A E = omega^2 B E  =>  B^{-1} A E = omega^2 E
            # Or A E = lambda B E.
            # Use scipy.linalg.eig
            try:
                w, v = np.linalg.eig(np.linalg.solve(B, np.diag(kg2)))
                # w are omega^2
                w = np.sort(np.real(w)) # Should be real
                # Filter out negative or zero (spurious)
                w = w[w > 1e-6]
                
                if len(w) >= 5:
                    omega4 = np.sqrt(w[3])
                    omega5 = np.sqrt(w[4])
                    max_omega4 = max(max_omega4, omega4)
                    min_omega5 = min(min_omega5, omega5)
            except:
                continue
                
        if min_omega5 == float('inf'):
            return -1.0
            
        if min_omega5 <= max_omega4:
            return -1.0
            
        return 2 * (min_omega5 - max_omega4) / (min_omega5 + max_omega4)

    # Initial eval
    C = make_full(q)
    current_score = evaluate_gap(C)
    if current_score > best_score:
        best_score = current_score
        best_q = q.copy()
        
    # Optimization
    np.random.seed(42)
    while time.time() < deadline:
        # Pick random pixel in q
        i = np.random.randint(16)
        j = np.random.randint(16)
        
        # Flip
        old_val = q[i, j]
        q[i, j] = 1 - old_val
        
        C = make_full(q)
        new_score = evaluate_gap(C)
        
        if new_score > current_score:
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best_q = q.copy()
        else:
            q[i, j] = old_val
            
    # Return best
    return make_full(best_q).tolist()
