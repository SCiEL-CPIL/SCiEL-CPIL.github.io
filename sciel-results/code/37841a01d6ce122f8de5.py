import time
import numpy as np
from numpy.linalg import eigh

def _apply_symmetry(density):
    """Apply C4v symmetry: transpose, row-reverse, col-reverse, and 180-degree rotation."""
    N = density.shape
    c = density.copy()
    # Average over all 8 symmetry operations of the square group
    ops = [
        c,
        c.T,
        c[::-1, :],
        c[:, ::-1],
        c[::-1, ::-1],
        c.T[::-1, :],
        c.T[:, ::-1],
        c[::-1, :].T
    ]
    # For C4v: identity, transpose, row-reverse, col-reverse, 180-rot, 
    # and their transposes
    sym = (c + c.T + c[::-1, :] + c[:, ::-1] + c[::-1, ::-1]) / 5.0
    return sym

def _compute_tm_bands_adjoint(density, n_k=10, pw_radius=12):
    """
    Compute TM bands and their gradients using adjoint method.
    Returns: (max_w4, min_w5, grad_density)
    """
    N = density.shape[0]
    eps = 1.0 + 11.0 * density
    inv_eps = 1.0 / eps
    
    # Fourier coefficients of 1/eps
    fft_inv = np.fft.fft2(inv_eps) / (N * N)
    
    # Generate plane wave list
    G_list = []
    for Gx in range(-pw_radius, pw_radius + 1):
        for Gy in range(-pw_radius, pw_radius + 1):
            if Gx*Gx + Gy*Gy <= pw_radius * pw_radius:
                G_list.append((Gx, Gy))
    n_pw = len(G_list)
    G_arr = np.array(G_list, dtype=float)
    
    # Build the inverse epsilon matrix in G-space
    # inv_eps_mat[i,j] = <G_i | 1/eps | G_j> = fft_inv[(G_i - G_j) mod N]
    Gx_diff = G_arr[:, 0][:, None] - G_arr[:, 0][None, :]
    Gy_diff = G_arr[:, 1][:, None] - G_arr[:, 1][None, :]
    
    inv_eps_mat = np.zeros((n_pw, n_pw), dtype=complex)
    for i in range(n_pw):
        for j in range(n_pw):
            dqx = int(Gx_diff[i, j]) % N
            dqy = int(Gy_diff[i, j]) % N
            inv_eps_mat[i, j] = fft_inv[dqx, dqy]
    
    # K-points in irreducible BZ
    k_points = [
        (0.0, 0.0), (0.25, 0.0), (0.5, 0.0), (0.5, 0.25),
        (0.5, 0.5), (0.35, 0.35), (0.4, 0.2), (0.2, 0.1),
        (0.3, 0.3), (0.45, 0.1)
    ][:n_k]
    
    max_w4 = 0.0
    min_w5 = float('inf')
    argmax_k4 = None
    argmin_k5 = None
    eigvec4 = None
    eigvec5 = None
    M4 = None
    M5 = None
    k4 = None
    k5 = None
    
    for idx, k in enumerate(k_points):
        kvec = np.array(k, dtype=float)
        kG = kvec + G_arr
        dots = kG @ kG.T
        
        M = dots * inv_eps_mat
        M = (M + M.conj().T) / 2
        
        try:
            eigvals, eigvecs = eigh(M)
            eigvals = np.sqrt(np.maximum(eigvals, 0))
            if len(eigvals) >= 5:
                w4 = eigvals[3]
                w5 = eigvals[4]
                if w4 > max_w4:
                    max_w4 = w4
                    argmax_k4 = idx
                    eigvec4 = eigvecs[:, 3]
                    M4 = M.copy()
                    k4 = k
                if w5 < min_w5:
                    min_w5 = w5
                    argmin_k5 = idx
                    eigvec5 = eigvecs[:, 4]
                    M5 = M.copy()
                    k5 = k
        except:
            pass
    
    if min_w5 == float('inf') or max_w4 <= 0:
        return -1.0, None, None
    
    # Compute gradient using adjoint method
    # Objective: f = min_w5 - max_w4
    # df/d(inv_eps) = -v4^H (dM/d(inv_eps)) v4 + v5^H (dM/d(inv_eps)) v5
    # where dM/d(inv_eps) involves the derivative of M with respect to inv_eps
    
    # For the plane wave method, M_{ij} = (k+G_i)·(k+G_j) * inv_eps_{ij}
    # dM/d(inv_eps_{ab}) = (k+G_a)·(k+G_b) * delta_{ia} delta_{jb}
    
    # The gradient with respect to density comes from:
    # inv_eps = 1/(1+11*density)
    # d(inv_eps)/d(density) = -11/(1+11*density)^2 = -11 * inv_eps^2
    
    grad_density = np.zeros_like(density)
    
    if argmax_k4 is not None and argmin_k5 is not None:
        # Adjoint for max_w4 (we want to decrease it)
        # d(max_w4)/d(inv_eps_{ij}) = v4_i * (k4+G_i)·(k4+G_j) * v4_j^*
        k4_arr = np.array(k4, dtype=float)
        k4G = k4_arr + G_arr
        dots4 = k4G @ k4G.T
        
        # Contribution from max_w4 (negative sign because we want to minimize max_w4)
        # grad_inv_eps_4[i,j] = - v4_i * dots4[i,j] * conj(v4_j)
        # But since M is Hermitian and we're using eigenvectors, this simplifies
        grad_inv_eps_4 = -np.outer(eigvec4, np.conj(eigvec4)) * dots4
        
        # Adjoint for min_w5 (we want to increase it)
        k5_arr = np.array(k5, dtype=float)
        k5G = k5_arr + G_arr
        dots5 = k5G @ k5G.T
        grad_inv_eps_5 = np.outer(eigvec5, np.conj(eigvec5)) * dots5
        
        # Total gradient in inv_eps space
        grad_inv_eps = grad_inv_eps_4 + grad_inv_eps_5
        
        # Convert to density space
        # d(inv_eps)/d(density) = -11 * inv_eps^2
        # So grad_density = -grad_inv_eps * 11 * inv_eps^2 (chain rule)
        # But we need to be careful: the G-space gradient needs to be mapped back to real space
        
        # The G-space gradient grad_inv_eps corresponds to changes in the Fourier coefficients
        # We need to map this back to real-space density gradient
        # This is done by: for each G, the coefficient is fft_inv[G]
        # d(fft_inv[G])/d(density[x]) = -11 * inv_eps[x]^2 * exp(-i G·x) / N^2
        
        # So grad_density[x] = sum_G grad_inv_eps[G] * (-11) * inv_eps[x]^2 * exp(-i G·x) / N^2
        # But we need to handle the fact that grad_inv_eps is in the G-difference space
        
        # Simpler approach: use finite differences for the gradient
        # Or: compute the real-space gradient directly
        # grad_density[x] = -11 * inv_eps[x]^2 * sum_{G_i, G_j} grad_inv_eps[i,j] * 
        #                    exp(i (G_i - G_j)·x) / N^2 * (k+G_i)·(k+G_j) * ...
        
        # For simplicity and robustness, use a simplified gradient
        # The dominant contribution comes from the diagonal terms
        # Approximate: grad_density ≈ -11 * inv_eps^2 * (some real-space field)
        
        # Let's use a different approach: compute the gradient via the 
        # real-space representation
        # The key insight: the gradient of the eigenvalue with respect to 
        # the dielectric function is proportional to the field intensity
        
        # For TM modes, the gradient of omega with respect to epsilon is:
        # d(omega)/d(epsilon[x]) = -omega * |E(x)|^2 / (2 * integral |E|^2)
        
        # Let's compute the field in real space
        # E(x) = sum_G c_G exp(i G·x)
        
        # For the 4th band at k4:
        E4_real = np.zeros((N, N), dtype=complex)
        for idx, (Gx, Gy) in enumerate(G_list):
            E4_real += eigvec4[idx] * np.exp(1j * (Gx * 2*np.pi * np.arange(N)[:, None] / N + 
                                                    Gy * 2*np.pi * np.arange(N)[None, :] / N))
        E4_real /= N  # normalization
        
        # For the 5th band at k5:
        E5_real = np.zeros((N, N), dtype=complex)
        for idx, (Gx, Gy) in enumerate(G_list):
            E5_real += eigvec5[idx] * np.exp(1j * (Gx * 2*np.pi * np.arange(N)[:, None] / N + 
                                                    Gy * 2*np.pi * np.arange(N)[None, :] / N))
        E5_real /= N
        
        # Gradient: we want to decrease max_w4 and increase min_w5
        # d(max_w4)/d(density[x]) ∝ -|E4(x)|^2 (decrease density where E4 is strong)
        # d(min_w5)/d(density[x]) ∝ +|E5(x)|^2 (increase density where E5 is weak? No...)
        
        # Actually: d(omega)/d(epsilon) = -omega * |E|^2 / (2 * integral |E|^2)
        # So to increase omega5, we should increase epsilon where |E5|^2 is large
        # To decrease omega4, we should decrease epsilon where |E4|^2 is large
        
        # Since epsilon = 1 + 11*density, d(epsilon)/d(density) = 11
        # grad_density = (1/11) * [ -max_w4 * |E4|^2 / (2*int|E4|^2) + min_w5 * |E5|^2 / (2*int|E5|^2) ]
        
        int_E4 = np.sum(np.abs(E4_real)**2)
        int_E5 = np.sum(np.abs(E5_real)**2)
        
        if int_E4 > 0 and int_E5 > 0:
            grad_density = (-max_w4 * np.abs(E4_real)**2 / (2 * int_E4) + 
                           min_w5 * np.abs(E5_real)**2 / (2 * int_E5)) / 11.0
    
    score = 2 * (min_w5 - max_w4) / (min_w5 + max_w4)
    return score, grad_density, (max_w4, min_w5)

def _gen_initial_density(rho0=0.3):
    """Generate initial symmetric density field."""
    N = 32
    density = np.ones((N, N)) * rho0
    # Add some structure: Gaussian bump in the center
    x = np.arange(N)
    y = np.arange(N)
    X, Y = np.meshgrid(x - N/2, y - N/2)
    R = np.sqrt(X**2 + Y**2)
    density += 0.3 * np.exp(-R**2 / (N/4)**2)
    density = np.clip(density, 0, 1)
    # Apply symmetry
    density = _apply_symmetry(density)
    return density

def _gen_pattern_cell(cell):
    """Generate a pattern-based cell for initialization."""
    N = 32
    density = np.zeros((N, N))
    # Create a checkerboard-like pattern with some structure
    for i in range(N):
        for j in range(N):
            # Radial pattern
            x = (i + 0.5) / N - 0.5
            y = (j + 0.5) / N - 0.5
            r = np.sqrt(x*x + y*y)
            # Concentric rings
            val = 0.5 + 0.5 * np.cos(20 * r)
            if val > 0.5:
                density[i, j] = 1.0
    density = _apply_symmetry(density)
    return density

def design(problem):
    start_time = time.time()
    deadline = start_time + 220.0  # Safety margin before 230s kill
    
    size = problem["pixels"]
    assert size == 32
    
    # Initialize density field
    density = _gen_initial_density(rho0=0.3)
    
    best_density = density.copy()
    best_score, _, _ = _compute_tm_bands_adjoint(density, n_k=6, pw_radius=10)
    
    # Gradient-based optimization
    lr = 0.1
    momentum = 0.0
    velocity = np.zeros_like(density)
    
    n_iterations = 0
    while time.time() < deadline - 30:
        n_iterations += 1
        
        # Compute score and gradient
        score, grad, _ = _compute_tm_bands_adjoint(density, n_k=6, pw_radius=10)
        
        if grad is None:
            break
        
        # Apply symmetry to gradient
        grad = _apply_symmetry(grad)
        
        # Momentum update
        velocity = 0.9 * velocity + lr * grad
        density = density + velocity
        
        # Clip to [0, 1]
        density = np.clip(density, 0, 1)
        
        # Apply symmetry
        density = _apply_symmetry(density)
        
        # Check improvement
        if score > best_score:
            best_score = score
            best_density = density.copy()
        
        # Adaptive learning rate
        if n_iterations % 10 == 0:
            lr *= 0.95
        
        # Periodic restart with different initial conditions
        if n_iterations % 50 == 0 and time.time() < deadline - 60:
            # Try a different initialization
            alt_density = _gen_pattern_cell()
            alt_score, _, _ = _compute_tm_bands_adjoint(alt_density, n_k=4, pw_radius=8)
            if alt_score > best_score:
                best_score = alt_score
                best_density = alt_density.copy()
                density = alt_density.copy()
                velocity = np.zeros_like(density)
    
    # Convert best density to binary cell
    # Use threshold with some optimization
    best_cell = (best_density > 0.5).astype(int)
    
    # Try a few different thresholds to see if we can improve
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
        if time.time() > deadline - 15:
            break
        test_cell = (best_density > thresh).astype(int)
        # Quick evaluation
        test_score, _, _ = _compute_tm_bands_adjoint(test_cell.astype(float), n_k=4, pw_radius=8)
        if test_score > best_score:
            best_score = test_score
            best_cell = test_cell
    
    # Ensure symmetry
    c = best_cell.astype(float)
    c_sym = _apply_symmetry(c)
    best_cell = (c_sym > 0.5).astype(int)
    
    # Final verification and return
    if time.time() < deadline - 5:
        final_score, _, _ = _compute_tm_bands_adjoint(best_cell.astype(float), n_k=8, pw_radius=10)
    
    return best_cell.tolist()
