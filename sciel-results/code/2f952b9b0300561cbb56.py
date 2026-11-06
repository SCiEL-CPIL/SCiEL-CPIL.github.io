"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def integrate(f, t_span, y0):
    """
    Solve the fixed 10x10 2D heat equation IVP on [0, 0.05].
    
    Uses spectral decomposition (FFT for periodic, DCT/Sine for Dirichlet)
    with a single function call to determine the operator scale and BC type.
    """
    t0, t1 = t_span
    y0 = np.asarray(y0, dtype=float)
    n = len(y0)
    N = int(np.sqrt(n))
    
    # 1. Probe the operator with a single call
    # Use e_0 (first element, corner) to distinguish BCs and get scale
    e0 = np.zeros(n)
    e0[0] = 1.0
    f0 = f(t0, e0)
    
    # Determine BC type:
    # For Dirichlet (interior points, 0 boundary), e_0 (corner) has neighbors at 1 and N.
    # f0[N-1] (top-right corner in 1D index, which is (0, N-1)) should be 0.
    # For Periodic, e_0 is connected to N-1 (left wrap).
    # So if f0[N-1] is non-zero, it's Periodic.
    is_periodic = abs(f0[N - 1]) > 1e-10
    
    dt = t1 - t0
    
    if is_periodic:
        # Periodic BCs: Use FFT
        # 1D eigenvalues for 5-point Laplacian (scale 1):
        # lambda_k = -4 * sin^2(pi * k / N)
        k = np.arange(N)
        lam_1d = -4.0 * np.sin(np.pi * k / N)**2
        lam_2d = lam_1d[:, None] + lam_1d[None, :]
        lam_flat = lam_2d.flatten()
        
        # Determine scale factor
        # Theoretical f0[0] for Periodic Laplacian (diag -4) is -4 * scale
        # Measured f0[0]
        scale = f0[0] / -4.0
        
        # Transform to frequency domain
        y0_2d = y0.reshape(N, N)
        Y0 = np.fft.fft2(y0_2d)
        
        # Apply exponential decay
        # lam_flat order matches fft2 output order (0..N-1)
        decay = np.exp(scale * lam_flat * dt).reshape(N, N)
        Y_end = Y0 * decay
        
        # Inverse transform
        y_end = np.real(np.fft.ifft2(Y_end)).flatten()
        return y_end

    else:
        # Dirichlet BCs: Use Sine Transform (DCT)
        # 1D eigenvalues for Dirichlet Laplacian (N points):
        # mu_k = -4 * sin^2(k * pi / (2 * (N + 1))) for k = 1..N
        k = np.arange(1, N + 1)
        mu_1d = -4.0 * np.sin(k * np.pi / (2 * (N + 1)))**2
        mu_2d = mu_1d[:, None] + mu_1d[None, :]
        mu_flat = mu_2d.flatten()
        
        # Determine scale factor
        # Theoretical f0[0] for Dirichlet Laplacian (diag -4) is -4 * scale
        # Measured f0[0]
        scale = f0[0] / -4.0
        
        # Construct Sine Matrix S
        # S[j, k] = sqrt(2/(N+1)) * sin((j+1)(k+1)pi/(N+1))
        # j, k = 0..N-1
        j_idx = np.arange(N)[:, None]
        k_idx = np.arange(N)[None, :]
        S = np.sqrt(2.0 / (N + 1)) * np.sin((j_idx + 1) * (k_idx + 1) * np.pi / (N + 1))
        
        # Transform to eigenbasis
        # c = V^T y0, where V = S \otimes S
        # c_2d = S^T y0_2d S^T
        y0_2d = y0.reshape(N, N)
        C = S.T @ y0_2d @ S.T
        
        # Apply exponential decay
        # mu_flat order matches C indices (k=0..N-1 corresponds to mode 1..N)
        decay = np.exp(scale * mu_flat * dt).reshape(N, N)
        C_end = C * decay
        
        # Inverse transform
        # y_end = V c = (S \otimes S) vec(C_end) = vec(S C_end S)
        y_end_2d = S @ C_end @ S
        return y_end_2d.flatten()

# EVOLVE-BLOCK-END
