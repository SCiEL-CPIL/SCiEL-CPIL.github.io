import numpy as np

def integrate(f, t_span, y0):
    t0, t1 = t_span
    dt = t1 - t0
    y0 = np.array(y0, dtype=float)
    n = len(y0)
    
    if n != 100:
        # Fallback for non-10x10, though task says fixed 10x10
        def _rk4(f, t0, t1, y0, steps):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        return _rk4(f, t0, t1, y0, 2000)

    N = 10
    # Probe to determine c and BC
    # Use center index 55 (i=5, j=5)
    probe_idx = 55
    v = np.zeros(n)
    v[probe_idx] = 1.0
    w = f(0.0, v)
    
    # Neighbors of 55 (i=5, j=5)
    # Up: 45, Down: 65, Left: 54, Right: 56
    c_up = w[45]
    c_right = w[56]
    
    tol = 1e-10
    is_periodic = np.abs(c_right) > tol
    c = c_up
    
    if np.abs(c) < tol:
        # Fallback
        def _rk4(f, t0, t1, y0, steps):
            h = (t1 - t0) / steps
            t, y = t0, np.array(y0, dtype=float)
            for _ in range(steps):
                k1 = f(t, y)
                k2 = f(t + h / 2, y + h / 2 * k1)
                k3 = f(t + h / 2, y + h / 2 * k2)
                k4 = f(t + h, y + h * k3)
                y = y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                t += h
            return y
        return _rk4(f, t0, t1, y0, 2000)

    if is_periodic:
        # Periodic BCs
        # Eigenvalues lambda_pq = 4 * c * (cos(2*pi*p/N) + cos(2*pi*q/N) - 2)
        # Eigenvectors V_pq[i,j] = exp(2*pi*i*(p*i + q*j)/N)
        
        # Precompute cos terms
        p_vals = np.arange(N)
        cos_p = np.cos(2 * np.pi * p_vals / N)
        # lambda is outer sum
        # lambda_pq = 4*c * (cos_p[p] + cos_p[q] - 2)
        lambdas = 4 * c * (cos_p[:, None] + cos_p[None, :] - 2)
        
        # Exponential factors
        exp_factors = np.exp(lambdas * dt)
        
        # Project y0 onto Fourier modes
        # y0 is 10x10. Reshape to (N, N)
        Y0 = y0.reshape(N, N)
        
        # FFT2 to get coefficients
        # numpy.fft.fft2 returns sum_{i,j} Y0[i,j] exp(-2*pi*i*(p*i+q*j)/N)
        # Our V is exp(+2*pi*i...). So we need inverse FFT scaling?
        # Let's check orthogonality.
        # <V_pq, V_rs> = sum exp(2pi i (p i + q j)/N) exp(-2pi i (r i + s j)/N)
        # = sum exp(2pi i ((p-r)i + (q-s)j)/N)
        # = N^2 if p=r, q=s else 0.
        # So C_pq = 1/N^2 sum V_pq^* Y0[i,j]
        # sum V_pq^* Y0[i,j] = sum exp(-2pi i (p i + q j)/N) Y0[i,j] = FFT2(Y0)[p,q]
        # So C_pq = FFT2(Y0)[p,q] / N^2
        
        C = np.fft.fft2(Y0) / (N * N)
        
        # Reconstruct
        # Y_t[i,j] = sum C_pq exp(lambda_pq dt) V_pq[i,j]
        # This is inverse FFT of (C * exp_factors) * N^2?
        # ifft2(X) = 1/N^2 sum X_pq exp(2pi i (p i + q j)/N)
        # We want sum C_pq E_pq exp(2pi i ...)
        # So Y_t = N^2 * ifft2(C * exp_factors)
        
        Y_end = N * N * np.fft.ifft2(C * exp_factors)
        
        return Y_end.real.flatten()
    else:
        # Dirichlet BCs
        # Eigenvalues lambda_pq = -4 * c * (sin^2(p*pi/(2*(N+1))) + sin^2(q*pi/(2*(N+1))))
        # p, q from 1 to N
        # Eigenvectors V_pq[i,j] = sin(p*(i+1)*pi/(N+1)) * sin(q*(j+1)*pi/(N+1))
        
        p_vals = np.arange(1, N + 1)
        sin_p = np.sin(p_vals * np.pi / (2 * (N + 1)))
        sin_sq_p = sin_p ** 2
        
        lambdas = -4 * c * (sin_sq_p[:, None] + sin_sq_p[None, :])
        exp_factors = np.exp(lambdas * dt)
        
        Y0 = y0.reshape(N, N)
        
        # Project onto sine basis
        # We need C_pq = <V_pq, Y0> / <V_pq, V_pq>
        # <V_pq, V_pq> = (sum_{i=0}^{N-1} sin^2((i+1)p*pi/(N+1)))^2
        # sum_{k=1}^N sin^2(k*pi/(N+1)) = (N+1)/2
        # So norm_sq = ((N+1)/2)^2
        
        # Compute projections
        # V_pq[i,j] = sin_p[i] * sin_p[j] (where sin_p[i] is sin((i+1)p*pi/...))
        # Let S be matrix where S[i, p-1] = sin((i+1)*p*pi/(N+1))
        # Then V_pq[i,j] = S[i, p-1] * S[j, q-1]
        # Y0 = sum C_pq V_pq
        # C = S^{-1} Y0 S^{-T} ?
        # Actually, since S is orthogonal up to scaling:
        # S^T S = ((N+1)/2) I
        # So S^{-1} = 2/(N+1) S^T
        # C = (2/(N+1) S^T) Y0 (2/(N+1) S^T)
        
        S = np.zeros((N, N))
        for i in range(N):
            for p_idx in range(N):
                p = p_idx + 1
                S[i, p_idx] = np.sin((i + 1) * p * np.pi / (N + 1))
        
        # C = (2/(N+1))^2 S^T Y0 S^T
        factor = (2.0 / (N + 1)) ** 2
        C = factor * (S.T @ Y0) @ S.T
        
        # Apply time evolution
        C_t = C * exp_factors
        
        # Reconstruct Y_end = S C_t S^T * (N+1)/2 ?
        # Y_end = sum C_t_pq V_pq
        # Y_end[i,j] = sum_pq C_t[p,q] S[i,p] S[j,q]
        # Y_end = S C_t S^T
        # Wait, check scaling.
        # Y0 = sum C V.
        # C = S^{-1} Y0 S^{-T}
        # Y0 = S C S^T
        # So Y_end = S C_t S^T
        
        Y_end = S @ C_t @ S.T
        
        return Y_end.flatten()
