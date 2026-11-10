    from scipy.special import gammaln
    # m = -10..10  -> index i = m + 10
    mvals = np.arange(-k, k + 1)
    am = np.abs(mvals)
    
    # Normalization N_m = sqrt(21/(4 pi) * (10-m)!/(10+m)!)
    log_N_sq = np.log(21.0 / (4.0 * np.pi)) + gammaln(k - am + 1) - gammaln(k + am + 1)
    N = np.sqrt(np.exp(log_N_sq))
    
    # Compute Associated Legendre Polynomials P_l^m(x) for all m
    # lpmv(m, v, x) with m array, x array -> shape (len(m), len(x))
    P = lpmv(am, k, x)  # Shape (21, 70)
    
    # Create meshgrid for x and phi
    # x: (70,), phi: (130,)
    # We want B[m, i, j] where i is x-index, j is phi-index
    X, Phi = np.meshgrid(x, phi, indexing='ij') # X (70, 130), Phi (70, 130)
    
    B_3d = np.zeros((n, 70, 130), dtype=np.float64)
    
    # m = 0 (index 10)
    B_3d[10, :, :] = N[10] * P[10, :, None]
    
    # m > 0 (indices 11..20)
    m_pos = np.arange(1, k + 1)
    idx_pos = m_pos + k
    # P[idx_pos] is (10, 70)
    # cos(m * phi) is (10, 130)
    # We need (10, 70, 130)
    cos_term = np.cos(m_pos[:, None, None] * Phi[None, :, :])
    B_3d[idx_pos, :, :] = np.sqrt(2.0) * N[idx_pos, None, None] * P[idx_pos, :, None] * cos_term
    
    # m < 0 (indices 0..9)
    m_neg_mag = np.arange(1, k + 1)
    idx_neg = k - m_neg_mag
    sin_term = np.sin(m_neg_mag[:, None, None] * Phi[None, :, :])
    B_3d[idx_neg, :, :] = np.sqrt(2.0) * N[idx_neg, None, None] * P[idx_neg, :, None] * sin_term
    
    # Flatten to (21, 70*130)
    B = B_3d.reshape(n, -1)
