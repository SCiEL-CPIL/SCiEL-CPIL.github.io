<<<<<<< SEARCH
def _compute_tm_gap(cell, N_pw=17, n_k=30):
    """
    Compute the TM band gap ratio between bands 4 and 5 for a given 32x32 cell.
    Uses plane-wave method with exact Fourier coefficients.
    """
    size = len(cell)
    eps = np.array(cell, dtype=np.float64)
    eps = np.where(eps > 0.5, 12.0, 1.0)
    
    # Compute inverse epsilon
    inv_eps = 1.0 / eps
    
    # Fourier coefficients of inverse epsilon
    # For square lattice, use 2D FFT
    # We need coefficients for wavevectors (m, n) where m, n in [-N_pw//2, N_pw//2]
    # But we need to be careful about the centering
    
    N = size
    # Compute DFT of inv_eps
    # The Fourier coefficient for wavevector (m, n) is:
    # c_{m,n} = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i + n*j)/N)
    
    # Use FFT for efficiency
    # Shift so that the origin is at the center
    # Actually, let's just use the standard FFT
    inv_eps_fft = np.fft.fft2(inv_eps)
    
    # The FFT gives us coefficients for wavevectors (0, 1, ..., N-1)
    # We need to map these to the range [-N/2, N/2]
    # For even N, the wavevector N/2 is the Nyquist frequency
    
    # Let's extract the needed coefficients
    # For a square lattice photonic crystal, the dielectric function is periodic
    # with period 1 (in lattice constant units)
    
    # The plane waves are indexed by (m, n) where m, n are integers
    # The wavevector is (2*pi*m, 2*pi*n) in units of 2*pi/a
    
    # We need coefficients c_{m,n} for |m|, |n| <= N_pw//2
    
    # Compute all Fourier coefficients of inv_eps for indices 0..N-1
    # c[m,n] = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i + n*j)/N)
    # This is exactly the 2D DFT / N^2
    c_full = inv_eps_fft / (N * N)  # shape (N, N), c_full[n_idx, m_idx] = c_{m,n}
    
    # Helper to get c_{m,n} for any integer m, n using periodicity
    def get_c(m, n):
        m_idx = m % N
        n_idx = n % N
        return c_full[n_idx, m_idx]
    
    # Build the plane-wave matrix
    # Number of plane waves
    n_pw = N_pw ** 2
    
    # Generate plane wave indices
    pw_indices = []
    for m in range(-N_half, N_half + 1):
        for n in range(-N_half, N_half + 1):
            pw_indices.append((m, n))
    
    n_pw = len(pw_indices)
=======
def _compute_tm_gap(cell, N_pw=17, n_k=30):
    """
    Compute the TM band gap ratio between bands 4 and 5 for a given 32x32 cell.
    Uses plane-wave method with exact Fourier coefficients.
    """
    size = len(cell)
    eps = np.array(cell, dtype=np.float64)
    eps = np.where(eps > 0.5, 12.0, 1.0)
    
    # Compute inverse epsilon
    inv_eps = 1.0 / eps
    
    N = size
    # Compute DFT of inv_eps
    inv_eps_fft = np.fft.fft2(inv_eps)
    
    # c[m,n] = (1/N^2) * sum_{i,j} inv_eps[i,j] * exp(-2*pi*i*(m*i + n*j)/N)
    # This is exactly the 2D DFT / N^2
    c_full = inv_eps_fft / (N * N)  # shape (N, N), c_full[n_idx, m_idx] = c_{m,n}
    
    # Helper to get c_{m,n} for any integer m, n using periodicity
    def get_c(m, n):
        m_idx = m % N
        n_idx = n % N
        return c_full[n_idx, m_idx]
    
    # Generate plane wave indices
    N_half = N_pw // 2
    pw_indices = []
    for m in range(-N_half, N_half + 1):
        for n in range(-N_half, N_half + 1):
            pw_indices.append((m, n))
    
    n_pw = len(pw_indices)
>>>>>>> REPLACE

