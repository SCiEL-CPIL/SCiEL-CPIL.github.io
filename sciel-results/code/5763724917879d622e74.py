import numpy as np

def construct():
    N = 24
    # Precompute Fourier matrix B where B[j, k] = exp(-i * 2 * pi * j * k / N)
    # Note: numpy.fft uses this convention
    j = np.arange(N)
    k = np.arange(N)
    B = np.exp(-1j * 2 * np.pi * np.outer(j, k) / N)
    
    def peak_norm(c):
        # c is a vector of +1/-1
        D = B @ c
        return np.max(np.abs(D))
    
    def local_search(c):
        c = c.copy()
        D = B @ c
        M = np.max(np.abs(D))
        improved = True
        while improved:
            improved = False
            # Compute effect of flipping each bit
            # R[k, j] = 2 * c[k] * B[j, k]
            R = 2 * c[:, np.newaxis] * B.T
            # D[None, :] - R[k, :] is the new DFT if we flip bit k
            # We want min over k of max over j of |D - R[k]|
            diffs = np.abs(D[np.newaxis, :] - R)
            peaks = np.max(diffs, axis=1)
            k_star = np.argmin(peaks)
            if peaks[k_star] < M - 1e-12:
                # Flip bit k_star
                c[k_star] *= -1
                D = D - 2 * c[k_star] * B[:, k_star] # Note: c[k_star] is now flipped, so we subtract 2 * new_c * B? 
                # Let's re-calculate D to be safe or update carefully.
                # Old D. Flip k. New D = Old D - 2 * old_c[k] * B[:, k].
                # old_c[k] was -new_c[k].
                # So New D = Old D + 2 * new_c[k] * B[:, k].
                # My update above: D = D - 2 * c[k_star] * B[:, k_star] where c is already flipped.
                # So D_new = D_old - 2 * (-c_old) * B = D_old + 2 * c_old * B. Correct.
                M = peaks[k_star]
                improved = True
        return c, M

    # Initial candidates
    candidates = []
    
    # 1. Golay-Rudin-Shapiro
    p, q = [1], [1]
    while len(p) < N:
        p, q = p + q, p + [-c for c in q]
    c_grs = np.array(p[:N])
    candidates.append(c_grs)
    
    # 2. Random restarts
    rng = np.random.RandomState(42)
    for _ in range(50):
        c_rand = rng.choice([-1, 1], size=N)
        candidates.append(c_rand)
        
    # Also try all +1 and all -1 (symmetric, so just all +1)
    candidates.append(np.ones(N))

    best_c = None
    best_M = float('inf')
    
    for c_init in candidates:
        c_opt, M_opt = local_search(c_init)
        if M_opt < best_M:
            best_M = M_opt
            best_c = c_opt
            
    # Convert to list of ints
    return list(best_c.astype(int))
