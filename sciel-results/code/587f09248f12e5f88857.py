<<<<<<< SEARCH
def _random_support_search(n, m, p, rank, deadline, U55, V55, W55):
    """
    Random search for a support pattern of the given rank.
    """
    U55 = np.array(U55, dtype=np.float64)
    V55 = np.array(V55, dtype=np.float64)
    W55 = np.array(W55, dtype=np.float64)
    
    for trial in range(1000):
        if time.time() > deadline:
            break
        
        idx = random.sample(range(55), rank)
        U_trial = U55[:, idx].copy()
        V_trial = V55[:, idx].copy()
        W_trial = W55[:, idx].copy()
        
        if _verify_fast(U_trial, V_trial, W_trial, n, m, p):
            if _is_valid_entries(U_trial.tolist(), V_trial.tolist(), W_trial.tolist()):
                return U_trial.tolist(), V_trial.tolist(), W_trial.tolist()
    
    return None
=======
def _random_support_search(n, m, p, rank, deadline, U55, V55, W55):
    """
    Random search for a support pattern of the given rank.
    Tries subsets of terms from the rank-55 decomposition.
    """
    U55 = np.asarray(U55, dtype=np.float64)
    V55 = np.asarray(V55, dtype=np.float64)
    W55 = np.asarray(W55, dtype=np.float64)
    
    for trial in range(2000):
        if time.time() > deadline:
            break
        
        idx = random.sample(range(55), rank)
        
        A, b = _build_linear_system(U55, V55, W55, n, m, p, idx)
        
        try:
            coeffs, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            continue
        
        if len(residuals) > 0 and residuals[0] > 1e-8:
            continue
        
        if np.any(np.abs(coeffs * 2 - np.round(coeffs * 2)) > 1e-6):
            continue
        
        U_new = U55[:, idx] * coeffs
        V_new = V55[:, idx] * coeffs
        W_new = W55[:, idx] * coeffs
        
        if _verify_fast(U_new, V_new, W_new, n, m, p):
            if _is_valid_entries(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
    
    return None
>>>>>>> REPLACE

