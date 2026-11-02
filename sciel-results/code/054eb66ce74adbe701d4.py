    candidates = []
    for factor in [2]: # half-integers
        U_cand = np.round(U_real * factor) / factor
        V_cand = np.round(V_real * factor) / factor
        W_cand = np.round(W_real * factor) / factor
        T_cand = np.einsum('ar,br,cr->abc', U_cand, V_cand, W_cand)
        err = np.sum((T_cand - T_target)**2)
        candidates.append((err, U_cand, V_cand, W_cand))
        
    # Also try integers
    for factor in [1]:
        U_cand = np.round(U_real * factor) / factor
        V_cand = np.round(V_real * factor) / factor
        W_cand = np.round(W_real * factor) / factor
        T_cand = np.einsum('ar,br,cr->abc', U_cand, V_cand, W_cand)
        err = np.sum((T_cand - T_target)**2)
        candidates.append((err, U_cand, V_cand, W_cand))
        
    candidates.sort(key=lambda x: x[0])
    best_err, U_best, V_best, W_best = candidates[0]
    
    if best_err < 1e-10:
        return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
