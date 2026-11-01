def _polish_solution(U, V, W, T_target):
    # Round to nearest 0.5
    U_r = np.round(U * 2) / 2
    V_r = np.round(V * 2) / 2
    W_r = np.round(W * 2) / 2
    
    # Check if rounded solution is still valid
    T_recon = np.einsum('ir,jr,kr->ijk', U_r, V_r, W_r)
    error = np.linalg.norm(T_recon - T_target)
    if error < 1e-9:
        return U_r, V_r, W_r
    return None
