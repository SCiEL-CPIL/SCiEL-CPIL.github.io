def _error_and_grad(params, n, m, p, r, lambda_reg=1e-3):
    """Compute error and gradient for a given set of parameters."""
    U = params[:n*m*r].reshape(n, m, r)
    V = params[n*m*r:n*m*r + m*p*r].reshape(m, p, r)
    W = params[n*m*r + m*p*r:].reshape(p, n, r)
    
    # Compute S_ijk = sum_r U_ijk V_jk W_ki
    S = np.einsum('ijc,jkc,kic->ijk', U, V, W)
    
    # Error is sum (S_ijk - 1)^2 + lambda * (sum U^2 + sum V^2 + sum W^2)
    err = np.sum((S - 1.0)**2) + lambda_reg * (np.sum(U**2) + np.sum(V**2) + np.sum(W**2))
    
    # Gradient
    # dE/dU_ijk = 2 * sum_k (S_ijk - 1) V_jk W_ki + 2 * lambda * U_ijk
    # dE/dV_jk = 2 * sum_i (S_ijk - 1) U_ijk W_ki + 2 * lambda * V_jk
    # dE/dW_ki = 2 * sum_j (S_ijk - 1) U_ijk V_jk + 2 * lambda * W_ki
    
    dU = 2 * np.einsum('ijk,jkc,kic->ijc', S - 1.0, V, W) + 2 * lambda_reg * U
    dV = 2 * np.einsum('ijk,ijc,kic->jkc', S - 1.0, U, W) + 2 * lambda_reg * V
    dW = 2 * np.einsum('ijk,ijc,jkc->kic', S - 1.0, U, V) + 2 * lambda_reg * W
    
    grad = np.concatenate([dU.flatten(), dV.flatten(), dW.flatten()])
    
    return err, grad
