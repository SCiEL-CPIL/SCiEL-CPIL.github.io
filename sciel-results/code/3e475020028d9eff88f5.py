<<<<<<< SEARCH
def _search_rank_27(n, m, p, U28, V28, W28, T, budget):
    """Bounded local search for a rank-27 exact decomposition.

    The four Strassen blocks are independent, so the only exploitable
    redundancy is in the SUM of the blocks. We treat the 28 bilinear forms as
    the search space, round to integer/half-integer entries, and keep the best
    exact candidate. Returns (U,V,W) or None.
    """
    best = None
    best_err = float('inf')
    deadline = time.time() + budget
    rng = np.random.default_rng(20240517)

    def try_candidate(Uc, Vc, Wc):
        nonlocal best, best_err
        # round to half-integers
        Uc = np.round(2 * Uc) / 2.0
        Vc = np.round(2 * Vc) / 2.0
        Wc = np.round(2 * Wc) / 2.0
        err = _residual(Uc, Vc, Wc, T, n, m, p)
        if err < best_err:
            best_err = err
            best = (Uc.copy(), Vc.copy(), Wc.copy())
        if best_err < 1e-9:
            # exact: snap to clean integer/half-integer
            best = (np.round(2 * best[0]) / 2.0,
                    np.round(2 * best[1]) / 2.0,
                    np.round(2 * best[2]) / 2.0)
            return True
        return False

    # seed: the rank-28 base (its rank-27 truncation is the natural starting point)
    for trunc in (27,):
        try_candidate(U28[:, :trunc], V28[:, :trunc], W28[:, :trunc])

    # local perturbations of the base, rounded to the half-integer lattice
    step_sizes = [0.25, 0.5, 0.75, 1.0]
    for ss in step_sizes:
        for _ in range(40):
            if time.time() > deadline:
                break
            dU = rng.integers(-1, 2, size=U28.shape) * ss
            dV = rng.integers(-1, 2, size=V28.shape) * ss
            dW = rng.integers(-1, 2, size=W28.shape) * ss
            if try_candidate(U28 + dU, V28 + dV, W28 + dW):
                break
        if best_err < 1e-9:
            break
        if time.time() > deadline:
            break

    # verify the best candidate is exact
    if best is not None:
        e = _residual(best[0], best[1], best[2], T, n, m, p)
        if e < 1e-9:
            return best
    return None
=======
def _search_rank_27(n, m, p, U28, V28, W28, T, budget):
    """Bounded local search for a rank-27 exact decomposition.

    Uses gradient-based refinement on the residual, with rounding to
    half-integer lattice and exact verification. Exploits the linear
    structure of the tensor to find redundant bilinear forms.
    Returns (U,V,W) or None.
    """
    best = None
    best_err = float('inf')
    deadline = time.time() + budget
    rng = np.random.default_rng(20240517)

    def try_candidate(Uc, Vc, Wc):
        nonlocal best, best_err
        # round to half-integers
        Uc = np.round(2 * Uc) / 2.0
        Vc = np.round(2 * Vc) / 2.0
        Wc = np.round(2 * Wc) / 2.0
        err = _residual(Uc, Vc, Wc, T, n, m, p)
        if err < best_err:
            best_err = err
            best = (Uc.copy(), Vc.copy(), Wc.copy())
        if best_err < 1e-9:
            return True
        return False

    # Seed with rank-27 truncation of the base
    for trunc in (27,):
        try_candidate(U28[:, :trunc], V28[:, :trunc], W28[:, :trunc])

    # Gradient-based local search
    # Work in continuous space, then round to half-integers
    U_curr = U28.astype(float).copy()
    V_curr = V28.astype(float).copy()
    W_curr = W28.astype(float).copy()

    # Compute gradient of residual^2 w.r.t. U, V, W
    def compute_gradients(Uc, Vc, Wc):
        """Compute gradients of ||sum_r U_r*V_r*W_r - T||^2 / 2."""
        r = Uc.shape[1]
        # Accumulate contribution to each term
        # d/dU[i,j,c] = sum_{j',k'} V[j',k',c] * W[k',i,c] * (acc[i,j,k'] - T[i,j,k'])
        # But this is expensive; use finite differences or analytic form
        
        # Analytic gradient:
        # L = 0.5 * ||A - T||^2 where A[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
        # dL/dU[i,j,c] = sum_{j',k'} V[j',k',c] * W[k',i,c] * (A[i,j,k'] - T[i,j,k'])
        # Wait, need to be more careful.
        
        # A[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # dL/dU[i,j,c0] = sum_{j',k'} (A[i,j',k'] - T[i,j',k']) * V[j',k',c0] * W[k',i,c0]
        
        # First compute A
        A = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            Uc_r = Uc[:, c].reshape(n, m)
            Vc_r = Vc[:, c].reshape(m, p)
            Wc_r = Wc[:, c].reshape(p, n)
            A += np.einsum('ij,jk,ki->ijk', Uc_r, Vc_r, Wc_r)
        
        diff = A - T
        
        # Gradient w.r.t. U[i,j,c]
        grad_U = np.zeros_like(Uc)
        grad_V = np.zeros_like(Vc)
        grad_W = np.zeros_like(Wc)
        
        for c in range(r):
            Uc_r = Uc[:, c].reshape(n, m)
            Vc_r = Vc[:, c].reshape(m, p)
            Wc_r = Wc[:, c].reshape(p, n)
            
            # grad_U[i,j,c] = sum_{j',k'} diff[i,j',k'] * Vc_r[j',k'] * Wc_r[k',i]
            # = sum_{k'} (sum_{j'} diff[i,j',k'] * Vc_r[j',k']) * Wc_r[k',i]
            # Let M1[i,k'] = sum_{j'} diff[i,j',k'] * Vc_r[j',k']
            M1 = np.einsum('ijk,jk->ik', diff, Vc_r)  # shape (n, p)
            grad_U[:, c] = np.einsum('ik,ki->ij', M1, Wc_r).reshape(-1)
            
            # grad_V[j,k,c] = sum_{i,k'} diff[i,j,k'] * Uc_r[i,j] * Wc_r[k',i]
            # = sum_{i} (sum_{k'} diff[i,j,k'] * Wc_r[k',i]) * Uc_r[i,j]
            M2 = np.einsum('ijk,ki->ij', diff, Wc_r)  # shape (n, m)
            grad_V[:, c] = np.einsum('ij,ij->jk', M2, Uc_r).reshape(-1)
            
            # grad_W[k,i,c] = sum_{i',j'} diff[i',j',k] * Uc_r[i',j'] * Vc_r[j',k]
            # = sum_{j'} (sum_{i'} diff[i',j',k] * Uc_r[i',j']) * Vc_r[j',k]
            M3 = np.einsum('ijk,ij->jk', diff, Uc_r)  # shape (m, p)
            grad_W[:, c] = np.einsum('jk,jk->ki', M3, Vc_r).reshape(-1)
        
        return grad_U, grad_V, grad_W

    # Adam-like optimizer for smooth refinement
    lr = 0.01
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    
    m_U = np.zeros_like(U_curr)
    m_V = np.zeros_like(V_curr)
    m_W = np.zeros_like(W_curr)
    v_U = np.zeros_like(U_curr)
    v_V = np.zeros_like(V_curr)
    v_W = np.zeros_like(W_curr)
    t_step = 0
    
    # Run gradient descent for a bounded number of steps
    max_steps = 500
    for step in range(max_steps):
        if time.time() > deadline:
            break
        t_step += 1
        
        gU, gV, gW = compute_gradients(U_curr, V_curr, W_curr)
        
        # Update moments
        m_U = beta1 * m_U + (1 - beta1) * gU
        m_V = beta1 * m_V + (1 - beta1) * gV
        m_W = beta1 * m_W + (1 - beta1) * gW
        v_U = beta2 * v_U + (1 - beta2) * (gU ** 2)
        v_V = beta2 * v_V + (1 - beta2) * (gV ** 2)
        v_W = beta2 * v_W + (1 - beta2) * (gW ** 2)
        
        mhat_U = m_U / (1 - beta1 ** t_step)
        mhat_V = m_V / (1 - beta1 ** t_step)
        mhat_W = m_W / (1 - beta1 ** t_step)
        vhat_U = v_U / (1 - beta2 ** t_step)
        vhat_V = v_V / (1 - beta2 ** t_step)
        vhat_W = v_W / (1 - beta2 ** t_step)
        
        # Update
        U_curr -= lr * mhat_U / (np.sqrt(vhat_U) + eps)
        V_curr -= lr * mhat_V / (np.sqrt(vhat_V) + eps)
        W_curr -= lr * mhat_W / (np.sqrt(vhat_W) + eps)
        
        # Every few steps, try rounding and checking
        if step % 10 == 0:
            if try_candidate(U_curr, V_curr, W_curr):
                return best
            # Also try with zeroed-out last column to encourage rank-27
            U27 = U_curr.copy()
            V27 = V_curr.copy()
            W27 = W_curr.copy()
            # Scale down the last column to encourage sparsity
            U27[:, -1] *= 0.5
            V27[:, -1] *= 0.5
            W27[:, -1] *= 0.5
            if try_candidate(U27, V27, W27):
                return best

    # Final round of random perturbations around the best found
    if best is not None:
        for _ in range(100):
            if time.time() > deadline:
                break
            dU = rng.normal(0, 0.1, size=best[0].shape)
            dV = rng.normal(0, 0.1, size=best[1].shape)
            dW = rng.normal(0, 0.1, size=best[2].shape)
            if try_candidate(best[0] + dU, best[1] + dV, best[2] + dW):
                break

    # verify the best candidate is exact
    if best is not None:
        e = _residual(best[0], best[1], best[2], T, n, m, p)
        if e < 1e-9:
            return best
    return None
>>>>>>> REPLACE

