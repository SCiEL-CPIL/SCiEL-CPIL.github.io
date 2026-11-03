"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    schoolbook_rank = n * m * p  # 32
    deadline = time.time() + 140  # safety margin before 150s kill

    def build_tensor():
        T = np.zeros((n * m, m * p, p * n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1.0
        return T

    T = build_tensor()

    def verify(U, V, W, tol=1e-8):
        recon = np.einsum('at,bt,ct->abc', U, V, W)
        return np.max(np.abs(recon - T)) < tol

    def to_lists(U, V, W, r):
        U_list = [[float(U[i, t]) for t in range(r)] for i in range(n * m)]
        V_list = [[float(V[i, t]) for t in range(r)] for i in range(m * p)]
        W_list = [[float(W[i, t]) for t in range(r)] for i in range(p * n)]
        return {"U": U_list, "V": V_list, "W": W_list}

    def strassen_2x2_decomp():
        U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
        # C[i,j] -> index j*2 + i (trace convention k*n+i with n=2, k=j, i=i)
        U[0,0]=1.0; U[3,0]=1.0; V[0,0]=1.0; V[3,0]=1.0; W[0,0]=1.0; W[3,0]=1.0
        U[2,1]=1.0; U[3,1]=1.0; V[0,1]=1.0; W[1,1]=1.0; W[3,1]=-1.0
        U[0,2]=1.0; V[1,2]=1.0; V[3,2]=-1.0; W[2,2]=1.0; W[3,2]=1.0
        U[3,3]=1.0; V[2,3]=1.0; V[0,3]=-1.0; W[0,3]=1.0; W[1,3]=1.0
        U[0,4]=1.0; U[1,4]=1.0; V[3,4]=1.0; W[0,4]=-1.0; W[2,4]=1.0
        U[2,5]=1.0; U[0,5]=-1.0; V[0,5]=1.0; V[1,5]=1.0; W[3,5]=1.0
        U[1,6]=1.0; U[3,6]=-1.0; V[2,6]=1.0; V[3,6]=1.0; W[0,6]=1.0
        return U, V, W

    def build_strassen28():
        U_s, V_s, W_s = strassen_2x2_decomp()
        full_rank = 28
        U_full = np.zeros((n * m, full_rank))
        V_full = np.zeros((m * p, full_rank))
        W_full = np.zeros((p * n, full_rank))
        blocks = [
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+lk,          lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+lk,      lambda lk,li: lk*2+li),
            (lambda li,lj: li*4+lj,          lambda lj,lk: lj*4+(lk+2),      lambda lk,li: (lk+2)*2+li),
            (lambda li,lj: li*4+(lj+2),      lambda lj,lk: (lj+2)*4+(lk+2),  lambda lk,li: (lk+2)*2+li),
        ]
        for bi,(ga,gb,gc) in enumerate(blocks):
            for t in range(7):
                col = bi*7+t
                for li in range(2):
                    for lj in range(2):
                        U_full[ga(li,lj), col] = U_s[li*2+lj, t]
                for lj in range(2):
                    for lk in range(2):
                        V_full[gb(lj,lk), col] = V_s[lj*2+lk, t]
                for lk in range(2):
                    for li in range(2):
                        W_full[gc(lk,li), col] = W_s[lk*2+li, t]
        return U_full, V_full, W_full

    U28, V28, W28 = build_strassen28()
    if not verify(U28, V28, W28, tol=1e-8):
        U = np.zeros((n*m, schoolbook_rank)); V = np.zeros((m*p, schoolbook_rank)); W = np.zeros((p*n, schoolbook_rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col]=1.0; V[j*p+k, col]=1.0; W[k*n+i, col]=1.0; col+=1
        return to_lists(U, V, W, schoolbook_rank)

    # === Linear Algebraic Peeling: find linear dependencies among the 28 rank-1 tensors ===
    # The tensor T = sum_{t=0}^{27} u_t ⊗ v_t ⊗ w_t
    # A linear dependence exists if there are coefficients c_t not all zero such that
    # sum_t c_t * u_t ⊗ v_t ⊗ w_t = 0
    # This is equivalent to: for all (a,b,c) indices, sum_t c_t * U[a,t]*V[b,t]*W[c,t] = 0
    # We can set up a system: flatten each rank-1 tensor u_t⊗v_t⊗w_t into a vector in R^{n*m*m*p*p*n}
    # and find the null space of the matrix whose columns are these vectors.
    
    # More efficiently: the condition sum_t c_t * U[:,t] ⊗ V[:,t] ⊗ W[:,t] = 0
    # can be checked by: for each c index, the matrix sum_t c_t * U[:,t] * V[:,t]^T * W[c,t] = 0
    # Or: reshape the tensor and use the fact that T = sum_t outer(u_t, outer(v_t, w_t))
    # so we need sum_t c_t * u_t ⊗ (v_t ⊗ w_t) = 0, i.e., a vector in R^{n*m} ⊗ R^{m*p} ⊗ R^{p*n}
    
    # Let's build the matrix M where M[:,t] = vec(u_t ⊗ v_t ⊗ w_t)
    # This is a (n*m*m*p*p*n) x 28 matrix = 32*16*8 x 28 = 4096 x 28
    
    def find_linear_dependence(U_mat, V_mat, W_mat, r):
        """Find coefficients c such that sum_t c_t * u_t ⊗ v_t ⊗ w_t = 0."""
        # Build matrix whose column t is the vectorization of u_t ⊗ v_t ⊗ w_t
        # u_t is shape (n*m,), v_t is (m*p,), w_t is (p*n,)
        # vec(u_t ⊗ v_t ⊗ w_t) has length n*m*m*p*p*n
        dim = (n * m) * (m * p) * (p * n)
        M = np.zeros((dim, r))
        for t in range(r):
            M[:, t] = np.kron(np.kron(U_mat[:, t], V_mat[:, t]), W_mat[:, t])
        # Find null space
        try:
            U_sv, s, Vt = np.linalg.svd(M, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        # The null space corresponds to singular values ~ 0
        tol_ns = 1e-10
        null_dim = np.sum(s < tol_ns)
        if null_dim == 0:
            # Check if any singular value is very small relative to the max
            if s[0] > 0 and s[-1] / s[0] < 1e-8:
                null_dim = 1
            else:
                return None
        # The null space vectors are the last null_dim columns of Vt
        null_vec = Vt[-null_dim:]  # shape (null_dim, r)
        # Pick the one with largest norm for stability
        norms = np.linalg.norm(null_vec, axis=1)
        idx = np.argmax(norms)
        c = null_vec[idx]
        c = c / np.linalg.norm(c)
        return c

    # Try to find a linear dependence and peel it
    c_dep = find_linear_dependence(U28, V28, W28, 28)
    
    if c_dep is not None:
        # We have sum_t c_t * u_t ⊗ v_t ⊗ w_t = 0
        # This means we can eliminate one term. Find the term with largest |c_t|
        # and express it in terms of the others.
        # Actually, for a valid rank reduction, we need to find a term that can be
        # replaced. The standard approach: if sum_t c_t * T_t = 0, then
        # T_j = -sum_{t≠j} (c_t/c_j) * T_t for any j with c_j ≠ 0.
        # But this doesn't directly give a rank reduction unless we can merge.
        # 
        # Actually, the linear dependence means the 28 tensors are linearly dependent,
        # so they span a space of dimension ≤ 27. But rank is about the number of
        # rank-1 terms needed, not the dimension of the span.
        #
        # A true rank reduction requires that one of the rank-1 tensors can be expressed
        # as a sum of other rank-1 tensors that are "simpler" - but in general,
        # linear dependence of the tensors doesn't imply rank reduction.
        #
        # However, if the dependence has a specific structure where one tensor's
        # factors are linear combinations of others, we might be able to reduce.
        #
        # Let me try a different approach: use the GD refinement to see if we can
        # actually achieve rank 27 by optimizing.
        pass

    # === Refinement: try to reduce rank to 27 via optimization ===
    # Strategy: start with rank-28 Strassen, add 2 free columns, optimize to reduce
    # residual, then snap to half-integers and verify.
    
    r = 28
    free_dim = 2
    fixed_slice = slice(0, r - free_dim)

    def refine(seed, max_iters=8000):
        rng = np.random.RandomState(seed)
        U = U28.copy(); V = V28.copy(); W = W28.copy()
        U[:, r-free_dim:] = rng.randn(n*m, free_dim) * 0.3
        V[:, r-free_dim:] = rng.randn(m*p, free_dim) * 0.3
        W[:, r-free_dim:] = rng.randn(p*n, free_dim) * 0.3

        base_U = U[:, fixed_slice]; base_V = V[:, fixed_slice]; base_W = W[:, fixed_slice]
        base = np.einsum('at,bt,ct->abc', base_U, base_V, base_W)

        # Smooth surrogate: log-sum-exp approximation to max|F|
        beta = 80.0

        def smooth_loss(F):
            absF = np.abs(F)
            m = absF.max()
            if m < 1e-15:
                return 0.0
            shifted = absF - m
            lse = m + np.log(np.sum(np.exp(beta * shifted))) / beta
            return lse

        def loss_and_grads(Uf, Vf, Wf):
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            loss = smooth_loss(F)
            absF = np.abs(F)
            m = absF.max()
            if m < 1e-15:
                dF = np.zeros_like(F)
            else:
                shifted = absF - m
                weights = np.exp(beta * shifted) / np.sum(np.exp(beta * shifted))
                dF = weights * np.sign(F)
            gU = np.einsum('abc,bt,ct->at', dF, Vf, Wf)
            gV = np.einsum('abc,at,ct->bt', dF, Uf, Wf)
            gW = np.einsum('abc,at,bt->ct', dF, Uf, Vf)
            return loss, gU, gV, gW

        best_loss = np.inf
        bestU = U[:, r-free_dim:].copy()
        bestV = V[:, r-free_dim:].copy()
        bestW = W[:, r-free_dim:].copy()

        for it in range(max_iters):
            if time.time() > deadline - 15:
                break
            Uf = U[:, r-free_dim:]; Vf = V[:, r-free_dim:]; Wf = W[:, r-free_dim:]
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            loss, gU, gV, gW = loss_and_grads(Uf, Vf, Wf)

            err_l2 = float(np.sum(F * F))
            lr = 0.1 / (1.0 + err_l2)
            gnorm = np.sqrt(np.sum(gU**2) + np.sum(gV**2) + np.sum(gW**2)) + 1e-12
            lr = min(lr, 0.3 / gnorm)

            U[:, r-free_dim:] -= lr * gU
            V[:, r-free_dim:] -= lr * gV
            W[:, r-free_dim:] -= lr * gW

            recon2 = base + np.einsum('at,bt,ct->abc', U[:, r-free_dim:], V[:, r-free_dim:], W[:, r-free_dim:])
            F2 = recon2 - T
            cur_loss = smooth_loss(F2)
            if cur_loss < best_loss:
                best_loss = cur_loss
                bestU = U[:, r-free_dim:].copy()
                bestV = V[:, r-free_dim:].copy()
                bestW = W[:, r-free_dim:].copy()
                if best_loss < 1e-12:
                    break

        # Final L2 polish
        U[:, r-free_dim:] = bestU; V[:, r-free_dim:] = bestV; W[:, r-free_dim:] = bestW
        for it in range(3000):
            if time.time() > deadline - 8:
                break
            Uf = U[:, r-free_dim:]; Vf = V[:, r-free_dim:]; Wf = W[:, r-free_dim:]
            recon = base + np.einsum('at,bt,ct->abc', Uf, Vf, Wf)
            F = recon - T
            gU = np.einsum('abc,bt,ct->at', F, Vf, Wf)
            gV = np.einsum('abc,at,ct->bt', F, Uf, Wf)
            gW = np.einsum('abc,at,bt->ct', F, Uf, Vf)
            err = float(np.sum(F * F))
            lr = 0.05 / (1.0 + err)
            U[:, r-free_dim:] -= lr * gU
            V[:, r-free_dim:] -= lr * gV
            W[:, r-free_dim:] -= lr * gW
            if err < 1e-16:
                break

        # Snap to half-integers and verify
        for snap_factor in [2, 1]:
            U2 = U.copy(); V2 = V.copy(); W2 = W.copy()
            U2[:, r-free_dim:] = np.round(U2[:, r-free_dim:] * snap_factor) / snap_factor
            V2[:, r-free_dim:] = np.round(V2[:, r-free_dim:] * snap_factor) / snap_factor
            W2[:, r-free_dim:] = np.round(W2[:, r-free_dim:] * snap_factor) / snap_factor
            if verify(U2, V2, W2, tol=1e-8):
                return U2, V2, W2

        # Try rounding the entire matrices
        Us = np.round(U * 2) / 2; Vs = np.round(V * 2) / 2; Ws = np.round(W * 2) / 2
        if verify(Us, Vs, Ws, tol=1e-8):
            return Us, Vs, Ws

        return None

    seeds = [0, 1, 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311, 313, 317, 331, 337, 347, 349, 353, 359, 367, 373, 379, 383, 389, 397, 401, 409, 419, 421, 431, 433, 439, 443, 449, 457, 461, 463, 467, 479, 487, 491, 499, 503, 509, 521, 523, 541, 547, 557, 563, 569, 571, 577, 587, 593, 599, 601, 607, 613, 617, 619, 631, 641, 643, 647, 653, 659, 661, 673, 677, 683, 691, 701, 709, 719, 727, 733, 739, 743, 751, 757, 761, 769, 773, 787, 797, 809, 811, 821, 823, 827, 829, 839, 853, 857, 859, 863, 877, 881, 883, 887, 907, 911, 919, 929, 937, 941, 947, 953, 967, 971, 977, 983, 991, 997]
    
    for seed in seeds:
        if time.time() > deadline - 15:
            break
        res = refine(seed)
        if res is not None:
            Uf, Vf, Wf = res
            return to_lists(Uf, Vf, Wf, r)  # rank 28 (or possibly 27 if optimization succeeds)

    # Fallback: verified rank-28 Strassen
    return to_lists(U28, V28, W28, 28)
# EVOLVE-BLOCK-END
