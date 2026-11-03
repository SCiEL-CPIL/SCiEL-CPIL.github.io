"""Matrix multiplication 2,4,5 — rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x5 multiplication.
    Uses randomised iterative refinement (gradient descent on the tensor
    identity) with multiple random restarts to try to find a lower-rank
    decomposition than the trivial n*m*p = 40.

    The tensor identity for matrix multiplication of size (n,m,p):
        For all (i,j,k): C_{ik} = sum_l A_{il} B_{lk}
    can be written as a rank-R decomposition:
        T_{ikl} = sum_c U_{(i,l),c} * V_{(k,i),c} * W_{(l,k),c}
    Wait, let me re-read the task.

    The task says: U of shape (n*m, rank), V of shape (m*p, rank), W of shape (p*n, rank).
    The output factor follows the "official trace convention W[k*n+i, r]".

    The tensor for matrix multiplication C = A * B where A is n x m and B is m x p:
    C[i,k] = sum_j A[i,j] * B[j,k]

    The decomposition must satisfy:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = delta_{i,k}  (or something similar)

    Actually, the standard formulation: we want to find U, V, W such that for all
    (i, j, k), the coefficient of A[i,j] * B[j,k] in C[i,k] equals 1. This is:
    sum_c U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c] = 1  for the (i,j,k) index... 

    Hmm, let me think more carefully. The matrix multiplication tensor T has indices
    (i, j, k) where i is row of A (and C), j is column of A / row of B, k is column
    of B (and C). The tensor entry T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).

    The rank-R decomposition means: T[i,j,k] = sum_{c=1}^{R} U[i,j,c] * V[j,k,c] * W[k,i,c]
    where U has indices (i,j,c), V has indices (j,k,c), W has indices (k,i,c).

    But the task specifies:
    - U: shape (n*m, rank) — so U[row, col] where row = i*m + j, col = c
    - V: shape (m*p, rank) — so V[row, col] where row = j*p + k, col = c
    - W: shape (p*n, rank) — so W[row, col] where row = k*n + i, col = c

    And the condition is:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1  for all i,j,k
    """
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0  # 145s to be safe
    
    # Start with the schoolbook solution (rank = n*m*p = 40)
    nmp = n * m * p
    schoolbook_U = [[0] * nmp for _ in range(n * m)]
    schoolbook_V = [[0] * nmp for _ in range(m * p)]
    schoolbook_W = [[0] * nmp for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                schoolbook_U[i * m + j][col] = 1
                schoolbook_V[j * p + k][col] = 1
                schoolbook_W[k * n + i][col] = 1
                col += 1

    best_rank = nmp
    best_result = {"U": schoolbook_U, "V": schoolbook_V, "W": schoolbook_W}

    # Precompute index arrays for the tensor identity
    # For each (i,j,k), we need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    nm = n * m
    mp = m * p
    pn = p * n

    # Build a list of (u_idx, v_idx, w_idx) for each (i,j,k)
    tensor_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                tensor_indices.append((i * m + j, j * p + k, k * n + i))

    def residual_norm(U, V, W, R):
        """Compute the residual: for each (i,j,k), |sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1|"""
        U = np.asarray(U)
        V = np.asarray(V)
        W = np.asarray(W)
        max_res = 0.0
        for (ui, vi, wi) in tensor_indices:
            prod = U[ui, :R] * V[vi, :R] * W[wi, :R]
            res = abs(np.sum(prod) - 1.0)
            if res > max_res:
                max_res = res
        return max_res

    def check_exact(U, V, W, R):
        """Check if the decomposition is exact (within tight tolerance)."""
        U = np.asarray(U, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        W = np.asarray(W, dtype=np.float64)
        for (ui, vi, wi) in tensor_indices:
            prod = U[ui, :R] * V[vi, :R] * W[wi, :R]
            if abs(np.sum(prod) - 1.0) > 1e-8:
                return False
        return True

    def polish(U, V, W, R, max_iters=200):
        """
        High-precision polish: use least-squares to refine entries so that
        sum_c U[ui,c]*V[vi,c]*W[wi,c] = 1 exactly (to machine precision).
        """
        U = np.array(U, dtype=np.float64)
        V = np.array(V, dtype=np.float64)
        W = np.array(W, dtype=np.float64)
        
        for it in range(max_iters):
            # Linearize: treat V*W as fixed, solve for U via least squares
            # For each tensor index (ui, vi, wi): sum_c U[ui,c] * (V[vi,c]*W[wi,c]) = 1
            # Build linear system: A_U * vec(U) = 1  where A_U is len(tensor_indices) x (nm * R)
            # But this is a huge system. Instead, do coordinate-wise updates.
            
            # Update U: for each ui, solve sum_c U[ui,c] * V[vi,c] * W[wi,c] = 1
            # Group by ui
            for ui in range(nm):
                # Collect all equations involving U[ui, :]
                eqs = []
                for (u_idx, v_idx, w_idx) in tensor_indices:
                    if u_idx == ui:
                        vw = V[v_idx, :R] * W[w_idx, :R]
                        eqs.append(vw)
                if eqs:
                    A = np.array(eqs)  # (num_eqs, R)
                    b = np.ones(len(eqs))
                    # Least squares
                    U[ui, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Similarly update V
            for vi in range(mp):
                eqs = []
                for (u_idx, v_idx, w_idx) in tensor_indices:
                    if v_idx == vi:
                        uw = U[u_idx, :R] * W[w_idx, :R]
                        eqs.append(uw)
                if eqs:
                    A = np.array(eqs)
                    b = np.ones(len(eqs))
                    V[vi, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # Similarly update W
            for wi in range(pn):
                eqs = []
                for (u_idx, v_idx, w_idx) in tensor_indices:
                    if w_idx == wi:
                        uv = U[u_idx, :R] * V[v_idx, :R]
                        eqs.append(uv)
                if eqs:
                    A = np.array(eqs)
                    b = np.ones(len(eqs))
                    W[wi, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        
        return U.tolist(), V.tolist(), W.tolist()

    # Try to find a lower-rank decomposition using random initializations + iterative refinement
    for rank_trial in [nmp - 1, nmp - 2, nmp - 3, nmp - 4, nmp - 5]:
        if rank_trial < 1:
            break
        if time.time() > deadline - 20:
            break
        
        R = rank_trial
        nm_rows = n * m
        mp_rows = m * p
        pn_rows = p * n
        
        # Number of equations = n*m*p = 40
        # Unknowns = R * (nm + mp + pn) = R * (8 + 20 + 10) = 38*R
        # For R = 39: 1482 unknowns, 40 equations — very underdetermined
        
        # Strategy: random init, then iterative LSQ refinement
        np.random.seed(42)
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            U = np.random.randn(nm_rows, R) * 0.1
            V = np.random.randn(mp_rows, R) * 0.1
            W = np.random.randn(pn_rows, R) * 0.1
            
            # Scale to help
            U[:, 0] += 1.0
            V[:, 0] += 1.0
            W[:, 0] += 1.0
            
            # Iterative refinement
            converged = False
            for it in range(100):
                if time.time() > deadline - 5:
                    break
                
                # Update U
                for ui in range(nm_rows):
                    eqs_vw = []
                    for (u_idx, v_idx, w_idx) in tensor_indices:
                        if u_idx == ui:
                            eqs_vw.append(V[v_idx, :R] * W[w_idx, :R])
                    if eqs_vw:
                        A = np.array(eqs_vw)
                        b = np.ones(len(eqs_vw))
                        try:
                            U[ui, :], _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                        except np.linalg.LinAlgError:
                            pass
                
                # Update V
                for vi in range(mp_rows):
                    eqs_uw = []
                    for (u_idx, v_idx, w_idx) in tensor_indices:
                        if v_idx == vi:
                            eqs_uw.append(U[u_idx, :R] * W[w_idx, :R])
                    if eqs_uw:
                        A = np.array(eqs_uw)
                        b = np.ones(len(eqs_uw))
                        try:
                            V[vi, :], _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                        except np.linalg.LinAlgError:
                            pass
                
                # Update W
                for wi in range(pn_rows):
                    eqs_uv = []
                    for (u_idx, v_idx, w_idx) in tensor_indices:
                        if w_idx == wi:
                            eqs_uv.append(U[u_idx, :R] * V[v_idx, :R])
                    if eqs_uv:
                        A = np.array(eqs_uv)
                        b = np.ones(len(eqs_uv))
                        try:
                            W[wi, :], _, _, _ = np.linalg.lstsq(A, b, rcond=1e-12)
                        except np.linalg.LinAlgError:
                            pass
                
                # Check convergence
                max_res = 0.0
                for (ui, vi, wi) in tensor_indices:
                    prod = U[ui, :R] * V[vi, :R] * W[wi, :R]
                    res = abs(np.sum(prod) - 1.0)
                    if res > max_res:
                        max_res = res
                
                if max_res < 1e-10:
                    converged = True
                    break
                
                if time.time() > deadline - 5:
                    break
            
            if converged:
                # We found a valid rank-R decomposition!
                best_rank = R
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                print(f"Found valid decomposition at rank {R}")
                # Try to go lower
                break
        
        if time.time() > deadline - 10:
            break

    return best_result
# EVOLVE-BLOCK-END

