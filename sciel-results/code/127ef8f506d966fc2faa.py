"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.linalg import null_space

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin before 120s limit
    
    # Schoolbook rank
    schoolbook_rank = n * m * p
    
    # Build the matrix representation of the tensor
    # T[i*m+j, j*p+k, k*n+i] = 1
    # We represent each rank-1 tensor e_a ⊗ e_b ⊗ e_c as a vector in R^{(nm)*(mp)*(pn)}
    # But that's huge. Instead, we use the matrix flattening approach.
    
    # The tensor M has entries M[a, b, c] where a in [0, nm), b in [0, mp), c in [0, pn)
    # M[a, b, c] = 1 if a = i*m+j, b = j'*p+k, c = k'*n+i' with j=j' and k=k' and i=i'
    # Actually: M[a, b, c] = 1 iff (a mod m) == (b div p) and (c mod n) == (a div m) and (b mod p) == (c div n)
    # Let me re-derive. a = i*m+j, b = j*p+k, c = k*n+i.
    # So j = a mod m, i = a div m, k = b mod p, j' = b div p, i' = c div n, k' = c mod n.
    # M[a,b,c] = 1 iff j == j' and k == k' and i == i'.
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build matrix B of shape (nm*mp, pn) where B[(a,b), c] = M[a,b,c]
    # Then rank of tensor = min over partitions... no.
    # The tensor rank is the minimum r such that M = sum_{t=1}^{r} u_t v_t^T w_t^T
    # where u_t in R^{nm}, v_t in R^{mp}, w_t in R^{pn}.
    # This is equivalent to: for all a, c: M[a,:,c] = sum_t u_t[a] v_t^T w_t[c]
    # i.e., M_ac = sum_t (u_t[a] * w_t[c]) * v_t
    # So for fixed a,c, M[a,:,c] is a vector in R^{mp} that lies in the span of {v_t}.
    
    # The border rank approach: the tensor rank >= max over all (a,c) of the rank of M[a,:,c]
    # But that's just 1 for most entries.
    
    # A better approach: use the fact that tensor rank >= border rank >= spectral norm related bounds.
    
    # Let's try a direct search: try to find rank r decompositions using iterative methods.
    # For each candidate rank r, we try to find U, V, W such that M = sum_t U[:,t] ⊗ V[:,t] ⊗ W[:,t]
    
    # First, let's establish a lower bound.
    # The matricization M_{ac -> b} has a certain rank structure.
    # Consider the "slice" matrices: for each fixed a, M[a] is an mp x pn matrix.
    # M[a] = sum_t u_t[a] v_t w_t^T
    # So M[a] lies in the span of {v_t w_t^T}.
    
    # Similarly for fixed c: M[:,c] is nm x mp, lying in span of {u_t v_t^T}.
    
    # Let's use the following: the tensor rank r satisfies that the matrix
    # [M_{a,b,c}] can be written as sum of r rank-1 tensors.
    
    # Practical approach: try increasing r from a lower bound up to schoolbook_rank,
    # and for each r, try to solve using alternating least squares or similar.
    
    # Lower bound: the rank of any matricization gives a lower bound on border rank.
    # M_{ab -> c} has rank at most... let's compute.
    
    # Actually, let's just try to find the minimal rank directly.
    # The schoolbook gives rank 48. Can we do better?
    
    # For small tensors, we can use the following approach:
    # 1. Compute the "tensor rank" lower bound via matricization ranks.
    # 2. Try to construct a decomposition of that rank using linear algebra.
    
    # Matricization ranks:
    # M_{ab->c}: flatten (a,b) to nm*mp, c stays. Matrix of shape (nm*mp, pn).
    # M_{ac->b}: shape (nm*pn, mp)
    # M_{bc->a}: shape (mp*pn, nm)
    
    # Let's build the full tensor as a 3D array first.
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    # Compute matricization ranks for lower bounds
    M_ab_c = T.reshape(nm * mp, pn)
    M_ac_b = T.transpose(0, 2, 1).reshape(nm * pn, mp)
    M_bc_a = T.transpose(1, 2, 0).reshape(mp * pn, nm)
    
    rank_ab_c = np.linalg.matrix_rank(M_ab_c)
    rank_ac_b = np.linalg.matrix_rank(M_ac_b)
    rank_bc_a = np.linalg.matrix_rank(M_bc_a)
    
    lower_bound = max(rank_ab_c, rank_ac_b, rank_bc_a)
    
    # Try ranks from lower_bound to schoolbook_rank
    for r in range(lower_bound, schoolbook_rank + 1):
        if time.time() > deadline - 5.0:
            break
        result = _try_rank(T, r, deadline)
        if result is not None:
            U, V, W = result
            # Convert to list of lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook
    U = np.zeros((nm, schoolbook_rank))
    V = np.zeros((mp, schoolbook_rank))
    W = np.zeros((pn, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1.0
                V[b, col] = 1.0
                W[c, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}


def _try_rank(T, r, deadline):
    """Try to find a rank-r decomposition of tensor T."""
    nm, mp, pn = T.shape
    t0 = time.time()
    
    # Strategy: use the fact that for rank r, we need U (nm x r), V (mp x r), W (pn x r)
    # such that T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
    
    # Use iterative approach: fix two matrices, solve for the third.
    # This is the "multiplicative update" or ALS approach.
    
    # Initialize
    rng = np.random.default_rng(42)
    best_err = np.inf
    best_result = None
    
    n_restarts = 3
    max_iter = 200
    
    for restart in range(n_restarts):
        if time.time() > deadline - 2.0:
            break
        U = rng.standard_normal((nm, r))
        V = rng.standard_normal((mp, r))
        W = rng.standard_normal((pn, r))
        
        # Normalize to avoid numerical issues
        for t in range(r):
            norm = np.linalg.norm(U[:, t])
            if norm > 1e-10:
                U[:, t] /= norm
                V[:, t] *= norm
        for t in range(r):
            norm = np.linalg.norm(V[:, t])
            if norm > 1e-10:
                V[:, t] /= norm
                W[:, t] *= norm
        
        for iteration in range(max_iter):
            if time.time() > deadline - 2.0:
                break
            
            # Solve for U: for each a, T[a,b,c] = sum_t U[a,t] V[b,t] W[c,t]
            # Fix V, W: T[:, :, :] = sum_t U[:, t] V[:, t] W[:, t]
            # For each a: T[a, b, c] over b,c = sum_t U[a,t] (V[b,t] W[c,t])
            # So T[a] = sum_t U[a,t] (V[:, t] W[:, t]^T)
            # This is a linear system: T[a] = [V W^T]_t U[a,t]
            # Let's build: for each t, the outer product V[:,t] W[:,t]^T is mp x pn
            # We want: for each a, vec(T[a]) = sum_t U[a,t] vec(V[:,t] W[:,t]^T)
            
            # Build matrix A: (mp*pn) x r where A[:, t] = vec(V[:,t] outer W[:,t])
            A_U = np.zeros((mp * pn, r))
            for t in range(r):
                A_U[:, t] = np.outer(V[:, t], W[:, t]).flatten()
            
            # Solve: T.reshape(nm, mp*pn) = U @ A_U^T (transposed)
            # T_2d[a, bc] = sum_t U[a,t] A_U[bc, t]
            # So T_2d = U @ A_U.T
            T_2d = T.reshape(nm, mp * pn)
            U_new, residuals, rank_sol, sv = np.linalg.lstsq(A_U.T, T_2d.T, rcond=None)
            U = U_new.T
            
            # Solve for V
            A_V = np.zeros((nm * pn, r))
            for t in range(r):
                A_V[:, t] = np.outer(U[:, t], W[:, t]).flatten()
            T_2d_v = T.transpose(1, 0, 2).reshape(mp, nm * pn)
            V_new, residuals, rank_sol, sv = np.linalg.lstsq(A_V.T, T_2d_v.T, rcond=None)
            V = V_new.T
            
            # Solve for W
            A_W = np.zeros((nm * mp, r))
            for t in range(r):
                A_W[:, t] = np.outer(U[:, t], V[:, t]).flatten()
            T_2d_w = T.transpose(2, 0, 1).reshape(pn, nm * mp)
            W_new, residuals, rank_sol, sv = np.linalg.lstsq(A_W.T, T_2d_w.T, rcond=None)
            W = W_new.T
            
            # Check convergence
            err = np.linalg.norm(T - _reconstruct(U, V, W))
            if err < 1e-6:
                if err < best_err:
                    best_err = err
                    best_result = (U.copy(), V.copy(), W.copy())
                return (U, V, W)
            
            if err < best_err:
                best_err = err
                best_result = (U.copy(), V.copy(), W.copy())
    
    if best_result is not None and best_err < 1e-4:
        return best_result
    return None


def _reconstruct(U, V, W):
    """Reconstruct tensor from factors."""
    nm, r = U.shape
    mp, _ = V.shape
    pn, _ = W.shape
    T = np.zeros((nm, mp, pn), dtype=U.dtype)
    for t in range(r):
        T += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
    return T
# EVOLVE-BLOCK-END
