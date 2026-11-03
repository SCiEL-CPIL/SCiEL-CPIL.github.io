"""Rank-minimal exact tensor decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np

N, M, P = 3, 3, 5

def _aidx(i, j):
    return i * M + j

def _bidx(j, k):
    return j * P + k

def _cidx(k, i):
    return k * N + i

def _build_target():
    """Build the matmul tensor T[n,j,k,m] = 1 if m==n, else 0."""
    T = np.zeros((N * M, M * P, P * N))
    for i in range(N):
        for j in range(M):
            for k in range(P):
                for n in range(N):
                    if n == i:
                        T[_aidx(i, j), _bidx(j, k), _cidx(k, n)] = 1.0
    return T

def _verify(U, V, W, T, tol=1e-9):
    """Check that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    r = U.shape[1]
    S = np.zeros((N * M, M * P, P * N))
    for c in range(r):
        S += np.einsum('a,b,c->abc', U[:, c], V[:, c], W[:, c])
    return bool(np.max(np.abs(S - T)) < tol)

def _to_exact(mat):
    """Convert float matrix to exact rational (Fraction) representation."""
    from fractions import Fraction
    out = []
    for row in mat:
        r = []
        for v in row:
            if abs(v) < 1e-9:
                r.append(Fraction(0, 1))
            else:
                f = Fraction(v).limit_denominator(1 << 20)
                if abs(float(f) - v) > 1e-7:
                    return None
                r.append(f)
        out.append(r)
    return out

def _structured_rank3():
    """
    Construct a rank-3 decomposition for matmul (3x3) * (3x5).
    This is likely invalid, but kept for structure.
    """
    rank = N  # 3
    U = np.zeros((N * M, rank))
    V = np.ones((M * P, rank))
    W = np.zeros((P * N, rank))
    
    for n in range(N):
        for j in range(M):
            U[_aidx(n, j), n] = 1.0
        for k in range(P):
            W[_cidx(k, n), n] = 1.0
    
    return U, V, W

def _try_lower_rank_search(time_budget):
    """
    Attempt to find a rank < 45 decomposition via structured search.
    We try ranks 30 to 44.
    """
    T = _build_target()
    
    # The theoretical lower bound is 15, but practical decompositions are harder.
    # We try to find a solution with rank 40-44.
    for rank in range(44, 29, -1):
        if time.time() > time_budget:
            break
        
        rng = np.random.default_rng(rank * 1000)
        
        for attempt in range(100):
            if time.time() > time_budget:
                break
            
            # Random initialization
            U = rng.normal(size=(N * M, rank))
            V = rng.normal(size=(M * P, rank))
            W = rng.normal(size=(P * N, rank))
            
            # Iterative refinement (ALS)
            for _ in range(50):
                # Fix U, W, solve for V
                # T[i,j,k,n] = sum_c U[i,j,c] V[j,k,c] W[k,n,c]
                # For fixed (j,k), V[j,k,:] is unknown.
                # Sum over i,n: T[i,j,k,n] = sum_c V[j,k,c] (sum_i U[i,j,c] W[k,n,c] ? No)
                # T[i,j,k,n] = sum_c V[j,k,c] (U[i,j,c] W[k,n,c])
                # This is linear in V[j,k,c].
                # We can solve for V by minimizing ||T - sum U V W||.
                
                # Construct linear system for V
                # Unknowns: V[j,k,c] for j=0..2, k=0..4, c=0..rank-1
                # Total unknowns: 3 * 5 * rank
                # Equations: 135 (for each i,j,k,n)
                
                num_unknowns = M * P * rank
                num_eqs = N * M * P * N
                
                A_mat = np.zeros((num_eqs, num_unknowns))
                b_vec = T.flatten()
                
                eq_idx = 0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            for n in range(N):
                                for c in range(rank):
                                    # Coefficient of V[j,k,c] in T[i,j,k,n]
                                    # T[i,j,k,n] = sum_c U[i,j,c] V[j,k,c] W[k,n,c]
                                    # So coeff is U[i,j,c] * W[k,n,c]
                                    A_mat[eq_idx, j * (P * rank) + k * rank + c] = U[_aidx(i, j), c] * W[_cidx(k, n), c]
                                eq_idx += 1
                
                try:
                    sol, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    V = sol.reshape(M, P, rank)
                except:
                    continue
                
                # Similarly solve for U and W
                # Solve for U
                A_mat_u = np.zeros((num_eqs, N * M * rank))
                eq_idx = 0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            for n in range(N):
                                for c in range(rank):
                                    A_mat_u[eq_idx, i * (M * rank) + j * rank + c] = V[j, k, c] * W[_cidx(k, n), c]
                                eq_idx += 1
                try:
                    sol_u, _, _, _ = np.linalg.lstsq(A_mat_u, b_vec, rcond=None)
                    U = sol_u.reshape(N, M, rank)
                except:
                    continue
                
                # Solve for W
                A_mat_w = np.zeros((num_eqs, P * N * rank))
                eq_idx = 0
                for i in range(N):
                    for j in range(M):
                        for k in range(P):
                            for n in range(N):
                                for c in range(rank):
                                    A_mat_w[eq_idx, k * (N * rank) + n * rank + c] = U[i, j, c] * V[j, k, c]
                                eq_idx += 1
                try:
                    sol_w, _, _, _ = np.linalg.lstsq(A_mat_w, b_vec, rcond=None)
                    W = sol_w.reshape(P, N, rank)
                except:
                    continue

            # Convert back to 2D
            U_2d = np.zeros((N * M, rank))
            V_2d = np.zeros((M * P, rank))
            W_2d = np.zeros((P * N, rank))
            for i in range(N):
                for j in range(M):
                    for c in range(rank):
                        U_2d[_aidx(i, j), c] = U[i, j, c]
            for j in range(M):
                for k in range(P):
                    for c in range(rank):
                        V_2d[_bidx(j, k), c] = V[j, k, c]
            for k in range(P):
                for n in range(N):
                    for c in range(rank):
                        W_2d[_cidx(k, n), c] = W[k, n, c]
            
            if _verify(U_2d, V_2d, W_2d, T, tol=1e-5):
                return U_2d, V_2d, W_2d
                
    return None

def construct():
    t0 = time.time()
    deadline = t0 + 145.0
    
    T = _build_target()
    
    # Try the rank-3 construction (will likely fail)
    U, V, W = _structured_rank3()
    if _verify(U, V, W, T):
        U_r = _to_exact(U)
        V_r = _to_exact(V)
        W_r = _to_exact(W)
        if U_r is not None and V_r is not None and W_r is not None:
            return {"U": U_r, "V": V_r, "W": W_r}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try search for lower rank
    res = _try_lower_rank_search(deadline)
    if res:
        U, V, W = res
        U_r = _to_exact(U)
        V_r = _to_exact(V)
        W_r = _to_exact(W)
        if U_r is not None and V_r is not None and W_r is not None:
            return {"U": U_r, "V": V_r, "W": W_r}
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: schoolbook
    rank = N * M * P
    U = np.zeros((N * M, rank))
    V = np.zeros((M * P, rank))
    W = np.zeros((P * N, rank))
    col = 0
    for i in range(N):
        for j in range(M):
            for k in range(P):
                U[_aidx(i, j), col] = 1.0
                V[_bidx(j, k), col] = 1.0
                W[_cidx(k, i), col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
