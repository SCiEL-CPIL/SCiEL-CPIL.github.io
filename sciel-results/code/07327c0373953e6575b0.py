import time
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _build_tensors(n, m, p):
    """Build the three matricizations of the multiplication tensor.
    T2: (n*m, m*p) with T2[i*m+j, j*p+k] = 1
    T1: (m*p, p*n) with T1[j*p+k, k*n+i] = 1 if i==k else 0  -- wait, this is not right.
    Let me reconsider.
    
    The tensor T has entries T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k]).
    
    Matricization T2 (flatten i,j vs k): Actually the standard is:
    T_1: flatten (i,k) vs j -> (n*p, m), T_1[i*p+k, j] = 1
    T_2: flatten (i,j) vs k -> (n*m, p), T_2[i*m+j, k] = 1  -- but this doesn't capture j-p+k
    T_3: flatten (j,k) vs i -> (m*p, n), T_3[j*p+k, i] = 1
    
    Hmm, let me use the convention from the task:
    U: (n*m, rank), indexed by (i,j)
    V: (m*p, rank), indexed by (j,k)
    W: (p*n, rank), indexed by (k,i)
    
    The tensor decomposition requires:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i,j,k
    
    So the matricizations are:
    M_UV: (n*m, m*p) where M_UV[i*m+j, j*p+k] = sum_r U[i*m+j,r]*V[j*p+k,r] = 1 for all i,j,k
    But wait, this sum must equal 1 for ALL i,j,k. But the LHS only depends on i*m+j and j*p+k,
    and the constraint is that for fixed (i,j,k), the product sum equals 1.
    
    Actually, the correct matricizations for verifying are:
    For fixed i: sum_r U[i*m+j, r] * V[j*p+k, r] should give... no.
    
    Let me think again. The tensor is T[i,j,k] = 1. We want:
    T[i,j,k] = sum_r U[i,j]_r * V[j,k]_r * W[k,i]_r
    
    So for the decomposition to be exact:
    sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for all i in [0,n), j in [0,m), k in [0,p)
    
    To verify, we can check:
    For each (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1
    
    This is O(n*m*p*rank) which is fine for small dimensions.
    """
    pass


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W (numpy arrays) exactly decompose the multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                if abs(s - 1.0) > tol:
                    return False
    return True


def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Faster verification using matrix operations."""
    # For each i: sum_j sum_k (U[i*m+j,:] * V[j*p+k,:] * W[k*n+i,:]) should be... no.
    # The constraint is per (i,j,k), not summed.
    # Let's just do the triple loop but with numpy for the inner product.
    for i in range(n):
        for j in range(m):
            for k in range(p):
                prod = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
                if abs(np.sum(prod) - 1.0) > tol:
                    return False
    return True


def _greedy_peel(n, m, p, target_rank, deadline):
    """Greedy rank-1 peeling. Returns (U, V, W) numpy arrays of shape (*, target_rank) or None."""
    # Build the full tensor in a form amenable to peeling.
    # We'll work with the 3-way array T[i,j,k] = 1.
    # Greedy: at each step, find (u, v, w) that best matches the residual.
    # Use SVD on the mode-2 matricization or similar.
    
    # T as (n, m, p) array
    T = np.ones((n, m, p), dtype=float)
    
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(target_rank):
        if time.time() > deadline:
            return None
        
        # Matricize T along mode 2: shape (n*p, m)
        # T_2[i*p+k, j] = T[i,j,k]
        T2 = T.transpose(0, 2, 1).reshape(n * p, m)
        
        # SVD
        try:
            U_svd, s, Vt = np.linalg.svd(T2, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        
        if len(s) == 0 or s[0] < 1e-12:
            break
        
        # The leading left singular vector gives a combination of (i,k)
        # The leading right singular vector gives a combination of j
        # We need to extract u (length n), v (length m), w (length p)
        
        # Right singular vector: v_vec (length m)
        v_vec = Vt[0, :]
        
        # Left singular vector: w_ik (length n*p), reshape to (n, p)
        w_ik = U_svd[:, 0].reshape(n, p)
        
        # For a rank-1 term, T[i,j,k] ≈ a * u[i] * v[j] * w[k]
        # We have v[j] from SVD. For u and w, we need to decompose w_ik into u[i]*w[k].
        # w_ik[i,k] ≈ u[i] * w[k]
        # Do SVD of w_ik
        U_w, s_w, Vt_w = np.linalg.svd(w_ik, full_matrices=False)
        if len(s_w) == 0 or s_w[0] < 1e-12:
            break
        u_vec = U_w[:, 0]  # length n
        w_vec = Vt_w[0, :]  # length p
        
        # Scale factor: T ≈ a * u outer v outer w
        # a = <T, u⊗v⊗w> / (||u||^2 ||v||^2 ||w||^2)
        # Since u, v, w are unit vectors: a = sum T[i,j,k] u[i]v[j]w[k]
        a = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a += T[i, j, k] * u_vec[i] * v_vec[j] * w_vec[k]
        
        if abs(a) < 1e-12:
            break
        
        # Subtract: T -= a * u outer v outer w
        T -= a * np.einsum('i,j,k->ijk', u_vec, v_vec, w_vec)
        
        U_list.append((a, u_vec))
        V_list.append((a, v_vec))  # Note: a is shared, but we store per-term
        W_list.append((a, w_vec))
        
        # Check if T is approximately zero
        if np.max(np.abs(T)) < 1e-10:
            break
    
    if len(U_list) == 0:
        return None
    
    actual_rank = len(U_list)
    if actual_rank < target_rank:
        # Pad with zeros
        while len(U_list) < target_rank:
            U_list.append((0.0, np.zeros(n)))
            V_list.append((0.0, np.zeros(m)))
            W_list.append((0.0, np.zeros(p)))
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    for r in range(target_rank):
        a, u_vec = U_list[r]
        _, v_vec = V_list[r]
        _, w_vec = W_list[r]
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = a * u_vec[i] * v_vec[j]  # This is wrong!
        
    # The above is incorrect. The decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # If we have T ≈ sum_r a_r * u_r[i] * v_r[j] * w_r[k], then we can set:
    # U[i*m+j, r] = u_r[i], V[j*p+k, r] = v_r[j], W[k*n+i, r] = a_r * w_r[k]
    # Or distribute a_r arbitrarily among the three factors.
    # Let's set U[i*m+j, r] = u_r[i], V[j*p+k, r] = v_r[j], W[k*n+i, r] = a_r * w_r[k]
    
    U = np.zeros((n * m, target_rank))
    V = np.zeros((m * p, target_rank))
    W = np.zeros((p * n, target_rank))
    
    for r in range(target_rank):
        a, u_vec = U_list[r]
        _, v_vec = V_list[r]
        _, w_vec = W_list[r]
        for i in range(n):
            for j in range(m):
                U[i * m + j, r] = u_vec[i]
        for j in range(m):
            for k in range(p):
                V[j * p + k, r] = v_vec[j]
        for k in range(p):
            for i in range(n):
                W[k * n + i, r] = a * w_vec[k]
    
    return U, V, W


def _project_to_half_integers(U, V, W, tol=1e-6):
    """Project U, V, W to nearest half-integer grid. Returns (U', V', W', max_err)."""
    U_p = np.round(2.0 * U) / 2.0
    V_p = np.round(2.0 * V) / 2.0
    W_p = np.round(2.0 * W) / 2.0
    
    err = max(np.max(np.abs(U_p - U)), np.max(np.abs(V_p - V)), np.max(np.abs(W_p - W)))
    return U_p, V_p, W_p, err


def _to_list(mat):
    """Convert numpy array to list of lists with half-integer snapping."""
    out = []
    for row in mat:
        r = []
        for val in row:
            hv = np.round(val * 2.0) / 2.0
            if abs(hv - round(hv)) < 1e-9:
                r.append(int(round(hv)))
            else:
                r.append(float(hv))
        out.append(r)
    return out


def _try_rank(r, n, m, p, deadline):
    """Try to find a rank-r decomposition. Returns result dict or None."""
    if time.time() > deadline:
        return None
    
    result = _greedy_peel(n, m, p, r, deadline)
    if result is None:
        return None
    
    U, V, W = result
    
    # Project to half-integers
    U_p, V_p, W_p, err = _project_to_half_integers(U, V, W)
    
    if err > 1e-4:
        return None
    
    # Verify the projected version
    if _verify_fast(U_p, V_p, W_p, n, m, p, tol=1e-6):
        return {"U": _to_list(U_p), "V": _to_list(V_p), "W": _to_list(W_p)}
    
    # Try without projection if the original is close to half-integers
    if err < 1e-6:
        if _verify_fast(U, V, W, n, m, p, tol=1e-6):
            return {"U": _to_list(U), "V": _to_list(V), "W": _to_list(W)}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 140.0  # Safety margin before 150s kill
    
    # Start with schoolbook as fallback
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try ranks from small to large
    # Theoretical lower bound for matrix multiplication rank is at least max(n,m,p)
    # For 2x3x7, known bounds suggest rank might be around 14-21
    min_rank = max(n, m, p)
    max_rank = min(best_rank, 30)
    
    for r in range(min_rank, max_rank + 1):
        if time.time() > deadline:
            break
        try:
            result = _try_rank(r, n, m, p, deadline)
            if result is not None:
                return result
        except Exception:
            continue
    
    return best

# EVOLVE-BLOCK-END
