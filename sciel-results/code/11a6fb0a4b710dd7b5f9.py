import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 115.0
    
    dim_u = n * m  # 16
    dim_v = m * p  # 28
    dim_w = p * n  # 28
    
    # Build target tensor
    T_target = np.zeros((dim_u, dim_v, dim_w), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_target[a, b, c] = 1
    
    # Get the known Kronecker decomposition (rank 97) as fallback and initialization guide
    U_kron, V_kron, W_kron = _kronecker_decomp(n, m, p)
    rank_kron = U_kron.shape[1]
    
    # Try greedy peeling to find a lower rank exact decomposition
    U_best, V_best, W_best, rank_best = _greedy_peel(T_target, dim_u, dim_v, dim_w, 
                                                       U_kron, V_kron, W_kron, deadline)
    
    # Verify the best result
    if rank_best < rank_kron:
        T_decomp = np.einsum('ar,br,cr->abc', U_best, V_best, W_best)
        if np.allclose(T_decomp, T_target, atol=1e-10):
            return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
    
    # Fallback to Kronecker decomposition
    T_decomp = np.einsum('ar,br,cr->abc', U_kron, V_kron, W_kron)
    if not np.allclose(T_decomp, T_target, atol=1e-10):
        return _schoolbook(n, m, p)
    
    return {"U": U_kron.tolist(), "V": V_kron.tolist(), "W": W_kron.tolist()}


def _greedy_peel(T, dim_u, dim_v, dim_w, U_init, V_init, W_init, deadline):
    """Greedy peeling: iteratively extract rank-1 terms from the residual tensor.
    
    Uses the Kronecker decomposition vectors as initialization hints for better
    first terms, then continues peeling until the residual is zero or time runs out.
    """
    R = T.copy()
    U_list = []
    V_list = []
    W_list = []
    
    max_terms = 120  # Can't do better than this many terms in practice
    
    for term_idx in range(max_terms):
        if time.time() > deadline - 5.0:
            break
        
        if not np.any(R):
            break
        
        # Try to find a good rank-1 term
        u, v, w = _find_rank1_term(R, dim_u, dim_v, dim_w, U_init, V_init, W_init, 
                                     term_idx, deadline)
        
        if u is None:
            break
        
        # Compute the coefficient: we want to subtract alpha * u ⊗ v ⊗ w from R
        # alpha = <R, u⊗v⊗w> / <u⊗v⊗w, u⊗v⊗w>
        numerator = np.einsum('abc,a,b,c', R, u, v, w)
        denominator = np.sum(u**2) * np.sum(v**2) * np.sum(w**2)
        
        if abs(denominator) < 1e-15:
            break
        
        alpha = numerator / denominator
        
        if abs(alpha) < 1e-15:
            # Try a different approach: just use sign-based
            u_s = np.sign(u)
            v_s = np.sign(v)
            w_s = np.sign(w)
            if not np.any(u_s) or not np.any(v_s) or not np.any(w_s):
                break
            numerator = np.einsum('abc,a,b,c', R, u_s, v_s, w_s)
            if abs(numerator) < 1e-15:
                break
            alpha = 1.0
            u, v, w = u_s, v_s, w_s
        
        # Subtract this rank-1 term
        R -= alpha * np.einsum('a,b,c->abc', u, v, w)
        
        # Check if we've made progress
        if np.max(np.abs(R)) < 1e-10:
            R = np.zeros_like(R)
            break
        
        U_list.append(alpha * u)
        V_list.append(v)
        W_list.append(w)
        
        # Periodic check: if we're making very slow progress, try a restart
        if term_idx > 0 and term_idx % 10 == 0:
            if time.time() > deadline - 5.0:
                break
    
    if not U_list:
        return None, None, None, 999
    
    # Final verification
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    rank = U.shape[1]
    
    T_decomp = np.einsum('ar,br,cr->abc', U, V, W)
    if not np.allclose(T_decomp, T, atol=1e-8):
        return None, None, None, 999
    
    return U, V, W, rank


def _find_rank1_term(R, dim_u, dim_v, dim_w, U_init, V_init, W_init, term_idx, deadline):
    """Find a good rank-1 term for the residual tensor R.
    
    Uses multiple strategies:
    1. Initialize from Kronecker vectors (if available)
    2. Use alternating maximization with sign constraint
    3. Try random restarts
    """
    best_u, best_v, best_w = None, None, None
    best_score = 0
    
    # Strategy 1: Use Kronecker vectors as initialization
    if U_init is not None and term_idx < U_init.shape[1]:
        u0 = U_init[:, term_idx].copy()
        v0 = V_init[:, term_idx].copy()
        w0 = W_init[:, term_idx].copy()
        
        # Refine with alternating maximization
        u, v, w = _alternating_max(R, dim_u, dim_v, dim_w, u0, v0, w0, deadline)
        score = np.einsum('abc,a,b,c', R, u, v, w)
        if score > best_score:
            best_u, best_v, best_w = u, v, w
            best_score = score
    
    # Strategy 2: Random initialization with alternating maximization
    for restart in range(5):
        if time.time() > deadline - 5.0:
            break
        
        u0 = np.random.choice([-1.0, 0.0, 1.0], size=dim_u)
        v0 = np.random.choice([-1.0, 0.0, 1.0], size=dim_v)
        w0 = np.random.choice([-1.0, 0.0, 1.0], size=dim_w)
        
        if not np.any(u0) or not np.any(v0) or not np.any(w0):
            continue
        
        u, v, w = _alternating_max(R, dim_u, dim_v, dim_w, u0, v0, w0, deadline)
        score = np.einsum('abc,a,b,c', R, u, v, w)
        if score > best_score:
            best_u, best_v, best_w = u, v, w
            best_score = score
    
    # Strategy 3: Greedy coordinate-based initialization
    # Find the largest entry in R and use it to seed
    if best_score <= 0:
        idx = np.unravel_index(np.argmax(np.abs(R)), R.shape)
        a, b, c = idx
        u0 = np.zeros(dim_u); u0[a] = 1.0
        v0 = np.zeros(dim_v); v0[b] = 1.0
        w0 = np.zeros(dim_w); w0[c] = 1.0
        
        u, v, w = _alternating_max(R, dim_u, dim_v, dim_w, u0, v0, w0, deadline)
        score = np.einsum('abc,a,b,c', R, u, v, w)
        if score > best_score:
            best_u, best_v, best_w = u, v, w
            best_score = score
    
    if best_score <= 0:
        return None, None, None
    
    return best_u, best_v, best_w


def _alternating_max(R, dim_u, dim_v, dim_w, u, v, w, deadline):
    """Alternating maximization to maximize <R, u⊗v⊗w> with u,v,w in {-1,0,1}."""
    for _ in range(30):
        if time.time() > deadline - 5.0:
            break
        
        # Update u
        M_u = np.einsum('abc,b,c->a', R, v, w)
        new_u = np.sign(M_u).astype(np.float64)
        
        # Update v
        M_v = np.einsum('abc,a,c->b', R, u, w)
        new_v = np.sign(M_v).astype(np.float64)
        
        # Update w
        M_w = np.einsum('abc,a,b->c', R, u, v)
        new_w = np.sign(M_w).astype(np.float64)
        
        if np.array_equal(new_u, u) and np.array_equal(new_v, v) and np.array_equal(new_w, w):
            break
        
        u, v, w = new_u, new_v, new_w
    
    return u, v, w


def _kronecker_decomp(n, m, p):
    """Build the Kronecker-based decomposition: Strassen for k=0..3, schoolbook for k=4..6."""
    U_4, V_4, W_4 = _strassen_4x4x4(n, m, p)
    U_3, V_3, W_3 = _schoolbook_4x4x3(n, m, p, k_start=4)
    
    U_all = np.hstack([U_4, U_3])
    V_all = np.hstack([V_4, V_3])
    W_all = np.hstack([W_4, W_3])
    
    return U_all, V_all, W_all


def _strassen_2x2x2():
    """Strassen decomposition of 2x2x2 matrix multiplication."""
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # M1 = (a11+a22)(b11+b22)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2 = (a21+a22)b11
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # M3 = a11(b12-b22)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = a22(b21-b11)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (a11+a12)b22
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # M6 = (a21-a11)(b11+b12)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # M7 = (a12-a22)(b21+b22)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W


def _strassen_4x4x4(n, m, p):
    """Build 4x4x4 decomposition via Kronecker of Strassen 2x2x2."""
    U2, V2, W2 = _strassen_2x2x2()
    
    U_list = []
    V_list = []
    W_list = []
    
    for r in range(7):
        for s in range(7):
            u = np.kron(U2[:, r], U2[:, s])
            
            v_kron = np.kron(V2[:, r], V2[:, s])
            v_full = np.zeros(m * p)
            for j in range(n):
                for k in range(4):
                    v_full[j * p + k] = v_kron[j * 4 + k]
            
            w_kron = np.kron(W2[:, r], W2[:, s])
            w_full = np.zeros(p * n)
            w_full[:16] = w_kron
            
            U_list.append(u)
            V_list.append(v_full)
            W_list.append(w_full)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W


def _schoolbook_4x4x3(n, m, p, k_start=4):
    """Schoolbook decomposition for columns k_start..p-1."""
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(k_start, p):
                u = np.zeros(n * m)
                v = np.zeros(m * p)
                w = np.zeros(p * n)
                
                a = i * m + j
                b = j * p + k
                c = k * n + i
                
                u[a] = 1
                v[b] = 1
                w[c] = 1
                
                U_list.append(u)
                V_list.append(v)
                W_list.append(w)
    
    U = np.array(U_list).T
    V = np.array(V_list).T
    W = np.array(W_list).T
    
    return U, V, W


def _schoolbook(n, m, p):
    """Return schoolbook decomposition of n x m times m x p."""
    rank = n * m * p
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    U = np.zeros((dim_u, rank))
    V = np.zeros((dim_v, rank))
    W = np.zeros((dim_w, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
