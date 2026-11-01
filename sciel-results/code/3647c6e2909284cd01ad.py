import time
import numpy as np
from scipy.optimize import least_squares

DEADLINE = 150.0
START = time.time()

def _time_left():
    return DEADLINE - (time.time() - START)

def _build_schoolbook(n, m, p):
    """Trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                U[a_idx, r] = 1.0
                V[b_idx, r] = 1.0
                W[c_idx, r] = 1.0
                r += 1
    return U, V, W

def _build_strassen_222():
    """Rank-7 Strassen for 2x2x2."""
    n, m, p = 2, 2, 2
    rank = 7
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    # P1 = (A11+A22)(B11+B22)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # P2 = (A21+A22)A11
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=-1
    # P3 = A11(B12-B22)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[1,2]=1; W[3,2]=1
    # P4 = A22(B21-B11)
    U[3,3]=1; V[2,3]=1; V[0,3]=-1; W[0,3]=1; W[2,3]=1
    # P5 = (A11+A12)B22
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[3,4]=1
    # P6 = (A21-A11)(B11+B12)
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # P7 = (A12-A22)(B21+B22)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[2,6]=-1
    return U, V, W

def _embed(t_U, t_V, t_W, n_src, m_src, p_src, n_tgt, m_tgt, p_tgt, n_map, m_map, p_map):
    """Embed a small tensor into a larger space."""
    rank = t_U.shape[1]
    U = np.zeros((n_tgt * m_tgt, rank))
    V = np.zeros((m_tgt * p_tgt, rank))
    W = np.zeros((p_tgt * n_tgt, rank))
    for i in range(n_src):
        for j in range(m_src):
            U[n_map[i]*m_tgt + m_map[j], :] = t_U[i*m_src+j, :]
    for j in range(m_src):
        for k in range(p_src):
            V[m_map[j]*p_tgt + p_map[k], :] = t_V[j*p_src+k, :]
    for k in range(p_src):
        for i in range(n_src):
            W[p_map[k]*n_tgt + n_map[i], :] = t_W[k*n_src+i, :]
    return U, V, W

def _tensor_sum(U_list, V_list, W_list, n, m, p):
    U = np.hstack(U_list)
    V = np.hstack(V_list)
    W = np.hstack(W_list)
    return U, V, W

def _verify(U, V, W, n=2, m=4, p=7, atol=1e-8):
    """Check decomposition correctness on random inputs."""
    np.random.seed(12345)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_true = A @ B
    C_calc = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for j in range(m):
                for r in range(U.shape[1]):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            C_calc[i, k] = s
    return np.allclose(C_calc, C_true, atol=atol)

def _find_223_rank10():
    """
    Try to find a rank-10 decomposition for 2x2x3 matrix multiplication.
    Uses iterative refinement from a random start, then rounds to half-integers.
    """
    n, m, p = 2, 2, 3
    rank = 10
    max_iters = 500
    tol = 1e-8
    
    # Known structure: 2x2x3 can be seen as 2x2x2 (Strassen) + 2x2x1 (Schoolbook) = 7+4=11.
    # We want to save 1 rank.
    # Let's try to optimize from a random start.
    
    rng = np.random.RandomState(42)
    U = rng.randn(n*m, rank)
    V = rng.randn(m*p, rank)
    W = rng.randn(p*n, rank)
    
    # Flatten tensors for optimization
    def flatten(U, V, W):
        return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    def unflatten(x):
        U_size = n*m*rank
        V_size = m*p*rank
        U = x[:U_size].reshape(n*m, rank)
        V = x[U_size:U_size+V_size].reshape(m*p, rank)
        W = x[U_size+V_size:].reshape(p*n, rank)
        return U, V, W
    
    def objective(x):
        U, V, W = unflatten(x)
        # Compute tensor contraction
        # T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We want T to be the identity tensor for matmul:
        # T[i,j,k] = 1 if i,j,k form a valid matmul term? 
        # No, the tensor T for matmul is defined such that 
        # C[i,k] = sum_j A[i,j] B[j,k].
        # The decomposition is:
        # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = delta? 
        # Actually, the tensor T has T[i,j,k] = 1 for all i,j,k? 
        # No. The matmul tensor T is such that 
        # <A, U_r> <B, V_r> <W_r, C> ...
        # The constraint is:
        # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all (i,j,k)?
        # Let's check the schoolbook:
        # U[i*m+j, r] is 1 if r corresponds to (i,j).
        # V[j*p+k, r] is 1 if r corresponds to (j,k).
        # W[k*n+i, r] is 1 if r corresponds to (k,i).
        # For a fixed r=(i0,j0,k0), U[i*m+j,r] is 1 only if i=i0, j=j0.
        # V[j*p+k,r] is 1 only if j=j0, k=k0.
        # W[k*n+i,r] is 1 only if k=k0, i=i0.
        # So the product is 1 only if i=i0, j=j0, k=k0.
        # So sum_r ... = 1 if (i,j,k) is in the set of atoms.
        # For matmul, every (i,j,k) is an atom? 
        # Yes, C[i,k] = sum_j A[i,j] B[j,k].
        # So T[i,j,k] = 1 for all i,j,k.
        
        residuals = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residuals[i, j, k] = 1.0 - s
        return residuals.flatten()
    
    x0 = flatten(U, V, W)
    
    # Use least squares
    try:
        result = least_squares(objective, x0, method='lm', max_nfev=2000)
        U_opt, V_opt, W_opt = unflatten(result.x)
        
        # Check if it's close enough
        if result.cost < tol:
            # Try to round to half-integers
            U_round = np.round(U_opt * 2) / 2
            V_round = np.round(V_opt * 2) / 2
            W_round = np.round(W_opt * 2) / 2
            
            if _verify(U_round, V_round, 2, 2, 3, atol=1e-6):
                return U_round, V_round, W_round
            
            # If rounding fails, try integer rounding
            U_round2 = np.round(U_opt)
            V_round2 = np.round(V_opt)
            W_round2 = np.round(W_opt)
            if _verify(U_round2, V_round2, 2, 2, 3, atol=1e-6):
                return U_round2, V_round2, W_round2
            
            # If still fails, return the float solution (might not be accepted if strict integer/half-integer required, but let's see)
            # The task says "integer/half-integer entries". So we must round.
            # If we can't round, we return None.
            return None
    except Exception:
        pass
    
    return None

def _build_48_rank():
    """Build the rank-48 decomposition using rank-10 for 2x2x3."""
    S_U, S_V, S_W = _build_strassen_222()
    
    # Get rank-10 for 2x2x3
    U223, V223, W223 = _find_223_rank10()
    if U223 is None:
        return None
        
    # Build 2x4x4: four 2x2x2 Strassen blocks
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        for p_off in [0, 2]:
            u, v, w = _embed(S_U, S_V, S_W, 2,2,2, 2,4,4,
                             [0,1], [m_off, m_off+1], [p_off, p_off+1])
            U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U244, V244, W244 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,4)
    
    # Build 2x4x3: two 2x2x3 blocks
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        u, v, w = _embed(U223, V223, W223, 2,2,3, 2,4,3,
                         [0,1], [m_off, m_off+1], [0,1,2])
        U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U243, V243, W243 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,3)
    
    # 2x4x7 = 2x4x4 (p=[0..3]) + 2x4x3 (p=[4..6])
    u1, v1, w1 = _embed(U244, V244, W244, 2,4,4, 2,4,7,
                        [0,1],[0,1,2,3],[0,1,2,3])
    u2, v2, w2 = _embed(U243, V243, W243, 2,4,3, 2,4,7,
                        [0,1],[0,1,2,3],[4,5,6])
    U = np.hstack([u1, u2])
    V = np.hstack([v1, v2])
    W = np.hstack([w1, w2])
    return U, V, W

def _build_50_rank():
    """Build the rank-50 decomposition (Strassen-based)."""
    S_U, S_V, S_W = _build_strassen_222()
    
    # Build 2x4x4: four 2x2x2 Strassen blocks
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        for p_off in [0, 2]:
            u, v, w = _embed(S_U, S_V, S_W, 2,2,2, 2,4,4,
                             [0,1], [m_off, m_off+1], [p_off, p_off+1])
            U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U244, V244, W244 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,4)
    
    # Build 2x4x3: two 2x2x3 blocks, each = Strassen(2x2x2) + schoolbook(2x2x1)
    sch_U, sch_V, sch_W = _build_schoolbook(2, 2, 1)
    u_s, v_s, w_s = _embed(S_U, S_V, S_W, 2,2,2, 2,2,3, [0,1],[0,1],[0,1])
    u_sc, v_sc, w_sc = _embed(sch_U, sch_V, sch_W, 2,2,1, 2,2,3, [0,1],[0,1],[2])
    U223 = np.hstack([u_s, u_sc])
    V223 = np.hstack([v_s, v_sc])
    W223 = np.hstack([w_s, w_sc])
    
    U_parts, V_parts, W_parts = [], [], []
    for m_off in [0, 2]:
        u, v, w = _embed(U223, V223, W223, 2,2,3, 2,4,3,
                         [0,1], [m_off, m_off+1], [0,1,2])
        U_parts.append(u); V_parts.append(v); W_parts.append(w)
    U243, V243, W243 = _tensor_sum(U_parts, V_parts, W_parts, 2,4,3)
    
    # 2x4x7 = 2x4x4 (p=[0..3]) + 2x4x3 (p=[4..6])
    u1, v1, w1 = _embed(U244, V244, W244, 2,4,4, 2,4,7,
                        [0,1],[0,1,2,3],[0,1,2,3])
    u2, v2, w2 = _embed(U243, V243, W243, 2,4,3, 2,4,7,
                        [0,1],[0,1,2,3],[4,5,6])
    U = np.hstack([u1, u2])
    V = np.hstack([v1, v2])
    W = np.hstack([w1, w2])
    return U, V, W

def construct():
    t0 = time.time()
    
    # First, get the known rank-50 solution
    U50, V50, W50 = _build_50_rank()
    if not _verify(U50, V50, W50):
        U, V, W = _build_schoolbook(2, 4, 7)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U50, V50, W50
    best_rank = 50
    
    # Try to find rank 48
    if _time_left() > 30:
        res48 = _build_48_rank()
        if res48 is not None:
            U48, V48, W48 = res48
            if _verify(U48, V48, W48):
                best_U, best_V, best_W = U48, V48, W48
                best_rank = 48
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

if __name__ == "__main__":
    res = construct()
    print(f"Rank: {len(res['U'][0])}")
