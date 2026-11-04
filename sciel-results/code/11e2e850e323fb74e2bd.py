import numpy as np
import time
import itertools

DIMENSIONS = [2, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 140

def _elapsed():
    return time.time() - START_TIME

def _deadline_reached():
    return _elapsed() > TIME_LIMIT

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition exactly reconstructs the tensor."""
    target = _build_target_tensor(n, m, p)
    recon = np.einsum('ac,bc,dc->abd', U, V, W)
    return np.allclose(recon, target, atol=tol)

def _schoolbook_decomposition(n, m, p):
    """Standard schoolbook decomposition with rank n*m*p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """Return rank-7 Strassen decomposition for 2x2x2 matrix multiplication."""
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[p,q],[r,s]]
    # Strassen:
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # p = m1+m4-m5+m7
    # q = m3+m5
    # r = m2+m4
    # s = m1-m2+m3+m6
    
    # U: A indices (a=0, b=1, c=2, d=3)
    # V: B indices (e=0, f=1, g=2, h=3)
    # W: C indices (p=0, q=1, r=2, s=3)
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # m1 = (a+d)(e+h)
    U[0,0]=1; U[3,0]=1
    V[0,0]=1; V[3,0]=1
    W[0,0]=1; W[3,0]=1
    
    # m2 = (c+d)e
    U[2,1]=1; U[3,1]=1
    V[0,1]=1
    W[2,1]=1; W[3,1]=-1
    
    # m3 = a(f-h)
    U[0,2]=1
    V[1,2]=1; V[3,2]=-1
    W[1,2]=1; W[3,2]=1
    
    # m4 = d(g-e)
    U[3,3]=1
    V[2,3]=1; V[0,3]=-1
    W[0,3]=1; W[2,3]=1
    
    # m5 = (a+b)h
    U[0,4]=1; U[1,4]=1
    V[3,4]=1
    W[0,4]=-1; W[1,4]=1
    
    # m6 = (c-a)(e+f)
    U[2,5]=1; U[0,5]=-1
    V[0,5]=1; V[1,5]=1
    W[3,5]=1
    
    # m7 = (b-d)(g+h)
    U[1,6]=1; U[3,6]=-1
    V[2,6]=1; V[3,6]=1
    W[0,6]=1; W[3,6]=1
    
    return U, V, W

def _build_2x4x6_decomposition():
    """
    Build 2x4x6 decomposition using Strassen for 2x2x2 blocks.
    
    Strategy: Split the 4 intermediate dimension into two 2-dim blocks,
    and the 6 output dimension into three 2-dim blocks.
    
    A = [A1 | A2] where A1, A2 are 2x2
    B = [[B11, B12, B13], [B21, B22, B23]] where each B_ij is 2x2
    C = A1*B11 + A2*B21, A1*B12 + A2*B22, A1*B13 + A2*B23 (each 2x2)
    
    Each 2x2 product pair can be computed with Strassen-like combinations.
    For each output block k, we need: C_k = A1*B1k + A2*B2k
    
    Using Strassen for each (A1, B1k) and (A2, B2k) separately gives 7+7=14 per k,
    total 3*14 = 42. But we can share some terms.
    
    Actually, let's use a different approach: treat it as 2x(4x6) and use
    the fact that we can combine the two A blocks.
    
    For each output column block k (0,1,2):
    C_k = A1*B1k + A2*B2k
    
    Let's use Strassen on the combined system. Define:
    S1 = A1 + A2 (2x2)
    S2 = B1k + B2k (2x2)
    
    Then A1*B1k + A2*B2k is not simply related to S1*S2.
    
    Let's just use 2x2x2 Strassen for each of the 6 products (A1*B1k for k=0,1,2
    and A2*B2k for k=0,1,2), giving 6*7 = 42 terms.
    """
    n, m, p = 2, 4, 6
    
    # We have 6 sub-problems: (A1,B11), (A1,B12), (A1,B13), (A2,B21), (A2,B22), (A2,B23)
    # Each is a 2x2x2 multiplication, solved with Strassen (rank 7)
    # Total rank: 6 * 7 = 42
    
    rank = 42
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    S_U, S_V, S_W = _strassen_2x2x2()
    
    # For each of the 6 sub-problems, embed the Strassen decomposition
    sub_problems = [
        # (A_block_start, B_block_row_start, B_block_col_start, C_block_col_start)
        (0, 0, 0, 0),  # A1 * B11 -> C cols 0,1
        (0, 0, 2, 2),  # A1 * B12 -> C cols 2,3
        (0, 0, 4, 4),  # A1 * B13 -> C cols 4,5
        (2, 2, 0, 0),  # A2 * B21 -> C cols 0,1
        (2, 2, 2, 2),  # A2 * B22 -> C cols 2,3
        (2, 2, 4, 4),  # A2 * B23 -> C cols 4,5
    ]
    
    for sp_idx, (a_start, b_row_start, b_col_start, c_col_start) in enumerate(sub_problems):
        base = sp_idx * 7
        for c in range(7):
            col = base + c
            # U: A indices
            for i in range(2):
                for j in range(2):
                    U[i*4 + (a_start + j), col] = S_U[i*2 + j, c]
            # V: B indices
            for j in range(2):
                for k in range(2):
                    V[(b_row_start + j)*6 + (b_col_start + k), col] = S_V[j*2 + k, c]
            # W: C indices
            for k in range(2):
                for i in range(2):
                    W[(c_col_start + k)*2 + i, col] = S_W[k*2 + i, c]
    
    return U, V, W

def _try_lower_rank_search(n, m, p, max_rank, max_iters=2000, seed=42):
    """
    Try to find a lower-rank decomposition using iterative refinement.
    Start from schoolbook and try to reduce rank.
    """
    rng = np.random.RandomState(seed)
    target = _build_target_tensor(n, m, p)
    
    # Try ranks from lower bound up
    lower_bound = max(n*m, m*p, p*n)
    start_rank = max(lower_bound, 20)
    
    best_U, best_V, best_W = None, None, None
    best_rank = n*m*p
    
    # First, get the schoolbook as baseline
    U_school, V_school, W_school = _schoolbook_decomposition(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n*m*p
    
    for rank in range(start_rank, min(best_rank, max_rank + 1)):
        if _deadline_reached():
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if _deadline_reached():
                break
            
            U = rng.randn(n*m, rank)
            V = rng.randn(m*p, rank)
            W = rng.randn(p*n, rank)
            
            # Iterative refinement
            for it in range(max_iters):
                if _deadline_reached():
                    break
                
                # Update U
                # T[a,b,d] = sum_c U[a,c] V[b,c] W[d,c]
                # For fixed a: T[a,:,:] = U[a,:] @ P where P[c,b,d] = V[b,c]*W[d,c]
                P = (V[:, None, :] * W[None, :, :]).reshape(rank, -1)
                P_T = P.T
                try:
                    P_pinv = np.linalg.pinv(P_T)
                    new_U = np.zeros((n*m, rank))
                    for a in range(n*m):
                        y = target[a, :, :].reshape(-1)
                        new_U[a, :] = y @ P_pinv
                    U = new_U
                except:
                    break
                
                if _deadline_reached():
                    break
                
                # Update V
                Q = (U[:, None, :] * W[None, :, :]).reshape(rank, -1)
                Q_T = Q.T
                try:
                    Q_pinv = np.linalg.pinv(Q_T)
                    new_V = np.zeros((m*p, rank))
                    for b in range(m*p):
                        y = target[:, b, :].reshape(-1)
                        new_V[b, :] = y @ Q_pinv
                    V = new_V
                except:
                    break
                
                if _deadline_reached():
                    break
                
                # Update W
                R = (U[:, None, :] * V[None, :, :]).reshape(rank, -1)
                R_T = R.T
                try:
                    R_pinv = np.linalg.pinv(R_T)
                    new_W = np.zeros((p*n, rank))
                    for d in range(p*n):
                        y = target[:, :, d].reshape(-1)
                        new_W[d, :] = y @ R_pinv
                    W = new_W
                except:
                    break
                
                # Check convergence
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.linalg.norm(target - recon)
                if err < 1e-9:
                    if rank < best_rank:
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        best_rank = rank
                    break
            
            if rank < best_rank and best_rank <= rank:
                break
    
    if best_rank < n*m*p:
        return best_U, best_V, best_W, best_rank
    
    return None, None, None, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # First, try the Strassen-based 42-rank decomposition
    U, V, W = _build_2x4x6_decomposition()
    
    if _verify_decomposition(U, V, W, n, m, p):
        # Try to improve with search
        if not _deadline_reached():
            U2, V2, W2, rank2 = _try_lower_rank_search(n, m, p, max_rank=42, max_iters=500, seed=42)
            if U2 is not None and rank2 < 42:
                U, V, W = U2, V2, W2
    
    # Fallback to schoolbook if anything fails
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook_decomposition(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
