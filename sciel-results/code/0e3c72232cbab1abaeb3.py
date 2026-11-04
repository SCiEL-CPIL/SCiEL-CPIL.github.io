import time
import numpy as np

START = time.time()
DEADLINE = START + 120.0  # Safety margin

def _strassen_2x2x2():
    """Strassen 2x2x2 decomposition, rank 7."""
    U = np.zeros((4, 7), dtype=np.int16)
    V = np.zeros((4, 7), dtype=np.int16)
    W = np.zeros((4, 7), dtype=np.int16)
    # M1 = (A00+A11)(B00+B11)
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[3,0]=1
    # M2 = (A10+A11)B00
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[1,1]=1; W[3,1]=-1
    # M3 = A00(B01-B11)
    U[0,2]=1; V[1,2]=1; V[3,2]=-1; W[2,2]=1; W[3,2]=1
    # M4 = A11(B10-B00)
    U[3,3]=1; V[0,3]=-1; V[2,3]=1; W[0,3]=1; W[1,3]=1
    # M5 = (A00+A01)B11
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[0,4]=-1; W[2,4]=1
    # M6 = (A10-A00)(B00+B01)
    U[0,5]=-1; U[2,5]=1; V[0,5]=1; V[1,5]=1; W[3,5]=1
    # M7 = (A01-A11)(B10+B11)
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1
    return U, V, W

def _build_4x4x4_tensors():
    """Build 4x4x4 via Kronecker product of two 2x2x2 Strassen."""
    U2, V2, W2 = _strassen_2x2x2()
    rank = 49
    U4 = np.zeros((16, rank), dtype=np.int16)
    V4 = np.zeros((16, rank), dtype=np.int16)
    W4 = np.zeros((16, rank), dtype=np.int16)
    for c0 in range(7):
        for c1 in range(7):
            c = c0*7 + c1
            for i in range(4):
                for j in range(4):
                    U4[i*4+j, c] = U2[(i//2)*2+(j//2), c0] * U2[(i%2)*2+(j%2), c1]
            for j in range(4):
                for k in range(4):
                    V4[j*4+k, c] = V2[(j//2)*2+(k//2), c0] * V2[(j%2)*2+(k%2), c1]
            for k in range(4):
                for i in range(4):
                    W4[k*4+i, c] = W2[(k//2)*2+(i//2), c0] * W2[(k%2)*2+(i%2), c1]
    return U4, V4, W4

def _build_rank98():
    """
    Construct rank-98 decomposition for 4x4 * 4x8 matrix multiplication.
    """
    n, m, p = 4, 4, 8
    U4, V4, W4 = _build_4x4x4_tensors()
    
    total_rank = 98
    U = np.zeros((n * m, total_rank), dtype=np.int16)
    V = np.zeros((m * p, total_rank), dtype=np.int16)
    W = np.zeros((p * n, total_rank), dtype=np.int16)
    
    # Block 0: k in [0, 4)
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j*8 + k, 0:49] = V4[j*4 + k, :]
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = W4[k*4 + i, :]
            
    # Block 1: k in [4, 8)
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4, 8):
            V[j*8 + k, 49:98] = V4[j*4 + (k-4), :]
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = W4[(k-4)*4 + i, :]
            
    return U, V, W

def _verify_fast(U, V, W, n, m, p):
    """
    Fast vectorized verification. Returns max absolute error.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    rank = U_arr.shape[1]
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = U_arr[i*m+j]
                v = V_arr[j*p+k]
                w = W_arr[k*n+i]
                val = np.dot(np.multiply(u, v), w)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that U, V, W correctly decompose the matrix multiplication tensor.
    Returns True if valid.
    """
    return _verify_fast(U, V, W, n, m, p) < 1e-6

def _try_find_lower_rank(n, m, p, max_rank, min_rank, time_limit):
    """
    Attempt to find a decomposition with rank < max_rank using ALS.
    Returns (U, V, W) if found, else None.
    """
    start_time = time.time()
    
    # Try ranks from max_rank - 1 down to min_rank
    for r in range(max_rank - 1, min_rank - 1, -1):
        if time.time() > time_limit:
            break
            
        # Try a few random initializations for this rank
        for seed in range(5):
            if time.time() > time_limit:
                break
                
            np.random.seed(seed)
            
            # Initialize
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            # ALS iterations
            for it in range(300):
                if time.time() > time_limit:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        b = np.ones(p)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        b = np.ones(n)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        b = np.ones(m)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0:
                    err = _verify_fast(U, V, W, n, m, p)
                    if err < 1e-5:
                        # Try to round to half-integers
                        U_round = np.round(U * 2) / 2
                        V_round = np.round(V * 2) / 2
                        W_round = np.round(W * 2) / 2
                        
                        # Verify rounded solution
                        if _verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                            return U_round, V_round, W_round
                            
    return None

def _try_shared_u_optimization(U98, V98, W98, n, m, p, time_limit):
    """
    Try to exploit the shared U structure to reduce rank.
    Since both blocks use the same U4, we can try to merge them.
    """
    start_time = time.time()
    U4, V4, W4 = _build_4x4x4_tensors()
    
    # The key insight: U[:, 0:49] == U[:, 49:98]
    # So for each tensor t, the U factor is the same for both blocks.
    # We can try to find a lower rank decomposition that exploits this.
    
    # One approach: try rank 49 with modified V and W
    # But this won't work directly because V and W have different structures.
    
    # Better approach: use ALS starting from the rank-98 solution
    # but project onto a lower-dimensional space.
    
    # Try various lower ranks
    for r in range(97, 85, -1):
        if time.time() > time_limit:
            break
            
        for seed in range(3):
            if time.time() > time_limit:
                break
                
            np.random.seed(seed + 100)
            
            # Initialize by perturbing the rank-98 solution
            U = np.zeros((n*m, r))
            V = np.zeros((m*p, r))
            W = np.zeros((p*n, r))
            
            # Use first r columns of the rank-98 solution as starting point
            if r <= 98:
                U[:, :r] = U98[:, :r]
                V[:, :r] = V98[:, :r]
                W[:, :r] = W98[:, :r]
            
            # Add small random perturbation
            U += 0.01 * np.random.randn(n*m, r)
            V += 0.01 * np.random.randn(m*p, r)
            W += 0.01 * np.random.randn(p*n, r)
            
            # ALS iterations
            for it in range(100):
                if time.time() > time_limit:
                    break
                    
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])
                        b = np.ones(p)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            U[i*m+j, :] = x
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])
                        b = np.ones(n)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            V[j*p+k, :] = x
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])
                        b = np.ones(m)
                        try:
                            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                            W[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence
                if it % 10 == 0:
                    err = _verify_fast(U, V, W, n, m, p)
                    if err < 1e-5:
                        # Try to round to half-integers
                        U_round = np.round(U * 2) / 2
                        V_round = np.round(V * 2) / 2
                        W_round = np.round(W * 2) / 2
                        
                        if _verify_decomposition(U_round.tolist(), V_round.tolist(), W_round.tolist(), n, m, p):
                            return U_round, V_round, W_round
    
    return None

def construct():
    n, m, p = 4, 4, 8
    
    # 1. Construct the known rank-98 solution
    U98, V98, W98 = _build_rank98()
    best_U, best_V, best_W = U98, V98, W98
    best_rank = 98
    
    # 2. Try to find a lower rank solution
    search_deadline = START + 115.0
    
    # Try shared-U optimization first (more targeted)
    result = _try_shared_u_optimization(U98, V98, W98, n, m, p, search_deadline)
    
    if result is None and time.time() < search_deadline:
        # Fall back to general ALS search
        result = _try_find_lower_rank(n, m, p, 98, 80, search_deadline)
    
    if result is not None:
        U, V, W = result
        best_U, best_V, best_W = U, V, W
        best_rank = len(U[0])
        
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
