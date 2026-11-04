"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
import time as _time
import numpy as _np

_START_TIME = _time.time()
_DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _time_left():
    return _DEADLINE - _time.time()

def _verify(U, V, W, n=2, m=4, p=4):
    """Verify that U, V, W exactly decompose the (n,m,p) matmul tensor."""
    rank = len(U[0])
    # Build the matmul tensor: T[i,j,k] = 1 if i=k (output row i, input col j, output col k)
    # Actually: C[i,k] = sum_j A[i,j]*B[j,k]
    # Tensor T_{i,j,k} where output index is (i,k), first input is (i,j), second is (j,k)
    # U has shape (n*m, rank): index i*m + j
    # V has shape (m*p, rank): index j*p + k
    # W has shape (p*n, rank): index k*n + i  (trace convention W[k*n+i, r])
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 if i==k... no
    # Actually the tensor is: for each (i,j,k), the coefficient is 1
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal 1 for all (i,j,k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return False
    return True

def _construct_strassen_28():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(a_idx, b_idx, c_idx, start_col):
        a, b, c, d = a_idx
        e, f, g, h = b_idx
        x00, x01, x10, x11 = c_idx
        
        col = start_col
        U[a][col] += 1; U[d][col] += 1; V[e][col] += 1; V[h][col] += 1
        W[x00][col] += 1; W[x11][col] += 1
        
        col = start_col + 1
        U[c][col] += 1; U[d][col] += 1; V[e][col] += 1
        W[x10][col] += 1; W[x11][col] -= 1
        
        col = start_col + 2
        U[a][col] += 1; V[f][col] += 1; V[h][col] -= 1
        W[x01][col] += 1; W[x11][col] += 1
        
        col = start_col + 3
        U[d][col] += 1; V[g][col] += 1; V[e][col] -= 1
        W[x00][col] += 1; W[x10][col] += 1
        
        col = start_col + 4
        U[a][col] += 1; U[b][col] += 1; V[h][col] += 1
        W[x00][col] -= 1; W[x01][col] += 1
        
        col = start_col + 5
        U[c][col] += 1; U[a][col] -= 1; V[e][col] += 1; V[f][col] += 1
        W[x11][col] += 1
        
        col = start_col + 6
        U[b][col] += 1; U[d][col] -= 1; V[g][col] += 1; V[h][col] += 1
        W[x00][col] += 1

    add_strassen_block([0, 1, 4, 5], [0, 1, 4, 5], [0, 2, 1, 3], 0)
    add_strassen_block([0, 1, 4, 5], [2, 3, 6, 7], [4, 6, 5, 7], 7)
    add_strassen_block([2, 3, 6, 7], [8, 9, 12, 13], [0, 2, 1, 3], 14)
    add_strassen_block([2, 3, 6, 7], [10, 11, 14, 15], [4, 6, 5, 7], 21)
    
    return {"U": U, "V": V, "W": W}

def _search_lower_rank():
    """Try to find a decomposition with rank < 28 using linear algebra over rationals."""
    n, m, p = DIMENSIONS
    num_tensors = n * m * p  # 32
    num_U = n * m  # 8
    num_V = m * p  # 16
    num_W = p * n  # 8
    
    # For a given rank r, we need to find U (8xr), V (16xr), W (8xr)
    # such that for each (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of 32 equations in r*(8+16+8) = 32r unknowns.
    # But it's nonlinear (products of variables from different matrices).
    # 
    # Approach: fix U and V, solve for W linearly. Or use iterative methods.
    # Simpler: try rank 27 and see if a solution exists via randomized search + linear solve.
    
    # For each rank r from 27 down to 20:
    for r in range(27, 19, -1):
        if _time_left() < 5:
            break
        
        # Strategy: randomize U and V, then solve for W via least squares
        # For each output index (k,i) in W, the constraint is:
        # For all (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
        # For fixed U,V, this is linear in W.
        
        for attempt in range(50):
            if _time_left() < 3:
                break
            
            # Random U, V with small integer entries
            U_rand = _np.random.randint(-2, 3, size=(num_U, r))
            V_rand = _np.random.randint(-2, 3, size=(num_V, r))
            
            # Build linear system for W
            # Unknowns: W has num_W * r = 8*r entries
            # Equations: num_tensors = 32
            # For equation (i,j,k): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # Group by W index: for each w_idx = k*n+i, the coefficient of W[w_idx,r] is U[i*m+j,r]*V[j*p+k,r]
            
            # Build matrix A (32 x 8*r) and vector b (32,)
            A = _np.zeros((num_tensors, num_W * r))
            b = _np.ones(num_tensors)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        eq_idx = i * m * p + j * p + k
                        for r_idx in range(r):
                            col_idx = (k * n + i) * r + r_idx
                            A[eq_idx, col_idx] = U_rand[i*m+j, r_idx] * V_rand[j*p+k, r_idx]
            
            # Solve least squares
            try:
                W_sol, residuals, rank_A, sv = _np.linalg.lstsq(A, b, rcond=None)
            except:
                continue
            
            # Check if solution is exact
            W_sol = W_sol.reshape(num_W, r)
            # Check residual
            max_err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for r_idx in range(r):
                            s += U_rand[i*m+j, r_idx] * V_rand[j*p+k, r_idx] * W_sol[k*n+i, r_idx]
                        max_err = max(max_err, abs(s - 1))
            
            if max_err < 1e-10:
                # Found a solution! Convert to integer/half-integer
                U_int = U_rand.tolist()
                V_int = V_rand.tolist()
                W_int = [[round(x, 10) for x in row] for row in W_sol.tolist()]
                # Verify
                if _verify(U_int, V_int, W_int):
                    return {"U": U_int, "V": V_int, "W": W_int}
    
    return None

def construct():
    # Try to find a lower-rank decomposition first
    result = _search_lower_rank()
    if result is not None:
        return result
    
    # Fallback to Strassen-28
    return _construct_strassen_28()
# EVOLVE-BLOCK-END
