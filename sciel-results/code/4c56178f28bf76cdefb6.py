import numpy as np
import time
from scipy.optimize import minimize

# Dimensions
n, m, p = 2, 3, 5
N_U = n * m  # 6
N_V = m * p  # 15
N_W = p * n  # 10

# Precompute indices for the tensor decomposition
# U index: i * m + j
# V index: j * p + k
# W index: k * n + i
_u_indices_list = []
_v_indices_list = []
_w_indices_list = []
for i in range(n):
    for j in range(m):
        for k in range(p):
            _u_indices_list.append(i * m + j)
            _v_indices_list.append(j * p + k)
            _w_indices_list.append(k * n + i)

U_INDICES = np.array(_u_indices_list)
V_INDICES = np.array(_v_indices_list)
W_INDICES = np.array(_w_indices_list)
NUM_TERMS = len(_u_indices_list)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin for 150s limit
    
    # Schoolbook fallback (rank 30)
    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_solution = get_schoolbook()
    best_rank = n * m * p
    
    # Helper to verify a solution
    def verify_solution(U, V, W):
        # U, V, W are numpy arrays
        terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
        sums = terms.sum(axis=1)
        return np.allclose(sums, 1.0, atol=1e-9)
    
    # Structured Strassen decomposition
    # Split 3x5 into 3x2 (k=0,1) and 3x3 (k=2,3,4)
    # For 2x2x2 part: use Strassen (7 multiplications)
    # For 2x3x3 part: use schoolbook (9 multiplications)
    # Total: 7 + 9 = 16 rank
    
    def get_strassen_decomposition():
        rank = 16
        U = np.zeros((N_U, rank))
        V = np.zeros((N_V, rank))
        W = np.zeros((N_W, rank))
        
        # Strassen for 2x2x2 (k=0,1)
        # A = [[a00, a01], [a10, a11]] (2x2)
        # B = [[b00, b01], [b10, b11]] (2x2)
        # C = A*B (2x2)
        # 
        # m1 = (a00+a11)*(b00+b11)
        # m2 = (a10+a11)*b00
        # m3 = a00*(b01-b11)
        # m4 = a11*(b10-b00)
        # m5 = (a00+a10)*b11
        # m6 = (a11-a10)*(b00+b10)
        # m7 = (a01-a11)*(b01+b11)
        #
        # C00 = m1 + m4 - m5 + m6
        # C01 = m3 + m5
        # C10 = m2 + m4
        # C11 = m1 - m2 + m3 + m7
        
        # U indices for A: i*m + j where i in {0,1}, j in {0,1}
        # V indices for B: j*p + k where j in {0,1,2}, k in {0,1}
        # W indices for C: k*n + i where k in {0,1}, i in {0,1}
        
        # m1: (a00+a11)*(b00+b11)
        # U: a00 at (0,0)=0, a11 at (1,1)=4
        # V: b00 at (0,0)=0, b11 at (1,1)=6
        # W: C00 at (0,0)=0, C11 at (1,1)=3
        U[0, 0] = 1  # a00
        U[4, 0] = 1  # a11
        V[0, 0] = 1  # b00
        V[6, 0] = 1  # b11
        W[0, 0] = 1  # C00
        W[3, 0] = 1  # C11
        
        # m2: (a10+a11)*b00
        # U: a10 at (1,0)=3, a11 at (1,1)=4
        # V: b00 at (0,0)=0
        # W: C10 at (0,1)=2
        U[3, 1] = 1  # a10
        U[4, 1] = 1  # a11
        V[0, 1] = 1  # b00
        W[2, 1] = 1  # C10
        
        # m3: a00*(b01-b11)
        # U: a00 at (0,0)=0
        # V: b01 at (0,1)=1, b11 at (1,1)=6
        # W: C01 at (1,0)=1
        U[0, 2] = 1  # a00
        V[1, 2] = 1  # b01
        V[6, 2] = -1  # b11
        W[1, 2] = 1  # C01
        
        # m4: a11*(b10-b00)
        # U: a11 at (1,1)=4
        # V: b10 at (1,0)=5, b00 at (0,0)=0
        # W: C00 at (0,0)=0, C10 at (0,1)=2
        U[4, 3] = 1  # a11
        V[5, 3] = 1  # b10
        V[0, 3] = -1  # b00
        W[0, 3] = 1  # C00
        W[2, 3] = 1  # C10
        
        # m5: (a00+a10)*b11
        # U: a00 at (0,0)=0, a10 at (1,0)=3
        # V: b11 at (1,1)=6
        # W: C01 at (1,0)=1
        U[0, 4] = 1  # a00
        U[3, 4] = 1  # a10
        V[6, 4] = 1  # b11
        W[1, 4] = 1  # C01
        
        # m6: (a11-a10)*(b00+b10)
        # U: a11 at (1,1)=4, a10 at (1,0)=3
        # V: b00 at (0,0)=0, b10 at (1,0)=5
        # W: C00 at (0,0)=0
        U[4, 5] = 1  # a11
        U[3, 5] = -1  # a10
        V[0, 5] = 1  # b00
        V[5, 5] = 1  # b10
        W[0, 5] = 1  # C00
        
        # m7: (a01-a11)*(b01+b11)
        # U: a01 at (0,1)=1, a11 at (1,1)=4
        # V: b01 at (0,1)=1, b11 at (1,1)=6
        # W: C11 at (1,1)=3
        U[1, 6] = 1  # a01
        U[4, 6] = -1  # a11
        V[1, 6] = 1  # b01
        V[6, 6] = 1  # b11
        W[3, 6] = 1  # C11
        
        # Schoolbook for 2x3x3 (k=2,3,4)
        # A is 2x3, B is 3x3, C is 2x3
        # C[i][k] = sum_j A[i][j]*B[j][k]
        # 
        # U indices: i*m + j where i in {0,1}, j in {0,1,2}
        # V indices: j*p + k where j in {0,1,2}, k in {2,3,4}
        # W indices: k*n + i where k in {2,3,4}, i in {0,1}
        
        for j in range(3):
            for k in range(3):
                col = 7 + j * 3 + k
                for i in range(2):
                    U[i * 3 + j, col] = 1  # A[i][j]
                    V[j * 5 + (k + 2), col] = 1  # B[j][k+2]
                    W[(k + 2) * 2 + i, col] = 1  # C[k+2][i]
        
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Verify the Strassen decomposition
    strassen_sol = get_strassen_decomposition()
    U_arr = np.array(strassen_sol["U"])
    V_arr = np.array(strassen_sol["V"])
    W_arr = np.array(strassen_sol["W"])
    
    if verify_solution(U_arr, V_arr, W_arr):
        best_solution = strassen_sol
        best_rank = 16
    
    # Try to find lower rank using random search with iterative refinement
    # The tensor T has entries T[i,j,k] = 1 for all valid indices
    # We need to find U, V, W such that sum_r U[i,j,r]*V[j,k,r]*W[k,i,r] = T[i,j,k]
    
    def try_random_rank(rank, max_iters=500, seed=42):
        """Try to find a rank-rank decomposition using random init + iterative refinement"""
        if rank >= best_rank:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = rng.randn(N_U, rank) * 0.1
        V = rng.randn(N_V, rank) * 0.1
        W = rng.randn(N_W, rank) * 0.1
        
        # The tensor T[i,j,k] = 1 for all i,j,k
        # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        # Use iterative multiplicative updates (like NMF but for tensor)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j,r), U[i*m+j, r] should satisfy
            # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
            # This is a linear system in U for each (i,j)
            # For fixed (i,j), let u_r = U[i*m+j, r]
            # Then sum_r u_r * (V[j*p+k, r] * W[k*n+i, r]) = 1 for k=0..p-1
            # This gives p equations in rank unknowns
            
            # Build the system: M[k, r] = V[j*p+k, r] * W[k*n+i, r]
            # Solve M @ u = 1 for u
            
            for i in range(n):
                for j in range(m):
                    # Build matrix M of shape (p, rank)
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve M @ u = ones(p)
                    # Use least squares if rank > p
                    if rank <= p:
                        # Underdetermined or square, use lstsq
                        u, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    else:
                        u, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(p), rcond=None)
                    
                    U[i*m+j, :] = u
            
            # Update V: for each (j,k,r), V[j*p+k, r]
            # For fixed (j,k), let v_r = V[j*p+k, r]
            # sum_r U[i*m+j, r] * v_r * W[k*n+i, r] = 1 for all i
            # M[i, r] = U[i*m+j, r] * W[k*n+i, r]
            # Solve M @ v = ones(n)
            
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    v, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(n), rcond=None)
                    V[j*p+k, :] = v
            
            # Update W: for each (k,i,r), W[k*n+i, r]
            # For fixed (k,i), let w_r = W[k*n+i, r]
            # sum_r U[i*m+j, r] * V[j*p+k, r] * w_r = 1 for all j
            # M[j, r] = U[i*m+j, r] * V[j*p+k, r]
            # Solve M @ w = ones(m)
            
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    w, residuals, rank_sol, sv = np.linalg.lstsq(M, np.ones(m), rcond=None)
                    W[k*n+i, :] = w
            
            # Check convergence
            terms = U[U_INDICES] * V[V_INDICES] * W[W_INDICES]
            sums = terms.sum(axis=1)
            error = np.max(np.abs(sums - 1.0))
            
            if error < 1e-9:
                return {"U": U, "V": V, "W": W}
        
        return None
    
    # Try ranks from 15 down to best_rank - 1
    for rank in range(min(best_rank - 1, 15), 14, -1):
        if time.time() > deadline - 10:
            break
        
        # Try multiple random seeds
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            result = try_random_rank(rank, max_iters=200, seed=seed + rank * 100)
            if result is not None:
                U_test = np.array(result["U"])
                V_test = np.array(result["V"])
                W_test = np.array(result["W"])
                
                if verify_solution(U_test, V_test, W_test):
                    best_solution = {"U": U_test.tolist(), "V": V_test.tolist(), "W": W_test.tolist()}
                    best_rank = rank
                    break
        
        if best_rank <= rank:
            break
    
    return best_solution
