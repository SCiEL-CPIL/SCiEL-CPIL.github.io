"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 2, 12]

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly represents matrix multiplication."""
    n, m, p = n, m, p
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    # The tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # But actually the decomposition is: (A*B)[i,k] = sum_j A[i,j] * B[j,k]
    # In tensor form: T_{i,j,k} where A is n x m, B is m x p, C is n x p
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] has T[i,j,k] = 1 if we view it as the multiplication tensor
    
    # The rank-r decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Wait, the interface says:
    # U: shape (n*m, rank) - indexed by [i*m+j, c]
    # V: shape (m*p, rank) - indexed by [j*p+k, c]  
    # W: shape (p*n, rank) - indexed by [k*n+i, c]
    # And the convention is W[k*n+i, r]
    
    # So the product is: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal T[i,j,k]
    # where T[i,j,k] = 1 for all i,j,k (since multiplication tensor is all 1s in this view)
    
    # Actually no. The multiplication tensor M_{n,m,p} has entries:
    # M[i,j,k] = 1 for all valid i,j,k (it's the identity-like tensor for matmul)
    # Wait, let me reconsider. For matrix multiplication A (n x m) times B (m x p) = C (n x p):
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor is T[i,j,k] = delta (just 1) because it's the structure tensor.
    
    # Actually the tensor M_{n,m,p} in the standard tensor decomposition sense:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # And we want: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Wait that doesn't seem right dimensionally. Let me re-read the contract.
    # "U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)"
    # "W[k*n+i, r]"
    
    # The standard Strassen-style decomposition:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # In rank-r form: C[i,k] = sum_c (sum_j U1[i,j,c] A[i,j]) * (sum_k ... )
    
    # Actually the correct tensor decomposition for matrix multiplication:
    # The tensor T has indices (i, j, k) where i indexes rows of A (and C), 
    # j indexes cols of A (and rows of B), k indexes cols of B (and C).
    # T[i,j,k] = 1 (the multiplication is linear in all three)
    # Rank-r decomposition: T[i,j,k] = sum_c u_c[i,j] v_c[j,k] w_c[k,i]
    
    # So: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    # Verify on random inputs
    np.random.seed(42)
    for trial in range(5):
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C_expected = A @ B
        C_decomp = np.zeros((n, p))
        for c in range(len(U[0])):
            # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
            # C[i,k] += U[i,j,c] * A[i,j] * V[j,k,c] * B[j,k] * W[k,i,c]
            # Actually: C[i,k] = sum_c (sum_j U[i*m+j,c] * A[i,j]) * (sum_k V[j*p+k,c] * B[j,k]) * W[k*n+i,c]
            # No, that's not right either. Let me think more carefully.
            pass
        
        # Simpler: directly check the tensor identity
    # Check tensor identity: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(len(U[0])):
                    s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_rank(r, n, m, p, max_time=30):
    """Try to find a rank-r decomposition using random initialization and NNLS polishing."""
    start_time = time.time()
    
    # Try several random initializations
    for attempt in range(20):
        if time.time() - start_time > max_time:
            return None
        
        np.random.seed(attempt * 12345 + 7)
        # Random initialization with small entries
        U = np.random.randn(n*m, r) * 0.5
        V = np.random.randn(m*p, r) * 0.5
        W = np.random.randn(p*n, r) * 0.5
        
        # Build the system: for each (i,j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # This is nonlinear. Use a simple iterative approach:
        # Fix V and W, solve for U, then fix U and W, solve for V, etc.
        
        for iteration in range(50):
            if time.time() - start_time > max_time:
                return None
            
            # Update U: for each (i,j), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all k
            # This is: U[i*m+j,:] @ diag(...) ... actually:
            # For fixed i,j: sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
            # Let M_k = V[j*p+k,:] * W[k*n+i,:] element-wise, then sum_c U[i*m+j,c]*M_k[c] = 1
            # This is a linear system: (matrix with rows M_k) @ U[i*m+j,:] = ones
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k,:] = V[j*p+k,:] * W[k*n+i,:]
                    if r > p:
                        # Use NNLS or least squares
                        try:
                            sol, _ = nnls(M, np.ones(p))
                        except:
                            sol = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                        U[i*m+j,:] = sol
            if time.time() - start_time > max_time:
                return None
            
            # Update V: for each (j,k), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i,:] = U[i*m+j,:] * W[k*n+i,:]
                    if r > n:
                        try:
                            sol, _ = nnls(M, np.ones(n))
                        except:
                            sol = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    else:
                        sol = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    V[j*p+k,:] = sol
            if time.time() - start_time > max_time:
                return None
            
            # Update W: for each (k,i), sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all j
            for k in range(p):
                for i in range(n):
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j,:] = U[i*m+j,:] * V[j*p+k,:]
                    if r > m:
                        try:
                            sol, _ = nnls(M, np.ones(m))
                        except:
                            sol = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    else:
                        sol = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    W[k*n+i,:] = sol
            if time.time() - start_time > max_time:
                return None
        
        # Check if we got a valid decomposition
        U_list = [[float(U[i,j]) for j in range(r)] for i in range(n*m)]
        V_list = [[float(V[i,j]) for j in range(r)] for i in range(m*p)]
        W_list = [[float(W[i,j]) for j in range(r)] for i in range(p*n)]
        
        if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
            # Try to snap to half-integers
            U_snap = np.round(U * 2) / 2
            V_snap = np.round(V * 2) / 2
            W_snap = np.round(W * 2) / 2
            U_snap_list = [[float(U_snap[i,j]) for j in range(r)] for i in range(n*m)]
            V_snap_list = [[float(V_snap[i,j]) for j in range(r)] for i in range(m*p)]
            W_snap_list = [[float(W_snap[i,j]) for j in range(r)] for i in range(p*n)]
            if _verify(U_snap_list, V_snap_list, W_snap_list, n, m, p, tol=1e-9):
                return U_snap_list, V_snap_list, W_snap_list
            return U_list, V_list, W_list
    
    return None

def _schoolbook(n, m, p):
    """Return the schoolbook decomposition of rank n*m*p."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s kill
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find lower rank decompositions
    for r in range(best_rank - 1, max(best_rank - 5, 0), -1):
        if time.time() > deadline:
            break
        
        result = _try_rank(r, n, m, p, max_time=min(25, deadline - time.time() - 5))
        if result is not None:
            U_list, V_list, W_list = result
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best = {"U": U_list, "V": V_list, "W": W_list}
                best_rank = r
                break  # Found minimal rank, stop
    
    # If we still have the schoolbook, try a few more random attempts at rank best_rank-1
    if best_rank == n * m * p and time.time() < deadline - 10:
        result = _try_rank(best_rank - 1, n, m, p, max_time=min(20, deadline - time.time() - 5))
        if result is not None:
            U_list, V_list, W_list = result
            if _verify(U_list, V_list, W_list, n, m, p, tol=1e-8):
                best = {"U": U_list, "V": V_list, "W": W_list}
    
    return best
# EVOLVE-BLOCK-END
