import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p, atol=1e-10):
    """Verify that sum_r U[:,r] V[:,r] W[:,r] equals the matmul tensor."""
    nm, mp, pn = n * m, m * p, p * n
    rank = U.shape[1]
    T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    T_target = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T_target[i * m + j, j * p + k, k * n + i] = 1.0
    return np.allclose(T_decomp, T_target, atol=atol)

def _build_target_tensor(n, m, p):
    """Build the matmul tensor T[i*m+j, j*p+k, k*n+i] = 1."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    return T

def _decomposition_residual(U, V, W, T_target):
    """Compute T_decomp - T_target efficiently."""
    nm, mp, pn = T_target.shape
    rank = U.shape[1]
    T_decomp = np.zeros((nm, mp, pn), dtype=np.float64)
    for r in range(rank):
        T_decomp += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
    return T_decomp - T_target

def _try_rank16_algebraic(n, m, p, time_budget):
    """
    Search for rank-16 decomposition by fixing V to a sparse/generic pattern
    and solving the resulting system for U and W.
    
    Strategy: For rank r, if we fix V (mp x r), the equations become:
      For each (i,j,k): sum_r U[i*m+j, r] * W[k*n+i, r] = T[i*m+j, j*p+k, k*n+i] / V[j*p+k, r]
    But this is still bilinear in U and W.
    
    Better approach: Fix V. Then for each (i,j), the p equations in k are:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1  for k=0..p-1
    
    If we also fix W, then U can be solved linearly. If we fix U, W can be solved linearly.
    
    Let's try: fix V and W randomly (sparse), then solve for U linearly.
    For fixed V and W, the equation for (i,j,k) is:
      sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
    
    So for each (i,j), we have p linear equations in the m*r variables U[i*m+j, :]:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1,  k=0..p-1
    
    This is a p x r linear system for U[i*m+j, :]. If p >= r, we can solve.
    Here p=3, r=16, so p < r. Underdetermined.
    
    Alternative: Fix U and W, solve for V. For each (j,k), n equations in m*r vars:
      sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1, i=0..n-1
    n=2, r=16. Also underdetermined.
    
    Let's try a different approach: Use the rank-17 solution and try to merge
    two terms into one using a Strassen-like trick.
    """
    nm, mp, pn = n * m, m * p, p * n
    deadline = time.time() + time_budget
    T_target = _build_target_tensor(n, m, p)
    
    # Start from the rank-17 solution and try to find a rank-16 by merging terms
    # The rank-17 has 7 Strassen + 4 schoolbook (col 2) + 6 schoolbook (col 3)
    # Try to merge some schoolbook terms using bilinear tricks
    
    # For the "col 3" part (A[:, 0:3] * B[0:3, 2] -> C[:, 2]):
    # We have 6 terms: A00*B02, A01*B12, A02*B22, A10*B02, A11*B12, A12*B22
    # Can we do this in fewer terms? For 2x3x1, the rank is at least 3 (since 2*3*1=6 entries,
    # but the tensor is T[i*3+j, j*3+2, 2*2+i] = T[i*3+j, j*3+2, 4+i] = 1 for i=0,1; j=0,1,2
    # That's 6 ones. The rank of this sub-tensor... 
    # Actually for 2x3x1, it's just outer product structure. Let me think...
    # C[2, i] = sum_j A[i,j] * B[j,2]
    # This is a 2x3x1 tensor with entries T[i*3+j, j*3+2, 4+i] = 1
    # Flattened: for (i,j,k=2): position (i*3+j, j*3+2, 4+i)
    # This tensor has rank 3 (it's the sum of 3 rank-1 tensors: for each j, A[i,j]*B[j,2])
    # But can we do better? The rank of a 2x3x1 tensor is at most min(2*3, 1*2, 2*1) = 2
    # Wait, rank of tensor T in I x J x K is at most min(|I|*|J|, |J|*|K|, |K|*|I|)
    # Here |I|=6, |J|=3 (only j*3+2 for j=0,1,2 so 3 values), |K|=2
    # So rank <= min(18, 6, 12) = 6. But the tensor is sparse...
    # Actually the nonzero entries are at specific positions. Let me just try random search.
    
    # Try random search for rank 16
    rng = np.random.default_rng(12345)
    
    for attempt in range(200):
        if time.time() > deadline - 5:
            break
        
        rank = 16
        U = rng.standard_normal((nm, rank))
        V = rng.standard_normal((mp, rank))
        W = rng.standard_normal((pn, rank))
        
        # Iterative refinement: alternate solving for one factor
        for iteration in range(100):
            if time.time() > deadline - 5:
                break
            
            # Solve for W: for each (k,n+i) index in W, the equations are:
            # For each (i,j): sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # For fixed (i), and varying k: sum_r (U[i*m+j,r] * V[j*p+k,r]) * W[k*n+i,r] = 1
            # This couples across j. Let's do it differently.
            
            # For fixed (i, k), the W entry W[k*n+i, r] appears in equations for all j:
            # sum_j sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] ... no, W is per (i,k)
            # Actually W[k*n+i, r] multiplies U[i*m+j, r] * V[j*p+k, r] for each j
            # But the target is T[i*m+j, j*p+k, k*n+i] = 1 only when the indices match
            # For the matmul tensor, T[i*m+j, j*p+k, k*n+i] = 1 always (it's the structure)
            # So the equation is: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for ALL (i,j,k)
            
            # For fixed (i,k), and all j: sum_r W[k*n+i,r] * (U[i*m+j,r] * V[j*p+k,r]) = 1
            # This is: for each j, sum_r W[k*n+i,r] * A_{ij}^{(k)}(r) = 1
            # where A_{ij}^{(k)}(r) = U[i*m+j,r] * V[j*p+k,r]
            # This is m equations in r unknowns for W[k*n+i, :]
            # m=3, r=16. Underdetermined. Use least squares.
            
            # Let's do least squares for W
            # For each (i,k): we have m equations: for j=0..m-1
            #   sum_r W[k*n+i,r] * U[i*m+j,r] * V[j*p+k,r] = 1
            # Matrix form: M_{j,r} = U[i*m+j,r] * V[j*p+k,r], solve M @ W[k*n+i,:] = ones(m)
            
            W_new = np.zeros_like(W)
            for i in range(n):
                for k in range(p):
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    # Solve M @ w = ones(m)
                    try:
                        w, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                        W_new[k*n+i, :] = w
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            W = W_new
            
            # Solve for U similarly
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        u, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                        U_new[i*m+j, :] = u
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            U = U_new
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        v, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                        V_new[j*p+k, :] = v
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            V = V_new
            
            # Check convergence
            residual = _decomposition_residual(U, V, W, T_target)
            if np.max(np.abs(residual)) < 1e-10:
                return U, V, W, rank
        
        # Try with different random init
        if time.time() > deadline - 10:
            break
    
    return None

def construct():
    n, m, p = DIMENSIONS
    
    start_time = time.time()
    total_budget = 120  # seconds total
    search_budget = min(90, total_budget - 10)
    
    # Try to find rank-16 decomposition
    result = _try_rank16_algebraic(n, m, p, search_budget)
    
    if result is not None:
        U, V, W, rank = result
        if _verify_decomposition(U, V, W, n, m, p, atol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: Rank 17 decomposition for 2x3x3 matrix multiplication
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Part 2: Schoolbook for A[0:2, 2] * B[2, 0:2] -> C[0:2, 0:2]
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1
    
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1
    
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1
    
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1
    
    # Part 3: Schoolbook for A[0:2, 0:3] * B[0:3, 2] -> C[0:2, 2]
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1
    
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1
    
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1
    
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1
    
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1
    
    U[5, 16] = 1
    V[8, 16] = 1
    W[5, 16] = 1
    
    # Verify and return
    if not _verify_decomposition(U, V, W, n, m, p, atol=1e-10):
        # Should not happen, but safety check
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
