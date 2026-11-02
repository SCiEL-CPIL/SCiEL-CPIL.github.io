"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _verify(U, V, W, n, m, p, tol=1e-6):
    """Check if U,V,W exactly decompose the matrix multiplication tensor."""
    # The tensor T[i,j,k] = delta (i.e., T_{i,j,k} = 1 for the multiplication tensor)
    # Actually, the matrix multiplication tensor is defined as:
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # The tensor is T_{i,j,k} = 1 (it's the "multiplication" tensor where each term is A_{ij} * B_{jk})
    # 
    # In the factorization: T_{i,j,k} = sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r}
    # Wait, let me re-read the convention.
    # 
    # U has shape (n*m, rank), indexed by (i,j) -> row i*m+j
    # V has shape (m*p, rank), indexed by (j,k) -> row j*p+k
    # W has shape (p*n, rank), indexed by (k,i) -> row k*n+i
    # 
    # The decomposition says: for the matrix multiplication tensor,
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should equal the tensor value.
    # The matrix multiplication tensor T[i,j,k] = 1 for all i,j,k (since C_ik = sum_j A_ij B_jk,
    # each term A_ij * B_jk contributes with coefficient 1).
    #
    # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k.
    
    rank = U.shape[1]
    # Check all (i,j,k) combinations
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _to_integer_or_half(U, V, W):
    """Try to convert to integer or half-integer entries."""
    def convert(M):
        M2 = M * 2
        rounded = np.round(M2)
        if np.allclose(M2, rounded, atol=1e-6):
            return rounded / 2.0
        return M
    return convert(U), convert(V), convert(W)


def _strassen_4x4x4():
    """
    Build a rank-63 decomposition for 4x4x4 using Strassen-like block structure.
    This is the standard Strassen decomposition applied to 2x2 blocks, giving rank 7^2 = 49? 
    No, Strassen for 2x2x2 has rank 7. For 4x4x4, we can use block Strassen:
    Split into 2x2 blocks of 2x2 matrices. The block multiplication uses 7 multiplications
    of 2x2 blocks, each of which uses rank 7. So total rank = 7 * 7 = 49.
    
    Actually, let me think more carefully. The 4x4x4 matrix multiplication tensor can be
    decomposed by viewing it as 2x2 blocks of 2x2 matrices. Each block multiplication
    C_block = A_block * B_block uses Strassen's 7 multiplications. But the Strassen
    decomposition itself involves additions of 2x2 matrices.
    
    Let me use the known result: the minimal rank for 4x4x4 is known to be at most 49
    (from Strassen) and at least 48 (from lower bounds). Let's use 49.
    
    Actually, I'll implement the standard Strassen-based 4x4x4 decomposition of rank 49.
    """
    # Strassen 2x2x2 rank-7 decomposition
    # For 2x2: A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # m1 = (a11+a22)(b11+b22)
    # m2 = (a21+a22)b11
    # m3 = a11(b12-b22)
    # m4 = a22(b21-b11)
    # m5 = (a11+a12)b22
    # m6 = (a21-a11)(b11+b12)
    # m7 = (a12-a22)(b21+b22)
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # For 4x4x4, we split into 2x2 blocks:
    # A = [[A11, A12], [A21, A22]] where each Aij is 2x2
    # B = [[B11, B12], [B21, B22]] where each Bij is 2x2
    # C = [[C11, C12], [C21, C22]] where each Cij is 2x2
    # 
    # Using Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    #
    # Each Mx is a 2x2 matrix multiplication, which has rank 7.
    # So total rank = 7 * 7 = 49.
    
    n = m = p = 4
    
    # The 2x2x2 Strassen decomposition in terms of (i,j,k) indices for 2x2:
    # i,j,k in {0,1}
    # U_2x2[i*2+j, r], V_2x2[j*2+k, r], W_2x2[k*2+i, r]
    
    # Strassen 2x2x2:
    # The 7 terms:
    # m1: (a00+a11)(b00+b11) -> contributes to C00
    # m2: (a10+a11)b00 -> contributes to C10
    # m3: a00(b01-b11) -> contributes to C01
    # m4: a11(b10-b00) -> contributes to C10 and C00
    # m5: (a00+a01)b11 -> contributes to C01 and C00
    # m6: (a10-a00)(b00+b01) -> contributes to C11
    # m7: (a01-a11)(b10+b11) -> contributes to C00 and C11
    
    # Let me define the Strassen 2x2x2 decomposition explicitly.
    # For 2x2x2, indices i,j,k in {0,1}
    # U[i*2+j, r], V[j*2+k, r], W[k*2+i, r]
    
    # The standard Strassen rank-7 decomposition:
    # Term 1: (a00+a11) * b00 -> no wait, let me be more careful.
    
    # Actually, let me just use the known factorization.
    # For 2x2x2 with Strassen:
    # The 7 rank-1 terms are:
    # r=0: U[0,0]=1, U[1,0]=0, U[2,0]=1, U[3,0]=0  (a00+a11)
    #       V[0,0]=1, V[1,0]=0, V[2,0]=1, V[3,0]=0  (b00+b11)
    #       W[0,0]=1, W[1,0]=0, W[2,0]=0, W[3,0]=0  (contributes to C00)
    # ... this is getting complicated. Let me use a different approach.
    
    # I'll use the fact that we can verify any decomposition numerically.
    # Let me try a random initialization and use gradient descent to find a low-rank decomposition.
    pass


def _random_decomposition(n, m, p, rank, seed=None):
    """Initialize a random rank-rank decomposition."""
    if seed is not None:
        np.random.seed(seed)
    U = np.random.randn(n * m, rank) * 0.5
    V = np.random.randn(m * p, rank) * 0.5
    W = np.random.randn(p * n, rank) * 0.5
    return U, V, W


def _gradient_step(U, V, W, n, m, p, lr=0.01, max_iter=1000, tol=1e-8, deadline=None):
    """
    Use gradient descent to minimize the error of the tensor decomposition.
    Error = sum_{i,j,k} (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
    """
    rank = U.shape[1]
    
    # Precompute indices
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Compute gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i, j, k in indices:
            ui = U[i*m+j, :]
            vj = V[j*p+k, :]
            wk = W[k*n+i, :]
            val = np.dot(ui, vj * wk)
            err = val - 1.0
            # d/dU: err * V[j*p+k,r] * W[k*n+i,r]
            dU[i*m+j, :] += err * vj * wk
            dV[j*p+k, :] += err * ui * wk
            dW[k*n+i, :] += err * ui * vj
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        # Check convergence
        if it % 50 == 0:
            max_err = 0
            for i, j, k in indices:
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                max_err = max(max_err, abs(val - 1.0))
            if max_err < tol:
                break
    
    return U, V, W


def _verify_fast(U, V, W, n, m, p, tol=1e-6):
    """Fast verification using tensor operations."""
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol, max_err


def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Leave 20s safety margin
    
    # Start with schoolbook
    U_school, V_school, W_school = _schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p  # 96
    
    # Try to find lower rank decompositions
    # Known: 4x4x6 matrix multiplication
    # Schoolbook rank = 96
    # Let's try various ranks
    
    # First, try to improve using gradient descent from schoolbook with reduced rank
    # This is unlikely to work directly, but let's try
    
    # Try random initializations with various ranks
    ranks_to_try = [96, 90, 85, 80, 75, 70, 65, 60, 55, 50]
    
    for target_rank in ranks_to_try:
        if time.time() > deadline - 30:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            U, V, W = _random_decomposition(n, m, p, target_rank, seed=target_rank * 100 + attempt)
            U, V, W = _gradient_step(U, V, W, n, m, p, lr=0.005, max_iter=2000, deadline=deadline - 15)
            
            valid, max_err = _verify_fast(U, V, W, n, m, p, tol=1e-4)
            if valid and target_rank < best_rank:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                print(f"Found valid decomposition with rank {target_rank}, max_err={max_err}")
                break
    
    # Try to polish the best solution
    if time.time() < deadline - 10:
        U, V, W = _gradient_step(best_U, best_V, best_W, n, m, p, lr=0.001, max_iter=1000, tol=1e-10, deadline=deadline - 5)
        valid, max_err = _verify_fast(U, V, W, n, m, p, tol=1e-6)
        if valid:
            best_U, best_V, best_W = U, V, W
    
    # Convert to integer or half-integer if possible
    U, V, W = _to_integer_or_half(best_U, best_V, best_W)
    
    # Final verification
    valid, max_err = _verify_fast(U, V, W, n, m, p, tol=1e-4)
    if not valid:
        # Fall back to schoolbook
        U, V, W = _schoolbook(n, m, p)
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
