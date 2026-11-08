import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 3, 4]

def _build_tensor():
    """Build the matrix multiplication tensor for 3x3 * 3x4."""
    n, m, p = DIMENSIONS
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    return T

def _verify_decomposition(U, V, W, tol=1e-9):
    """Verify that U, V, W exactly decompose the tensor."""
    T = _build_tensor()
    rank = U.shape[1]
    T_recon = np.zeros_like(T)
    for r in range(rank):
        T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
    return np.allclose(T_recon, T, atol=tol)

def _get_rank_34_baseline():
    """Return a verified rank-34 decomposition."""
    n, m, p = DIMENSIONS
    rank = 34
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Left half (columns 0,1 of B) - Strassen-like for 3x3 * 3x2
    # M1 = (A00+A11)*(B00+B11) -> contributes to C00, C11
    U[0][col]=1; U[4][col]=1; V[0][col]=1; V[5][col]=1; W[0][col]=1; W[4][col]=1; col+=1
    # M2 = A11*(B00) -> correction
    U[3][col]=1; U[4][col]=1; V[0][col]=1; W[1][col]=1; W[4][col]=-1; col+=1
    # M3 = A00*(B11-B01) -> 
    U[0][col]=1; V[1][col]=1; V[5][col]=-1; W[3][col]=1; W[4][col]=1; col+=1
    # M4 = A11*(B01-B10) ->
    U[4][col]=1; V[4][col]=1; V[0][col]=-1; W[0][col]=1; W[1][col]=1; col+=1
    # M5 = (A00+A01)*(B11) ->
    U[0][col]=1; U[1][col]=1; V[5][col]=1; W[0][col]=-1; W[3][col]=1; col+=1
    # M6 = (A10-A00)*(B00+B01) ->
    U[3][col]=1; U[0][col]=-1; V[0][col]=1; V[1][col]=1; W[4][col]=1; col+=1
    # M7 = (A01-A11)*(B10+B11) ->
    U[1][col]=1; U[4][col]=-1; V[4][col]=1; V[5][col]=1; W[0][col]=1; col+=1
    # Direct terms for row 2 of A (A20, A21) with left half of B
    U[2][col]=1; V[8][col]=1; W[0][col]=1; col+=1
    U[2][col]=1; V[9][col]=1; W[3][col]=1; col+=1
    U[5][col]=1; V[8][col]=1; W[1][col]=1; col+=1
    U[5][col]=1; V[9][col]=1; W[4][col]=1; col+=1
    # Direct terms for rows 2,3 of A (A20,A21,A22) - wait, n=3 so rows 0,1,2
    # Actually for 3x3, rows are 0,1,2. A20 is row2,col0 = index 2*3+0=6
    U[6][col]=1; V[0][col]=1; W[2][col]=1; col+=1
    U[6][col]=1; V[1][col]=1; W[5][col]=1; col+=1
    U[7][col]=1; V[4][col]=1; W[2][col]=1; col+=1
    U[7][col]=1; V[5][col]=1; W[5][col]=1; col+=1
    U[8][col]=1; V[8][col]=1; W[2][col]=1; col+=1
    U[8][col]=1; V[9][col]=1; W[5][col]=1; col+=1
    # Right half (columns 2,3 of B) - same pattern
    U[0][col]=1; U[4][col]=1; V[2][col]=1; V[7][col]=1; W[6][col]=1; W[10][col]=1; col+=1
    U[3][col]=1; U[4][col]=1; V[2][col]=1; W[7][col]=1; W[10][col]=-1; col+=1
    U[0][col]=1; V[3][col]=1; V[7][col]=-1; W[9][col]=1; W[10][col]=1; col+=1
    U[4][col]=1; V[6][col]=1; V[2][col]=-1; W[6][col]=1; W[7][col]=1; col+=1
    U[0][col]=1; U[1][col]=1; V[7][col]=1; W[6][col]=-1; W[9][col]=1; col+=1
    U[3][col]=1; U[0][col]=-1; V[2][col]=1; V[3][col]=1; W[10][col]=1; col+=1
    U[1][col]=1; U[4][col]=-1; V[6][col]=1; V[7][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[10][col]=1; W[6][col]=1; col+=1
    U[2][col]=1; V[11][col]=1; W[9][col]=1; col+=1
    U[5][col]=1; V[10][col]=1; W[7][col]=1; col+=1
    U[5][col]=1; V[11][col]=1; W[10][col]=1; col+=1
    U[6][col]=1; V[2][col]=1; W[8][col]=1; col+=1
    U[6][col]=1; V[3][col]=1; W[11][col]=1; col+=1
    U[7][col]=1; V[6][col]=1; W[8][col]=1; col+=1
    U[7][col]=1; V[7][col]=1; W[11][col]=1; col+=1
    U[8][col]=1; V[10][col]=1; W[8][col]=1; col+=1
    U[8][col]=1; V[11][col]=1; W[11][col]=1; col+=1
    
    return {"U": U, "V": V, "W": W}

def _try_rank_32_structured():
    """
    Try to construct a rank-32 decomposition.
    
    For 3x3 * 3x4, we can think of it as:
    - Split B into B_left (3x2) and B_right (3x2)
    - C_left = A * B_left (3x2), C_right = A * B_right (3x2)
    
    For 3x3 * 3x2, the naive rank is 18, but we can do better.
    The key insight: for 3x3 * 3x2, we can use a decomposition with rank 16
    by sharing some structure between the two columns.
    
    Actually, let me think more carefully. The minimal rank for 3x3 * 3x2 is unknown
    but bounded. Let me try a different approach: use the fact that we have 4 columns
    and try to share computation across all of them.
    
    Alternative: Use a block decomposition where we process 2 columns at a time
    with shared Strassen-like terms.
    """
    n, m, p = DIMENSIONS
    # Try rank 32: 16 per half
    # For each half (2 columns), use a rank-16 decomposition
    # This would give total rank 32
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # For 3x3 * 3x2, here's a rank-16 approach:
    # Standard: 9*2 = 18 terms
    # Save 2 terms by sharing: 
    # C[i,0] = sum_j A[i,j]*B[j,0]
    # C[i,1] = sum_j A[i,j]*B[j,1]
    # Combined: C[i,:] = A[i,:] * B[:,0:2]
    # We can define M_j = A[i,j] * (B[j,0] + B[j,1]) type terms
    
    # Actually, for 3x3 * 3x2, let's use:
    # 7 Strassen-like terms for the 2x2 submatrix (rows 0,1 of A; cols 0,1 of B)
    # + 3*2 = 6 direct terms for row 2 of A
    # + 1 shared term = 14? No, that's not right.
    
    # Let me just use the naive 18 per half = 36 total, which is worse than 34.
    # So rank 32 requires a smarter approach.
    
    # For rank 32, we need 16 per half.
    # 3x3 * 3x2 has 18 naive terms. To get 16, we need to save 2.
    
    # One approach: 
    # M1 = A00*(B00+B10) -> C00 gets A00*B00, C10 gets A00*B10... no that's not right.
    
    # Let me try: for the 2x2 block (A[0:2,0:2] * B[0:2,0:2]), use 7 terms (Strassen)
    # For the remaining terms involving row 2 or col 2 of the 3x3... 
    # Actually 3x3 * 3x2: A is 3x3, B is 3x2
    # C[i,k] = sum_j A[i,j]*B[j,k] for i in 0..2, k in 0..1
    # Total 6 output elements, 18 input pairs.
    
    # Strassen for the 2x2 * 2x2 part (A[0:2,0:2] * B[0:2,0:2]): 7 terms
    # Remaining: A[0:2,2] * B[2,0:2] (2 terms) + A[2,0:3] * B[0:2,0:2] (4 terms) + A[2,2]*B[2,0:2] (2 terms) = 8 terms
    # Total: 7 + 8 = 15 terms per half! That gives 30 total.
    
    # Wait, let me recount:
    # A[0:2,0:2] * B[0:2,0:2]: 2x2 * 2x2 -> 7 Strassen terms
    # A[0:2,2] * B[2,0:2]: 2x1 * 1x2 -> 2 terms (A[0,2]*B[2,0], A[1,2]*B[2,1])... 
    #   Actually: C[0,0] += A[0,2]*B[2,0], C[0,1] += A[0,2]*B[2,1], C[1,0] += A[1,2]*B[2,0], C[1,1] += A[1,2]*B[2,1]
    #   That's 4 terms, but we can share: M = A[0,2]*B[2,0] + ... no.
    #   Direct: 4 terms
    # A[2,0:3] * B[0:2,0:2]: 1x3 * 3x2 -> C[2,0] = A[2,0]*B[0,0]+A[2,1]*B[1,0]+A[2,2]*B[2,0], C[2,1] = A[2,0]*B[0,1]+A[2,1]*B[1,1]+A[2,2]*B[2,1]
    #   That's 6 terms, but we can use Strassen-like: 
    #   M1 = A[2,0]*(B[0,0]+B[1,0]), M2 = A[2,1]*(B[0,1]-B[0,0]), etc.
    #   Actually for 1x3 * 3x2, it's just 6 scalar multiplications. Can we do better?
    #   C[2,0] = A20*B00 + A21*B10 + A22*B20
    #   C[2,1] = A20*B01 + A21*B11 + A22*B21
    #   We can define: X = A20*(B00+B01), Y = A21*(B10+B11), Z = A22*(B20+B21)
    #   Then C[2,0] = X + Y + Z - A20*B01 - A21*B11 - A22*B21... that doesn't help.
    #   Or: C[2,0] = A20*B00 + A21*B10 + A22*B20
    #   C[2,1] = A20*B01 + A21*B11 + A22*B21
    #   No obvious sharing. 6 terms.
    
    # Total per half: 7 + 4 + 6 = 17 terms. Total: 34. Same as baseline.
    
    # To get 16 per half, we need to save 1 more term.
    # Maybe we can share between the two halves?
    
    # For now, let's just return the baseline and try random search for improvement.
    return None

def _random_search_for_rank(rank, max_time=10.0, seed=42):
    """Random search for a decomposition of given rank."""
    T = _build_tensor()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    np.random.seed(seed)
    start_time = time.time()
    
    for attempt in range(200):
        if time.time() - start_time > max_time:
            break
        
        # Initialize with small random values
        U = np.random.randn(nm, rank) * 0.5
        V = np.random.randn(mp, rank) * 0.5
        W = np.random.randn(pn, rank) * 0.5
        
        # Gradient descent
        lr = 0.01
        for step in range(500):
            if time.time() - start_time > max_time:
                break
            
            # Compute error
            T_recon = np.zeros_like(T)
            for r in range(rank):
                T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = T_recon - T
            
            if np.linalg.norm(err) < 1e-10:
                break
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for r in range(rank):
                dU[:, r] = np.einsum('abc,b,c->a', err, V[:, r], W[:, r])
                dV[:, r] = np.einsum('abc,a,c->b', err, U[:, r], W[:, r])
                dW[:, r] = np.einsum('abc,a,b->c', err, U[:, r], V[:, r])
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        # Check if we found a valid decomposition
        T_recon = np.zeros_like(T)
        for r in range(rank):
            T_recon += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
        
        if np.allclose(T_recon, T, atol=1e-8):
            # Try to round to integers/half-integers
            U_round = np.round(U * 2) / 2
            V_round = np.round(V * 2) / 2
            W_round = np.round(W * 2) / 2
            
            T_recon2 = np.zeros_like(T)
            for r in range(rank):
                T_recon2 += np.einsum('a,b,c->abc', U_round[:, r], V_round[:, r], W_round[:, r])
            
            if np.allclose(T_recon2, T, atol=1e-9):
                return {
                    "U": U_round.tolist(),
                    "V": V_round.tolist(),
                    "W": W_round.tolist()
                }
    
    return None

def construct():
    deadline = time.time() + 130  # Safety margin
    
    # Start with baseline
    best_result = _get_rank_34_baseline()
    best_rank = 34
    
    # Verify baseline
    U_arr = np.array(best_result["U"], dtype=float)
    V_arr = np.array(best_result["V"], dtype=float)
    W_arr = np.array(best_result["W"], dtype=float)
    if not _verify_decomposition(U_arr, V_arr, W_arr):
        # Fallback: naive rank 36
        n, m, p = DIMENSIONS
        rank = n * m * p // 4  # This is wrong, let me just use direct
        # Direct decomposition: rank = n*m*p / something... 
        # Actually naive is sum over j for each (i,k): 3*3*4 = 36 terms? No.
        # C[i,k] = sum_j A[i,j]*B[j,k], so for each (i,k) pair, we have one term per j.
        # Total terms = n * p * m = 3 * 4 * 3 = 36
        rank = 36
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a][col] = 1
                    V[b][col] = 1
                    W[c][col] = 1
                    col += 1
        best_result = {"U": U, "V": V, "W": W}
        best_rank = 36
    
    # Try to find better decompositions
    for rank in range(28, best_rank):
        if time.time() > deadline - 20:
            break
        remaining_time = deadline - time.time() - 15
        if remaining_time < 2:
            break
        result = _random_search_for_rank(rank, max_time=min(10.0, remaining_time))
        if result is not None:
            best_result = result
            best_rank = rank
            # Continue searching for even better
    
    return best_result
