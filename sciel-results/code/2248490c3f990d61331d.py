import numpy as np
import time

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify if U, V, W form a valid decomposition."""
    R = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                val = 0.0
                for r in range(R):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if abs(val - expected) > 1e-9:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return the trivial rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1.0
                V[j*p+k, col] = 1.0
                W[k*n+i, col] = 1.0
                col += 1
    return U, V, W, rank

def _strassen_2x2x2():
    """
    Return a rank-7 decomposition for 2x2x2 matrix multiplication.
    The 7 Strassen terms with integer coefficients.
    """
    n, m, p = 2, 2, 2
    # Strassen's 7 multiplications for 2x2x2
    # U rows indexed by (i,j) in [0,1]x[0,1] -> i*2+j
    # V rows indexed by (j,k) in [0,1]x[0,1] -> j*2+k
    # W rows indexed by (k,i) in [0,1]x[0,1] -> k*2+i
    
    # The 7 Strassen terms:
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U[i,j] coefficient for each term
    # i,j in {0,1}
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Term 1: (a00+a11)(b00+b11)
    # U: a00=1, a11=1
    U[0,0] = 1; U[3,0] = 1
    # V: b00=1, b11=1
    V[0,0] = 1; V[3,0] = 1
    # W: c00 += M1, c11 += M1
    W[0,0] = 1; W[3,0] = 1
    
    # Term 2: (a10+a11)b00
    # U: a10=1, a11=1
    U[2,1] = 1; U[3,1] = 1
    # V: b00=1
    V[0,1] = 1
    # W: c10 += M2, c11 += M2
    W[2,1] = 1; W[3,1] = 1
    
    # Term 3: a00(b01-b11)
    # U: a00=1
    U[0,2] = 1
    # V: b01=1, b11=-1
    V[1,2] = 1; V[3,2] = -1
    # W: c01 += M3
    W[1,2] = 1
    
    # Term 4: a11(b10-b00)
    # U: a11=1
    U[3,3] = 1
    # V: b10=1, b00=-1
    V[2,3] = 1; V[0,3] = -1
    # W: c10 += M4
    W[2,3] = 1
    
    # Term 5: (a00+a01)b11
    # U: a00=1, a01=1
    U[0,4] = 1; U[1,4] = 1
    # V: b11=1
    V[3,4] = 1
    # W: c01 += M5, c11 += M5
    W[1,4] = 1; W[3,4] = 1
    
    # Term 6: (a10-a00)(b00+b01)
    # U: a10=1, a00=-1
    U[2,5] = 1; U[0,5] = -1
    # V: b00=1, b01=1
    V[0,5] = 1; V[1,5] = 1
    # W: c10 += M6
    W[2,5] = 1
    
    # Term 7: (a01-a11)(b10+b11)
    # U: a01=1, a11=-1
    U[1,6] = 1; U[3,6] = -1
    # V: b10=1, b11=1
    V[2,6] = 1; V[3,6] = 1
    # W: c01 += M7
    W[1,6] = 1
    
    return U, V, W, 7

def _build_3x3x3_from_strassen():
    """
    Build a 3x3x3 decomposition using the 2x2x2 Strassen decomposition.
    This gives rank 27 (same as schoolbook) but let's verify it works.
    Actually, we can't easily get below 27 for 3x3x3 with simple block methods.
    The known rank is 23 (Landsberg) but the coefficients are complex.
    
    For 3x3x3, the schoolbook rank is 27.
    """
    n, m, p = 3, 3, 3
    # Use schoolbook for 3x3x3
    U, V, W, rank = _schoolbook(n, m, p)
    return U, V, W, rank

def _build_3x3x6_direct():
    """
    Build a 3x3x6 decomposition directly.
    The multiplication of a 3x3 matrix by a 3x6 matrix can be viewed as
    two 3x3x3 multiplications (for the two 3x3 blocks of the 3x6 matrix).
    
    If we use schoolbook for each 3x3x3 block, we get rank 54.
    But we can potentially do better by sharing some terms.
    
    Actually, the 3x3x6 tensor has a specific structure. Let's think about it.
    T[i,j,k] = 1 for all i in [0,2], j in [0,2], k in [0,5].
    
    The rank of this tensor... it's not simply 2*27 because the two blocks
    share the same A matrix (3x3) but have different B matrices (two 3x3 blocks).
    
    Actually, for multiplying A (3x3) by B (3x6), the result C = A*B is 3x6.
    C[:, 0:3] = A * B[:, 0:3]
    C[:, 3:6] = A * B[:, 3:6]
    
    So it's two independent 3x3x3 multiplications that share the same A.
    The tensor rank of this combined operation...
    
    If we use the same 27-term decomposition for both blocks, we get 54 terms.
    But since the A part is shared, maybe we can do better?
    
    Actually no - in a tensor decomposition, each term is a rank-1 tensor.
    The fact that A is shared doesn't directly help reduce the rank in the
    standard tensor decomposition framework, because each term still needs
    its own U, V, W vectors.
    
    However, we can try to find a decomposition with fewer than 54 terms.
    
    Let's try a randomized search for a lower rank.
    """
    n, m, p = 3, 3, 6
    
    # Start with schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    
    # Try to find a lower rank decomposition using random initialization
    # and iterative refinement
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    deadline = time.time() + 120
    
    # Try ranks from 53 down to 40
    for target_rank in range(53, 39, -1):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 5:
                break
            
            # Initialize with random values
            U_rand = np.random.randn(n*m, target_rank)
            V_rand = np.random.randn(m*p, target_rank)
            W_rand = np.random.randn(p*n, target_rank)
            
            # Normalize to help convergence
            for r in range(target_rank):
                U_rand[:, r] /= (np.linalg.norm(U_rand[:, r]) + 1e-12)
                V_rand[:, r] /= (np.linalg.norm(V_rand[:, r]) + 1e-12)
                W_rand[:, r] /= (np.linalg.norm(W_rand[:, r]) + 1e-12)
            
            # Iterate to reduce error
            converged = False
            for it in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute current tensor
                T_current = np.zeros((n, m, p))
                for r in range(target_rank):
                    T_current += np.einsum('i,j,k->ijk', 
                                           U_rand[:, r], V_rand[:, r], W_rand[:, r])
                
                # Target tensor
                T_target = np.ones((n, m, p))
                
                # Compute error
                error = np.linalg.norm(T_target - T_current)
                if error < 1e-10:
                    converged = True
                    break
                
                # Update each factor
                for r in range(target_rank):
                    # Update U
                    U_rand[:, r] = np.einsum('j,k,ijk->i', 
                                            V_rand[:, r], W_rand[:, r], T_target - T_current)
                    U_rand[:, r] /= (np.linalg.norm(U_rand[:, r]) + 1e-12)
                    
                    # Update V
                    V_rand[:, r] = np.einsum('i,k,ijk->j', 
                                            U_rand[:, r], W_rand[:, r], T_target - T_current)
                    V_rand[:, r] /= (np.linalg.norm(V_rand[:, r]) + 1e-12)
                    
                    # Update W
                    W_rand[:, r] = np.einsum('i,j,ijk->k', 
                                            U_rand[:, r], V_rand[:, r], T_target - T_current)
                    W_rand[:, r] /= (np.linalg.norm(W_rand[:, r]) + 1e-12)
                
                if time.time() > deadline - 5:
                    break
            
            if converged:
                best_rank = target_rank
                best_U, best_V, best_W = U_rand, V_rand, W_rand
                break
    
    # Round to nearest half-integer if close
    def _round_to_half(x):
        return np.round(x * 2) / 2
    
    # Check if we can round
    if best_rank < 54:
        U_rounded = _round_to_half(best_U)
        V_rounded = _round_to_half(best_V)
        W_rounded = _round_to_half(best_W)
        if _verify(U_rounded, V_rounded, W_rounded, n, m, p):
            best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    return best_U, best_V, best_W, best_rank

def construct():
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as baseline
    U_sb, V_sb, W_sb, rank_sb = _schoolbook(n, m, p)
    
    # Try to find a better decomposition
    U_best, V_best, W_best, rank_best = _build_3x3x6_direct()
    
    # Verify the best result
    if not _verify(U_best, V_best, W_best, n, m, p):
        # Fall back to schoolbook
        U_best, V_best, W_best, rank_best = U_sb, V_sb, W_sb, rank_sb
    
    return {"U": U_best.tolist(), "V": V_best.tolist(), "W": W_best.tolist()}
