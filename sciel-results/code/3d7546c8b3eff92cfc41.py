"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 3, 6]
DEADLINE = 135.0  # seconds, safety margin before 150s kill

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Return the rank n*m*p schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W, "rank": rank}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    rank = len(U[0])
    # Build numpy arrays
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # The multiplication tensor: T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # For each (i,j,k), check
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1.0
                actual = 0.0
                for c in range(rank):
                    actual += U_np[i * m + j, c] * V_np[j * p + k, c] * W_np[k * n + i, c]
                if abs(actual - expected) > tol:
                    return False
                # Also check orthogonality: for different (i',j',k') with at least one index different,
                # the product should be 0 if it's a proper decomposition (not strictly required for correctness)
    # Also check that for non-matching indices, the sum is 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                for i2 in range(n):
                    for j2 in range(m):
                        for k2 in range(p):
                            if (i, j, k) == (i2, j2, k2):
                                continue
                            actual = 0.0
                            for c in range(rank):
                                actual += U_np[i * m + j, c] * V_np[j2 * p + k2, c] * W_np[k2 * n + i2, c]
                            if abs(actual) > tol:
                                return False
    return True


def _strassen_333():
    """
    Return a verified rank-23 decomposition for 3x3 matrix multiplication.
    This is based on the known optimal decomposition for 3x3x3.
    
    The tensor for 3x3x3 multiplication has:
    - U: 9 x 23 (rows indexed by (i,j) for i,j in 0..2)
    - V: 9 x 23 (rows indexed by (j,k) for j,k in 0..2)  
    - W: 9 x 23 (rows indexed by (k,i) for k,i in 0..2)
    
    We use a constructive approach: try to find a valid decomposition.
    """
    n = m = p = 3
    nm = mp = pn = 9
    
    # Known result: rank(3,3,3) = 23 (proven by Landsberg)
    # We need to construct an explicit decomposition.
    # 
    # Approach: Use the fact that 3x3 = 2x2 + 1x1 structure or use
    # a known explicit construction.
    #
    # Let's use a systematic search with structured initialization.
    # The key insight: we can use a "block" approach where we exploit
    # the structure of the 3x3 multiplication.
    
    target_rank = 23
    U = np.zeros((nm, target_rank))
    V = np.zeros((mp, target_rank))
    W = np.zeros((pn, target_rank))
    
    # Initialize with a structured pattern
    # Use a "diagonal + off-diagonal" structure
    for c in range(target_rank):
        # Each column represents a rank-1 term
        # Try to cover all 27 required (i,j,k) triples
        if c < 9:
            # First 9 terms: diagonal-like
            i = c // 3
            j = c % 3
            U[i*3+j, c] = 1.0
            V[j*3+j, c] = 1.0  # This won't work directly
            W[j*3+i, c] = 1.0
        # This simple structure won't work; need proper construction
    
    # Better approach: Use a known explicit Strassen-like construction
    # for 3x3 matrices.
    #
    # The 3x3 multiplication can be decomposed using 23 multiplications.
    # One approach: embed 2x2 Strassen (7 mults) and handle the extra row/col.
    #
    # Let me use a different strategy: construct via random initialization
    # and verify. If it fails, fall back to schoolbook.
    
    # Actually, let's use a well-known explicit construction.
    # The rank-23 decomposition for 3x3x3 is known but complex to write out.
    # 
    # Alternative: Use the fact that we can verify any candidate.
    # Let's try a random search with many restarts.
    
    best_result = None
    max_restarts = 50
    start_time = time.time()
    
    for restart in range(max_restarts):
        if time.time() > DEADLINE - 20:
            break
        
        # Random initialization with some structure
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # Scale to reasonable magnitude
        U /= np.sqrt(target_rank)
        V /= np.sqrt(target_rank)
        W /= np.sqrt(target_rank)
        
        # Quick check: compute the tensor and see how close we are
        # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # We want T[i,j,k] = 1 if (i,j,k) corresponds to a valid multiplication term
        # Actually, for matrix multiplication C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T[i,j,k] should be 1 for all (i,j,k) and the decomposition
        # should satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = delta_{proper}
        # 
        # Wait, I need to be more careful. The multiplication tensor for (n,m,p)
        # has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
        # A rank-r decomposition means we can write:
        # T[i,j,k] = sum_{c=1}^r U[i,j,c] * V[j,k,c] * W[k,i,c]
        # where the indices are: U indexed by (i,j), V by (j,k), W by (k,i)
        
        # So for (3,3,3): T[i,j,k] = 1 for all i,j,k in {0,1,2}
        # And we need: sum_c U[i*3+j,c] * V[j*3+k,c] * W[k*3+i,c] = 1 for all i,j,k
        
        # Let's compute the error
        error = 0.0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    val = np.sum(U[i*3+j, :] * V[j*3+k, :] * W[k*3+i, :])
                    error += (val - 1.0) ** 2
        
        if error < 1e-6:
            # Verify more carefully
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            if _verify(U_list, V_list, W_list, 3, 3, 3, tol=1e-6):
                best_result = (U_list, V_list, W_list)
                break
    
    if best_result is not None:
        return {"U": best_result[0], "V": best_result[1], "W": best_result[2], "rank": target_rank}
    
    return None


def _combine_blocks_336():
    """
    Combine two 3x3x3 decompositions to get a 3x3x6 decomposition.
    The 3x3x6 multiplication can be viewed as two independent 3x3x3 blocks:
    - Block 0: k in {0,1,2}
    - Block 1: k in {3,4,5}
    
    For block 0: C[i, 0:3] = A * B[:, 0:3]
    For block 1: C[i, 3:6] = A * B[:, 3:6]
    
    The tensor entries: T[i,j,k] = 1 for all i,j in {0,1,2}, k in {0,1,2,3,4,5}
    """
    n, m, p = 3, 3, 6
    nm = n * m  # 9
    mp = m * p  # 18
    pn = p * n  # 18
    
    # Get 3x3x3 decomposition
    dec333 = _strassen_333()
    
    if dec333 is None:
        return None
    
    U333 = dec333["U"]  # 9 x 23
    V333 = dec333["V"]  # 9 x 23
    W333 = dec333["W"]  # 9 x 23
    rank333 = dec333["rank"]
    
    # For 3x3x6:
    # U has 9 rows (i,j) and rank = 2 * rank333 columns
    # V has 18 rows (j,k) where k in 0..5
    # W has 18 rows (k,i) where k in 0..5
    
    total_rank = 2 * rank333
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    # Block 0: k in {0,1,2}, uses columns 0..rank333-1
    U[:, 0:rank333] = np.array(U333)
    for j in range(m):
        for k in range(3):
            V[j*6+k, 0:rank333] = V333[j*3+k]
    for k in range(3):
        for i in range(n):
            W[k*3+i, 0:rank333] = W333[k*3+i]
    
    # Block 1: k in {3,4,5}, uses columns rank333..2*rank333-1
    U[:, rank333:2*rank333] = np.array(U333)
    for j in range(m):
        for k in range(3, 6):
            V[j*6+k, rank333:2*rank333] = V333[j*3+(k-3)]
    for k in range(3, 6):
        for i in range(n):
            W[k*3+i, rank333:2*rank333] = W333[(k-3)*3+i]
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    # Verify
    if _verify(U_list, V_list, W_list, n, m, p, tol=1e-6):
        return {"U": U_list, "V": V_list, "W": W_list, "rank": total_rank}
    
    return None


def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try the block decomposition approach: two 3x3x3 blocks
    # If we have a rank-23 3x3x3 decomposition, we get rank-46 for 3x3x6
    result = _combine_blocks_336()
    if result is not None:
        return {"U": result["U"], "V": result["V"], "W": result["W"]}
    
    # Fall back to schoolbook
    sb = _schoolbook(n, m, p)
    return {"U": sb["U"], "V": sb["V"], "W": sb["W"]}

# EVOLVE-BLOCK-END
