import time
import numpy as np

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Trivial rank-54 decomposition as fallback
    def trivial_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    # Verify a decomposition by checking if it reconstructs the multiplication tensor
    def verify_decomp(U, V, W, tol=1e-9):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        
        # Build the expected tensor: T[i,j,k] = 1 for all i,j,k
        # In flattened form: T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
        # We check: sum_s U[a,s]*V[b,s]*W[c,s] should equal 1 for valid (a,b,c)
        # and 0 for invalid ones.
        
        # Check all valid (i,j,k) positions
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0.0
                    for s in range(rank):
                        val += U[a, s] * V[b, s] * W[c, s]
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check some invalid positions (sample)
        # For a position (a,b,c) to be valid, there must exist i,j,k such that
        # a = i*m+j, b = j*p+k, c = k*n+i
        # Let's check that for random invalid positions, the value is ~0
        import random
        random.seed(42)
        for _ in range(200):
            a = random.randint(0, n*m-1)
            b = random.randint(0, m*p-1)
            c = random.randint(0, p*n-1)
            # Check if this is a valid position
            valid = False
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        if a == i*m+j and b == j*p+k and c == k*n+i:
                            valid = True
                            break
                    if valid:
                        break
                if valid:
                    break
            if not valid:
                val = 0.0
                for s in range(rank):
                    val += U[a, s] * V[b, s] * W[c, s]
                if abs(val) > tol:
                    return False
        
        return True
    
    # Try to find a lower-rank decomposition using optimization
    # Strategy: use the fact that the tensor has structure
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Key insight: For each fixed i, the mapping from B to C[i,:] is linear.
    # B is 3x9, so we can think of B as 3 vectors of length 9 (rows).
    # C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:] + A[i,2]*B[2,:]
    # 
    # This is a linear combination of the 3 rows of B, with coefficients from row i of A.
    # 
    # For the tensor decomposition, we need:
    # T[i,j,k] = sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # 
    # Let's try a structured ansatz. Since the multiplication is separable in j,
    # we can try to use rank-3 terms for each (i, k-group).
    
    # Alternative approach: Use the known result that matmul(2,3,9) has rank <= 18
    # by exploiting the structure of the 3x9 matrix.
    
    # Let me try a direct construction with rank 18.
    # The idea: for each of the 3 "blocks" of 3 columns in B, and for each of the 2 rows of A,
    # we need to compute a 2x3 matrix product. The rank of 2x3x3 matmul is 6 (trivial).
    # So 2 * 3 * 3 = 18 terms.
    
    # For block g (columns 3g, 3g+1, 3g+2 of B):
    # C[i, 3g+h] = sum_j A[i,j] * B[j, 3g+h] for h=0,1,2
    # This is a 2x3x3 matmul, which has rank 6 (one term for each (i,j,h)).
    
    # So total rank = 2 * 3 * 3 = 18? No, 2 rows * 3 blocks * 3 (j,h) = 18.
    # Wait: for each (i, g, j, h), we have a term. That's 2*3*3*3 = 54. Still trivial.
    
    # I need a smarter approach. Let me use the fact that for 2x3x3, the rank is actually 6,
    # not 18. The 6 terms correspond to the 6 entries of the 2x3 result, but that's not right either.
    
    # Actually, for 2x3x3 matmul: C[i,h] = sum_j A[i,j]*B[j,h]
    # The tensor T[i,j,h] = 1 for all i,j,h. In flattened form (6, 9, 6), the valid positions are
    # (i*3+j, j*3+h, h*2+i) for i in {0,1}, j in {0,1,2}, h in {0,1,2}. That's 18 positions.
    # The rank of this tensor is 18 (trivial) or less.
    
    # I don't have a known lower-rank construction for 2x3x3. Let me just return the trivial decomposition.
    
    if time.time() > deadline:
        return trivial_decomp()
    
    return trivial_decomp()
