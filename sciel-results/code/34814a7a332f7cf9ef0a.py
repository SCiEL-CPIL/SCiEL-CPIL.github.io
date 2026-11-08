import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def construct():
    start_time = time.time()
    timeout = 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    # n=3, m=5, p=5
    
    # We'll build the decomposition using:
    # 1. Strassen 2x2x2 blocks where applicable
    # 2. Schoolbook for remaining terms
    # 
    # Strategy: Partition the 3x5x5 multiplication into blocks
    # Rows: i in {0,1} and i=2
    # Cols: j in {0,1,2,3,4}
    # Output: k in {0,1,2,3,4}
    #
    # Block partitioning:
    # - For i in {0,1}, j in {0,1}, k in {0,1}: Strassen 2x2x2 (rank 7)
    # - For i in {0,1}, j in {2,3}, k in {0,1}: Strassen 2x2x2 (rank 7)
    # - For i in {0,1}, j in {0,1}, k in {2,3}: Strassen 2x2x2 (rank 7)
    # - For i in {0,1}, j in {2,3}, k in {2,3}: Strassen 2x2x2 (rank 7)
    # - For i in {0,1}, j in {4}, k in {0,1,2,3,4}: schoolbook (5 terms)
    # - For i in {0,1}, j in {0,1,2,3}, k in {4}: schoolbook (8 terms)
    # - For i=2, j in {0,1,2,3,4}, k in {0,1,2,3,4}: schoolbook (25 terms)
    #
    # Total: 4*7 + 5 + 8 + 25 = 28 + 38 = 66? Let me recount.
    # Actually for i in {0,1}, j in {4}: 2 rows * 5 k-values = 10 terms
    # For i in {0,1}, j in {0,1,2,3}, k in {4}: 2*4*1 = 8 terms
    # For i=2: 5*5 = 25 terms
    # Total: 28 + 10 + 8 + 25 = 71. That's the same as before.
    
    # Let me try a different partitioning to get lower rank.
    # 
    # Alternative: Use Strassen more cleverly.
    # 
    # Actually, let's try:
    # - i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}: 4 Strassen blocks (rank 28)
    # - i in {0,1}, j in {4}, k in {0,1,2,3,4}: 10 schoolbook terms
    # - i in {0,1}, j in {0,1,2,3}, k in {4}: 8 schoolbook terms  
    # - i=2, j in {0,1,2,3,4}, k in {0,1,2,3,4}: 25 schoolbook terms
    # Total: 28 + 10 + 8 + 25 = 71
    #
    # Hmm, still 71. Let me think about this differently.
    #
    # What if I use a 2x2x2 Strassen for i in {0,1}, j in {0,1}, k in {0,1}
    # and similarly for other 2x2x2 blocks, but overlap them?
    # No, they can't overlap in the decomposition.
    #
    # Let me try yet another approach:
    # - Use 2x2x2 Strassen for the four blocks covering i in {0,1}, j in {0,3}, k in {0,3}
    #   That's blocks: (0,0,0), (0,2,0), (0,0,2), (0,2,2) - same as before, rank 28
    # - Remaining terms:
    #   i in {0,1}, j=4, k in {0,1,2,3,4}: 10 terms
    #   i in {0,1}, j in {0,1,2,3}, k=4: 8 terms
    #   i=2, all j,k: 25 terms
    #   Total remaining: 43, grand total: 71
    #
    # To get below 71, I need a different strategy.
    # 
    # Idea: Can I use Strassen for blocks involving i=2?
    # If I pair i=2 with i=0 or i=1, I could use a 2x2x2 Strassen.
    # 
    # Let's try:
    # - i in {0,2}, j in {0,1}, k in {0,1}: Strassen 2x2x2 (rank 7)
    # - i in {0,2}, j in {2,3}, k in {0,1}: Strassen 2x2x2 (rank 7)
    # - i in {0,2}, j in {0,1}, k in {2,3}: Strassen 2x2x2 (rank 7)
    # - i in {0,2}, j in {2,3}, k in {2,3}: Strassen 2x2x2 (rank 7)
    # - i in {0,2}, j=4, k in {0,1,2,3,4}: 10 terms
    # - i in {0,2}, j in {0,1,2,3}, k=4: 8 terms
    # - i=1, j in {0,1,2,3,4}, k in {0,1,2,3,4}: 25 terms
    # Total: 28 + 10 + 8 + 25 = 71. Still 71!
    #
    # It seems hard to beat 71 with this approach. Let me try to be more creative.
    #
    # What about using a 3x2x2 or similar asymmetric block?
    # 
    # Actually, the key insight is that for a 3x5x5 matrix multiplication,
    # we might be able to use a tensor decomposition that exploits the
    # rectangular structure more effectively.
    #
    # Let me try: Partition into 2x2x2 Strassen blocks and handle the rest.
    # But arrange the blocks to cover more of the tensor with fewer terms.
    #
    # Alternative: Use the fact that M(3,5,5) <= M(3,6,6) <= ... 
    # But I don't have a known decomposition for M(3,6,6) that's better.
    #
    # Let me just implement the 71-rank solution cleanly and see if there's
    # a way to reduce it. For now, I'll go with the 4 Strassen blocks + schoolbook.
    
    U = [[0.0] * 71 for _ in range(n * m)]
    V = [[0.0] * 71 for _ in range(m * p)]
    W = [[0.0] * 71 for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i0, j0, k0, col_offset):
        nonlocal col
        c = col_offset + 0
        U[i0*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 1
        U[(i0+1)*m + j0][c] += 1
        U[(i0+1)*m + (j0+1)][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += -1
        
        c = col_offset + 2
        U[i0*m + j0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += -1
        W[(k0+1)*n + i0][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 3
        U[(i0+1)*m + (j0+1)][c] += 1
        V[(j0+1)*p + k0][c] += 1
        V[j0*p + k0][c] += -1
        W[k0*n + i0][c] += 1
        W[k0*n + (i0+1)][c] += 1
        
        c = col_offset + 4
        U[i0*m + j0][c] += 1
        U[i0*m + (j0+1)][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += -1
        W[(k0+1)*n + i0][c] += 1
        
        c = col_offset + 5
        U[(i0+1)*m + j0][c] += 1
        U[i0*m + j0][c] += -1
        V[j0*p + k0][c] += 1
        V[j0*p + (k0+1)][c] += 1
        W[(k0+1)*n + (i0+1)][c] += 1
        
        c = col_offset + 6
        U[i0*m + (j0+1)][c] += 1
        U[(i0+1)*m + (j0+1)][c] += -1
        V[(j0+1)*p + k0][c] += 1
        V[(j0+1)*p + (k0+1)][c] += 1
        W[k0*n + i0][c] += 1
    
    # Four 2x2x2 Strassen blocks
    blocks = [(0, 0, 0), (0, 2, 0), (0, 0, 2), (0, 2, 2)]
    
    for i0, j0, k0 in blocks:
        add_strassen_block(i0, j0, k0, col)
        col += 7
    
    covered = [[[False]*p for _ in range(m)] for _ in range(n)]
    for i0, j0, k0 in blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered[i0+di][j0+dj][k0+dk] = True
    
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if not covered[i][j][k]:
                    U[i*m+j][col] += 1
                    V[j*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    if time.time() - start_time > timeout:
        # Should not happen, but safety check
        pass
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    n, m, p = DIMENSIONS
    
    # Verify correctness
    def check(U, V, W):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # C[k*n+i, :] should equal sum over r of U[i*m+j, r] * V[j*p+k, r]
                    # But wait, the convention is W[k*n+i, r]
                    # Let me re-read: "output factor follows the official trace convention W[k*n+i,r]"
                    # 
                    # The tensor decomposition is:
                    # T_{i,j,k} = sum_r U_{i,j,r} * V_{j,k,r} * W_{k,i,r}
                    # where T is the matrix multiplication tensor.
                    #
                    # For matrix multiplication C = A * B:
                    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
                    #
                    # So the tensor T_{i,j,k} = 1 if this is a valid term, else 0.
                    # Actually, T_{i,j,k} corresponds to the coefficient of A_{i,j} * B_{j,k} in C_{i,k}.
                    # So T_{i,j,k} = delta_{i,i} * delta_{k,k} ... no.
                    #
                    # The matrix multiplication tensor has T_{i,j,k} = 1 for all i,j,k in the appropriate ranges,
                    # and the decomposition is:
                    # sum_r U_{i*m+j, r} * V_{j*p+k, r} * W_{k*n+i, r} = 1 for all i,j,k
                    #
                    # Wait, that's not right either. Let me think again.
                    #
                    # The bilinear form is: C_{i,k} = sum_j A_{i,j} B_{j,k}
                    #
                    # In tensor form, the multiplication tensor T satisfies:
                    # T(A,B)_{i,k} = sum_{j} T_{i,j,k} A_{i,j} B_{j,k}
                    # where T_{i,j,k} = 1.
                    #
                    # The tensor decomposition expresses T as:
                    # T_{i,j,k} = sum_r U_{i*m+j, r} V_{j*p+k, r} W_{k*n+i, r}
                    #
                    # So we need: for all i, j, k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                    #
                    # Let's verify this.
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(val - 1.0) > 1e-10:
                        return False, (i, j, k, val)
        return True, None
    
    ok, info = check(U, V, W)
    if ok:
        print(f"Rank: {U.shape[1]} - CORRECT")
    else:
        print(f"Rank: {U.shape[1]} - INCORRECT at {info}")
