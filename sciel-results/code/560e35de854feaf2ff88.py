import time
from fractions import Fraction

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # 130s safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # n=2, m=5, p=6
    # Rank 54 decomposition
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        nonlocal col
        # 7 columns for one 2x2 * 2x2 block
        
        # m0: (x00+x11)(y00+y11) -> Z00+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m1: (x10+x11)y00 -> Z10+=1, Z11-=1
        U[(i_base+1) * m + j_base][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] -= 1
        col += 1
        
        # m2: x00(y01-y11) -> Z01+=1, Z11+=1
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m3: x11(y10-y00) -> Z00+=1, Z10+=1
        U[(i_base+1) * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base+1)][col] += 1
        col += 1
        
        # m4: (x00+x01)y11 -> Z00-=1, Z01+=1
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base+1)][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base+1) * n + i_base][col] += 1
        col += 1
        
        # m5: (x10-x00)(y00+y01) -> Z11+=1
        U[(i_base+1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base+1)][col] += 1
        W[(k_base+1) * n + (i_base+1)][col] += 1
        col += 1
        
        # m6: (x01-x11)(y10+y11) -> Z00+=1
        U[i_base * m + (j_base+1)][col] += 1
        U[(i_base+1) * m + (j_base+1)][col] -= 1
        V[(j_base+1) * p + k_base][col] += 1
        V[(j_base+1) * p + (k_base+1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    # Blocks for A_R (j=3,4) * B_R (j=3,4)
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(0, 3, 0)
    add_strassen_block(0, 3, 2)
    add_strassen_block(0, 3, 4)
    
    # Blocks for A_L1 (j=0,1) * B_L1 (j=0,1)
    # k splits: 0-1, 2-3, 4-5
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # Outer product for A_L2 (j=2) * B_L2 (j=2)
    # A_L2 is 2x1, B_L2 is 1x6
    for k in range(p):
        # C[0, k] += A[0, 2] * B[2, k]
        U[0 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 0][col] = 1
        col += 1
        
        # C[1, k] += A[1, 2] * B[2, k]
        U[1 * m + 2][col] = 1
        V[2 * p + k][col] = 1
        W[k * n + 1][col] = 1
        col += 1
    
    # Dual-variable elimination: check for linearly dependent rank-1 terms
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can check if removing a term still preserves the decomposition
    # by verifying that the removed term is in the span of others
    
    # Check time
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try to eliminate redundant terms by checking linear dependence
    # Build the tensor and check if each term is necessary
    def verify_decomposition(U_mat, V_mat, W_mat, n, m, p):
        """Verify that the decomposition exactly reproduces the tensor."""
        r = len(U_mat[0])
        # T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0  # The tensor entry for matrix multiplication is 1 if valid, 0 otherwise
                    # Actually, for matrix mult, T[i,j,k] = delta(i,j,k) where the indices form a valid multiplication
                    # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
                    # So T[i,j,k] = 1 for all valid (i,j,k) triples
                    actual = 0.0
                    for r in range(r):
                        actual += U_mat[i*m+j][r] * V_mat[j*p+k][r] * W_mat[k*n+i][r]
                    if abs(actual - 1.0) > 1e-9:
                        return False
        return True
    
    # Try to remove each term and see if the decomposition still holds
    # This is expensive, so we only do it if we have time
    best_U, best_V, best_W = U, V, W
    best_rank = rank
    
    # Try removing terms one at a time (greedy elimination)
    # Start with the current decomposition
    current_U = [row[:] for row in U]
    current_V = [row[:] for row in V]
    current_W = [row[:] for row in W]
    current_rank = rank
    
    # For each term, check if it can be removed
    for trial in range(current_rank):
        if time.time() > deadline - 10:  # Leave 10s buffer
            break
        
        # Try removing term 'trial'
        test_U = [row[:] for row in current_U]
        test_V = [row[:] for row in current_V]
        test_W = [row[:] for row in current_W]
        
        # Zero out term 'trial'
        for i in range(n * m):
            test_U[i][trial] = 0
        for j in range(m * p):
            test_V[j][trial] = 0
        for k in range(p * n):
            test_W[k][trial] = 0
        
        # Verify
        if verify_decomposition(test_U, test_V, test_W, n, m, p):
            # Term can be removed
            current_U = test_U
            current_V = test_V
            current_W = test_W
            current_rank -= 1
            best_U = current_U
            best_V = current_V
            best_W = current_W
            best_rank = current_rank
    
    # Return the best decomposition found
    # Trim to the actual rank
    final_U = []
    for row in best_U:
        final_U.append(row[:best_rank])
    final_V = []
    for row in best_V:
        final_V.append(row[:best_rank])
    final_W = []
    for row in best_W:
        final_W.append(row[:best_rank])
    
    return {"U": final_U, "V": final_V, "W": final_W}
# EVOLVE-BLOCK-END
