import time
import numpy as np
from scipy.optimize import nnls

DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # safety margin before 150s kill
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    def naive_solution():
        rank = n * m * p  # 64
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return {"U": U, "V": V, "W": W}
    
    def verify_decomposition(U, V, W, rank):
        """Verify that the decomposition exactly computes matrix multiplication."""
        np.random.seed(42)
        X = np.random.randn(n, m)
        Y = np.random.randn(m, p)
        Z_expected = X @ Y
        Z_computed = np.zeros((n, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for r in range(rank):
                        Z_computed[i, k] += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
        return np.allclose(Z_expected, Z_computed, atol=1e-10)
    
    def verify_decomposition_fast(U, V, W, rank):
        """Faster verification using matrix operations."""
        np.random.seed(42)
        X = np.random.randn(n, m)
        Y = np.random.randn(m, p)
        Z_expected = X @ Y
        Z_computed = np.zeros((n, p))
        U_arr = np.array(U)  # shape (n*m, rank)
        V_arr = np.array(V)  # shape (m*p, rank)
        W_arr = np.array(W)  # shape (p*n, rank)
        
        for r in range(rank):
            # Extract the r-th column
            u_r = U_arr[:, r]  # shape (n*m,)
            v_r = V_arr[:, r]  # shape (m*p,)
            w_r = W_arr[:, r]  # shape (p*n,)
            
            # Reshape to 2D
            u_mat = u_r.reshape(n, m)
            v_mat = v_r.reshape(m, p)
            w_mat = w_r.reshape(p, n)
            
            # The contribution to Z from this rank-1 term
            # Z[i,k] += sum_j u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
            # = sum_j X[i,j] * Y[j,k] * w_mat[k,i] when u_mat[i,j]=X[i,j], v_mat[j,k]=Y[j,k]
            # But we need to check the tensor identity, not just for one X,Y.
            # Actually the verification should check: for all i,j,k,
            #   sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            pass
        
        # Direct tensor check
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(rank))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition
    strassen_U = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, -1, 0, 1],
    ]
    strassen_V = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 1, 0, 1],
    ]
    strassen_W = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [-1, 0, 0, 0],
    ]
    
    # Build rank-56 Strassen solution as baseline
    def build_strassen_solution():
        rank = 56
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        for j in range(4):
            # First 2x2 block: X[0:2, j*2:j*2+2] * Y[j*2:j*2+2, :] -> but wait
            # Actually for 2x4 * 4x8, we need to be more careful.
            # Let's use the standard block decomposition.
            # X is 2x4, Y is 4x8.
            # We can view X as [X_left | X_right] where X_left is 2x2, X_right is 2x2
            # Y as [Y_top; Y_bottom] where Y_top is 2x8, Y_bottom is 2x8
            # Then Z = X_left @ Y_top + X_right @ Y_bottom
            # Each 2x2 @ 2x8 can be done with 7*4 = 28 multiplications? No...
            # 2x2 @ 2x8: for each of 8 columns, we do a 2x2 @ 2x1 which takes 7 mults.
            # So 2x2 @ 2x8 takes 7*8 = 56 mults? No, that's not right either.
            # Actually 2x2 @ 2x8: result is 2x8. Each element C[i,k] = sum_j A[i,j]*B[j,k].
            # Using Strassen for each 2x2 subblock: A is 2x2, B[:,k:k+2] is 2x2.
            # So for each pair of output columns, we do one Strassen 2x2x2.
            # 8 columns = 4 pairs, so 4 Strassen operations = 4*7 = 28 mults per half.
            # Two halves: 56 mults total. That matches rank 56.
            
            # Let me restructure:
            # For columns 0,1 of Y (i.e., k=0,1):
            #   Z[:,0:2] = X[:,0:2] @ Y[0:2,0:2] + X[:,2:4] @ Y[2:4,0:2]
            # First term: 2x2 @ 2x2, use Strassen (7 mults)
            # Second term: 2x2 @ 2x2, use Strassen (7 mults)
            # Total for k=0,1: 14 mults
            # Similarly for k=2,3: 14 mults, k=4,5: 14, k=6,7: 14
            # Total: 56 mults
            
            for k_pair in range(4):
                k_start = k_pair * 2
                # First half: X[:, 0:2] @ Y[0:2, k_start:k_start+2]
                base1 = k_pair * 14
                for r in range(7):
                    cr = base1 + r
                    for idx in range(4):
                        li, lj = divmod(idx, 2)
                        U[li * m + lj][cr] += strassen_U[r][idx]
                    for idx in range(4):
                        lj, lk = divmod(idx, 2)
                        V[lj * p + k_start + lk][cr] += strassen_V[r][idx]
                    for idx in range(4):
                        li, lk = divmod(idx, 2)
                        W[(k_start + lk) * n + li][cr] += strassen_W[r][idx]
                
                # Second half: X[:, 2:4] @ Y[2:4, k_start:k_start+2]
                base2 = k_pair * 14 + 7
                for r in range(7):
                    cr = base2 + r
                    for idx in range(4):
                        li, lj = divmod(idx, 2)
                        U[li * m + 2 + lj][cr] += strassen_U[r][idx]
                    for idx in range(4):
                        lj, lk = divmod(idx, 2)
                        V[(2 + lj) * p + k_start + lk][cr] += strassen_V[r][idx]
                    for idx in range(4):
                        li, lk = divmod(idx, 2)
                        W[(k_start + lk) * n + li][cr] += strassen_W[r][idx]
        
        return U, V, W, rank
    
    U, V, W, rank = build_strassen_solution()
    
    if not verify_decomposition(U, V, W, rank):
        return naive_solution()
    
    # Now try to reduce the rank by identifying linearly dependent columns
    # and merging them.
    
    # Approach: Try to find a lower rank solution by solving a linear system.
    # For a given rank R, we need to find U, V, W such that:
    # sum_{r=1}^{R} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # This is a nonlinear system. Instead, let's try a different approach:
    # Fix W to be identity-like and solve for U and V.
    
    # Alternative: Use the fact that the tensor is separable in a certain way.
    # For fixed i,k, the equation is:
    # sum_j [sum_r U[i*m+j,r] * W[k*n+i,r]] * V[j*p+k,r] ... no, that's not separable.
    
    # Let me try a direct rank-reduction approach:
    # Start with rank 56 and try to eliminate one rank term at a time.
    # For each rank term r, check if it can be expressed as a linear combination
    # of the others.
    
    # Actually, a simpler approach: try different known low-rank decompositions.
    
    # The theoretical lower bound for 2x4 x 4x8 is at least 2*4*8/2 = 32
    # (by the standard lower bound). Upper bound is 56 (Strassen).
    # Known results suggest the rank might be around 48-52.
    
    # Let me try to construct a rank-48 solution using a different strategy.
    # Idea: Precompute 6 linear combinations of each row of X.
    
    # For the 2x4 x 4x8 problem, we can view it as:
    # Z[i,k] = sum_j X[i,j] * Y[j,k]
    # = X[i,0]*Y[0,k] + X[i,1]*Y[1,k] + X[i,2]*Y[2,k] + X[i,3]*Y[3,k]
    
    # Precompute for each i:
    # S[i,0] = X[i,0] + X[i,1]
    # S[i,1] = X[i,0] - X[i,1]
    # S[i,2] = X[i,2] + X[i,3]
    # S[i,3] = X[i,2] - X[i,3]
    # S[i,4] = X[i,0] + X[i,2]
    # S[i,5] = X[i,1] + X[i,3]
    
    # Then Z[i,k] can be computed from these 6 values and appropriate linear
    # combinations of Y.
    
    # This gives 6 precomputation steps per row of X, so 12 total.
    # For each of the 8 columns of Y, we need to compute 2 outputs.
    # If each output takes 4 multiplications (using the precomputed values),
    # total = 12 + 8*2*4 = 12 + 64 = 76. That's worse than 56.
    
    # Let me try a different precomputation strategy.
    
    # Actually, let me just try to find any valid solution with rank < 56
    # by random search or by solving the linear system.
    
    # For rank R, the system has 64 equations and 3*R*(n*m + m*p + p*n) / (some factor) unknowns.
    # Actually, the unknowns are U (n*m*R), V (m*p*R), W (p*n*R) = R*(8+32+16) = 56R.
    # For R=48, that's 2688 unknowns and 64 equations. Very underdetermined.
    
    # The challenge is the nonlinearity (products of U, V, W).
    
    # Let me try a heuristic: fix W to have a simple structure and solve for U, V.
    
    # If W[k*n+i, r] = delta_{k, r mod p} * something... let's try:
    # W[k*n+i, r] = 1 if r < p and k == r, else 0. But that doesn't work for all i.
    
    # Let me try: W[k*n+i, r] = 1 for all k,i when r < n*p = 16, and 0 otherwise.
    # Then we need: sum_{r=0}^{15} U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j,k.
    # This is a linear system in U and V (bilinear).
    
    # For fixed k, we need: sum_{r=0}^{15} U[i*m+j, r] * V[j*p+k, r] = 1 for all i,j.
    # This is 8 equations (for i in {0,1}, j in {0,1,2,3}) with 16*8 = 128 unknowns
    # (U has 8*16 = 128 entries, V has 32*16 = 512 entries, but only V[:,k,:] for fixed k).
    
    # Actually, for fixed k, V[j*p+k, r] for j in {0,1,2,3}, r in {0,...,15} is 4*16 = 64 unknowns.
    # U[i*m+j, r] for i in {0,1}, j in {0,1,2,3}, r in {0,...,15} is 8*16 = 128 unknowns.
    # Total 192 unknowns, 8 equations. Very underdetermined.
    
    # But we need this to hold for ALL k, so the same U must work for all k.
    
    # Let me try a specific construction:
    # W[k*n+i, r] = 1 if r == k, else 0. (So W is like an identity in k, replicated over i)
    # Then: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = U[i*m+j, k] * V[j*p+k, k] = 1
    # So we need U[i*m+j, k] * V[j*p+k, k] = 1 for all i,j,k.
    # This means U[i*m+j, k] = 1 and V[j*p+k, k] = 1 for all i,j,k.
    # But U has shape (8, R) and we're using R >= 8 (since k goes up to 7).
    # If R=8, U[i*m+j, k] = 1 for all i,j,k. But U has only 8 columns, so
    # U[i*m+j, r] for r in {0,...,7} = 1 for all i,j,r. That's fine.
    # V[j*p+k, k] = 1 for all j,k. V has shape (32, 8), so V[j*8+k, k] = 1.
    # W[k*n+i, k] = 1 for all k,i. W has shape (16, 8), so W[k*2+i, k] = 1.
    
    # Let's verify: for i=0, j=0, k=0:
    # sum_r U[0*4+0, r] * V[0*8+0, r] * W[0*2+0, r]
    # = sum_r U[0, r] * V[0, r] * W[0, r]
    # U[0, r] = 1 for all r (0-7)
    # V[0, r] = 1 if r == 0 (since V[j*8+k, k] = 1 means V[0, 0] = 1, V[0, r] = 0 for r != 0)
    # Wait, V[j*p+k, k] = 1 means V[0*8+0, 0] = 1, V[0*8+1, 1] = 1, etc.
    # So V[0, 0] = 1, V[0, 1] = 0, V[0, 2] = 0, ..., V[0, 7] = 0.
    # W[0*2+0, 0] = 1, W[0*2+0, 1] = 0, ..., W[0*2+0, 7] = 0.
    # So the sum = U[0,0]*V[0,0]*W[0,0] + 0 + ... + 0 = 1*1*1 = 1. Good.
    
    # For i=0, j=1, k=0:
    # sum_r U[0*4+1, r] * V[1*8+0, r] * W[0*2+0, r]
    # = sum_r U[1, r] * V[8, r] * W[0, r]
    # U[1, r] = 1 for all r
    # V[8, r]: V[j*p+k, k] = 1 means V[1*8+0, 0] = 1, so V[8, 0] = 1, V[8, r] = 0 for r != 0.
    # W[0, r] = 1 if r == 0, else 0.
    # Sum = U[1,0]*V[8,0]*W[0,0] = 1*1*1 = 1. Good.
    
    # For i=1, j=0, k=3:
    # sum_r U[1*4+0, r] * V[0*8+3, r] * W[3*2+1, r]
    # = sum_r U[4, r] * V[3, r] * W[7, r]
    # U[4, r] = 1 for all r
    # V[3, r]: V[j*p+k, k] = 1 means V[0*8+3, 3] = 1, so V[3, 3] = 1, V[3, r] = 0 for r != 3.
    # W[7, r]: W[k*n+i, k] = 1 means W[3*2+1, 3] = 1, so W[7, 3] = 1, W[7, r] = 0 for r != 3.
    # Sum = U[4,3]*V[3,3]*W[7,3] = 1*1*1 = 1. Good.
    
    # This gives a rank-8 solution! Let me verify more carefully.
    
    def try_rank8_solution():
        rank = 8
        U = [[0] * rank for _ in range(n * m)]  # 8x8
        V = [[0] * rank for _ in range(m * p)]  # 32x8
        W = [[0] * rank for _ in range(p * n)]  # 16x8
        
        # U[i*m+j, r] = 1 for all i, j, r
        for i in range(n):
            for j in range(m):
                for r in range(rank):
                    U[i * m + j][r] = 1
        
        # V[j*p+k, r] = 1 if r == k, else 0
        for j in range(m):
            for k in range(p):
                V[j * p + k][k] = 1
        
        # W[k*n+i, r] = 1 if r == k, else 0
        for k in range(p):
            for i in range(n):
                W[k * n + i][k] = 1
        
        if verify_decomposition(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        return None
    
    result = try_rank8_solution()
    if result is not None:
        return result
    
    # If rank 8 doesn't work (it should, but let's be safe), try rank 16.
    def try_rank16_solution():
        rank = 16
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        # U[i*m+j, r] = 1 for all i, j, r < 8
        for i in range(n):
            for j in range(m):
                for r in range(8):
                    U[i * m + j][r] = 1
        
        # V[j*p+k, r] = 1 if r == k, else 0
        for j in range(m):
            for k in range(p):
                V[j * p + k][k] = 1
        
        # W[k*n+i, r] = 1 if r == k, else 0
        for k in range(p):
            for i in range(n):
                W[k * n + i][k] = 1
        
        if verify_decomposition(U, V, W, rank):
            return {"U": U, "V": V, "W": W}
        return None
    
    result = try_rank16_solution()
    if result is not None:
        return result
    
    # Return the Strassen solution
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
