import numpy as np
import time
import random

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin (evaluator kills at 150s)

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    def snap_to_half(x):
        return np.round(x * 2) / 2.0

    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that the tensor decomposition is exact."""
        U_3d = U.reshape(n, m, -1)
        V_3d = V.reshape(m, p, -1)
        W_3d = W.reshape(p, n, -1)
        reconstructed = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
        return np.allclose(reconstructed, 1.0, atol=tol)

    def verify_and_snap(U, V, W, tol=1e-8):
        """Snap to half-integers and verify."""
        U_s = snap_to_half(U)
        V_s = snap_to_half(V)
        W_s = snap_to_half(W)
        if verify_decomposition(U_s, V_s, W_s, tol):
            return U_s, V_s, W_s, True
        return None, None, None, False

    def strassen_2x2x2():
        """Return the Strassen decomposition for 2x2x2 matrix multiplication.
        Rank 7, with entries in {0, 1, -1}."""
        # The 7 terms of Strassen's algorithm
        # U[i,j], V[j,k], W[k,i] for i,j,k in {0,1}
        terms = [
            # (U_idx, V_idx, W_idx) where idx is (i,j), (j,k), (k,i)
            # Term 1: (A11)(B11-B22) -> C11, C12
            ((0,0), (0,0), (0,0)),  # A11 * B11 -> C11
            ((0,0), (0,0), (1,0)),  # A11 * B11 -> C21 (wait, this isn't right)
        ]
        # Let me use the standard Strassen formulation
        # C11 = M1 - M4 + M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M3 + M6 + M7
        # where:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        #
        # Each M is a product of two linear combinations.
        # We need to express this as a rank-7 decomposition.
        #
        # For each term c, we have U[:,c], V[:,c], W[:,c]
        # U[i,j,c] * V[j,k,c] * W[k,i,c] should sum to C[i,k] = sum_j A[i,j]B[j,k]
        #
        # The trick: each M_t is (sum_a alpha_a A[a,b]) * (sum_b beta_b B[b,c])
        # So for term t: U[i,j] = alpha_{i,j}, V[j,k] = beta_{j,k}, W[k,i] = 1 (or distributed)
        # Actually, we need W to route the product to the right position.
        
        # Standard approach: 
        # For each Strassen term t, define:
        # U_t[i,j] = coefficient of A[i,j] in the first factor
        # V_t[j,k] = coefficient of B[j,k] in the second factor  
        # W_t[k,i] = coefficient that routes to C[i,k]
        # But this doesn't quite work because the product U*V*W gives
        # sum_{i,j,k} U[i,j]V[j,k]W[k,i] which is a scalar, not a matrix.
        #
        # The correct interpretation: the tensor T[i,j,k] = delta (the multiplication tensor)
        # T[i,j,k] = 1 if the (i,k) entry of A*B involves A[i,j]*B[j,k], else 0
        # Actually T[i,j,k] is the tensor such that (A*B)[i,k] = sum_{j} A[i,j] * B[j,k]
        # So T[i,j,k] = 1 for all valid i,j,k (it's the identity in the middle index)
        #
        # For the decomposition: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k] = 1
        #
        # For Strassen on 2x2:
        # We need 7 terms. Let me construct them explicitly.
        
        U = np.zeros((2, 2, 7))
        V = np.zeros((2, 2, 7))
        W = np.zeros((2, 2, 7))
        
        # M1 = (A11+A22)(B11+B22) contributes to C11, C12, C21, C22
        # In the tensor: U[i,j,0]*V[j,k,0]*W[k,i,0]
        # We want this to contribute to all four entries of C.
        # First factor: A11+A22 -> U[0,0]=1, U[1,1]=1
        # Second factor: B11+B22 -> V[0,0]=1, V[1,1]=1
        # But then W needs to be 1 for all (k,i) to contribute to all entries? No...
        # The issue is that U*V*W sums over j, so:
        # sum_j U[i,j,0] V[j,k,0] W[k,i,0]
        # For i=0,k=0: sum_j U[0,j,0]V[j,0,0]W[0,0,0] = (U[0,0]V[0,0] + U[0,1]V[1,0]) * W[0,0]
        # We want this to equal the contribution of M1 to C[0,0]
        
        # This is getting complicated. Let me use a known explicit construction.
        # 
        # Actually, the standard way: the multiplication tensor for 2x2 is rank 7.
        # One explicit decomposition:
        #
        # Let me use the formulation where each term corresponds to a "slice"
        
        # Term 0: A11 * (B11 - B22)
        U[0,0,0] = 1; V[0,0,0] = 1; V[0,1,0] = -1; W[0,0,0] = 1; W[1,0,0] = 1
        # Check: for C[i,k] = sum_j U[i,j]V[j,k]W[k,i]
        # C[0,0]: j=0: U[0,0]V[0,0]W[0,0] = 1*1*1 = 1; j=1: U[0,1]V[1,0]W[0,0] = 0
        # C[0,1]: j=0: U[0,0]V[0,1]W[1,0] = 1*(-1)*1 = -1; j=1: 0
        # C[1,0]: j=0: U[1,0]V[0,0]W[0,1] = 0; j=1: 0  -> 0
        # C[1,1]: j=0: U[1,0]V[0,1]W[1,1] = 0; j=1: 0 -> 0
        # So this term contributes (1,-1,0,0) to C[0,0],C[0,1],C[1,0],C[1,1]
        # That's A11*B11 - A11*B22, which is correct for the (0,0) and (0,1) positions.
        
        # Term 1: A22 * (B12 - B11)
        U[1,1,1] = 1; V[1,0,1] = -1; V[1,1,1] = 1; W[0,1,1] = 1; W[1,1,1] = 1
        # C[1,0]: j=1: U[1,1]V[1,0]W[0,1] = 1*(-1)*1 = -1
        # C[1,1]: j=1: U[1,1]V[1,1]W[1,1] = 1*1*1 = 1
        # Contributes (0,0,-1,1) -> A22*(-B11) + A22*B12
        
        # Term 2: (A11+A22) * (B11+B22)
        U[0,0,2] = 1; U[1,1,2] = 1; V[0,0,2] = 1; V[1,1,2] = 1
        W[0,0,2] = 1; W[0,1,2] = 1; W[1,0,2] = 1; W[1,1,2] = 1
        # C[0,0]: j=0: U[0,0]V[0,0]W[0,0]=1; j=1: U[0,1]V[1,0]W[0,0]=0 -> 1
        # C[0,1]: j=0: U[0,0]V[0,1]W[1,0]=0; j=1: U[0,1]V[1,1]W[1,0]=0 -> 0
        # Hmm, that's not right. Let me reconsider.
        
        # I think I'm overcomplicating this. Let me use a verified Strassen decomposition.
        
        # Reset
        U = np.zeros((2, 2, 7))
        V = np.zeros((2, 2, 7))
        W = np.zeros((2, 2, 7))
        
        # Using the standard Strassen decomposition for 2x2x2:
        # The 7 bilinear forms are:
        # z1 = (a11+a22)(b11+b22)
        # z2 = (a21+a22)b11
        # z3 = a11(b21-b22)
        # z4 = a22(b12-b11)
        # z5 = (a11+a12)b22
        # z6 = (a21-a11)(b11+b12)
        # z7 = (a12-a22)(b21+b22)
        #
        # c11 = z1 - z4 + z5 + z7
        # c12 = z3 + z5
        # c21 = z2 + z4
        # c22 = z1 - z3 + z6 + z7
        #
        # For the tensor decomposition, term c has:
        # U_c[i,j] = coeff of a[i,j] in first factor of z_c
        # V_c[j,k] = coeff of b[j,k] in second factor of z_c
        # W_c[k,i] = coeff of z_c in c[i,k]
        
        # z1 = (a11+a22)(b11+b22)
        U[0,0,0] = 1; U[1,1,0] = 1
        V[0,0,0] = 1; V[1,1,0] = 1
        W[0,0,0] = 1; W[0,1,0] = 0; W[1,0,0] = 0; W[1,1,0] = 1
        # c11 gets +1, c22 gets +1
        
        # z2 = (a21+a22)b11
        U[1,0,1] = 1; U[1,1,1] = 1
        V[0,0,1] = 1
        W[1,0,1] = 1  # c21 gets +1
        
        # z3 = a11(b21-b22)
        U[0,0,2] = 1
        V[1,0,2] = 1; V[1,1,2] = -1
        W[0,1,2] = 1; W[1,1,2] = -1  # c12 gets +1, c22 gets -1
        
        # z4 = a22(b12-b11)
        U[1,1,3] = 1
        V[0,0,3] = -1; V[0,1,3] = 1
        W[0,0,3] = -1; W[1,0,3] = 1  # c11 gets -1, c21 gets +1
        
        # z5 = (a11+a12)b22
        U[0,0,4] = 1; U[0,1,4] = 1
        V[1,1,4] = 1
        W[0,0,4] = 1; W[0,1,4] = 1  # c11 gets +1, c12 gets +1
        
        # z6 = (a21-a11)(b11+b12)
        U[1,0,5] = 1; U[0,0,5] = -1
        V[0,0,5] = 1; V[0,1,5] = 1
        W[1,1,5] = 1  # c22 gets +1
        
        # z7 = (a12-a22)(b21+b22)
        U[0,1,6] = 1; U[1,1,6] = -1
        V[1,0,6] = 1; V[1,1,6] = 1
        W[0,0,6] = 1; W[1,1,6] = 1  # c11 gets +1, c22 gets +1
        
        # Verify
        U_flat = U.reshape(4, 7)
        V_flat = V.reshape(4, 7)
        W_flat = W.reshape(4, 7)
        # T[i,j,k] should be 1 for all i,j,k in {0,1}
        recon = np.einsum('ijc,jkc,kic->ijk', U, V, W)
        if not np.allclose(recon, 1.0, atol=1e-10):
            # Fallback: use schoolbook for 2x2x2 (rank 4)
            U = np.zeros((2, 2, 4))
            V = np.zeros((2, 2, 4))
            W = np.zeros((2, 2, 4))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i, j, i*2+j*2+k] = 1  # This is wrong, let me redo
            # Schoolbook for 2x2: rank 4
            U = np.zeros((2, 2, 4))
            V = np.zeros((2, 2, 4))
            W = np.zeros((2, 2, 4))
            col = 0
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        U[i, j, col] = 1
                        V[j, k, col] = 1
                        W[k, i, col] = 1
                        col += 1
            return U.reshape(4, 4), V.reshape(4, 4), W.reshape(4, 4)
        
        return U.reshape(4, 7), V.reshape(4, 7), W.reshape(4, 7)

    def build_block_decomposition():
        """Build a decomposition by splitting p=6 into three p=2 blocks.
        For each block, we need to multiply 3x4 by 4x2.
        Split m=4 into two m=2 blocks.
        For each (m_block, p_block) pair, we have a 3x2 by 2x2 multiplication.
        Split n=3 into n=2 and n=1.
        For the n=2 part: use Strassen-like or known rank for 2x2x2 (rank 7)
        For the n=1 part: rank 4 (just dot products)
        
        Actually, let me think more carefully.
        3x4 * 4x6:
        Split columns of A (m=4) into two groups of 2: A = [A1 | A2] where A1,A2 are 3x2
        Split rows of B (m=4) and columns of B (p=6) into blocks:
        B = [[B11, B12, B13], [B21, B22, B23]] where Bij are 2x2
        C = A1*B_col1 + A2*B_col2 where B_col_i = [B1i; B2i] is 4x2
        
        So C = [A1; A2] * B where A1 is 3x2, A2 is 3x2, B is 4x6
        = A1 * B_top + A2 * B_bottom where B_top is 2x6, B_bottom is 2x6
        
        Split B_top and B_bottom into three 2x2 blocks each:
        B_top = [Bt1, Bt2, Bt3], B_bottom = [Bb1, Bb2, Bb3]
        
        C = A1*Bt1 + A1*Bt2 + A1*Bt3 + A2*Bb1 + A2*Bb2 + A2*Bb3
        
        Each term is (3x2) * (2x2) = 3x2
        For 3x2 * 2x2: split n=3 into n=2 and n=1
        n=2 part: 2x2 * 2x2, rank 7 (Strassen)
        n=1 part: 1x2 * 2x2, rank 4 (schoolbook: 1*2*2=4)
        Total per term: 7 + 4 = 11
        6 terms: 6 * 11 = 66
        
        This matches the known rank 66!
        """
        # Strassen for 2x2x2
        U_s, V_s, W_s = strassen_2x2x2()
        
        # Build the full decomposition
        # Total rank: 6 * 11 = 66
        total_rank = 66
        U = np.zeros((n * m, total_rank))  # (12, 66)
        V = np.zeros((m * p, total_rank))  # (24, 66)
        W = np.zeros((p * n, total_rank))  # (18, 66)
        
        # 6 blocks: (m_block in {0,1}, p_block in {0,1,2})
        # m_block 0: rows 0,1 of B (cols 0,1 of A)
        # m_block 1: rows 2,3 of B (cols 2,3 of A)
        # p_block k: cols 2k, 2k+1 of B (and C)
        
        col_offset = 0
        for mb in range(2):  # m_block: 0 or 1
            for pb in range(3):  # p_block: 0, 1, or 2
                # A indices: rows 0,1,2 (n=3), cols 2*mb, 2*mb+1 (m=2)
                # B indices: rows 2*mb, 2*mb+1 (m=2), cols 2*pb, 2*pb+1 (p=2)
                # C indices: rows 0,1,2 (n=3), cols 2*pb, 2*pb+1 (p=2)
                
                # n=2 part (rows 0,1 of C): Strassen rank 7
                for c in range(7):
                    # U: A[i, j] where i in {0,1}, j in {2*mb, 2*mb+1}
                    U_idx_i = [0, 1]
                    U_idx_j = [2*mb, 2*mb+1]
                    for ii in range(2):
                        for jj in range(2):
                            U[U_idx_i[ii] * m + U_idx_j[jj], col_offset + c] = U_s[ii * 2 + jj, c]
                    
                    # V: B[j, k] where j in {2*mb, 2*mb+1}, k in {2*pb, 2*pb+1}
                    V_idx_j = [2*mb, 2*mb+1]
                    V_idx_k = [2*pb, 2*pb+1]
                    for jj in range(2):
                        for kk in range(2):
                            V[V_idx_j[jj] * p + V_idx_k[kk], col_offset + c] = V_s[jj * 2 + kk, c]
                    
                    # W: C[k, i] where k in {2*pb, 2*pb+1}, i in {0,1}
                    W_idx_k = [2*pb, 2*pb+1]
                    W_idx_i = [0, 1]
                    for kk in range(2):
                        for ii in range(2):
                            W[W_idx_k[kk] * n + W_idx_i[ii], col_offset + c] = W_s[kk * 2 + ii, c]
                
                col_offset += 7
                
                # n=1 part (row 2 of C): schoolbook rank 4
                for c in range(4):
                    j_local = c // 2
                    k_local = c % 2
                    j = 2 * mb + j_local
                    k = 2 * pb + k_local
                    
                    U[2 * m + j, col_offset + c] = 1
                    V[j * p + k, col_offset + c] = 1
                    W[k * n + 2, col_offset + c] = 1
                
                col_offset += 4
        
        assert col_offset == total_rank, f"col_offset={col_offset}, expected {total_rank}"
        
        # Verify
        if verify_decomposition(U, V, W, tol=1e-10):
            return U, V, W
        
        # Fallback to schoolbook
        return None, None, None

    # Try the block decomposition first
    U, V, W = build_block_decomposition()
    if U is not None:
        # Snap and verify
        U_s, V_s, W_s, success = verify_and_snap(U, V, W, tol=1e-8)
        if success:
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        # If it's already integer, return directly
        if verify_decomposition(U, V, W, tol=1e-10):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Fallback: try to find a lower rank via optimization
    def optimize_rank(rank, max_iter=3000, lr=0.02):
        """Use gradient descent to find a rank-r decomposition."""
        np.random.seed(int(time.time() * 1000) % 100000)
        U = np.random.randn(n * m, rank) * 0.1
        V = np.random.randn(m * p, rank) * 0.1
        W = np.random.randn(p * n, rank) * 0.1
        
        momentum_U = np.zeros_like(U)
        momentum_V = np.zeros_like(V)
        momentum_W = np.zeros_like(W)
        beta = 0.9
        lr_current = lr
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                return None
            
            U_3d = U.reshape(n, m, rank)
            V_3d = V.reshape(m, p, rank)
            W_3d = W.reshape(p, n, rank)
            
            recon = np.einsum('ijc,jkc,kic->ijk', U_3d, V_3d, W_3d)
            err = 1.0 - recon
            
            grad_U_3d = -2 * np.einsum('ijk,jkc,kic->ijc', err, V_3d, W_3d)
            grad_U = grad_U_3d.reshape(n * m, rank)
            
            grad_V_3d = -2 * np.einsum('ijk,ijc,kic->jkc', err, U_3d, W_3d)
            grad_V = grad_V_3d.reshape(m * p, rank)
            
            grad_W_3d = -2 * np.einsum('ijk,ijc,jkc->kic', err, U_3d, V_3d)
            grad_W = grad_W_3d.reshape(p * n, rank)
            
            momentum_U = beta * momentum_U + (1 - beta) * grad_U
            momentum_V = beta * momentum_V + (1 - beta) * grad_V
            momentum_W = beta * momentum_W + (1 - beta) * grad_W
            
            U -= lr_current * momentum_U
            V -= lr_current * momentum_V
            W -= lr_current * momentum_W
            
            # Adaptive learning rate
            if it % 500 == 0 and it > 0:
                lr_current *= 0.9
            
            # Check for convergence
            if it % 200 == 0:
                U_s, V_s, W_s, success = verify_and_snap(U, V, W, tol=1e-6)
                if success:
                    return U_s, V_s, W_s
        
        # Final check
        U_s, V_s, W_s, success = verify_and_snap(U, V, W, tol=1e-6)
        if success:
            return U_s, V_s, W_s
        return None

    # Try ranks below 66
    for r in [60, 62, 64, 65]:
        if time.time() > deadline - 15:
            break
        for restart in range(2):
            if time.time() > deadline - 10:
                break
            res = optimize_rank(r, max_iter=2000, lr=0.03)
            if res:
                U, V, W = res
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Final fallback: schoolbook
    return get_schoolbook()
# EVOLVE-BLOCK-END
