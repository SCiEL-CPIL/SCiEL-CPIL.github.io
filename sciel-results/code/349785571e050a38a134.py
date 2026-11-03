"""
Strategy: Deterministic Vectorized Algebraic Reduction.
1. Construct a valid decomposition for M_{4,4,7} using block Strassen.
2. Use vectorized NumPy operations for O(1) cost verification per candidate.
3. Perform exhaustive iterative greedy elimination of rank-1 terms.
4. If time permits, try randomized local perturbations (though deterministic is primary).
"""
import time
import numpy as np
import random

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    start_time = time.time()
    deadline = start_time + 140  # Safety margin

    def strassen_2x2():
        """
        Returns U, V, W for 2x2 x 2x2 Strassen decomposition.
        Shapes: (4, 7), (4, 7), (4, 7)
        Indices: U[i*2+j], V[i*2+j], W[i*2+j] where i,j in 0..1
        """
        U = np.zeros((4, 7), dtype=float)
        V = np.zeros((4, 7), dtype=float)
        W = np.zeros((4, 7), dtype=float)
        
        # p1 = (a+d)*(e+h)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # p2 = (c+d)*e
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = 1
        
        # p3 = a*(f-h)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # p4 = d*(g-e)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # p5 = (a+b)*h
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # p6 = (c-a)*(e+f)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # p7 = (b-d)*(g+h)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W

    def block_indices(br, bc, size=4):
        """Get flat indices for 2x2 block (br, bc) in a size x size matrix."""
        indices = []
        for r in range(2):
            for c in range(2):
                row = br * 2 + r
                col = bc * 2 + c
                indices.append(row * size + col)
        return indices

    def strassen_4x4x4():
        """
        Returns U, V, W for M_{4,4,4} using recursive Strassen, rank 49.
        U: (16, 49), V: (16, 49), W: (16, 49)
        """
        rank = 49
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        
        U2, V2, W2 = strassen_2x2()
        
        A11_idx = block_indices(0, 0)
        A12_idx = block_indices(0, 1)
        A21_idx = block_indices(1, 0)
        A22_idx = block_indices(1, 1)
        
        B11_idx = block_indices(0, 0)
        B12_idx = block_indices(0, 1)
        B21_idx = block_indices(1, 0)
        B22_idx = block_indices(1, 1)
        
        C11_idx = block_indices(0, 0)
        C12_idx = block_indices(0, 1)
        C21_idx = block_indices(1, 0)
        C22_idx = block_indices(1, 1)
        
        col = 0
        # P1 = (A11 + A22) * (B11 + B22)
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P2 = (A21 + A22) * B11
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C21_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] -= W2[idx % 4, r]
            col += 1
        
        # P3 = A11 * (B12 - B22)
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B12_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] -= V2[idx % 4, r]
            for idx in C12_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C22_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P4 = A22 * (B21 - B11)
        for r in range(7):
            for idx in A22_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B21_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B11_idx: V[idx, col] -= V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            for idx in C21_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P5 = (A11 + A12) * B22
        for r in range(7):
            for idx in A11_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] -= W2[idx % 4, r]
            for idx in C12_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P6 = (A21 - A11) * (B11 + B12)
        for r in range(7):
            for idx in A21_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A11_idx: U[idx, col] -= U2[idx % 4, r]
            for idx in B11_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B12_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C22_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        # P7 = (A12 - A22) * (B21 + B22)
        for r in range(7):
            for idx in A12_idx: U[idx, col] += U2[idx % 4, r]
            for idx in A22_idx: U[idx, col] -= U2[idx % 4, r]
            for idx in B21_idx: V[idx, col] += V2[idx % 4, r]
            for idx in B22_idx: V[idx, col] += V2[idx % 4, r]
            for idx in C11_idx: W[idx, col] += W2[idx % 4, r]
            col += 1
        
        assert col == 49
        return U, V, W

    def schoolbook_4x4x3():
        """
        Returns U, V, W for M_{4,4,3} using schoolbook, rank 48.
        U: (16, 48), V: (12, 48), W: (12, 48)
        """
        rank = 48
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((12, rank), dtype=float)   # 4*3 = 12
        W = np.zeros((12, rank), dtype=float)   # 3*4 = 12
        
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(3):
                    U[i * 4 + j, col] = 1
                    V[j * 3 + k, col] = 1
                    W[k * 4 + i, col] = 1
                    col += 1
        
        assert col == 48
        return U, V, W

    def verify_decomposition_fast(U, V, W, n, m, p):
        """
        Vectorized verification.
        U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
        Returns True if decomposition is exact.
        """
        rank = U.shape[1]
        # Reshape to 3D tensors for broadcasting
        # T_U[i, j, r] = U[i*m+j, r]
        # T_V[j, k, r] = V[j*p+k, r]
        # T_W[k, i, r] = W[k*n+i, r]
        
        # We need sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] == 1 if (i,j,k) is valid
        # Note: The problem definition says W[k*n+i, r].
        # Indices:
        # U: row i*m+j
        # V: row j*p+k
        # W: row k*n+i
        
        # Construct the tensor T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # Efficient way:
        # 1. Expand U to (n, m, 1, p, 1, rank) ? No, too big.
        # 2. Use einsum.
        
        # U_flat: (n*m, rank) -> U_3d: (n, m, rank)
        U_3d = U.reshape(n, m, rank)
        # V_flat: (m*p, rank) -> V_3d: (m, p, rank)
        V_3d = V.reshape(m, p, rank)
        # W_flat: (p*n, rank) -> W_3d: (p, n, rank)
        W_3d = W.reshape(p, n, rank)
        
        # We want T[i, j, k] = sum_r U_3d[i, j, r] * V_3d[j, k, r] * W_3d[k, i, r]
        
        # Step 1: Compute M[i, j, k, r] = U_3d[i, j, r] * V_3d[j, k, r]
        # This is (n, m, 1, p, rank) * (1, m, p, 1, rank) ? 
        # Let's use einsum for clarity and speed.
        # 'ijr,jkr->ijk r'
        UV = np.einsum('ijr,jkr->ijk r', U_3d, V_3d)
        
        # Step 2: Multiply by W and sum over r
        # 'ijk r,ki r->ijk'
        T = np.einsum('ijk r,ki r->ijk', UV, W_3d)
        
        # Check target entries: T[i, j, k] should be 1 for all i,j,k
        # Wait, the tensor is M_{n,m,p}. 
        # The entries are defined for i in [0,n), j in [0,m), k in [0,p).
        # So T should be all 1s.
        
        # Check non-target entries:
        # The decomposition must be zero for any (u_idx, v_idx, w_idx) that doesn't form a valid triple.
        # A valid triple is defined by i = u_idx//m, j = u_idx%m, j2 = v_idx//p, k = v_idx%p, k2 = w_idx//n, i2 = w_idx%n.
        # Valid if i==i2, j==j2, k==k2.
        
        # Instead of checking all (n*m * m*p * p*n) entries (which is 16*28*28 = 12544),
        # we can just check if the tensor T is exactly the identity-like structure.
        # But T is only n x m x p. The "non-target" entries are implicitly zero if the tensor is correctly formed?
        # No. The tensor decomposition defines a 3-way tensor of size (n*m) x (m*p) x (p*n).
        # The condition is:
        # Sum_r U[u, r] V[v, r] W[w, r] = 1 if (u,v,w) corresponds to a valid (i,j,k).
        # Sum_r U[u, r] V[v, r] W[w, r] = 0 otherwise.
        
        # My T calculation above only checks the "valid" entries (where indices align).
        # I need to verify the "invalid" entries are zero.
        
        # Let's construct the full tensor check if rank is small enough, or sample.
        # Given rank ~93, full tensor is 16x28x28 = 12544 entries.
        # We can compute the full tensor using einsum on the flat matrices?
        # T_full[u, v, w] = sum_r U[u, r] V[v, r] W[w, r]
        
        # This is a rank-3 tensor of size (16, 28, 28).
        # We can compute it as:
        # Temp[u, v, r] = U[u, r] * V[v, r]  (16 x 28 x rank)
        # T_full[u, v, w] = sum_r Temp[u, v, r] * W[w, r]
        
        # Let's do it.
        # U: (16, R), V: (28, R), W: (28, R)
        
        # 1. U[:, None, :] * V[None, :, :] -> (16, 28, R)
        UV_full = U[:, np.newaxis, :] * V[np.newaxis, :, :]
        
        # 2. np.einsum('uvr,wr->uvw', UV_full, W) -> (16, 28, 28)
        T_full = np.einsum('uvr,wr->uvw', UV_full, W)
        
        # Now check T_full
        # Target entries:
        # u = i*m + j, v = j*p + k, w = k*n + i
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    if abs(T_full[u, v, w] - 1.0) > 1e-9:
                        return False
        
        # Non-target entries:
        # Create a mask of target entries
        target_mask = np.zeros((n*m, m*p, p*n), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    target_mask[u, v, w] = True
        
        # Check if all non-target entries are ~0
        non_target_vals = T_full[~target_mask]
        if np.any(np.abs(non_target_vals) > 1e-9):
            return False
            
        return True

    # 1. Construct Baseline
    U1, V1, W1 = strassen_4x4x4()
    U2, V2, W2 = schoolbook_4x4x3()
    
    rank_total = 49 + 48
    U_full = np.zeros((16, rank_total), dtype=float)
    V_full = np.zeros((28, rank_total), dtype=float)
    W_full = np.zeros((28, rank_total), dtype=float)
    
    # Part 1: 4x4x4 (B cols 0-3, C cols 0-3)
    # V1 is (16, 49), but V_full needs (28, 49). Rows 0-15 are B[0:4, 0:4]. Rows 16-27 are B[0:4, 4:7] which are 0.
    U_full[:, :49] = U1
    V_full[:16, :49] = V1
    W_full[:16, :49] = W1
    
    # Part 2: 4x4x3 (B cols 4-6, C cols 4-6)
    # U2 is (16, 48)
    # V2 is (12, 48) -> corresponds to B[0:4, 4:7]. In V_full, these are rows 16-27.
    # W2 is (12, 48) -> corresponds to C[4:7, 0:4]. In W_full, these are rows 16-27.
    U_full[:, 49:] = U2
    V_full[16:28, 49:] = V2
    W_full[16:28, 49:] = W2
    
    # Verify baseline
    if not verify_decomposition_fast(U_full, V_full, W_full, n, m, p):
        # Fallback to pure schoolbook if construction failed
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    current_U = U_full.copy()
    current_V = V_full.copy()
    current_W = W_full.copy()
    current_rank = rank_total
    
    # 2. Iterative Greedy Elimination
    # We try to remove terms one by one.
    # Since verification is vectorized, it's fast.
    # We iterate until no more terms can be removed.
    
    improved = True
    while improved:
        improved = False
        for term_idx in range(current_rank - 1, -1, -1):
            if time.time() > deadline:
                break
            
            # Try removing term_idx
            mask = [i for i in range(current_rank) if i != term_idx]
            U_new = current_U[:, mask]
            V_new = current_V[:, mask]
            W_new = current_W[:, mask]
            
            if verify_decomposition_fast(U_new, V_new, W_new, n, m, p):
                current_U = U_new
                current_V = V_new
                current_W = W_new
                current_rank -= 1
                improved = True
                # Restart loop from end? Or continue?
                # Continuing is fine, but indices shift. 
                # Since we are iterating backwards, removing an element at term_idx 
                # doesn't affect indices < term_idx. So we can continue.
        
        if time.time() > deadline:
            break

    # 3. If time remains, try a few random restarts with different initial constructions?
    # Or try "swap" operations?
    # Given the budget, the greedy elimination is the main driver.
    
    # Final check
    if not verify_decomposition_fast(current_U, current_V, current_W, n, m, p):
        # Should not happen, but fallback
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=float)
        V = np.zeros((m*p, rank), dtype=float)
        W = np.zeros((p*n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    return {"U": current_U.tolist(), "V": current_V.tolist(), "W": current_W.tolist()}
# EVOLVE-BLOCK-END
