"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 by 3x6 matrix multiplication.
    Strategy: Use Strassen-based block decomposition (rank 33) as baseline, then search
    for rank-32 using structured pattern search with constraint satisfaction.
    """
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety

    n, m, p = DIMENSIONS
    dim_a = n * m  # 6
    dim_b = m * p  # 18
    dim_c = p * n  # 12

    # Build the target tensor T[a, b, c]
    T = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the tensor T."""
        rank = U.shape[1]
        result = np.zeros((dim_a, dim_b, dim_c), dtype=np.int64)
        for c in range(rank):
            uc = U[:, c]
            vc = V[:, c]
            wc = W[:, c]
            result += np.einsum('i,j,k->ijk', uc, vc, wc)
        return np.array_equal(result, T)

    def schoolbook():
        """Return the schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    U[a, col] = 1
                    V[b, col] = 1
                    W[c, col] = 1
                    col += 1
        return U, V, W

    def strassen_block_decomposition():
        """
        Return rank-33 decomposition: 3 blocks of (Strassen 2x2x2 + Standard 2x1x2).
        Each block handles a 2x2 sub-block of the 3x6 B matrix plus the remaining column.
        """
        rank = 33
        U = np.zeros((dim_a, rank), dtype=np.int64)
        V = np.zeros((dim_b, rank), dtype=np.int64)
        W = np.zeros((dim_c, rank), dtype=np.int64)
        
        # Strassen 2x2x2 factors
        S_U = [
            [[1, 0], [0, 1]],
            [[0, 0], [1, 1]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 1], [0, -1]]
        ]
        S_V = [
            [[1, 0], [0, 1]],
            [[1, 0], [0, 0]],
            [[0, 1], [0, -1]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 1], [0, 0]],
            [[0, 0], [1, 1]]
        ]
        S_W = [
            [[1, 0], [0, 1]],
            [[0, 1], [0, -1]],
            [[0, 0], [1, 1]],
            [[1, 1], [0, 0]],
            [[-1, 0], [1, 0]],
            [[0, 0], [0, 1]],
            [[1, 0], [0, 0]]
        ]
        
        # Standard 2x1x2 factors
        Std_U = [
            [[1], [0]],
            [[1], [0]],
            [[0], [1]],
            [[0], [1]]
        ]
        Std_V = [
            [[1, 0]],
            [[0, 1]],
            [[1, 0]],
            [[0, 1]]
        ]
        Std_W = [
            [[1, 0], [0, 0]],
            [[0, 0], [1, 0]],
            [[0, 1], [0, 0]],
            [[0, 0], [0, 1]]
        ]
        
        col = 0
        for b in range(3):
            # Strassen part: handles columns 2b, 2b+1 of B (and corresponding C)
            for r in range(7):
                # Map U (2x2) to global U (rows for i=0,1 and j=0,1)
                for i in range(2):
                    for j in range(2):
                        U[i * m + j, col] = S_U[r][i][j]
                
                # Map V (2x2) to global V (rows for j=0,1 and k=2b, 2b+1)
                for j in range(2):
                    for kp in range(2):
                        k_global = 2 * b + kp
                        V[j * p + k_global, col] = S_V[r][j][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = S_W[r][kp][i]
                
                col += 1
            
            # Standard part: handles column j=2 of A/B, columns 2b, 2b+1 of B
            for r in range(4):
                # Map U (2x1) to global U (rows for i=0,1 and j=2)
                for i in range(2):
                    U[i * m + 2, col] = Std_U[r][i][0]
                
                # Map V (1x2) to global V (row for j=2 and k=2b, 2b+1)
                for kp in range(2):
                    k_global = 2 * b + kp
                    V[2 * p + k_global, col] = Std_V[r][0][kp]
                
                # Map W (2x2) to global W (rows for k=2b, 2b+1 and i=0,1)
                for kp in range(2):
                    for i in range(2):
                        k_global = 2 * b + kp
                        W[k_global * n + i, col] = Std_W[r][kp][i]
                
                col += 1
        
        return U, V, W

    def search_rank_32():
        """
        Search for a rank-32 decomposition by modifying the rank-33 baseline.
        Uses structured pattern search with constraint satisfaction.
        """
        # Start from the rank-33 baseline
        U33, V33, W33 = strassen_block_decomposition()
        
        # Strategy 1: Try merging pairs of rank-1 terms
        # For each pair (r1, r2), try to find if they can be replaced by one term
        # that satisfies the same constraints
        
        # Build constraint equations from T
        # T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        
        # Try removing one term and adjusting others
        for remove_idx in range(33):
            if time.time() > deadline - 15:
                break
            
            U = U33.copy()
            V = V33.copy()
            W = W33.copy()
            
            # Remove term remove_idx
            U = np.delete(U, remove_idx, axis=1)
            V = np.delete(V, remove_idx, axis=1)
            W = np.delete(W, remove_idx, axis=1)
            
            # Now we have rank 32, but it won't satisfy T exactly
            # Compute residual
            residual = T.astype(np.float64).copy()
            for r in range(32):
                residual -= np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
            
            # Try to fix by adjusting one term
            for adjust_idx in range(32):
                if time.time() > deadline - 15:
                    break
                
                U_test = U.copy()
                V_test = V.copy()
                W_test = W.copy()
                
                # Try various adjustments to term adjust_idx
                # Scale factors
                for scale_u in [-2, -1, 0, 1, 2]:
                    for scale_v in [-2, -1, 0, 1, 2]:
                        for scale_w in [-2, -1, 0, 1, 2]:
                            if time.time() > deadline - 15:
                                break
                            
                            U_test[:, adjust_idx] = U[:, adjust_idx] * (scale_u if scale_u != 0 else 1)
                            V_test[:, adjust_idx] = V[:, adjust_idx] * (scale_v if scale_v != 0 else 1)
                            W_test[:, adjust_idx] = W[:, adjust_idx] * (scale_w if scale_w != 0 else 1)
                            
                            # Check if this fixes the residual
                            test_residual = T.astype(np.float64).copy()
                            for r in range(32):
                                test_residual -= np.einsum('i,j,k->ijk', U_test[:, r], V_test[:, r], W_test[:, r])
                            
                            if np.allclose(test_residual, 0, atol=1e-9):
                                U_int = np.round(U_test).astype(np.int64)
                                V_int = np.round(V_test).astype(np.int64)
                                W_int = np.round(W_test).astype(np.int64)
                                if verify_decomposition(U_int, V_int, W_int):
                                    return U_int, V_int, W_int
        
        # Strategy 2: Try modifying values in specific positions
        # Focus on the boundary between Strassen and Standard blocks
        for b in range(3):
            if time.time() > deadline - 15:
                break
            
            # The boundary terms are at indices 7*b to 7*b+6 (Strassen) and 7*b+7 to 7*b+10 (Standard)
            for modify_idx in range(7 * b, 7 * b + 11):
                if time.time() > deadline - 15:
                    break
                
                U = U33.copy()
                V = V33.copy()
                W = W33.copy()
                
                # Try changing values in this term
                for du in [-2, -1, 0, 1, 2]:
                    for dv in [-2, -1, 0, 1, 2]:
                        for dw in [-2, -1, 0, 1, 2]:
                            if time.time() > deadline - 15:
                                break
                            
                            # Apply small perturbation
                            U_test = U.copy()
                            V_test = V.copy()
                            W_test = W.copy()
                            
                            # Modify non-zero entries
                            nz_u = np.nonzero(U[:, modify_idx])[0]
                            nz_v = np.nonzero(V[:, modify_idx])[0]
                            nz_w = np.nonzero(W[:, modify_idx])[0]
                            
                            if len(nz_u) > 0:
                                U_test[nz_u[0], modify_idx] += du
                            if len(nz_v) > 0:
                                V_test[nz_v[0], modify_idx] += dv
                            if len(nz_w) > 0:
                                W_test[nz_w[0], modify_idx] += dw
                            
                            # Check
                            test_residual = T.astype(np.float64).copy()
                            for r in range(33):
                                test_residual -= np.einsum('i,j,k->ijk', U_test[:, r], V_test[:, r], W_test[:, r])
                            
                            if np.allclose(test_residual, 0, atol=1e-9):
                                # Try to remove a redundant term
                                for remove_idx in range(33):
                                    if remove_idx == modify_idx:
                                        continue
                                    
                                    U_rm = np.delete(U_test, remove_idx, axis=1)
                                    V_rm = np.delete(V_test, remove_idx, axis=1)
                                    W_rm = np.delete(W_test, remove_idx, axis=1)
                                    
                                    # Check if rank-32 still works
                                    rm_residual = T.astype(np.float64).copy()
                                    for r in range(32):
                                        rm_residual -= np.einsum('i,j,k->ijk', U_rm[:, r], V_rm[:, r], W_rm[:, r])
                                    
                                    if np.allclose(rm_residual, 0, atol=1e-9):
                                        U_int = np.round(U_rm).astype(np.int64)
                                        V_int = np.round(V_rm).astype(np.int64)
                                        W_int = np.round(W_rm).astype(np.int64)
                                        if verify_decomposition(U_int, V_int, W_int):
                                            return U_int, V_int, W_int
        
        return None

    def random_search_rank_32(num_trials=100):
        """
        Random search for rank-32 decomposition using sparse patterns.
        """
        for trial in range(num_trials):
            if time.time() > deadline - 20:
                break
            
            # Generate random sparse pattern
            U = np.zeros((dim_a, 32), dtype=np.int64)
            V = np.zeros((dim_b, 32), dtype=np.int64)
            W = np.zeros((dim_c, 32), dtype=np.int64)
            
            for r in range(32):
                # Random sparse support
                nz_u = np.random.choice(dim_a, size=np.random.randint(1, 4), replace=False)
                nz_v = np.random.choice(dim_b, size=np.random.randint(1, 4), replace=False)
                nz_w = np.random.choice(dim_c, size=np.random.randint(1, 4), replace=False)
                
                U[nz_u, r] = np.random.choice([-2, -1, 1, 2], size=len(nz_u))
                V[nz_v, r] = np.random.choice([-2, -1, 1, 2], size=len(nz_v))
                W[nz_w, r] = np.random.choice([-2, -1, 1, 2], size=len(nz_w))
            
            # Check if this works
            if verify_decomposition(U, V, W):
                return U, V, W
        
        return None

    # Start with the known rank-33 decomposition
    best_U, best_V, best_W = strassen_block_decomposition()
    best_rank = 33
    
    # Verify the baseline
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook if Strassen construction is wrong
        best_U, best_V, best_W = schoolbook()
        best_rank = n * m * p
    else:
        # Try to find rank-32
        result_32 = search_rank_32()
        if result_32 is not None:
            best_U, best_V, best_W = result_32
            best_rank = 32
        else:
            # Try random search
            result_rand = random_search_rank_32()
            if result_rand is not None:
                best_U, best_V, best_W = result_rand
                best_rank = 32

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
# EVOLVE-BLOCK-END
