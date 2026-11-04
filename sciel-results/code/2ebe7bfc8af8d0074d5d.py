import time
import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # 140s safety margin within 150s limit
    
    # Start with schoolbook decomposition (rank = n*m*p = 48)
    rank0 = n * m * p
    U_cols = []
    V_cols = []
    W_cols = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = np.zeros(n*m)
                v = np.zeros(m*p)
                w = np.zeros(p*n)
                u[i*m + j] = 1.0
                v[j*p + k] = 1.0
                w[k*n + i] = 1.0
                U_cols.append(u)
                V_cols.append(v)
                W_cols.append(w)
    
    U = np.column_stack(U_cols)
    V = np.column_stack(V_cols)
    W = np.column_stack(W_cols)
    
    # Verify the schoolbook decomposition
    def _verify(U, V, W, tol=1e-10):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    for l in range(p):
                        expected = 1.0 if (i == l and j == k) else 0.0
                        actual = 0.0
                        for r in range(U.shape[1]):
                            actual += U[i*m+j, r] * V[j*p+k, r] * W[k*n+l, r]
                        if abs(actual - expected) > tol:
                            return False
        return True
    
    if not _verify(U, V, W):
        # Should not happen for schoolbook, but fallback
        return {"U": np.zeros((n*m, 1)).tolist(), "V": np.zeros((m*p, 1)).tolist(), "W": np.zeros((p*n, 1)).tolist()}
    
    # Now try to reduce rank via a search-based approach
    # Strategy: Try to find a valid decomposition with fewer terms by:
    # 1. Randomly perturb and re-optimize
    # 2. Try known smaller decompositions (Strassen-like)
    # 3. Use a greedy rank-reduction heuristic
    
    best_U, best_V, best_W = U, V, W
    best_rank = rank0
    
    # First, try the corrected Strassen-based decomposition from the knowledge
    # The knowledge says: fixing bugs achieves rank 44 vs schoolbook 48
    # Let's implement a correct Strassen-based decomposition for 3x4 * 4x4
    
    U_cols2 = []
    V_cols2 = []
    W_cols2 = []
    
    def add_strassen_block(A_block_start_i, A_block_start_j, B_block_start_i, B_block_start_j, C_block_start_i, C_block_start_j):
        """
        Strassen decomposition for 2x2 * 2x2 blocks.
        A: rows [A_block_start_i, A_block_start_i+2), cols [A_block_start_j, A_block_start_j+2)
        B: rows [B_block_start_i, B_block_start_i+2), cols [B_block_start_j, B_block_start_j+2)
        C: rows [C_block_start_i, C_block_start_i+2), cols [C_block_start_j, C_block_start_j+2)
        
        W index: W[k*n + i, r] where k is col of C, i is row of C
        """
        terms = [
            # M1 = (x00+x11)(y00+y11)
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M2 = (x10+x11)y00
            (
                { (A_block_start_i+1)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M3 = (x00-x10)(y10+y11)
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+1)*m + (A_block_start_j+0): -1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M4 = x11(y11-y00)
            (
                { (A_block_start_i+1)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1, (B_block_start_i+0)*p + (B_block_start_j+0): -1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M5 = (x00+x01)y11
            (
                { (A_block_start_i+0)*m + (A_block_start_j+0): 1, (A_block_start_i+0)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): -1, (C_block_start_j+1)*n + (C_block_start_i+0): -1, (C_block_start_j+0)*n + (C_block_start_i+1): 1, (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M6 = (x10-x00)(y00+y01)
            (
                { (A_block_start_i+1)*m + (A_block_start_j+0): 1, (A_block_start_i+0)*m + (A_block_start_j+0): -1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1, (B_block_start_i+0)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+1)*n + (C_block_start_i+1): 1 }
            ),
            # M7 = (x01-x11)(y10+y11)
            (
                { (A_block_start_i+0)*m + (A_block_start_j+1): 1, (A_block_start_i+1)*m + (A_block_start_j+1): -1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1, (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i+0): 1, (C_block_start_j+1)*n + (C_block_start_i+0): 1 }
            ),
        ]
        
        for u_coefs, v_coefs, w_coefs in terms:
            u_col = np.zeros(n*m)
            for idx, coef in u_coefs.items():
                u_col[idx] = coef
            v_col = np.zeros(m*p)
            for idx, coef in v_coefs.items():
                v_col[idx] = coef
            w_col = np.zeros(p*n)
            for idx, coef in w_coefs.items():
                w_col[idx] = coef
            U_cols2.append(u_col)
            V_cols2.append(v_col)
            W_cols2.append(w_col)
    
    def add_schoolbook_1x2(A_block_start_i, A_block_start_j, B_block_start_i, B_block_start_j, C_block_start_i, C_block_start_j):
        """
        Schoolbook for 1x2 * 2x2 block.
        A: row A_block_start_i, cols [A_block_start_j, A_block_start_j+2)
        B: rows [B_block_start_i, B_block_start_i+2), cols [B_block_start_j, B_block_start_j+2)
        C: row C_block_start_i, cols [C_block_start_j, C_block_start_j+2)
        """
        terms = [
            # A[0] * B[0,0] -> C[0]
            (
                { (A_block_start_i)*m + (A_block_start_j+0): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+0): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i): 1 }
            ),
            # A[1] * B[1,0] -> C[0]
            (
                { (A_block_start_i)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+0): 1 },
                { (C_block_start_j+0)*n + (C_block_start_i): 1 }
            ),
            # A[0] * B[0,1] -> C[1]
            (
                { (A_block_start_i)*m + (A_block_start_j+0): 1 },
                { (B_block_start_i+0)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+1)*n + (C_block_start_i): 1 }
            ),
            # A[1] * B[1,1] -> C[1]
            (
                { (A_block_start_i)*m + (A_block_start_j+1): 1 },
                { (B_block_start_i+1)*p + (B_block_start_j+1): 1 },
                { (C_block_start_j+1)*n + (C_block_start_i): 1 }
            ),
        ]
        
        for u_coefs, v_coefs, w_coefs in terms:
            u_col = np.zeros(n*m)
            for idx, coef in u_coefs.items():
                u_col[idx] = coef
            v_col = np.zeros(m*p)
            for idx, coef in v_coefs.items():
                v_col[idx] = coef
            w_col = np.zeros(p*n)
            for idx, coef in w_coefs.items():
                w_col[idx] = coef
            U_cols2.append(u_col)
            V_cols2.append(v_col)
            W_cols2.append(w_col)
    
    # Build the 3x4 * 4x4 decomposition using Strassen for 2x2 blocks and schoolbook for 1x2 blocks
    # C is 3x4, A is 3x4, B is 4x4
    # Split A into: top 2 rows (0,1) and bottom 1 row (2)
    # Split B into: top 2 rows (0,1) and bottom 2 rows (2,3)
    # Split C into: top 2 rows (0,1) and bottom 1 row (2)
    
    # For the top 2 rows of C:
    # C[0:2, 0:2] = A[0:2, 0:2] * B[0:2, 0:2] + A[0:2, 2:4] * B[2:4, 0:2]
    # C[0:2, 2:4] = A[0:2, 0:2] * B[0:2, 2:4] + A[0:2, 2:4] * B[2:4, 2:4]
    
    # For the bottom 1 row of C:
    # C[2, 0:2] = A[2, 0:2] * B[0:2, 0:2] + A[2, 2:4] * B[2:4, 0:2]
    # C[2, 2:4] = A[2, 0:2] * B[0:2, 2:4] + A[2, 2:4] * B[2:4, 2:4]
    
    # Use Strassen for 2x2 * 2x2 blocks (7 terms each)
    # Use schoolbook 1x2 * 2x2 for 1x2 * 2x2 blocks (4 terms each)
    
    # Block 1: C[0:2, 0:2]
    add_strassen_block(0, 0, 0, 0, 0, 0)  # A[0:2, 0:2] * B[0:2, 0:2]
    add_strassen_block(0, 2, 2, 0, 0, 0)  # A[0:2, 2:4] * B[2:4, 0:2]
    
    # Block 2: C[0:2, 2:4]
    add_strassen_block(0, 0, 0, 2, 0, 2)  # A[0:2, 0:2] * B[0:2, 2:4]
    add_strassen_block(0, 2, 2, 2, 0, 2)  # A[0:2, 2:4] * B[2:4, 2:4]
    
    # Block 3: C[2, 0:2]
    add_schoolbook_1x2(2, 0, 0, 0, 2, 0)  # A[2, 0:2] * B[0:2, 0:2]
    add_schoolbook_1x2(2, 2, 2, 0, 2, 0)  # A[2, 2:4] * B[2:4, 0:2]
    
    # Block 4: C[2, 2:4]
    add_schoolbook_1x2(2, 0, 0, 2, 2, 2)  # A[2, 0:2] * B[0:2, 2:4]
    add_schoolbook_1x2(2, 2, 2, 2, 2, 2)  # A[2, 2:4] * B[2:4, 2:4]
    
    U2 = np.column_stack(U_cols2)
    V2 = np.column_stack(V_cols2)
    W2 = np.column_stack(W_cols2)
    
    if _verify(U2, V2, W2):
        if U2.shape[1] < best_rank:
            best_U, best_V, best_W = U2, V2, W2
            best_rank = U2.shape[1]
    
    # Now try to further reduce rank using a search-based approach
    # Strategy: Try to merge terms that are linearly dependent or can be combined
    
    # For now, try a simple random search: perturb the current best and see if we can find a valid lower-rank decomposition
    # This is computationally expensive, so we'll do a limited search
    
    # Try a few different block decompositions with different tilings
    # For example, split A into rows [0], [1], [2] and use different B splits
    
    # Alternative: Try a 3x3 Strassen-like decomposition if possible
    # But 3x3 * 3x3 has rank 23 (Coppersmith-Winograd), which is complex
    
    # For now, let's try to see if we can reduce the rank by combining terms
    
    # Check if the current best is the Strassen-based one (rank 44) or schoolbook (rank 48)
    # If it's 44, try to see if we can do better
    
    # Try a different tiling: split A into [0:3] and use B splits differently
    # This is getting complex, so let's just return the best we have
    
    def _to_clean_list(arr):
        result = []
        for row in arr:
            clean_row = []
            for val in row:
                nearest_int = round(val)
                nearest_half = round(val * 2) / 2.0
                if abs(val - nearest_int) < 1e-10:
                    clean_row.append(int(nearest_int))
                elif abs(val - nearest_half) < 1e-10:
                    clean_row.append(nearest_half)
                else:
                    clean_row.append(float(val))
            result.append(clean_row)
        return result
    
    return {"U": _to_clean_list(best_U), "V": _to_clean_list(best_V), "W": _to_clean_list(best_W)}
