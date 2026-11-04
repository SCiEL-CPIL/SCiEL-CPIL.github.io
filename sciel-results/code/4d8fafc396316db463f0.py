import time
import numpy as np
from itertools import product

def construct():
    t0 = time.time()
    deadline = t0 + 110.0  # safety margin under 150s kill
    
    n, m, p = 2, 4, 4
    
    def add_block(col_offset, X_rows, Y_rows, Z_rows, U, V, W, rank):
        """Add a 7-term Strassen 2x2x2 block multiply at given column offset."""
        # Term 0: (X0+X3)(Y0+Y3) -> Z0, Z3
        U[X_rows[0]][col_offset + 0] += 1
        U[X_rows[3]][col_offset + 0] += 1
        V[Y_rows[0]][col_offset + 0] += 1
        V[Y_rows[3]][col_offset + 0] += 1
        W[Z_rows[0]][col_offset + 0] += 1
        W[Z_rows[3]][col_offset + 0] += 1
        
        # Term 1: (X2+X3)Y0 -> Z2, Z3
        U[X_rows[2]][col_offset + 1] += 1
        U[X_rows[3]][col_offset + 1] += 1
        V[Y_rows[0]][col_offset + 1] += 1
        W[Z_rows[2]][col_offset + 1] += 1
        W[Z_rows[3]][col_offset + 1] -= 1
        
        # Term 2: X0(Y1-Y3) -> Z1, Z3
        U[X_rows[0]][col_offset + 2] += 1
        V[Y_rows[1]][col_offset + 2] += 1
        V[Y_rows[3]][col_offset + 2] -= 1
        W[Z_rows[1]][col_offset + 2] += 1
        W[Z_rows[3]][col_offset + 2] += 1
        
        # Term 3: X3(Y2-Y0) -> Z0, Z2
        U[X_rows[3]][col_offset + 3] += 1
        V[Y_rows[2]][col_offset + 3] += 1
        V[Y_rows[0]][col_offset + 3] -= 1
        W[Z_rows[0]][col_offset + 3] += 1
        W[Z_rows[2]][col_offset + 3] += 1
        
        # Term 4: (X0+X1)Y3 -> Z0, Z1
        U[X_rows[0]][col_offset + 4] += 1
        U[X_rows[1]][col_offset + 4] += 1
        V[Y_rows[3]][col_offset + 4] += 1
        W[Z_rows[0]][col_offset + 4] -= 1
        W[Z_rows[1]][col_offset + 4] += 1
        
        # Term 5: (X2-X0)(Y0+Y1) -> Z3
        U[X_rows[2]][col_offset + 5] += 1
        U[X_rows[0]][col_offset + 5] -= 1
        V[Y_rows[0]][col_offset + 5] += 1
        V[Y_rows[1]][col_offset + 5] += 1
        W[Z_rows[3]][col_offset + 5] += 1
        
        # Term 6: (X1-X3)(Y2+Y3) -> Z0
        U[X_rows[1]][col_offset + 6] += 1
        U[X_rows[3]][col_offset + 6] -= 1
        V[Y_rows[2]][col_offset + 6] += 1
        V[Y_rows[3]][col_offset + 6] += 1
        W[Z_rows[0]][col_offset + 6] += 1

    def verify(U, V, W, rank):
        """Verify decomposition with random test matrices."""
        np.random.seed(42)
        A_test = np.random.randn(n, m)
        B_test = np.random.randn(m, p)
        C_expected = A_test @ B_test
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        U3 = U_arr.reshape(n, m, rank)
        V3 = V_arr.reshape(m, p, rank)
        W3 = W_arr.reshape(p, n, rank)
        
        C_recon = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for r in range(rank):
                    a_part = np.dot(U3[i, :, r], A_test[i, :])
                    b_part = np.dot(V3[:, j, r], B_test[:, j])
                    C_recon[i, j] += W3[j, i, r] * a_part * b_part
        
        if np.allclose(C_recon, C_expected, atol=1e-10):
            return True
        
        # Try alternative convention
        W3b = W_arr.reshape(n, p, rank)
        C_recon2 = np.zeros((n, p))
        for i in range(n):
            for j in range(p):
                for r in range(rank):
                    a_part = np.dot(U3[i, :, r], A_test[i, :])
                    b_part = np.dot(V3[:, j, r], B_test[:, j])
                    C_recon2[i, j] += W3b[i, j, r] * a_part * b_part
        
        return np.allclose(C_recon2, C_expected, atol=1e-10)

    # Build initial rank-28 decomposition using 4 Strassen blocks
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    X1 = [0, 1, 4, 5];  Y1 = [0, 1, 4, 5];   Z1 = [0, 2, 1, 3]
    X2 = [2, 3, 6, 7];  Y2 = [8, 9, 12, 13]; Z2 = [0, 2, 1, 3]
    X3 = [0, 1, 4, 5];  Y3 = [2, 3, 6, 7];   Z3 = [4, 6, 5, 7]
    X4 = [2, 3, 6, 7];  Y4 = [10, 11, 14, 15]; Z4 = [4, 6, 5, 7]
    
    add_block(0,  X1, Y1, Z1, U, V, W, rank)
    add_block(7,  X2, Y2, Z2, U, V, W, rank)
    add_block(14, X3, Y3, Z3, U, V, W, rank)
    add_block(21, X4, Y4, Z4, U, V, W, rank)
    
    # Verify initial decomposition
    if not verify(U, V, W, rank):
        return {"U": U, "V": V, "W": W}
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by merging/canceling terms across blocks
    # Strategy: look for pairs of terms that can be combined using linearity
    
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Try various reduction strategies
    for target_rank in range(27, 23, -1):
        if time.time() > deadline - 5:
            break
        
        # Strategy 1: Try to find linear dependencies among the rank-28 terms
        # that allow us to reduce the rank
        
        # Check if any two terms are linearly dependent (same U, V vectors up to scalar)
        merged = False
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                if time.time() > deadline - 5:
                    break
                
                # Check if U[:, r1] and U[:, r2] are proportional
                u1 = U_arr[:, r1]
                u2 = U_arr[:, r2]
                if np.allclose(u1, u2, atol=1e-10) or np.allclose(u1, -u2, atol=1e-10):
                    # Check V
                    v1 = V_arr[:, r1]
                    v2 = V_arr[:, r2]
                    if np.allclose(v1, v2, atol=1e-10) or np.allclose(v1, -v2, atol=1e-10):
                        # Check W
                        w1 = W_arr[:, r1]
                        w2 = W_arr[:, r2]
                        if np.allclose(w1, w2, atol=1e-10) or np.allclose(w1, -w2, atol=1e-10):
                            # These two terms are identical or negatives - can merge
                            merged = True
                            break
            
            if merged or time.time() > deadline - 5:
                break
        
        if merged:
            # Remove duplicate terms
            new_U = []
            new_V = []
            new_W = []
            seen = set()
            for r in range(rank):
                key = (tuple(U_arr[:, r]), tuple(V_arr[:, r]), tuple(W_arr[:, r]))
                if key not in seen:
                    seen.add(key)
                    new_U.append(U[r])
                    new_V.append(V[r])
                    new_W.append(W[r])
            
            new_rank = len(new_U)
            if new_rank < best_rank and verify(new_U, new_V, new_W, new_rank):
                best_rank = new_rank
                best_U, best_V, best_W = new_U, new_V, new_W
                rank = new_rank
                U, V, W = new_U, new_V, new_W
                U_arr = np.array(U, dtype=float)
                V_arr = np.array(V, dtype=float)
                W_arr = np.array(W, dtype=float)
    
    # If we found a better rank, return it
    if best_rank < 28:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": U, "V": V, "W": W}
