import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill

    # Index mapping:
    # A[i,j] where i in {0,1}, j in {0,1,2} -> U row index = i*3 + j
    #   A00=0, A01=1, A02=2, A10=3, A11=4, A12=5
    # B[j,k] where j in {0,1,2}, k in {0,1,2} -> V row index = j*3 + k
    #   B00=0, B01=1, B02=2, B10=3, B11=4, B12=5, B20=6, B21=7, B22=8
    # C output: W[k*n + i, r] where k is column of C, i is row of C
    #   C[0,0]=0, C[1,0]=1, C[0,1]=2, C[1,1]=3, C[0,2]=4, C[1,2]=5

    def verify(U_mat, V_mat, W_mat, n_val, m_val, p_val):
        """Vectorized verification using numpy einsum"""
        rank = U_mat.shape[1]
        U_r = U_mat.reshape(n_val, m_val, rank)
        V_r = V_mat.reshape(m_val, p_val, rank)
        W_r = W_mat.reshape(p_val, n_val, rank)
        T = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        # Target: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B contributes to C[i,k]
        # Actually for matrix mult: C[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] should be 1 for each (i,j,k) triple where A[i,j]*B[j,k] contributes to C[i,k]
        # The tensor T[i,j,k] = 1 means the product A[i,j]*B[j,k] appears in C[i,k]
        target = np.ones((n_val, m_val, p_val))
        return np.allclose(T, target, atol=1e-10)

    def make_target(n_val, m_val, p_val):
        """Create the target multiplication tensor"""
        T = np.zeros((n_val, m_val, p_val))
        for i in range(n_val):
            for j in range(m_val):
                for k in range(p_val):
                    T[i, j, k] = 1.0
        return T

    target = make_target(n, m, p)

    # Rank 16 construction: Strassen (7) + 9 additional terms
    R = 16
    U = np.zeros((n * m, R))  # 6 x 16
    V = np.zeros((m * p, R))  # 9 x 16
    W = np.zeros((p * n, R))  # 6 x 16

    # Part 1: Strassen for A[0:2,0:2] * B[0:2,0:2] -> C[0:2,0:2]
    # M1: (A00+A11)*(B00+B11) -> contributes to C00 and C11
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1

    # M2: (A10+A11)*B00 -> C10 +, C11 -
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1

    # M3: A00*(B01-B11) -> C01 +, C11 +
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1

    # M4: A11*(B10-B00) -> C00 +, C10 +
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1

    # M5: (A00+A01)*B11 -> C00 -, C01 +
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1

    # M6: (A10-A00)*(B00+B01) -> C11 +
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1

    # M7: (A01-A11)*(B10+B11) -> C00 +
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1

    # Part 2: Terms involving A02, A12 with B20, B21 (contributing to C00,C01,C10,C11)
    # M8: A02*B20 -> C00
    U[2, 7] = 1
    V[6, 7] = 1
    W[0, 7] = 1

    # M9: A02*B21 -> C01
    U[2, 8] = 1
    V[7, 8] = 1
    W[2, 8] = 1

    # M10: A12*B20 -> C10
    U[5, 9] = 1
    V[6, 9] = 1
    W[1, 9] = 1

    # M11: A12*B21 -> C11
    U[5, 10] = 1
    V[7, 10] = 1
    W[3, 10] = 1

    # Part 3: Terms contributing to C02 and C12
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22

    # M12: A00*B02 -> C02
    U[0, 11] = 1
    V[2, 11] = 1
    W[4, 11] = 1

    # M13: A01*B12 -> C02
    U[1, 12] = 1
    V[5, 12] = 1
    W[4, 12] = 1

    # M14: A02*B22 -> C02
    U[2, 13] = 1
    V[8, 13] = 1
    W[4, 13] = 1

    # M15: A10*B02 -> C12
    U[3, 14] = 1
    V[2, 14] = 1
    W[5, 14] = 1

    # M16: A11*B12 -> C12
    U[4, 15] = 1
    V[5, 15] = 1
    W[5, 15] = 1

    # Verify rank-16 solution
    if verify(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # If rank 16 doesn't verify, we're missing A12*B22 -> C12
    # Let's check what's wrong and try to fix by adding it back (rank 17)
    # or find a way to absorb it

    # Compute current tensor
    U_r = U.reshape(n, m, R)
    V_r = V.reshape(m, p, R)
    W_r = W.reshape(p, n, R)
    T_current = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    residual = target - T_current

    # The missing term is A12*B22 -> C12, i.e., residual[1,2,2] should be 1
    if not np.allclose(residual, 0.0, atol=1e-10):
        # Try to absorb the missing term into existing terms
        # The missing contribution is to C[1,2] from A[1,2]*B[2,2]
        # We need to add this without increasing rank
        
        # Strategy: Try perturbing existing terms to absorb the missing contribution
        # The missing term has U[5,r]=1, V[8,r]=1, W[5,r]=1 for some new r
        
        # Try modifying term 16 (index 15) which already contributes to C12
        # Term 15: U[4,15]=1, V[5,15]=1, W[5,15]=1 (A11*B12 -> C12)
        # We could try to make this term also capture A12*B22
        
        # Alternative: Try all pairs of terms and see if we can modify one
        # to cover the missing A12*B22 -> C12 contribution
        
        # Let's try a more systematic approach: for each term r, try adding
        # small perturbations to its U, V, W entries to cover the residual
        
        best_solution = None
        
        for r in range(R):
            if time.time() > deadline - 10:
                break
            
            # Try perturbing U entries of term r
            for idx_u in range(n * m):
                if U[idx_u, r] == 0 and time.time() > deadline - 5:
                    continue
                for delta in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                    U_try = U.copy()
                    U_try[idx_u, r] += delta
                    U_try_r = U_try.reshape(n, m, R)
                    T_try = np.einsum('ijr,jkr,kir->ijk', U_try_r, V_r, W_r)
                    if np.allclose(T_try, target, atol=1e-10):
                        best_solution = (U_try.tolist(), V.tolist(), W.tolist())
                        break
                if best_solution:
                    break
            if best_solution:
                break
            
            # Try perturbing V entries of term r
            if not best_solution:
                for idx_v in range(m * p):
                    if V[idx_v, r] == 0 and time.time() > deadline - 5:
                        continue
                    for delta in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                        V_try = V.copy()
                        V_try[idx_v, r] += delta
                        V_try_r = V_try.reshape(m, p, R)
                        T_try = np.einsum('ijr,jkr,kir->ijk', U_r, V_try_r, W_r)
                        if np.allclose(T_try, target, atol=1e-10):
                            best_solution = (U.tolist(), V_try.tolist(), W.tolist())
                            break
                    if best_solution:
                        break
                if best_solution:
                    break
            
            # Try perturbing W entries of term r
            if not best_solution:
                for idx_w in range(p * n):
                    if W[idx_w, r] == 0 and time.time() > deadline - 5:
                        continue
                    for delta in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
                        W_try = W.copy()
                        W_try[idx_w, r] += delta
                        W_try_r = W_try.reshape(p, n, R)
                        T_try = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_try_r)
                        if np.allclose(T_try, target, atol=1e-10):
                            best_solution = (U.tolist(), V.tolist(), W_try.tolist())
                            break
                    if best_solution:
                        break
                if best_solution:
                    break
        
        if best_solution:
            return {"U": best_solution[0], "V": best_solution[1], "W": best_solution[2]}
        
        # If we can't fix rank 16, fall back to rank 17 with the missing term
        R17 = 17
        U17 = np.zeros((n * m, R17))
        V17 = np.zeros((m * p, R17))
        W17 = np.zeros((p * n, R17))
        
        # Copy rank 16 terms
        U17[:, :R] = U
        V17[:, :R] = V
        W17[:, :R] = W
        
        # Add missing term: A12*B22 -> C12
        U17[5, 16] = 1
        V17[8, 16] = 1
        W17[5, 16] = 1
        
        if verify(U17, V17, W17, n, m, p):
            return {"U": U17.tolist(), "V": V17.tolist(), "W": W17.tolist()}
    
    # Fallback: full schoolbook rank 18
    R18 = 18
    U18 = np.zeros((n * m, R18))
    V18 = np.zeros((m * p, R18))
    W18 = np.zeros((p * n, R18))
    
    r = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                ia = i * m + j
                ib = j * p + k
                out_idx = k * n + i
                U18[ia, r] = 1
                V18[ib, r] = 1
                W18[out_idx, r] = 1
                r += 1
    
    return {"U": U18.tolist(), "V": V18.tolist(), "W": W18.tolist()}
# EVOLVE-BLOCK-END
