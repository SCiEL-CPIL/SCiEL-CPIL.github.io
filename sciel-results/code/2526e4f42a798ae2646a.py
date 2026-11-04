import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    # Convention: 
    # U2 rows: (i, j) -> i*2 + j
    # V2 rows: (j, k) -> j*2 + k
    # W2 rows: (k, i) -> k*2 + i
    
    # Standard Strassen factors (one common set)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map these to U, V, W matrices.
    # U[i,j] contributes to A_ij
    # V[j,k] contributes to B_jk
    # W[k,i] contributes to C_ik (note: task says W[k*n+i,r])
    
    # Let's construct U2, V2, W2 explicitly for the 7 terms.
    
    # Term 1: (A11+A22)(B11+B22)
    # A part: A11, A22 -> U2[0,0]=1, U2[3,0]=1 (if 0-based: 0,0 and 1,1)
    # B part: B11, B22 -> V2[0,0]=1, V2[3,0]=1
    # C part: C11, C22 -> W2[0,0]=1, W2[3,0]=1 (C11 is 0,0; C22 is 1,1 -> idx 0 and 3)
    
    # Let's use 0-based indexing for 2x2:
    # i,j,k in {0,1}
    # U2 row idx: i*2+j
    # V2 row idx: j*2+k
    # W2 row idx: k*2+i
    
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11) -> C00 += 1, C11 += 1
    # A: A00 (idx 0), A11 (idx 3)
    U2[0, 0] = 1; U2[3, 0] = 1
    # B: B00 (idx 0), B11 (idx 3)
    V2[0, 0] = 1; V2[3, 0] = 1
    # C: C00 (idx 0), C11 (idx 3)
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2: (A10+A11)B00 -> C10 += 1
    # A: A10 (idx 2), A11 (idx 3)
    U2[2, 1] = 1; U2[3, 1] = 1
    # B: B00 (idx 0)
    V2[0, 1] = 1
    # C: C10 (idx 2) -> k=1, i=0 -> 1*2+0 = 2
    W2[2, 1] = 1
    
    # M3: A00(B01-B11) -> C01 += 1
    # A: A00 (idx 0)
    U2[0, 2] = 1
    # B: B01 (idx 1), B11 (idx 3)
    V2[1, 2] = 1; V2[3, 2] = -1
    # C: C01 (idx 1) -> k=0, i=1 -> 0*2+1 = 1
    W2[1, 2] = 1
    
    # M4: A11(B10-B00) -> C11 += 1, C10 += 1 ? No.
    # M4 = A11(B10-B00)
    # C11 = ... + M4 ...
    # C10 = ... + M4 ...
    # Wait, standard Strassen:
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    # My indices:
    # C00 (top-left) = C11 in 1-based?
    # Let's stick to 0-based consistently.
    # C00 (i=0, k=0)
    # C01 (i=0, k=1)
    # C10 (i=1, k=0)
    # C11 (i=1, k=1)
    
    # Recalculate M4 contribution:
    # M4 = A11(B10-B00)
    # A: A11 (idx 3)
    U2[3, 3] = 1
    # B: B10 (idx 2), B00 (idx 0)
    V2[2, 3] = 1; V2[0, 3] = -1
    # C:
    # C11 gets +M4. C11 is i=1, k=1 -> idx 3.
    # C10 gets +M4. C10 is i=1, k=0 -> idx 2.
    # Wait, does M4 contribute to both?
    # Let's check the formula again.
    # C11 = M1 + M4 - M5 + M7
    # C21 = M2 + M4
    # So yes, M4 contributes to C11 and C21 (which is C10 in 0-based).
    W2[3, 3] = 1
    W2[2, 3] = 1
    
    # M5: (A00+A01)B11 -> C01 += 1, C00 -= 1
    # A: A00 (idx 0), A01 (idx 1)
    U2[0, 4] = 1; U2[1, 4] = 1
    # B: B11 (idx 3)
    V2[3, 4] = 1
    # C:
    # C01 gets +M5. C01 is i=0, k=1 -> idx 1.
    # C00 gets -M5. C00 is i=0, k=0 -> idx 0.
    W2[1, 4] = 1
    W2[0, 4] = -1
    
    # M6: (A10-A00)(B00+B01) -> C11 += 1
    # A: A10 (idx 2), A00 (idx 0)
    U2[2, 5] = 1; U2[0, 5] = -1
    # B: B00 (idx 0), B01 (idx 1)
    V2[0, 5] = 1; V2[1, 5] = 1
    # C:
    # C11 gets +M6. C11 is i=1, k=1 -> idx 3.
    W2[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11) -> C00 += 1
    # A: A01 (idx 1), A11 (idx 3)
    U2[1, 6] = 1; U2[3, 6] = -1
    # B: B10 (idx 2), B11 (idx 3)
    V2[2, 6] = 1; V2[3, 6] = 1
    # C:
    # C00 gets +M7. C00 is i=0, k=0 -> idx 0.
    W2[0, 6] = 1

    # Verify 2x2 decomposition
    # C_ik = sum_j A_ij B_jk = sum_r U[i,j] V[j,k] W[k,i] ?
    # No, the decomposition is:
    # T_{ijk} = sum_r U_{ij}^r V_{jk}^r W_{ki}^r
    # where T_{ijk} is 1 if i=k? No.
    # Matrix mult tensor: T_{i,j,k} is coefficient of A_{ij} B_{jk} in C_{ik}.
    # So T_{i,j,k} = 1 if we are computing C_{ik} from A_{ij} B_{jk}.
    # Actually, the identity is:
    # C_{ik} = sum_{j,r} A_{ij} B_{jk} W_{kr} ? No.
    # The standard form is:
    # sum_r U_{ij}^r V_{jk}^r W_{ki}^r = delta_{?}
    # Let's check the definition in the prompt.
    # "exactly decompose the matrix multiplication tensor"
    # The tensor M has indices (i, j, k) and value 1 if ... ?
    # Usually, the multiplication tensor is defined such that:
    # <M, A \otimes B \otimes C> = trace(C^T A B)
    # This implies M_{i,j,k} = 1 if k=i? No.
    # Let's look at the verification loop in the original code:
    # val = np.dot(ui, vj * wk)
    # ui = U[i*4+j, :]
    # vj = V[j*4+k, :]
    # wk = W[k*4+i, :]
    # This computes sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # For the decomposition to be exact, this sum must be 1 for all i,j,k?
    # No, that would mean A B = I?
    # Wait. The tensor decomposition of matrix multiplication means:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # The bilinear form is:
    # sum_{i,j,k} C_{ik} A_{ij} B_{jk} = sum_{i,j,k} (sum_r U_{ij}^r V_{jk}^r W_{ki}^r) A_{ij} B_{jk}
    # This must hold for all A, B.
    # This implies that for each r, the term is U_{ij}^r V_{jk}^r W_{ki}^r.
    # The sum over r must reproduce the product.
    # Specifically, for a fixed i,j,k, the coefficient of A_{ij} B_{jk} in the expression for C_{ik} is 1.
    # But the decomposition sums over r.
    # So, sum_r U_{ij}^r V_{jk}^r W_{ki}^r must be equal to 1 for ALL i,j,k?
    # Let's test this with schoolbook.
    # Schoolbook: rank n^3.
    # For each (i,j,k), have a term.
    # U[i,j, i*n^2+j*n+k] = 1
    # V[j,k, i*n^2+j*n+k] = 1
    # W[k,i, i*n^2+j*n+k] = 1
    # Sum_r ...
    # For a specific i,j,k, only the term with r corresponding to (i,j,k) is non-zero.
    # Sum = 1*1*1 = 1.
    # For other i',j',k', the sum is 0.
    # So YES, the condition is:
    # sum_r U[i,j,r] V[j,k,r] W[k,i,r] = 1 for all i,j,k.
    
    # Let's verify my U2, V2, W2 satisfy this.
    # I will do this check in the code.

    # Construct 4x4 factors via Kronecker product
    U_kron = np.kron(U2, U2)
    V_kron = np.kron(V2, V2)
    W_kron = np.kron(W2, W2)
    
    # Permute rows to standard ordering
    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)
    
    for S in range(16):
        # U: (i, j)
        i = S // 4
        j = S % 4
        i_h, i_l = i // 2, i % 2
        j_h, j_l = j // 2, j % 2
        # Kron index: (i_h*2 + j_h)*4 + (i_l*2 + j_l)
        kron_idx_u = (i_h * 2 + j_h) * 4 + (i_l * 2 + j_l)
        U[S, :] = U_kron[kron_idx_u, :]
        
        # V: (j, k)
        j = S // 4
        k = S % 4
        j_h, j_l = j // 2, j % 2
        k_h, k_l = k // 2, k % 2
        # Kron index: (j_h*2 + k_h)*4 + (j_l*2 + k_l)
        kron_idx_v = (j_h * 2 + k_h) * 4 + (j_l * 2 + k_l)
        V[S, :] = V_kron[kron_idx_v, :]
        
        # W: (k, i)
        k = S // 4
        i = S % 4
        k_h, k_l = k // 2, k % 2
        i_h, i_l = i // 2, i % 2
        # Kron index: (k_h*2 + i_h)*4 + (k_l*2 + i_l)
        kron_idx_w = (k_h * 2 + i_h) * 4 + (k_l * 2 + i_l)
        W[S, :] = W_kron[kron_idx_w, :]

    # Verify
    if time.time() - t0 > deadline - 10.0:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    max_err = 0.0
    for i in range(4):
        for j in range(4):
            for k in range(4):
                ui = U[i * 4 + j, :]
                vj = V[j * 4 + k, :]
                wk = W[k * 4 + i, :]
                val = np.dot(ui, vj * wk)
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err

    if max_err > 1e-9:
        # Fallback: schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=float)
        V_fb = np.zeros((16, rank_fb), dtype=float)
        W_fb = np.zeros((16, rank_fb), dtype=float)
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i * 4 + j, col] = 1.0
                    V_fb[j * 4 + k, col] = 1.0
                    W_fb[k * 4 + i, col] = 1.0
                    col += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
