"""Matrix multiplication 2,5,6 rank-minimal tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    n, m, p = DIMENSIONS  # 2, 5, 6
    
    # Target Rank 54 based on Strassen decomposition of 2x2 blocks
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_base, j_base, k_base):
        """
        Adds a rank-7 Strassen decomposition for a 2x2 * 2x2 block.
        i_base: base index for i (rows of A, rows of C)
        j_base: base index for j (cols of A, rows of B)
        k_base: base index for k (cols of B, cols of C)
        """
        nonlocal col
        
        # M0: (A00+A11)(B00+B11)
        # U indices: (i_base, j_base), (i_base+1, j_base+1)
        # V indices: (j_base, k_base), (j_base+1, k_base+1)
        # W indices: (k_base, i_base), (k_base+1, i_base+1)
        U[i_base * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M1: (A10+A11)B00
        # U indices: (i_base+1, j_base), (i_base+1, j_base+1)
        # V indices: (j_base, k_base)
        # W indices: (k_base, i_base+1), (k_base+1, i_base+1) [Note: W11 gets -1 in standard Strassen? 
        # Let's check the standard Strassen formulas for C = AB.
        # C00 = M0 - M1 + M3 + M4
        # C01 = M1 + M2
        # C10 = M3 + M4
        # C11 = M0 - M2 + M1 + M5 ? No.
        
        # Standard Strassen:
        # M1 = (A10 + A11) B00
        # M2 = A00 (B01 - B11)
        # M3 = A11 (B10 - B00)
        # M4 = (A00 + A01) B11
        # M5 = (A10 - A00) (B00 + B01)
        # M6 = (A01 - A11) (B10 + B11)
        
        # C00 = M0 - M1 + M3 + M4
        # C01 = M1 + M2
        # C10 = M3 + M4
        # C11 = M0 - M2 + M1 + M5 ? 
        # Wait, C11 = M0 - M2 + M1 + M5 is not standard.
        # C11 = M0 - M2 + M1 + M5? 
        # Let's re-verify C11.
        # C11 = A10 B01 + A11 B11.
        # M0 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = A00(B01-B11) = A00B01 - A00B11
        # M1 = (A10+A11)B00 = A10B00 + A11B00
        # M5 = (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01
        
        # M0 - M2 + M1 + M5 
        # = (A00B00 + A00B11 + A11B00 + A11B11) - (A00B01 - A00B11) + (A10B00 + A11B00) + (A10B00 + A10B01 - A00B00 - A00B01)
        # A00B00: 1 - 1 = 0
        # A00B11: 1 + 1 = 2 ? No, M2 has -A00B11, so -(-A00B11) = +A00B11. So 1+1=2. 
        # This is getting confusing. Let's stick to the tensor decomposition definition.
        # The tensor T[a,b,c] = delta_{a+k, b} is not quite right for Strassen.
        # Strassen decomposes the multiplication tensor T_{2,2,2}.
        # The entries of U, V, W must satisfy sum_r U[i,j,r] V[j,k,r] W[k,l,r] = delta_{j,k} ... no.
        # The condition is that for any A, B, C = AB.
        # C[i,l] = sum_j A[i,j] B[j,l].
        # In tensor form: T[(i,j),(j,k),(k,i)] ... wait.
        # The tensor is defined such that <T, A \otimes B \otimes W> = <C, W>.
        # Actually, the decomposition U, V, W satisfies:
        # T[i,j,k] = sum_r U[i,r] V[j,r] W[k,r] ? No.
        # The indices are:
        # U is (n*m, rank). Index (i*m+j).
        # V is (m*p, rank). Index (j*p+k).
        # W is (p*n, rank). Index (k*n+i).
        # The tensor T is 3-way with dimensions (n*m, m*p, p*n).
        # T[(i*m+j), (j*p+k), (k*n+i)] = 1 if i,j,k are valid, else 0.
        # Wait, the tensor is sparse.
        # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k.
        # So T[(i*m+j), (j*p+k), (k*n+i)] = 1.
        
        # The decomposition requires:
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r].
        
        # For a specific block of i,j,k, we are decomposing the sub-tensor.
        # The Strassen algorithm provides a rank-7 decomposition of the 2x2x2 multiplication tensor.
        # The formulas in the inspiration code are standard Strassen coefficients.
        # I will trust the inspiration code's coefficients as they were verified to score 1.1111 (rank 54).
        
        # M1: (A10+A11)B00
        U[(i_base + 1) * m + j_base][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] -= 1
        col += 1
        
        # M2: A00(B01-B11)
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M3: A11(B10-B00)
        U[(i_base + 1) * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[j_base * p + k_base][col] -= 1
        W[k_base * n + i_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # M4: (A00+A01)B11
        U[i_base * m + j_base][col] += 1
        U[i_base * m + (j_base + 1)][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] -= 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # M5: (A10-A00)(B00+B01)
        U[(i_base + 1) * m + j_base][col] += 1
        U[i_base * m + j_base][col] -= 1
        V[j_base * p + k_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1
        
        # M6: (A01-A11)(B10+B11)
        U[i_base * m + (j_base + 1)][col] += 1
        U[(i_base + 1) * m + (j_base + 1)][col] -= 1
        V[(j_base + 1) * p + k_base][col] += 1
        V[(j_base + 1) * p + (k_base + 1)][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1

    def add_naive_2112_block(i_base, j_base, k_base):
        """
        Adds a rank-4 naive decomposition for a 2x1 * 1x2 block.
        i_base: base index for i
        j_base: single index for j
        k_base: base index for k
        """
        nonlocal col
        
        # 1: A00 B00 -> C00
        U[i_base * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + i_base][col] += 1
        col += 1
        
        # 2: A00 B01 -> C01
        U[i_base * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + i_base][col] += 1
        col += 1
        
        # 3: A10 B00 -> C10
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + k_base][col] += 1
        W[k_base * n + (i_base + 1)][col] += 1
        col += 1
        
        # 4: A10 B01 -> C11
        U[(i_base + 1) * m + j_base][col] += 1
        V[j_base * p + (k_base + 1)][col] += 1
        W[(k_base + 1) * n + (i_base + 1)][col] += 1
        col += 1

    # Apply Strassen blocks for j pairs (0,1) and (2,3)
    # j_base = 0
    add_strassen_block(0, 0, 0)
    add_strassen_block(0, 0, 2)
    add_strassen_block(0, 0, 4)
    
    # j_base = 2
    add_strassen_block(0, 2, 0)
    add_strassen_block(0, 2, 2)
    add_strassen_block(0, 2, 4)
    
    # Apply naive blocks for j=4
    add_naive_2112_block(0, 4, 0)
    add_naive_2112_block(0, 4, 2)
    add_naive_2112_block(0, 4, 4)
    
    # Verify col == 54
    if col != 54:
        # Fallback to schoolbook if logic error
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Optional: Verify the decomposition
    # Convert to numpy for quick check
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
                
    T_dec = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    if not np.allclose(T_dec, T, atol=1e-10, rtol=1e-10):
        # If verification fails, return schoolbook
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
