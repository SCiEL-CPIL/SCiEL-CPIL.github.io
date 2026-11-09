import time
import sys

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 3, 4, 6
    # Total rank estimate: 6 blocks of 2x2x2 (rank 7 each) + 6 blocks of 1x2x2 (rank 4 each)
    # 6*7 + 6*4 = 42 + 24 = 66
    rank = 66
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term to U, V, W
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        U[u_idx][col] += u_val
        V[v_idx][col] += v_val
        W[w_idx][col] += w_val
        col += 1

    # Strassen decomposition for 2x2x2
    # Local indices: i in 0,1; j in 0,1; k in 0,1
    # Global indices: i_g = i_off + i; j_g = j_off + j; k_g = k_off + k
    # U index: i_g * m + j_g
    # V index: j_g * p + k_g
    # W index: k_g * n + i_g
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11; V: B00, B11; W: C00, C11
        # This is a product of sums, so it's not a single rank-1 term in the original basis?
        # No, Strassen works by defining new linear combinations.
        # The rank-1 terms correspond to the products of the linear combinations.
        # Term 1: U has 1 at (i_off, j_off) and (i_off+1, j_off+1).
        #         V has 1 at (j_off, k_off) and (j_off+1, k_off+1).
        #         W has 1 at (k_off, i_off) and (k_off+1, i_off+1) with signs for reconstruction.
        # Wait, the W matrix entries determine how the product M contributes to C.
        # C_ik = sum_t U_it V_tk W_ki? No, C_ik = sum_t U_{i,t} V_{t,k} W_{k,i}?
        # Standard convention: C = U V W^T? Or C_ik = sum_t U_{i,t} V_{t,k} W_{k,i}?
        # The task says: W[k*n+i, r].
        # Tensor T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r].
        
        # So for a term r:
        # U[i1, j1] = 1, U[i2, j2] = 1
        # V[j1, k1] = 1, V[j2, k2] = 1
        # W[k1, i1] = c1, W[k2, i2] = c2
        # This contributes c1 * A[i1,j1]B[j1,k1] * C[k1,i1]? No.
        # The term is A[i1,j1]B[j1,k1] * (coefficient for C[k1,i1])?
        # Actually, the rank-1 decomposition is:
        # T = sum_r (u_r \otimes v_r \otimes w_r)
        # where u_r is a vector in A-space, v_r in B-space, w_r in C-space.
        # u_r[i,j] is U[i*m+j, r].
        # v_r[j,k] is V[j*p+k, r].
        # w_r[k,i] is W[k*n+i, r].
        
        # For Strassen M1 = (A00+A11)(B00+B11):
        # u has 1 at (0,0) and (1,1).
        # v has 1 at (0,0) and (1,1).
        # The product u \otimes v has non-zero entries at (0,0,0,0), (0,0,1,1), (1,1,0,0), (1,1,1,1).
        # This corresponds to C00, C01, C10, C11 contributions.
        # Specifically:
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # We need to assign coefficients to W such that the sum of all 7 terms gives the correct C.
        
        # Strassen Reconstruction:
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We will add 7 columns.
        
        # Term 1: M1
        # u: (0,0)=1, (1,1)=1
        # v: (0,0)=1, (1,1)=1
        # w: (0,0)=1, (1,1)=1  (Contributes to C00 and C11)
        add_term(i_off*m + j_off, j_off*p + k_off, k_off*n + i_off, 1, 1, 1)
        add_term((i_off+1)*m + (j_off+1), (j_off+1)*p + (k_off+1), (k_off+1)*n + (i_off+1), 1, 1, 1)
        # Wait, add_term adds to the same column.
        # My add_term function above increments col. I need a way to add multiple entries to the SAME column.
        
        # Let's rewrite the logic to build the column vector by vector.
        
    # Reset and use a different approach: build lists of non-zeros for each column.
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_block(i_off, j_off, k_off):
        nonlocal col
        # M1: (A00+A11)(B00+B11)
        # U: (i0,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1, (j1,k1)=1
        # W: (k0,i0)=1, (k1,i1)=1
        U[i_off*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        col += 1
        
        # M2: (A10+A11)B00
        # U: (i1,j0)=1, (i1,j1)=1
        # V: (j0,k0)=1
        # W: (k0,i1)=1
        U[(i_off+1)*m + j_off][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # M3: A00(B10-B11)
        # U: (i0,j0)=1
        # V: (j1,k0)=1, (j1,k1)=-1
        # W: (k0,i0)=1, (k1,i0)=1
        U[i_off*m + j_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = -1
        W[k_off*n + i_off][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M4: A11(B01-B00)
        # U: (i1,j1)=1
        # V: (j0,k1)=1, (j0,k0)=-1
        # W: (k1,i1)=1, (k0,i1)=1
        U[(i_off+1)*m + (j_off+1)][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        V[j_off*p + k_off][col] = -1
        W[(k_off+1)*n + (i_off+1)][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # M5: (A00+A01)B11
        # U: (i0,j0)=1, (i0,j1)=1
        # V: (j1,k1)=1
        # W: (k1,i0)=1
        U[i_off*m + j_off][col] = 1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        
        # M6: (A10-A00)(B00+B10)
        # U: (i1,j0)=1, (i0,j0)=-1
        # V: (j0,k0)=1, (j1,k0)=1
        # W: (k0,i1)=1
        U[(i_off+1)*m + j_off][col] = 1
        U[i_off*m + j_off][col] = -1
        V[j_off*p + k_off][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + (i_off+1)][col] = 1
        col += 1
        
        # M7: (A01-A11)(B11+B01)
        # U: (i0,j1)=1, (i1,j1)=-1
        # V: (j1,k1)=1, (j0,k1)=1
        # W: (k1,i0)=1
        U[i_off*m + (j_off+1)][col] = 1
        U[(i_off+1)*m + (j_off+1)][col] = -1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1

    def add_schoolbook_1x2x2_block(i_off, j_off, k_off):
        # i is single, j is 2, k is 2
        nonlocal col
        # C_ik = sum_j A_ij B_jk
        # 4 terms:
        # j=0, k=0
        U[i_off*m + j_off][col] = 1
        V[j_off*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        # j=0, k=1
        U[i_off*m + j_off][col] = 1
        V[j_off*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1
        # j=1, k=0
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + k_off][col] = 1
        W[k_off*n + i_off][col] = 1
        col += 1
        # j=1, k=1
        U[i_off*m + (j_off+1)][col] = 1
        V[(j_off+1)*p + (k_off+1)][col] = 1
        W[(k_off+1)*n + i_off][col] = 1
        col += 1

    # Apply blocks
    # i_off in {0, 2}
    # j_off in {0, 2}
    # k_off in {0, 2, 4}
    
    for i_off in [0, 2]:
        for j_off in [0, 2]:
            for k_off in [0, 2, 4]:
                if i_off == 0:
                    # 2x2x2 block
                    add_strassen_block(i_off, j_off, k_off)
                else:
                    # 1x2x2 block
                    add_schoolbook_1x2x2_block(i_off, j_off, k_off)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
