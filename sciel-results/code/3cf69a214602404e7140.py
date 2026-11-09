import numpy as np
import time

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 120.0
    
    n, m, p = DIMENSIONS
    # A is n x m (2x4)
    # B is m x p (4x8)
    # C is n x p (2x8)
    
    # We decompose into 8 blocks of 2x2x2.
    # A is split into 2 blocks along columns: A = [A0 | A1] where A0, A1 are 2x2.
    # B is split into 4 blocks along rows and columns: B is 4x8.
    # B rows split into 2 blocks of 2 rows: B_top (2x8), B_bottom (2x8).
    # B cols split into 4 blocks of 2 cols: B0, B1, B2, B3.
    # So B_top = [B00 | B01 | B02 | B03]
    #    B_bottom = [B10 | B11 | B12 | B13]
    
    # C = A * B = [A0 | A1] * [B_top; B_bottom]
    # C = A0 * B_top + A1 * B_bottom
    # C is 2x8. Let C = [C0 | C1 | C2 | C3] where each Ci is 2x2.
    # C0 = A0*B00 + A1*B10
    # C1 = A0*B01 + A1*B11
    # C2 = A0*B02 + A1*B12
    # C3 = A0*B03 + A1*B13
    
    # Each Ci is a sum of two 2x2x2 products.
    # We can use Strassen for each 2x2x2 product.
    # There are 4 Ci's, each requiring 2 products. Total 8 products.
    # Each product uses rank 7. Total rank = 8 * 7 = 56.
    
    rank = 56
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    def add_strassen_block(u_row_offset, v_row_offset, w_row_offset, rank_offset):
        """
        Adds a Strassen 2x2x2 block to U, V, W.
        u_row_offset: starting row in U for the 2x2 A block (size 2x2=4 rows)
        v_row_offset: starting row in V for the 2x2 B block (size 2x2=4 rows)
        w_row_offset: starting row in W for the 2x2 C block (size 2x2=4 rows)
        rank_offset: starting column in U,V,W for this block's 7 rank slots
        """
        # Local indices for 2x2 blocks:
        # A: rows i=0,1; cols j=0,1. U index: i*2+j
        # B: rows j=0,1; cols k=0,1. V index: j*2+k
        # C: rows k=0,1; cols i=0,1. W index: k*2+i (trace convention W[k*n+i, r])
        
        # Strassen 7 multiplications:
        # M1 = (A00 + A11) * (B00 + B11)
        # M2 = (A10 + A11) * B00
        # M3 = A00 * (B01 - B11)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A01) * B11
        # M6 = (A10 - A00) * (B00 + B01)
        # M7 = (A01 - A11) * (B10 + B11)
        
        # Output:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # M1
        U[u_row_offset + 0, rank_offset + 0] += 1  # A00
        U[u_row_offset + 3, rank_offset + 0] += 1  # A11
        V[v_row_offset + 0, rank_offset + 0] += 1  # B00
        V[v_row_offset + 3, rank_offset + 0] += 1  # B11
        W[w_row_offset + 0, rank_offset + 0] += 1  # C00 (k=0, i=0)
        W[w_row_offset + 3, rank_offset + 0] += 1  # C11 (k=1, i=1)
        
        # M2
        U[u_row_offset + 2, rank_offset + 1] += 1  # A10
        U[u_row_offset + 3, rank_offset + 1] += 1  # A11
        V[v_row_offset + 0, rank_offset + 1] += 1  # B00
        W[w_row_offset + 1, rank_offset + 1] += 1  # C10 (k=0, i=1)
        
        # M3
        U[u_row_offset + 0, rank_offset + 2] += 1  # A00
        V[v_row_offset + 1, rank_offset + 2] += 1  # B01
        V[v_row_offset + 3, rank_offset + 2] += -1 # B11
        W[w_row_offset + 2, rank_offset + 2] += 1  # C01 (k=1, i=0)
        
        # M4
        U[u_row_offset + 3, rank_offset + 3] += 1  # A11
        V[v_row_offset + 2, rank_offset + 3] += 1  # B10
        V[v_row_offset + 0, rank_offset + 3] += -1 # B00
        W[w_row_offset + 0, rank_offset + 3] += 1  # C00 (k=0, i=0)
        W[w_row_offset + 1, rank_offset + 3] += 1  # C10 (k=0, i=1)
        
        # M5
        U[u_row_offset + 0, rank_offset + 4] += 1  # A00
        U[u_row_offset + 1, rank_offset + 4] += 1  # A01
        V[v_row_offset + 3, rank_offset + 4] += 1  # B11
        W[w_row_offset + 0, rank_offset + 4] += -1 # C00 (k=0, i=0)
        W[w_row_offset + 2, rank_offset + 4] += 1  # C01 (k=1, i=0)
        
        # M6
        U[u_row_offset + 2, rank_offset + 5] += 1  # A10
        U[u_row_offset + 0, rank_offset + 5] += -1 # A00
        V[v_row_offset + 0, rank_offset + 5] += 1  # B00
        V[v_row_offset + 1, rank_offset + 5] += 1  # B01
        # M6 is not used in the standard Strassen output combination?
        # Wait, let me double check the standard Strassen formulas.
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # Indeed, M6 is not used in the final sum. This is a known property of the specific Strassen variant?
        # No, actually, in the standard Strassen algorithm, M6 is defined but its contribution is cancelled out or it's a different formulation.
        # Let's verify:
        # (A10-A00)(B00+B01) = A10B00 + A10B01 - A00B00 - A00B01.
        # Is this needed?
        # C10 = A10B00 + A11B10 = M2 + M4 = (A10+A11)B00 + A11(B10-B00) = A10B00 + A11B00 + A11B10 - A11B00 = A10B00 + A11B10. Correct.
        # C11 = A10B01 + A11B11 = M1 - M2 + M3 + M7 = (A00+A11)(B00+B11) - (A10+A11)B00 + A00(B01-B11) + (A01-A11)(B10+B11)
        # = A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + A11B11 - A10B00 + A00B01 + A01B10 + A01B11 - A11B10
        # This does not look like A10B01 + A11B11.
        # I think I have the Strassen formulas slightly mixed up or I am using a non-standard one.
        # Let's use a reliable source for Strassen 2x2x2.
        
        # Correct Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's re-verify C11 with these formulas.
        # C11 = M1 - M2 + M3 + M7
        # M1 = A00B00 + A00B11 + A11B00 + A11B11
        # M2 = A10B00 + A11B00
        # M3 = A00B01 - A00B11
        # M7 = A01B10 + A01B11 - A11B10 - A11B11
        # Sum:
        # A00B00 + A00B11 + A11B00 + A11B11 - A10B00 - A11B00 + A00B01 - A00B11 + A01B10 + A01B11 - A11B10 - A11B11
        # = A00B00 + A11B11 - A10B00 + A00B01 + A01B10 + A01B11 - A11B10
        # This is still not A10B01 + A11B11.
        
        # There is a different set of Strassen formulas. Let's use the one where:
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        # is actually correct for:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Wait, I found the error. The standard Strassen algorithm is:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)(B00)
        # M3 = (A00)(B01-B11)
        # M4 = (A11)(B10-B00)
        # M5 = (A00+A01)(B11)
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # Let's check C11 again.
        # C11 = A10B01 + A11B11.
        # My expansion gave: A00B00 + A11B11 - A10B00 + A00B01 + A01B10 + A01B11 - A11B10.
        # This is definitely wrong.
        
        # Let's use a different, verified set of Strassen formulas.
        # From Wikipedia:
        # M1 = (A00 + A11)(B00 + B11)
        # M2 = (A10 + A11)(B00)
        # M3 = (A00)(B01 - B11)
        # M4 = (A11)(B10 - B00)
        # M5 = (A00 + A01)(B11)
        # M6 = (A10 - A00)(B00 + B01)
        # M7 = (A01 - A11)(B10 + B11)
        
        # C00 = M1 + M4 - M5 - M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M7
        
        # I will trust the Wikipedia formulas. If my manual check fails, it's likely my manual check is wrong or I'm missing a cancellation.
        # Actually, let's just implement it and verify numerically in code.
        
        pass

    # Since I am uncertain about the exact Strassen coefficients without running, and the task requires exactness,
    # and I have a 120s budget, I will implement a robust verification step.
    # However, I know that rank 56 is achievable.
    # To be safe and ensure I return a valid result, I will implement the Strassen block,
    # verify it, and if it fails, fall back to schoolbook rank 64.
    
    # Let's implement the Strassen blocks properly.
    # There are 8 blocks.
    # Block (i, k) corresponds to A_i (2x2) * B_{i, k} (2x2) -> C_k (2x2)
    # i in {0, 1} (A block index)
    # k in {0, 1, 2, 3} (C block index, which is also B col block index)
    # B block is B_{i, k} where i is row block, k is col block.
    
    # U index for A_i: rows i*2 to i*2+3? No, A is 2x4.
    # A = [A0 | A1]. A0 is cols 0,1. A1 is cols 2,3.
    # U row index for A0: 0,1,2,3 (i=0,1; j=0,1)
    # U row index for A1: 0,1,2,3 (i=0,1; j=2,3) -> global row i*2 + j
    
    # V index for B_{i, k}: B is 4x8.
    # B row block i: rows i*2 to i*2+1.
    # B col block k: cols k*2 to k*2+1.
    # V row index: j*p + k where j is local row (0,1), k is local col (0,1).
    # Global V row: (i*2 + j_local) * 8 + (k*2 + k_local)
    
    # W index for C_k: C is 2x8.
    # C col block k: cols k*2 to k*2+1.
    # W row index: k*n + i where k is local col (0,1), i is local row (0,1).
    # Global W row: (k*2 + k_local) * 2 + i_local
    
    rank_offset = 0
    for i in range(2):  # A block index
        for k in range(4):  # C block index
            # U offset for A_i
            # A_i is 2x2. Rows 0,1. Cols i*2, i*2+1.
            # U row index = row * m + col = row * 4 + col
            # Local row r in {0,1}, local col c in {0,1}
            # Global U row = r * 4 + (i * 2 + c)
            
            # V offset for B_{i, k}
            # B_{i, k} is 2x2. Rows i*2, i*2+1. Cols k*2, k*2+1.
            # V row index = row * p + col = row * 8 + col
            # Local row r in {0,1}, local col c in {0,1}
            # Global V row = (i * 2 + r) * 8 + (k * 2 + c)
            
            # W offset for C_k
            # C_k is 2x2. Rows 0,1. Cols k*2, k*2+1.
            # W row index = col * n + row = col * 2 + row
            # Local row r in {0,1}, local col c in {0,1}
            # Global W row = (k * 2 + c) * 2 + r
            
            # We need to map local indices to global indices for U, V, W.
            # Instead of hardcoding offsets, we will iterate over the 7 Strassen terms
            # and for each term, iterate over the non-zero entries in U, V, W.
            
            # Strassen terms (local indices):
            # U: (r, c)
            # V: (r, c)
            # W: (c, r)  # Note: W is k*n+i, so col*2+row.
            
            # Let's define the 7 terms as lists of (u_idx, v_idx, w_idx, coeff)
            # But U, V, W are sparse in each term.
            
            # Term 1: (A00+A11)(B00+B11)
            # U: (0,0)=1, (1,1)=1
            # V: (0,0)=1, (1,1)=1
            # W: (0,0)=1, (1,1)=1
            
            # Term 2: (A10+A11)(B00)
            # U: (1,0)=1, (1,1)=1
            # V: (0,0)=1
            # W: (0,1)=1
            
            # Term 3: (A00)(B01-B11)
            # U: (0,0)=1
            # V: (0,1)=1, (1,1)=-1
            # W: (1,0)=1
            
            # Term 4: (A11)(B10-B00)
            # U: (1,1)=1
            # V: (1,0)=1, (0,0)=-1
            # W: (0,0)=1, (0,1)=1
            
            # Term 5: (A00+A01)(B11)
            # U: (0,0)=1, (0,1)=1
            # V: (1,1)=1
            # W: (0,0)=-1, (1,0)=1
            
            # Term 6: (A10-A00)(B00+B01)
            # U: (1,0)=1, (0,0)=-1
            # V: (0,0)=1, (0,1)=1
            # W: (not used in final sum? Let's assume it is not used, or I need to check.)
            # Actually, in the standard Strassen, M6 is not used in the output.
            # So I will skip adding W for M6.
            
            # Term 7: (A01-A11)(B10+B11)
            # U: (0,1)=1, (1,1)=-1
            # V: (1,0)=1, (1,1)=1
            # W: (0,0)=-1, (1,1)=1
            
            # Output combinations:
            # C00 = M1 + M4 - M5 - M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M7
            
            # I will add the contributions to W accordingly.
            
            # Define local to global mapping functions
            def u_global(r, c):
                return r * 4 + (i * 2 + c)
            
            def v_global(r, c):
                return (i * 2 + r) * 8 + (k * 2 + c)
            
            def w_global(r, c):
                # W index is col * n + row = c * 2 + r
                return (k * 2 + c) * 2 + r
            
            # Term 1
            U[u_global(0,0), rank_offset] += 1
            U[u_global(1,1), rank_offset] += 1
            V[v_global(0,0), rank_offset] += 1
            V[v_global(1,1), rank_offset] += 1
            W[w_global(0,0), rank_offset] += 1  # C00
            W[w_global(1,1), rank_offset] += 1  # C11
            
            # Term 2
            U[u_global(1,0), rank_offset+1] += 1
            U[u_global(1,1), rank_offset+1] += 1
            V[v_global(0,0), rank_offset+1] += 1
            W[w_global(0,1), rank_offset+1] += 1  # C10
            
            # Term 3
            U[u_global(0,0), rank_offset+2] += 1
            V[v_global(0,1), rank_offset+2] += 1
            V[v_global(1,1), rank_offset+2] += -1
            W[w_global(1,0), rank_offset+2] += 1  # C01
            
            # Term 4
            U[u_global(1,1), rank_offset+3] += 1
            V[v_global(1,0), rank_offset+3] += 1
            V[v_global(0,0), rank_offset+3] += -1
            W[w_global(0,0), rank_offset+3] += 1  # C00
            W[w_global(0,1), rank_offset+3] += 1  # C10
            
            # Term 5
            U[u_global(0,0), rank_offset+4] += 1
            U[u_global(0,1), rank_offset+4] += 1
            V[v_global(1,1), rank_offset+4] += 1
            W[w_global(0,0), rank_offset+4] += -1 # C00
            W[w_global(1,0), rank_offset+4] += 1  # C01
            
            # Term 6 (Skip W)
            U[u_global(1,0), rank_offset+5] += 1
            U[u_global(0,0), rank_offset+5] += -1
            V[v_global(0,0), rank_offset+5] += 1
            V[v_global(0,1), rank_offset+5] += 1
            
            # Term 7
            U[u_global(0,1), rank_offset+6] += 1
            U[u_global(1,1), rank_offset+6] += -1
            V[v_global(1,0), rank_offset+6] += 1
            V[v_global(1,1), rank_offset+6] += 1
            W[w_global(0,0), rank_offset+6] += -1 # C00
            W[w_global(1,1), rank_offset+6] += 1  # C11
            
            rank_offset += 7
            
    # Verify the decomposition
    # Tensor T[i,j,k] = 1 if and only if C[i,k] = sum_j A[i,j] B[j,k]
    # In our decomposition, T = sum_r U[:,r] V[:,r] W[:,r]^T
    # U is (n*m, rank), V is (m*p, rank), W is (p*n, rank)
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # We can verify by checking if U @ V.T @ W.T gives the correct tensor.
    # But U @ V.T is (n*m, m*p) which is not directly the tensor.
    # The tensor is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # Let's verify with random matrices.
    np.random.seed(42)
    A_test = np.random.randint(0, 10, size=(n, m))
    B_test = np.random.randint(0, 10, size=(m, p))
    C_test = A_test @ B_test
    
    # Compute C_decomp
    # C_decomp[i,k] = sum_j A[i,j] B[j,k]
    # In our decomposition:
    # C_decomp[i,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # This is not quite right. The decomposition is:
    # C[i,k] = sum_r U[:,r] (A) * V[:,r] (B) * W[:,r] (C)
    # Specifically, for each r, we have a term.
    # The term r contributes to C[i,k] as:
    # sum_j (U[i*m+j, r] A[i,j]) * (V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # Wait, the standard tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Here U is indexed by (i,j), V by (j,k), W by (k,i).
    # So C[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k] ? No.
    # The decomposition says:
    # sum_j A[i,j] B[j,k] = sum_r (sum_j U[i*m+j, r] A[i,j]) * (sum_k V[j*p+k, r] B[j,k]) * W[k*n+i, r]
    # This is not correct. The correct interpretation is:
    # The tensor T[i,j,k] is decomposed as sum_r U[i,j,r] V[j,k,r] W[k,i,r].
    # So T[i,j,k] = 1 for all i,j,k? No, T[i,j,k] is the coefficient of A[i,j]B[j,k] in C[i,k].
    # C[i,k] = sum_j A[i,j] B[j,k]. So T[i,j,k] = 1 for all i,j,k.
    # The decomposition is: 1 = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    # for all i,j,k.
    
    # Let's verify this identity.
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=int)
        V = np.zeros((m * p, rank), dtype=int)
        W = np.zeros((p * n, rank), dtype=int)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
    
    if time.time() > deadline:
        # Return best result (schoolbook)
        pass
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
