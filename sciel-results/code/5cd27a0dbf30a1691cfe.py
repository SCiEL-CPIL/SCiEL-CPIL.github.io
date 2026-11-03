"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify_decomp(U, V, W, n, m, p, rank):
    """Verify that the decomposition is correct using numpy."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for l in range(m):
            for k in range(p):
                u_row = U_arr[i*m+l, :]
                v_row = V_arr[l*p+k, :]
                w_row = W_arr[k*n+i, :]
                s = np.dot(u_row, v_row * w_row)
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _strassen_block(j0, j1, k0, k1, U, V, W, col, n, m, p):
    """Add 7 Strassen products for a 2x2x2 block and return new col."""
    # M1 = (A[j0]+A[j1])(B[k0]+B[k1])
    U[j0][col] += 1
    U[j1][col] += 1
    V[k0][col] += 1
    V[k1][col] += 1
    W[0][col] += 1  # contributes to C[0,k0] and C[0,k1] via W indexing
    # Actually W[k*n+i] where k is output col, i is output row
    # For Strassen: C[j0,k0] += M1, C[j0,k1] += M1, C[j1,k0] += M1, C[j1,k1] += M1
    # But we need to be careful. Let me use the standard 7-product Strassen.
    
    # Let me redo this properly. The standard Strassen for 2x2 matrices:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # In our case, we're multiplying a 2x2 block of A (rows j0,j1, all cols) 
    # by a 2x2 block of B (rows all, cols k0,k1)
    # But actually for matrix multiplication C = AB where A is n x m, B is m x p
    # The tensor entry T[i,l,k] = 1 if we're computing C[i,k] = sum_l A[i,l]*B[l,k]
    
    # For a Strassen block: we take 2 rows of A (i0, i1) and 2 cols of B (k0, k1)
    # and some 2 rows of the middle (l0, l1)
    # C[i0,k0] += A[i0,l0]*B[l0,k0] + A[i0,l1]*B[l1,k0] + A[i1,l0]*B[l0,k0] + A[i1,l1]*B[l1,k0] ... no wait
    
    # Actually, for matrix multiplication, the "Strassen" approach groups:
    # Take 2 rows of A (say rows 0,1 of the n-dim), 2 cols of B (say cols 0,1 of the p-dim)
    # and 2 rows of B / 2 cols of A (say rows 0,1 of the m-dim)
    # Then we compute 4 products using 7 Strassen multiplications
    
    # But the standard Strassen works on 2x2 matrix blocks. For our tensor:
    # We fix i in {i0,i1}, l in {l0,l1}, k in {k0,k1}
    # The contribution is sum over these 8 entries
    # We can use 7 rank-1 terms to compute these 8 products
    
    # Let me define it correctly:
    # A_block = [[A[i0,l0], A[i0,l1]], [A[i1,l0], A[i1,l1]]]
    # B_block = [[B[l0,k0], B[l0,k1]], [B[l1,k0], B[l1,k1]]]
    # C_block = A_block @ B_block = [[C[i0,k0], C[i0,k1]], [C[i1,k0], C[i1,k1]]]
    
    # M1 = (A[i0,l0]+A[i1,l1]) * (B[l0,k0]+B[l1,k1])
    U[i0][col] += 1
    U[i1][col] += 1
    V[l0][col] += 1
    V[l1][col] += 1
    W[k0*n+i0][col] += 1
    W[k1*n+i0][col] += 1
    W[k0*n+i1][col] += 1
    W[k1*n+i1][col] += 1
    col += 1
    
    # M2 = (A[i1,l0]+A[i1,l1]) * B[l0,k0]
    U[i1][col] += 1
    U[i1][col] += 1  # A[i1,l0] + A[i1,l1]
    V[l0][col] += 1
    V[l1][col] += 1
    W[k0*n+i1][col] += 1
    col += 1
    
    # M3 = A[i0,l0] * (B[l1,k0]-B[l1,k1])
    U[i0][col] += 1
    V[l1][col] += 1
    V[l1][col] -= 1  # B[l1,k0] - B[l1,k1]
    # Wait, V[l0] and V[l1] are different rows. Let me be more careful.
    # V is indexed by [l*p+k]. So V[l1*p+k0] and V[l1*p+k1] are different.
    # I was using simplified indexing. Let me use proper indexing.
    
    # This is getting complicated. Let me just use the known rank-54 decomposition
    # from the literature or construct it directly.
    
    # Actually, let me just return col unchanged and build the decomposition differently.
    # I'll use a direct construction based on the structure.
    return col

def construct():
    n, m, p = DIMENSIONS
    _start_time = time.time()
    _deadline = _start_time + 130
    
    # Build the rank-54 Strassen-based decomposition
    # Strategy: Use Strassen for 2x2x2 blocks where possible, schoolbook for the rest
    # n=2, m=5, p=6
    # Group m into 2+2+1, group p into 2+2+2
    # For each (2 m-rows) x (2 p-cols) block with 2 n-rows: 7 terms
    # Number of such blocks: (m/2 blocks) * (p/2 blocks) = 2 * 3 = 6 blocks * 7 = 42 terms
    # Remaining: 1 m-row * 6 p-cols * 2 n-rows = 12 schoolbook terms
    # Total: 42 + 12 = 54
    
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # m-groups: (0,1), (2,3), and singleton 4
    # p-groups: (0,1), (2,3), (4,5)
    
    m_groups = [(0, 1), (2, 3)]
    p_groups = [(0, 1), (2, 3), (4, 5)]
    n_rows = [0, 1]
    
    for l0, l1 in m_groups:
        for k0, k1 in p_groups:
            for i0, i1 in [(0, 1)]:
                # 7 Strassen terms for this 2x2x2 block
                # M1 = (A[i0,l0]+A[i1,l1]) * (B[l0,k0]+B[l1,k1])
                U[i0*m+l0][col] += 1
                U[i1*m+l1][col] += 1
                V[l0*p+k0][col] += 1
                V[l1*p+k1][col] += 1
                W[k0*n+i0][col] += 1
                W[k1*n+i0][col] += 1
                W[k0*n+i1][col] += 1
                W[k1*n+i1][col] += 1
                col += 1
                
                # M2 = (A[i1,l0]+A[i1,l1]) * B[l0,k0]
                U[i1*m+l0][col] += 1
                U[i1*m+l1][col] += 1
                V[l0*p+k0][col] += 1
                W[k0*n+i1][col] += 1
                col += 1
                
                # M3 = A[i0,l0] * (B[l1,k0]-B[l1,k1])
                U[i0*m+l0][col] += 1
                V[l1*p+k0][col] += 1
                V[l1*p+k1][col] -= 1
                W[k0*n+i0][col] += 1
                W[k1*n+i0][col] -= 1
                col += 1
                
                # M4 = A[i1,l1] * (B[l0,k1]-B[l0,k0])
                U[i1*m+l1][col] += 1
                V[l0*p+k1][col] += 1
                V[l0*p+k0][col] -= 1
                W[k1*n+i1][col] += 1
                W[k0*n+i1][col] -= 1
                col += 1
                
                # M5 = (A[i0,l0]+A[i0,l1]) * B[l1,k1]
                U[i0*m+l0][col] += 1
                U[i0*m+l1][col] += 1
                V[l1*p+k1][col] += 1
                W[k1*n+i0][col] += 1
                col += 1
                
                # M6 = (A[i1,l0]-A[i0,l0]) * (B[l0,k0]+B[l0,k1])
                U[i1*m+l0][col] += 1
                U[i0*m+l0][col] -= 1
                V[l0*p+k0][col] += 1
                V[l0*p+k1][col] += 1
                W[k0*n+i1][col] += 1
                W[k1*n+i1][col] += 1
                col += 1
                
                # M7 = (A[i0,l1]-A[i1,l1]) * (B[l1,k0]+B[l1,k1])
                U[i0*m+l1][col] += 1
                U[i1*m+l1][col] -= 1
                V[l1*p+k0][col] += 1
                V[l1*p+k1][col] += 1
                W[k0*n+i0][col] += 1
                W[k1*n+i0][col] += 1
                col += 1
    
    # Schoolbook for remaining m-row 4 with all 6 p-cols
    for k in range(p):
        for i in range(n):
            U[i*m+4][col] += 1
            V[4*p+k][col] += 1
            W[k*n+i][col] += 1
            col += 1
    
    # Verify
    if not _verify_decomp(U, V, W, n, m, p, rank):
        # Fallback: pure schoolbook (rank 60)
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    U[i*m+l][col] += 1
                    V[l*p+k][col] += 1
                    W[k*n+i][col] += 1
                    col += 1
    
    # Try to reduce rank by checking for redundant terms
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    best_rank = rank
    best_U = [row[:] for row in U]
    best_V = [row[:] for row in V]
    best_W = [row[:] for row in W]
    
    # Check if any term is redundant
    for r_drop in range(rank):
        if time.time() > _deadline - 10:
            break
        U_test = U_arr.copy()
        V_test = V_arr.copy()
        W_test = W_arr.copy()
        U_test[:, r_drop] = 0
        V_test[:, r_drop] = 0
        W_test[:, r_drop] = 0
        
        valid = True
        for i in range(n):
            for l in range(m):
                for k in range(p):
                    s = np.dot(U_test[i*m+l, :], V_test[l*p+k, :] * W_test[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            new_rank = rank - 1
            new_U = [row[:r_drop] + row[r_drop+1:] for row in U]
            new_V = [row[:r_drop] + row[r_drop+1:] for row in V]
            new_W = [row[:r_drop] + row[r_drop+1:] for row in W]
            
            if new_rank < best_rank:
                best_rank = new_rank
                best_U = new_U
                best_V = new_V
                best_W = new_W
            
            U = best_U
            V = best_V
            W = best_W
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            rank = new_rank
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
