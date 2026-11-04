"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 5
    deadline = time.time() + 115.0  # safety margin before 150s kill

    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 Strassen decomposition."""
        U = [
            [1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 1],
        ]
        V = [
            [1, 0, 0, 0, 0, 1, 1],
            [0, 1, -1, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0],
            [1, 1, 0, -1, 1, 0, 0],
        ]
        W = [
            [1, 0, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 1, -1, 0, 0],
        ]
        return U, V, W

    def verify(U, V, W, n, m, p, tol=1e-9):
        """Verify that the decomposition is correct."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Build the tensor from decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        T_decomp = np.zeros((n, m, p))
        for c in range(rank):
            u = U[:, c].reshape(n, m)
            v = V[:, c].reshape(m, p)
            w = W[:, c].reshape(p, n)
            # T[i,j,k] += u[i,j]*v[j,k]*w[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T_decomp[i, j, k] += u[i, j] * v[j, k] * w[k, i]
        # Target tensor is all ones
        return np.allclose(T_decomp, np.ones((n, m, p)), atol=tol, rtol=0)

    def verify_fast(U, V, W, n, m, p, tol=1e-9):
        """Faster verification using einsum."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        rank = U.shape[1]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_r = U.reshape(n, m, rank)
        V_r = V.reshape(m, p, rank)
        W_r = W.reshape(p, n, rank)
        # T[i,j,k] = sum_c U_r[i,j,c] * V_r[j,k,c] * W_r[k,i,c]
        T = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r, optimize=True)
        return np.allclose(T, np.ones((n, m, p)), atol=tol, rtol=0)

    # Start with schoolbook baseline
    best_result = schoolbook()
    best_rank = n * m * p

    # Try to construct a better decomposition using Strassen for 2x2 blocks
    # Split rows into {0,1} and {2}, split middle into pairs (0,1) and (2,3), and single 4
    # For rows {0,1}: use Strassen for middle pairs (0,1) and (2,3), schoolbook for middle 4
    # For row {2}: schoolbook for all middle indices
    
    # Let's construct the hybrid decomposition:
    # Total rank: 7 (Strassen for rows 0-1, mid 0-1) + 7 (Strassen for rows 0-1, mid 2-3) 
    #              + 2 (schoolbook for rows 0-1, mid 4) 
    #              + 5 (schoolbook for row 2, mid 0-4)
    # = 21
    
    # But wait, we need to be careful with the indexing.
    # The output W has indices k*n+i. For n=3, that's k*3+i.
    # For Strassen on rows 0-1, the output rows are k*3+0 and k*3+1.
    # For schoolbook on row 2, the output rows are k*3+2.
    
    # Let's build the matrices column by column.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # 1. Strassen for rows {0,1}, middle {0,1}
    # This handles T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1}
    # But Strassen 2x2x2 handles a 2x2 matrix multiply: C = A*B where A,B are 2x2.
    # Here, for fixed k, we're multiplying a 2x2 submatrix (rows 0-1, cols 0-1 of A) 
    # by a 2x1 submatrix (col k of B)? No, that's not right.
    
    # Actually, the tensor T[i,j,k] corresponds to C[i,k] = sum_j A[i,j] B[j,k].
    # For i in {0,1}, k in {0,1}, j in {0,1}: this is a 2x2x2 sub-tensor.
    # Strassen can decompose this 2x2x2 sub-tensor into 7 rank-1 terms.
    # But these terms only contribute to T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1}.
    # They do NOT contribute to T[i,j,k] for j in {2,3,4} or k in {2,3,4}.
    
    # So we can use 7 terms for the (0,1)x(0,1)x(0,1) sub-tensor.
    # Similarly, 7 terms for the (0,1)x(2,3)x(0,1) sub-tensor? No, that's not a standard sub-tensor.
    
    # Let's think differently. The matrix multiplication tensor is:
    # T[i,j,k] = delta_{i,j,k} in the sense that it's 1 for all (i,j,k).
    # We want to decompose this into rank-1 terms.
    
    # A rank-1 term is defined by vectors u (length 12), v (length 20), w (length 15)
    # such that T[i,j,k] = sum_c u[i*m+j,c] * v[j*p+k,c] * w[k*n+i,c].
    
    # For the schoolbook, each term has u, v, w with a single 1.
    # For Strassen, each term has multiple non-zero entries in u, v, w.
    
    # Let's try to use Strassen for the 2x2x2 block where i in {0,1}, j in {0,1}, k in {0,1}.
    # The Strassen decomposition for this block gives 7 terms.
    # For each term c:
    #   u[i*m+j,c] is non-zero for i in {0,1}, j in {0,1}
    #   v[j*p+k,c] is non-zero for j in {0,1}, k in {0,1}
    #   w[k*n+i,c] is non-zero for k in {0,1}, i in {0,1}
    
    # Let's extract the Strassen U, V, W and map them to our indexing.
    # Strassen U is 4x7, indexed by (i,j) for i,j in {0,1}.
    # In our U, the index is i*m+j = i*4+j.
    # So U_strassen[i,j] -> U[i*4+j, c]
    
    # Strassen V is 4x7, indexed by (j,k) for j,k in {0,1}.
    # In our V, the index is j*p+k = j*5+k.
    # So V_strassen[j,k] -> V[j*5+k, c]
    
    # Strassen W is 4x7, indexed by (k,i) for k,i in {0,1}.
    # In our W, the index is k*n+i = k*3+i.
    # So W_strassen[k,i] -> W[k*3+i, c]
    
    U_s, V_s, W_s = strassen_2x2x2()
    
    # Add these 7 terms
    for c in range(7):
        U_col = [0.0] * (n * m)
        V_col = [0.0] * (m * p)
        W_col = [0.0] * (p * n)
        
        for i in range(2):
            for j in range(2):
                U_col[i*m+j] = U_s[i*2+j][c]
        
        for j in range(2):
            for k in range(2):
                V_col[j*p+k] = V_s[j*2+k][c]
        
        for k in range(2):
            for i in range(2):
                W_col[k*n+i] = W_s[k*2+i][c]
        
        U_cols.append(U_col)
        V_cols.append(V_col)
        W_cols.append(W_col)
    
    # Now we need to handle the rest of the tensor.
    # The remaining entries are those where i in {0,1} and (j in {2,3,4} or k in {2,3,4})
    # plus all entries where i=2.
    
    # This is getting complicated. Let's just use schoolbook for the rest.
    # But we need to be careful not to double-count the (0,1)x(0,1)x(0,1) block.
    
    # Actually, the Strassen terms only contribute to T[i,j,k] for i in {0,1}, j in {0,1}, k in {0,1}.
    # For other (i,j,k), the Strassen terms contribute 0.
    # So we can add schoolbook terms for all other (i,j,k) combinations.
    
    # But wait, if we add schoolbook terms for (i,j,k) where i in {0,1}, j in {0,1}, k in {0,1},
    # we would be double-counting. So we skip those.
    
    # Let's add schoolbook terms for all (i,j,k) EXCEPT those already covered by Strassen.
    # Strassen covers: i in {0,1}, j in {0,1}, k in {0,1}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if already covered by Strassen
                if i < 2 and j < 2 and k < 2:
                    continue
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[i*m+j] = 1.0
                V_col[j*p+k] = 1.0
                W_col[k*n+i] = 1.0
                U_cols.append(U_col)
                V_cols.append(V_col)
                W_cols.append(W_col)
    
    # Total rank: 7 + (60 - 8) = 7 + 52 = 59. That's worse than schoolbook (60).
    # Actually, schoolbook is 60, and we're at 59. That's a slight improvement!
    
    # Let's verify this decomposition.
    U_hybrid = [[U_cols[c][r] for c in range(len(U_cols))] for r in range(n*m)]
    V_hybrid = [[V_cols[c][r] for c in range(len(V_cols))] for r in range(m*p)]
    W_hybrid = [[W_cols[c][r] for c in range(len(W_cols))] for r in range(p*n)]
    
    if verify_fast(U_hybrid, V_hybrid, W_hybrid, n, m, p):
        best_result = {"U": U_hybrid, "V": V_hybrid, "W": W_hybrid}
        best_rank = len(U_cols)
    
    # Try more Strassen blocks
    # Can we use Strassen for other 2x2x2 blocks?
    # For example, i in {0,1}, j in {2,3}, k in {0,1}
    # But this block is not independent; it shares indices with other blocks.
    
    # Let's try a different approach: use Strassen for all 2x2x2 blocks
    # where i in {0,1}, j in {2r, 2r+1}, k in {2s, 2s+1}
    
    # For n=3, m=4, p=5:
    # i can be {0,1} (one block) or {2} (single)
    # j can be {0,1}, {2,3} (two blocks)
    # k can be {0,1}, {2,3} (two blocks) or {4} (single)
    
    # If we use Strassen for i in {0,1}, j in {0,1}, k in {0,1}: 7 terms
    # i in {0,1}, j in {0,1}, k in {2,3}: 7 terms
    # i in {0,1}, j in {2,3}, k in {0,1}: 7 terms
    # i in {0,1}, j in {2,3}, k in {2,3}: 7 terms
    # Total: 28 terms for i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
    
    # Then schoolbook for:
    # i in {0,1}, j in {0,1,2,3}, k in {4}: 8 terms
    # i in {2}, j in {0,1,2,3}, k in {0,1,2,3,4}: 20 terms
    # Total: 28 + 8 + 20 = 56 terms
    
    # Let's construct this.
    
    U_cols2 = []
    V_cols2 = []
    W_cols2 = []
    
    # Strassen blocks for i in {0,1}, j in {j0, j0+1}, k in {k0, k0+1}
    for j0 in [0, 2]:
        for k0 in [0, 2]:
            U_s, V_s, W_s = strassen_2x2x2()
            for c in range(7):
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                
                for i in range(2):
                    for j in range(2):
                        U_col[i*m + (j0+j)] = U_s[i*2+j][c]
                
                for j in range(2):
                    for k in range(2):
                        V_col[(j0+j)*p + (k0+k)] = V_s[j*2+k][c]
                
                for k in range(2):
                    for i in range(2):
                        W_col[(k0+k)*n + i] = W_s[k*2+i][c]
                
                U_cols2.append(U_col)
                V_cols2.append(V_col)
                W_cols2.append(W_col)
    
    # Schoolbook for remaining entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if i in {0,1} and j in {0,1,2,3} and k in {0,1,2,3}
                if i < 2 and k < 4:
                    continue
                U_col = [0.0] * (n * m)
                V_col = [0.0] * (m * p)
                W_col = [0.0] * (p * n)
                U_col[i*m+j] = 1.0
                V_col[j*p+k] = 1.0
                W_col[k*n+i] = 1.0
                U_cols2.append(U_col)
                V_cols2.append(V_col)
                W_cols2.append(W_col)
    
    U_hybrid2 = [[U_cols2[c][r] for c in range(len(U_cols2))] for r in range(n*m)]
    V_hybrid2 = [[V_cols2[c][r] for c in range(len(V_cols2))] for r in range(m*p)]
    W_hybrid2 = [[W_cols2[c][r] for c in range(len(W_cols2))] for r in range(p*n)]
    
    if verify_fast(U_hybrid2, V_hybrid2, W_hybrid2, n, m, p):
        if len(U_cols2) < best_rank:
            best_result = {"U": U_hybrid2, "V": V_hybrid2, "W": W_hybrid2}
            best_rank = len(U_cols2)
    
    # Try to use Strassen for i in {0,1}, j in {0,1}, k in {0,1} and 
    # i in {0,1}, j in {2,3}, k in {0,1} and 
    # i in {0,1}, j in {0,1,2,3}, k in {2,3}
    # This is the same as above.
    
    # Let's try a different partition: use Strassen for i in {0,1}, all j, all k
    # But Strassen is for 2x2x2, not 2x4x5.
    
    # For 2x4x5, we can use the fact that rank(2x4x5) <= rank(2x2x5) + rank(2x2x5)
    # And rank(2x2x5) <= 7*2 + 2 = 16 (using Strassen for k in {0,1} and {2,3}, schoolbook for k=4)
    # So rank(2x4x5) <= 32.
    # Then rank(3x4x5) <= rank(2x4x5) + rank(1x4x5) <= 32 + 20 = 52.
    
    # Let's construct 2x4x5 with rank 32.
    
    U_245 = []
    V_245 = []
    W_245 = []
    
    # For i in {0,1}, j in {0,1}, k in {0,1}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * 8  # 2*4
        V_col = [0.0] * 20  # 4*5
        W_col = [0.0] * 10  # 5*2
        
        for i in range(2):
            for j in range(2):
                U_col[i*4+j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*5+k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*2+i] = W_s[k*2+i][c]
        
        U_245.append(U_col)
        V_245.append(V_col)
        W_245.append(W_col)
    
    # For i in {0,1}, j in {2,3}, k in {0,1}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * 8
        V_col = [0.0] * 20
        W_col = [0.0] * 10
        
        for i in range(2):
            for j in range(2):
                U_col[i*4+(2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*5+k] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[k*2+i] = W_s[k*2+i][c]
        
        U_245.append(U_col)
        V_245.append(V_col)
        W_245.append(W_col)
    
    # For i in {0,1}, j in {0,1}, k in {2,3}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * 8
        V_col = [0.0] * 20
        W_col = [0.0] * 10
        
        for i in range(2):
            for j in range(2):
                U_col[i*4+j] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[j*5+(2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*2+i] = W_s[k*2+i][c]
        
        U_245.append(U_col)
        V_245.append(V_col)
        W_245.append(W_col)
    
    # For i in {0,1}, j in {2,3}, k in {2,3}: Strassen
    U_s, V_s, W_s = strassen_2x2x2()
    for c in range(7):
        U_col = [0.0] * 8
        V_col = [0.0] * 20
        W_col = [0.0] * 10
        
        for i in range(2):
            for j in range(2):
                U_col[i*4+(2+j)] = U_s[i*2+j][c]
        for j in range(2):
            for k in range(2):
                V_col[(2+j)*5+(2+k)] = V_s[j*2+k][c]
        for k in range(2):
            for i in range(2):
                W_col[(2+k)*2+i] = W_s[k*2+i][c]
        
        U_245.append(U_col)
        V_245.append(V_col)
        W_245.append(W_col)
    
    # For i in {0,1}, j in {0,1,2,3}, k in {4}: schoolbook
    for i in range(2):
        for j in range(4):
            U_col = [0.0] * 8
            V_col = [0.0] * 20
            W_col = [0.0] * 10
            U_col[i*4+j] = 1.0
            V_col[j*5+4] = 1.0
            W_col[4*2+i] = 1.0
            U_245.append(U_col)
            V_245.append(V_col)
            W_245.append(W_col)
    
    # Now add the i=2 part: schoolbook for 1x4x5
    U_345 = []
    V_345 = []
    W_345 = []
    
    # Copy U_245, V_245, W_245 to U_345, V_345, W_345
    for c in range(len(U_245)):
        U_col = [0.0] * 12  # 3*4
        V_col = V_245[c][:]
        W_col = [0.0] * 15  # 5*3
        
        for i in range(2):
            for j in range(4):
                U_col[i*4+j] = U_245[c][i*4+j]
        for k in range(5):
            for i in range(2):
                W_col[k*3+i] = W_245[c][k*2+i]
        
        U_345.append(U_col)
        V_345.append(V_col)
        W_345.append(W_col)
    
    # Add schoolbook for i=2
    for j in range(4):
        for k in range(5):
            U_col = [0.0] * 12
            V_col = [0.0] * 20
            W_col = [0.0] * 15
            U_col[2*4+j] = 1.0
            V_col[j*5+k] = 1.0
            W_col[k*3+2] = 1.0
            U_345.append(U_col)
            V_345.append(V_col)
            W_345.append(W_col)
    
    U_hybrid3 = [[U_345[c][r] for c in range(len(U_345))] for r in range(n*m)]
    V_hybrid3 = [[V_345[c][r] for c in range(len(V_345))] for r in range(m*p)]
    W_hybrid3 = [[W_345[c][r] for c in range(len(W_345))] for r in range(p*n)]
    
    if verify_fast(U_hybrid3, V_hybrid3, W_hybrid3, n, m, p):
        if len(U_345) < best_rank:
            best_result = {"U": U_hybrid3, "V": V_hybrid3, "W": W_hybrid3}
            best_rank = len(U_345)
    
    return best_result
# EVOLVE-BLOCK-END
