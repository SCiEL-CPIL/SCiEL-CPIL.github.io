"""Find rank-minimal exact tensor decomposition for 3x4x5 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145.0
    
    # Build the target tensor T with shape (n*m, m*p, p*n)
    # T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i for valid i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0
    
    def verify(U, V, W, tol=1e-8):
        """Check if decomposition is exact."""
        r = U.shape[1]
        T_approx = np.einsum('ar,br,cr->abc', U, V, W)
        return np.max(np.abs(T_approx - T)) < tol
    
    def schoolbook():
        """Standard O(nmp) decomposition."""
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def strassen_decomposition_3x4x5():
        """
        Construct 3x4x5 MM using Strassen for 2x2x2 sub-blocks.
        
        Strategy: Identify 4 disjoint 2x2x2 blocks within the 3x4x5 tensor.
        For each block, use Strassen's rank-7 decomposition instead of schoolbook rank-8.
        Remaining terms use schoolbook.
        
        Blocks (i, j, k ranges):
        - Block 1: i in {0,1}, j in {0,1}, k in {0,1}
        - Block 2: i in {0,1}, j in {2,3}, k in {0,1}  
        - Block 3: i in {0,1}, j in {0,1}, k in {2,3}
        - Block 4: i in {0,1}, j in {2,3}, k in {2,3}
        
        Total: 4*7 + 28 = 56
        """
        U = np.zeros((n*m, 56))
        V = np.zeros((m*p, 56))
        W = np.zeros((p*n, 56))
        
        col = 0
        
        # Strassen's 7 terms for 2x2x2 with local indices (ii, jj, kk) in {0,1}
        # Global indices: i = i_offset + ii, j = j_offset + jj, k = k_offset + kk
        # a = i*m + j, b = j*p + k, c = k*n + i
        
        # Strassen terms: (list of (ii,jj) for U, list of (jj,kk) for V, list of (kk,ii) for W)
        # Using 1-based Strassen notation mapped to 0-based:
        # a11=(0,0), a12=(0,1), a21=(1,0), a22=(1,1)
        # b11=(0,0), b12=(0,1), b21=(1,0), b22=(1,1)
        # c11=(0,0), c12=(0,1), c21=(1,0), c22=(1,1)
        
        # M1 = (a11+a22)(b11+b22)
        # M2 = (a21+a22)b11
        # M3 = a11(b12-b22)
        # M4 = a22(b21-b11)
        # M5 = (a11+a12)b22
        # M6 = (a21-a11)(b11+b12)
        # M7 = (a12-a22)(b21+b22)
        #
        # c11 = M1+M4-M5+M7
        # c12 = M3+M5
        # c21 = M2+M4
        # c22 = M1-M2+M3+M6
        
        strassen_terms = [
            # term: (u_entries, v_entries, w_entries)
            # u_entries: list of (ii, jj, coeff)
            # v_entries: list of (jj, kk, coeff)
            # w_entries: list of (kk, ii, coeff)
            
            # M1 = (a11+a22)(b11+b22)
            ([((0,0), 1.0), ((1,1), 1.0)],
             [((0,0), 1.0), ((1,1), 1.0)],
             [((0,0), 1.0), ((1,1), 1.0)]),
            
            # M2 = (a21+a22)b11
            ([((1,0), 1.0), ((1,1), 1.0)],
             [((0,0), 1.0)],
             [((0,1), 1.0)]),
            
            # M3 = a11(b12-b22)
            ([((0,0), 1.0)],
             [((0,1), 1.0), ((1,1), -1.0)],
             [((1,0), 1.0), ((1,1), 1.0)]),
            
            # M4 = a22(b21-b11)
            ([((1,1), 1.0)],
             [((1,0), 1.0), ((0,0), -1.0)],
             [((0,0), 1.0), ((0,1), 1.0)]),
            
            # M5 = (a11+a12)b22
            ([((0,0), 1.0), ((0,1), 1.0)],
             [((1,1), 1.0)],
             [((1,0), 1.0), ((0,0), -1.0)]),
            
            # M6 = (a21-a11)(b11+b12)
            ([((1,0), 1.0), ((0,0), -1.0)],
             [((0,0), 1.0), ((0,1), 1.0)],
             [((1,1), 1.0)]),
            
            # M7 = (a12-a22)(b21+b22)
            ([((0,1), 1.0), ((1,1), -1.0)],
             [((1,0), 1.0), ((1,1), 1.0)],
             [((0,0), 1.0)]),
        ]
        
        # Define the 4 blocks: (i_offset, j_offset, k_offset)
        blocks = [
            (0, 0, 0),   # i in {0,1}, j in {0,1}, k in {0,1}
            (0, 2, 0),   # i in {0,1}, j in {2,3}, k in {0,1}
            (0, 0, 2),   # i in {0,1}, j in {0,1}, k in {2,3}
            (0, 2, 2),   # i in {0,1}, j in {2,3}, k in {2,3}
        ]
        
        for i_off, j_off, k_off in blocks:
            for term in strassen_terms:
                u_entries, v_entries, w_entries = term
                
                for (ii, jj), coeff in u_entries:
                    i_global = i_off + ii
                    j_global = j_off + jj
                    a = i_global * m + j_global
                    U[a, col] = coeff
                
                for (jj, kk), coeff in v_entries:
                    j_global = j_off + jj
                    k_global = k_off + kk
                    b = j_global * p + k_global
                    V[b, col] = coeff
                
                for (kk, ii), coeff in w_entries:
                    k_global = k_off + kk
                    i_global = i_off + ii
                    c = k_global * n + i_global
                    W[c, col] = coeff
                
                col += 1
        
        # Now add schoolbook terms for remaining (i,j,k) combinations
        # Blocks cover: i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3}
        # Remaining: i=2 with any j,k; i in {0,1} with j in {0,1,2,3}, k=4
        # Actually let's enumerate all (i,j,k) not covered by the 4 blocks
        
        covered = set()
        for i_off, j_off, k_off in blocks:
            for ii in range(2):
                for jj in range(2):
                    for kk in range(2):
                        covered.add((i_off + ii, j_off + jj, k_off + kk))
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if (i, j, k) not in covered:
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        U[a, col] = 1.0
                        V[b, col] = 1.0
                        W[c, col] = 1.0
                        col += 1
        
        assert col == 56, f"Expected 56 columns, got {col}"
        return U, V, W, 56
    
    # Try the Strassen-based approach
    U_str, V_str, W_str, rank_str = strassen_decomposition_3x4x5()
    
    # Verify and set best
    if verify(U_str, V_str, W_str, tol=1e-8):
        best_U, best_V, best_W = U_str.copy(), V_str.copy(), W_str.copy()
        best_rank = rank_str
    else:
        U_sb, V_sb, W_sb, rank_sb = schoolbook()
        best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
        best_rank = rank_sb
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
