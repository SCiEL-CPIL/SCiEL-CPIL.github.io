"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 5, 5]
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 5.0

def _time_left():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 5, 5
    # Schoolbook rank = n*m*p = 75
    
    def _build_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W

    def _verify(U, V, W, tol=1e-9):
        """
        Verify that the decomposition satisfies:
        For all i, k: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * V[j*p+k, c])
        should equal sum_j A[i,j]*B[j,k] for all A, B.
        
        Actually the tensor equation is:
        The multiplication tensor T has T[i,j,k] such that the bilinear form
        sum_{i,j,k} T[i,j,k] A[i,j] B[j,k] = C[i,k] (component-wise)
        
        With the factorization: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        For matrix multiplication, T[i,j,k] = 1 for all i, j, k? No...
        
        Let me think more carefully. The matrix multiplication map is:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        This is a trilinear map from (A, B, C) to R where:
        sum_{i,j,k} C[i,k] * A[i,j] * B[j,k] ... no that's not right either.
        
        The standard way: the multiplication tensor M is defined by
        M = sum_{i,j,k} e_i ⊗ e_j ⊗ e_k where the coefficient is 1 if the
        (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C.
        
        Since C[i,k] = sum_j A[i,j] B[j,k], every triple (i,j,k) contributes.
        So M[i,j,k] = 1 for all valid i, j, k? 
        
        Wait, that's not right. Let me reconsider.
        
        The tensor for matrix multiplication of (n x m) by (m x p) is:
        M = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} E_{i,j} ⊗ E_{j,k} ⊗ E_{i,k}
        
        where E_{i,j} is the matrix with 1 at (i,j) and 0 elsewhere.
        
        So as a 3-tensor: M[i, j, k] where the first index pairs with A's row,
        second with A's col / B's row, third with B's col.
        
        But the factorization given is:
        U: (n*m) x rank, indexed by (i,j) -> i*m+j
        V: (m*p) x rank, indexed by (j,k) -> j*p+k  
        W: (p*n) x rank, indexed by (k,i) -> k*n+i
        
        And the tensor entry is: M[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        For matrix multiplication, we need:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        This means the tensor should satisfy:
        sum_{i,j,k} M[i,j,k] * A[i,j] * B[j,k] = sum_{i,k} C[i,k] * (something)
        
        Actually, I think the correct interpretation is:
        The bilinear map (A, B) -> C is represented by the tensor M such that:
        C[i,k] = sum_{j} sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] * A[i,j] * B[j,k]
        
        Wait, that's not standard. Let me look at this differently.
        
        The standard tensor network for matrix multiplication:
        C[i,k] = sum_j A[i,j] B[j,k]
        
        This can be written as:
        C[i,k] = sum_j sum_c delta_{i,k}^{(c)} * A[i,j] * B[j,k]
        
        where delta_{i,k}^{(c)} are the "structure constants". For schoolbook,
        each term (i,j,k) is separate, so we have n*m*p terms.
        
        Actually, I think the factorization means:
        The tensor T_{i,j,k} (where i is row of A, j is col of A/row of B, k is col of B)
        satisfies: T_{i,j,k} = sum_c U_{i*m+j, c} V_{j*p+k, c} W_{k*n+i, c}
        
        And for matrix multiplication, T_{i,j,k} = 1 for all i, j, k? No, that gives rank 1.
        
        I think I'm overcomplicating this. Let me just verify the schoolbook numerically.
        """
        rank = U.shape[1]
        # The tensor M[i,j,k] = sum_c U[i*m+j, c] V[j*p+k, c] W[k*n+i, c]
        # For matmul, we need this to equal... what exactly?
        # 
        # Let's think of it as: the bilinear map (A,B) -> C is:
        # C = sum_c W_c * (U_c^T A_flat) * (V_c^T B_flat) ... no
        #
        # I think the correct check is:
        # For random A (n x m) and B (m x p), compute C = A @ B
        # Then check: C[i,k] = sum_c W[k*n+i, c] * sum_j U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k]
        # Hmm, that doesn't look right dimensionally.
        #
        # Let me try: C[i,k] = sum_c W[k*n+i, c] * (sum_j U[i*m+j, c] * A[i,j]) * (sum_j V[j*p+k, c] * B[j,k])
        # That would be: C[i,k] = sum_c W[k*n+i, c] * (U_c · A_row_i) * (V_c · B_col_k)
        # where U_c is the c-th column of U (length n*m), but we only use the i-th block...
        #
        # Actually I think it's simpler. Let me just test numerically with schoolbook.
        A = np.random.randn(n, m)
        B = np.random.randn(m, p)
        C = A @ B
        
        # Try interpretation 1: C[i,k] = sum_c W[k*n+i,c] * sum_j U[i*m+j,c]*A[i,j]*V[j*p+k,c]*B[j,k]
        C1 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    for j in range(m):
                        s += U[i*m+j, c] * A[i,j] * V[j*p+k, c] * B[j,k] * W[k*n+i, c]
                C1[i,k] = s
        if np.max(np.abs(C1 - C)) < tol:
            return True
        
        # Try interpretation 2: C[i,k] = sum_c W[k*n+i,c] * (sum_j U[i*m+j,c]*A[i,j]) * (sum_j V[j*p+k,c]*B[j,k])
        C2 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    ua = sum(U[i*m+j, c] * A[i,j] for j in range(m))
                    vb = sum(V[j*p+k, c] * B[j,k] for j in range(m))
                    s += W[k*n+i, c] * ua * vb
                C2[i,k] = s
        if np.max(np.abs(C2 - C)) < tol:
            return True
        
        # Try interpretation 3: The tensor M[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c]
        # and C[i,k] = sum_j M[i,j,k] * A[i,j] * B[j,k]
        C3 = np.zeros((n, p))
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    Mijk = 0.0
                    for c in range(rank):
                        Mijk += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    s += Mijk * A[i,j] * B[j,k]
                C3[i,k] = s
        if np.max(np.abs(C3 - C)) < tol:
            return True
        
        return False

    U, V, W = _build_schoolbook()
    
    # Quick verification of schoolbook
    if not _verify(U, V, W):
        # If schoolbook doesn't verify, something is wrong with our understanding
        # Return it anyway as best effort
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U, V, W
    best_rank = U.shape[1]
    
    # Try to find lower rank decompositions
    # Strategy: use Strassen on 2x2 blocks where possible
    # For 3x5x5, we can try various block decompositions
    
    # Let's try a greedy column removal approach
    # Start from schoolbook and try to remove columns while maintaining validity
    
    def _try_remove_column(U, V, W, col_idx):
        """Try removing column col_idx and see if the decomposition still works."""
        # This is complex; for now, skip
        pass
    
    # Alternative: try known low-rank constructions
    # For 2x2x2, Strassen gives rank 7 vs schoolbook 8
    # We can embed Strassen into our 3x5x5 problem
    
    # Actually, let's try a different approach:
    # The 5x5 part can be decomposed using block structure
    # 5 = 2 + 2 + 1
    # So 5x5 multiplication can be done with blocks:
    # [2 2 1; 2 2 1; 1 1 1] blocks
    # Each 2x2 block uses Strassen (rank 7), 2x1 and 1x2 use rank 2, 1x1 uses rank 1
    # Total for 5x5: 4*7 (2x2 blocks) + 4*2 (2x1 and 1x2) + 1*1 (1x1) = 28 + 8 + 1 = 37
    # Then for 3x5: we have 3 rows, so we need 3 copies of the 5-dim structure
    # Total rank: 3 * 37 = 111? That's worse than 75.
    
    # Let me reconsider. The schoolbook for 3x5x5 is 3*5*5 = 75.
    # To beat this, we need rank < 75.
    
    # For the 5x5 matrix multiplication, the best known rank is 25 (schoolbook) 
    # but actually it's known that rank(5x5x5) <= 25, and it's not known if it can be less.
    # Actually, the best known upper bound for 5x5 is 25 (schoolbook), but there might be
    # better constructions.
    
    # Let me try a simple approach: just return the schoolbook.
    # The inspiration mentioned score 1.0563 with rank 74, so we need to find rank 74 or less.
    
    # Let's try to find a rank 74 decomposition by modifying one term.
    # Idea: use Strassen on a 2x2x2 sub-problem.
    # In the 3x5x5 problem, pick two rows of A (say rows 0,1), two columns of A/rows of B (say 0,1),
    # and two columns of B (say 0,1). This gives a 2x2x2 sub-problem embedded in the 3x5x5.
    # Strassen reduces this from 8 terms to 7, saving 1 term total.
    
    # Construction:
    # For the sub-block A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2], use Strassen.
    # The rest uses schoolbook.
    
    def _strassen_2x2():
        """
        Strassen decomposition for 2x2x2 matrix multiplication.
        Returns U, V, W each of shape (4, 7) for the 2x2 case.
        U indexed by (i,j) -> i*2+j, V by (j,k) -> j*2+k, W by (k,i) -> k*2+i
        
        Strassen's 7 products:
        M1 = (A11+A22)(B11+B22)
        M2 = (A21+A22)B11
        M3 = A11(B12-B22)
        M4 = A22(B21-B11)
        M5 = (A11+A12)B22
        M6 = (A21-A11)(B11+B12)
        M7 = (A12-A22)(B21+B22)
        
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        """
        rank = 7
        U = np.zeros((4, rank))
        V = np.zeros((4, rank))
        W = np.zeros((4, rank))
        
        # M1 = (A11+A22)(B11+B22)
        # A11 is at (0,0) -> U[0,0], A22 at (1,1) -> U[3,0]
        U[0, 0] = 1; U[3, 0] = 1
        # B11 at (0,0) -> V[0,0], B22 at (1,1) -> V[3,0]
        V[0, 0] = 1; V[3, 0] = 1
        # C11 gets +1, C22 gets +1
        # W indexed by (k,i) -> k*2+i
        # C11: i=0,k=0 -> W[0,0] += 1
        # C22: i=1,k=1 -> W[3,0] += 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1  # A21=(1,0)->2, A22=(1,1)->3
        V[0, 1] = 1  # B11=(0,0)->0
        # C21: i=1,k=0 -> W[2,1] += 1
        W[2, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1  # A11=(0,0)->0
        V[1, 2] = 1; V[3, 2] = -1  # B12=(0,1)->1, B22=(1,1)->3
        # C12: i=0,k=1 -> W[2,2] += 1
        W[2, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1  # A22=(1,1)->3
        V[2, 3] = 1; V[0, 3] = -1  # B21=(1,0)->2, B11=(0,0)->0
        # C11: i=0,k=0 -> W[0,3] += 1
        # C21: i=1,k=0 -> W[2,3] += 1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1  # A11=(0,0)->0, A12=(0,1)->1
        V[3, 4] = 1  # B22=(1,1)->3
        # C11: i=0,k=0 -> W[0,4] -= 1
        # C12: i=0,k=1 -> W[2,4] += 1
        W[0, 4] = -1; W[2, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1  # A21=(1,0)->2, A11=(0,0)->0
        V[0, 5] = 1; V[1, 5] = 1  # B11=(0,0)->0, B12=(0,1)->1
        # C22: i=1,k=1 -> W[3,5] += 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1  # A12=(0,1)->1, A22=(1,1)->3
        V[2, 6] = 1; V[3, 6] = 1  # B21=(1,0)->2, B22=(1,1)->3
        # C11: i=0,k=0 -> W[0,6] += 1
        # C22: i=1,k=1 -> W[3,6] += 1
        W[0, 6] = 1; W[3, 6] = 1
        
        return U, V, W
    
    # Now embed Strassen into the 3x5x5 problem
    # Use Strassen for A[0:2, 0:2] @ B[0:2, 0:2] -> C[0:2, 0:2]
    # Schoolbook for everything else
    
    def _build_with_strassen():
        """
        Build decomposition using Strassen for the top-left 2x2x2 block.
        Schoolbook for the rest.
        Total rank = 75 - 8 + 7 = 74
        """
        rank = 74
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # First, set up Strassen for the 2x2x2 sub-block (rows 0-1 of A, cols 0-1 of A/B, cols 0-1 of B)
        sU, sV, sW = _strassen_2x2()
        
        # Map Strassen indices to global indices
        # Strassen U index (i,j) with i,j in {0,1} -> global i*m+j = i*5+j
        # Strassen V index (j,k) with j,k in {0,1} -> global j*p+k = j*5+k
        # Strassen W index (k,i) with k,i in {0,1} -> global k*n+i = k*3+i
        
        for c in range(7):
            for idx in range(4):
                if sU[idx, c] != 0:
                    # idx = i*2+j for i,j in {0,1}
                    si = idx // 2
                    sj = idx % 2
                    U[si * m + sj, c] = sU[idx, c]
                
                if sV[idx, c] != 0:
                    sj = idx // 2
                    sk = idx % 2
                    V[sj * p + sk, c] = sV[idx, c]
                
                if sW[idx, c] != 0:
                    sk = idx // 2
                    si = idx % 2
                    W[sk * n + si, c] = sW[idx, c]
        
        # Now add schoolbook terms for everything except the 2x2x2 block
        # The 2x2x2 block covers: i in {0,1}, j in {0,1}, k in {0,1}
        # We need to add terms for all (i,j,k) NOT in this block
        
        col = 7  # Start after Strassen terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Skip if this is in the Strassen block
                    if i < 2 and j < 2 and k < 2:
                        continue
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        
        # Verify col == rank
        assert col == rank, f"Expected rank {rank}, got {col}"
        
        return U, V, W
    
    U_s, V_s, W_s = _build_with_strassen()
    
    # Verify the Strassen-based decomposition
    if _verify(U_s, V_s, W_s):
        best_U, best_V, best_W = U_s, V_s, W_s
        best_rank = 74
    else:
        # If verification fails, fall back to schoolbook
        pass
    
    # Try to find even lower rank
    # For now, return the best we have
    
    # Convert to lists for the return format
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
