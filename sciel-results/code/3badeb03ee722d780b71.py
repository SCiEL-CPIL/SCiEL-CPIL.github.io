"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 5, 5]
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 5, 5
    # Schoolbook rank is 100. Try to reduce using block Strassen.
    
    # Partition: rows of A: 2+2, cols of A/rows of B: 2+3, cols of B: 2+3
    # A (4x5) = [A11 A12; A21 A22] where A11,A21 are 2x2, A12,A22 are 2x3
    # B (5x5) = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x3, B21 is 3x2, B22 is 3x3
    # C (4x5) = [C11 C12; C21 C22]
    # C11 = A11*B11 + A12*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C12 = A11*B12 + A12*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    # C21 = A21*B11 + A22*B21  (2x2 = 2x2*2x2 + 2x3*3x2)
    # C22 = A21*B12 + A22*B22  (2x3 = 2x2*2x3 + 2x3*3x3)
    
    # For each 2x2 * 2x2 product, use Strassen (rank 7)
    # For other products, use schoolbook
    
    # Let's build the decomposition term by term
    # Each term is (U_col, V_col, W_col)
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u, v, w):
        """u, v, w are lists of (index, value) for sparse entries"""
        U_cols.append(u)
        V_cols.append(v)
        W_cols.append(w)
    
    # Strassen for 2x2: 
    # M1 = (A11+A22)*(B11+B22)
    # M2 = (A21+A22)*B11
    # M3 = A11*(B12-B22)
    # M4 = A22*(B21-B11)
    # M5 = (A11+A12)*B22
    # M6 = (A21-A11)*(B11+B12)
    # M7 = (A12-A22)*(B21+B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # But our blocks are not all 2x2. Let me handle this differently.
    
    # Simpler approach: just use schoolbook but check if we can merge any terms.
    # Actually, let me just return schoolbook for now and verify correctness.
    # The goal is to find ANY valid decomposition with rank < 100.
    
    # Let me try a random search: initialize with rank r, random matrices, 
    # and check if they decompose the tensor.
    
    deadline = time.time() + 100  # leave margin
    
    def tensor_check(U, V, W):
        """Check if U,V,W correctly decompose the (n,m,p) matrix mult tensor."""
        n, m, p = DIMENSIONS
        # The tensor T[i,j,k,l] = delta(i, row of C) * delta(j, col of B) * delta(k, col of A) * delta(l, row of B)
        # Actually, the matrix mult tensor: C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor form: T[i,j,k] = A[i,j]*B[j,k] contributes to C[i,k]
        # So T is a 3-tensor with indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p)
        # T[i,j,k] = 1 if the entry A[i,j]*B[j,k] contributes to C[i,k], which is always true.
        # Wait, the tensor is defined as: for each (i,j,k), the bilinear form is A[i,j]*B[j,k] -> C[i,k]
        # So the tensor T has entries T[i,j,k] = 1 (it's the structure tensor)
        # A decomposition of rank r means: T[i,j,k] = sum_{c=0}^{r-1} U[i,j,c] * V[j,k,c] * W[k,i,c]
        # Wait, let me re-read the interface.
        
        # U has shape (n*m, rank), indexed by (i*m + j, c)
        # V has shape (m*p, rank), indexed by (j*p + k, c)  
        # W has shape (p*n, rank), indexed by (k*n + i, c)
        
        # The decomposition: for all i,j,k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
        # (since the tensor entry T[i,j,k] = 1 for the matrix multiplication)
        
        # Actually wait - the tensor for matrix multiplication: 
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The structure tensor has T[i,j,k] = 1 (each triple (i,j,k) has a 1)
        # So we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
        
        # Let's verify this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(len(U[0])):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Start with schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j][col] = 1
                V_sb[j*p+k][col] = 1
                W_sb[k*n+i][col] = 1
                col += 1
    
    # Verify schoolbook
    if not tensor_check(U_sb, V_sb, W_sb):
        # Should not happen
        pass
    
    # Now try to reduce rank by zeroing out some columns and adjusting
    # Or try random rank-r decompositions
    
    best = (rank_schoolbook, U_sb, V_sb, W_sb)
    
    # Try various ranks below schoolbook
    for target_rank in [90, 85, 80, 75, 70]:
        if time.time() > deadline - 20:
            break
        
        found = False
        for attempt in range(50):
            if time.time() > deadline - 20:
                break
            
            U = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.randint(-2, 2) for _ in range(target_rank)] for _ in range(p*n)]
            
            if tensor_check(U, V, W):
                found = True
                best = (target_rank, U, V, W)
                break
        
        if found:
            break
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
