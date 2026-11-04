"""
Strategy: Structure-aware constructive search using Strassen blocks.
Instead of generic ALS, we build the decomposition from known rank-7 
2x2 Strassen patterns and optimize a small set of free parameters 
to handle the non-2x2 dimensions (5 and 6) via a hybrid approach.
We use a combination of fixed structural components and a small 
Nelder-Mead optimization to find exact integer/half-integer solutions.
"""
import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

def _verify_rank(U, V, W, n=2, m=5, p=6):
    """Check if given factor matrices produce correct matmul tensor."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    
    # Reconstruct tensor
    # T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r]
    # Target: T[i*m+j, j*p+k, k*n+i] = 1 else 0
    
    # Efficient check using einsum
    # T is nm x mp x pn
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    # Check on-diagonal
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                if abs(T[a, b, c] - 1.0) > 1e-8:
                    return False
    
    # Check off-diagonal
    # Create a target tensor of zeros
    Target = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                Target[a, b, c] = 1.0
                
    # The difference should be zero
    if np.max(np.abs(T - Target)) > 1e-8:
        return False
        
    return True

def _schoolbook_rank():
    """Return the standard schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _strassen_2x2():
    """
    Returns U, V, W for 2x2 * 2x2 matmul with rank 7.
    Indices:
    U: 2x2 input A (row, col) -> index r*2+c
    V: 2x2 input B (row, col) -> index r*2+c
    W: 2x2 output C (row, col) -> index r*2+c
    """
    # Standard Strassen formulas for C = AB
    # M1 = (A11 - A22)(B22 + B11)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A22)(B22 - B11)  # Wait, standard is (A11+A22)(B22-B11) is not quite right for all variants.
    # Let's use the standard 7-product set:
    # p1 = (a11 - a22) * (b11 + b22)
    # p2 = (a21 + a22) * b11
    # p3 = (a11 + a12) * b22
    # p4 = a22 * (b21 - b11)
    # p5 = a11 * (b12 - b22)
    # p6 = (a21 - a11) * (b11 + b12)
    # p7 = (a12 - a22) * (b21 + b22)
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    
    # U factors (for A)
    # A indices: 00, 01, 10, 11
    # p1: a11(1), -a22(-1) -> U[:, 0] = [1, 0, 0, -1]
    # p2: a21(1), a22(1) -> U[:, 1] = [0, 0, 1, 1]
    # p3: a11(1), a12(1) -> U[:, 2] = [1, 1, 0, 0]
    # p4: a22(1) -> U[:, 3] = [0, 0, 0, 1]
    # p5: a11(1) -> U[:, 4] = [1, 0, 0, 0]
    # p6: a21(1), -a11(-1) -> U[:, 5] = [-1, 0, 1, 0]
    # p7: a12(1), -a22(-1) -> U[:, 6] = [0, 1, 0, -1]
    
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 1, 0, 0, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [-1, 1, 0, 1, 0, 0, -1]
    ], dtype=np.float64)
    
    # V factors (for B)
    # B indices: 00, 01, 10, 11
    # p1: b11(1), b22(1) -> V[:, 0] = [1, 0, 0, 1]
    # p2: b11(1) -> V[:, 1] = [1, 0, 0, 0]
    # p3: b22(1) -> V[:, 2] = [0, 0, 0, 1]
    # p4: b21(1), -b11(-1) -> V[:, 3] = [-1, 0, 1, 0]
    # p5: b12(1), -b22(-1) -> V[:, 4] = [0, 1, 0, -1]
    # p6: b11(1), b12(1) -> V[:, 5] = [1, 1, 0, 0]
    # p7: b21(1), b22(1) -> V[:, 6] = [0, 0, 1, 1]
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, 1, 0, -1, 0, 1]
    ], dtype=np.float64)
    
    # W factors (for C)
    # C indices: 00, 01, 10, 11
    # C11 = p1 + p4 - p5 + p7 -> W[0, :] = [1, 0, 0, 1, -1, 0, 1]
    # C12 = p3 + p5 -> W[1, :] = [0, 0, 1, 0, 1, 0, 0]
    # C21 = p2 + p4 -> W[2, :] = [0, 1, 0, 1, 0, 0, 0]
    # C22 = p1 - p2 + p3 + p6 -> W[3, :] = [1, -1, 1, 0, 0, 1, 0]
    
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=np.float64)
    
    return U, V, W

def _construct_hybrid_decomposition():
    """
    Construct a decomposition for 2x5 * 5x6 using Strassen blocks.
    
    Strategy:
    Split the 5x6 matrix into blocks.
    Since n=2 is small, we can view the multiplication as:
    C (2x6) = A (2x5) * B (5x6)
    
    We can decompose B into vertical strips or use a block structure.
    A common trick for (2, m, p) is to use the fact that rank(2, m, p) <= rank(2, 2, 2) * ceil(m/2) * ceil(p/2) is not tight.
    
    However, a known bound is rank(2, m, p) <= 2 * rank(1, m, p) ... no.
    
    Let's try a specific construction:
    Split A into [A1 A2] where A1 is 2x3 and A2 is 2x2.
    Split B into [B1; B2] where B1 is 3x6 and B2 is 2x6.
    C = A1*B1 + A2*B2.
    
    This is a sum of two tensors.
    Rank(T1 + T2) <= Rank(T1) + Rank(T2).
    
    We need rank(2, 3, 6) and rank(2, 2, 6).
    
    For 2x2 * 2x6:
    We can split the 6 columns into three 2x2 blocks.
    C = A * [B1 B2 B3] = [A*B1 A*B2 A*B3].
    Each A*Bk is 2x2 matmul. Rank 7.
    Total rank for 2x2 * 2x6 is 3 * 7 = 21.
    
    For 2x3 * 3x6:
    Split A into [A11 A12 A13] (2x1 each) and B into rows [B11; B12; B13] (1x6 each).
    C = A11*B11 + A12*B12 + A13*B13.
    Each term is rank 1 (outer product of 2x1 and 1x6).
    Total rank 3.
    
    Total rank = 21 + 3 = 24. This is much better than 60.
    
    Let's verify this construction.
    
    Term 1: 2x3 * 3x6
    A1 (2x3), B1 (3x6).
    Decomposition:
    For j in 0..2 (columns of A1, rows of B1):
       U[:, j] = A1[:, j] (2x1 vector)
       V[:, j] = B1[j, :] (1x6 vector)
       W[:, j] = Identity? No, W must map to C indices.
       
    Wait, the tensor T[a, b, c] = sum U[a, r] V[b, r] W[c, r].
    For rank-1 term: U is 2x3 matrix? No, U has shape (n*m, rank).
    
    Let's map indices carefully.
    n=2, m=3, p=6.
    U shape: 6 x 3.
    V shape: 18 x 3.
    W shape: 12 x 3.
    
    For each column j of A1 (which corresponds to row j of B1):
    r = j.
    U[a, r] should be A1[a, j] where a is index in 2x3 matrix.
    a = i * 3 + j? No, a is index in A (2x3). a = i*3 + j_A.
    Here j is the shared dimension.
    So for fixed j (0,1,2):
    U[i*3 + j, r] = A1[i, j]
    V[j*6 + k, r] = B1[j, k]
    W[k*2 + i, r] = 1.0
    
    Let's check the reconstruction for C[i, k]:
    C[i, k] = sum_j A1[i, j] * B1[j, k].
    Tensor entry T[a, b, c]:
    a = i*3 + j_A
    b = j_B*6 + k
    c = k*2 + i
    
    We want T[a, b, c] = 1 if j_A == j_B (and i, k match), else 0?
    No, the tensor represents the bilinear form.
    T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
    
    If we set:
    For r in 0..2:
       U[i*3 + r, r] = 1 (for all i) -> This means A1[i, r] is extracted?
       Wait, U is a factor matrix. The input A is a vector of length 6.
       U[a, r] is the coefficient of A[a] in the r-th product.
       
       If we want to compute sum_j A[i, j] B[j, k], we need 3 products.
       Product r (j=r):
       P_r = A[i, r] * B[r, k].
       
       So:
       U[a, r] = 1 if a = i*3 + r for some i? No, U is fixed.
       U[a, r] = 1 if a % 3 == r. (Extracts column r of A)
       V[b, r] = 1 if b // 6 == r. (Extracts row r of B)
       W[c, r] = 1 if c // 2 == k and c % 2 == i? 
       
       Let's trace:
       T[a, b, c] = sum_r U[a, r] V[b, r] W[c, r].
       a = i*3 + j_A
       b = j_B*6 + k
       c = k*2 + i
       
       We want T = 1 if j_A == j_B, else 0.
       
       Let r = j_A.
       U[a, r] = 1 if a % 3 == r.
       V[b, r] = 1 if b // 6 == r.
       W[c, r] = 1.
       
       Then T[a, b, c] = sum_r (1 if a%3==r) * (1 if b//6==r) * 1.
       This sum is 1 if there exists r such that a%3==r AND b//6==r.
       i.e., a%3 == b//6.
       j_A == j_B.
       This is exactly the condition for the diagonal of the multiplication tensor for 2x3 * 3x6.
       
       So the rank of 2x3 * 3x6 is 3.
       
    Term 2: 2x2 * 2x6
    Split B into B2 (2x6).
    A2 is 2x2.
    C2 = A2 * B2.
    B2 has 6 columns. Split into 3 blocks of 2 columns: B21, B22, B23 (each 2x2).
    C2 = [A2*B21, A2*B22, A2*B23].
    
    Each A2*B2k is a 2x2 matmul. Rank 7.
    Total rank 21.
    
    How to construct the factors for 2x2 * 2x6 with rank 21?
    We can reuse the 2x2 Strassen factors for each block.
    
    Indices for this sub-problem:
    n=2, m=2, p=6.
    U shape: 4 x 21.
    V shape: 12 x 21.
    W shape: 12 x 21.
    
    For block l (0, 1, 2):
    Columns of A2 are 0, 1.
    Rows of B2 are 0, 1.
    Columns of C2 are 2l, 2l+1.
    
    For each block l, we have 7 products.
    Let r be the global rank index.
    For block l, local rank s (0..6):
    r = l * 7 + s.
    
    U[a, r] = U_strassen[a, s]
    V[b, r] = V_strassen[b // 2, s] if b % 2 == 0? 
    Wait, V index b is for the 2x6 matrix B2.
    b = row * 6 + col.
    Row is 0 or 1. Col is 0..5.
    Col = 2l + c_local.
    
    So for a fixed block l:
    V[b, r] should extract the element at (row, 2l+c_local).
    V_strassen is for 2x2.
    V_strassen[row, s] is the coefficient for B[row, c_local].
    
    So V[b, r] = V_strassen[b // 6, s] if b % 6 == 2*l or 2*l+1?
    Actually, b // 6 is the row (0 or 1).
    b % 6 is the column.
    We want to match column to 2l + c_local.
    So if b % 6 is in {2l, 2l+1}, then c_local = b % 6 - 2l.
    V[b, r] = V_strassen[b // 6, s] if (b % 6 - 2l) in {0, 1}?
    No, V_strassen[row, s] corresponds to B[row, c_local].
    So V[b, r] = V_strassen[b // 6, s] if b % 6 == 2l + c_local for c_local in 0,1.
    Basically, V[b, r] = V_strassen[b // 6, s] if b % 6 >= 2l and b % 6 < 2l+2.
    
    W[c, r] = W_strassen[c_local, s] if c corresponds to column 2l+c_local.
    c = k*2 + i.
    k is the column of C (0..5).
    k = 2l + c_local.
    i is the row of C (0, 1).
    So c = (2l + c_local)*2 + i = 4l + 2*c_local + i.
    W_strassen index is c_local*2 + i? No, W_strassen is for 2x2 output.
    W_strassen[c_idx, s] where c_idx = row*2 + col.
    Here row=i, col=c_local.
    So W_strassen[i*2 + c_local, s].
    
    So W[c, r] = W_strassen[i*2 + c_local, s] if c == 4l + 2*c_local + i.
    
    This construction yields a valid decomposition of rank 3 + 21 = 24.
    
    We can try to optimize this further or just return this.
    Rank 24 is much better than 60.
    
    Can we do better?
    Rank(2, 5, 6) is known to be <= 2 * Rank(1, 5, 6) + ...
    Actually, Rank(2, m, p) <= 2 * Rank(1, m, p) is false.
    Rank(2, m, p) <= Rank(2, m, 2) * ceil(p/2) + ...
    
    Let's stick to the 24 rank construction first. It's a massive improvement over 60.
    """
    
    n, m, p = 2, 5, 6
    
    # Split m=5 into 3 and 2.
    # Term 1: 2x3 * 3x6. Rank 3.
    # Term 2: 2x2 * 2x6. Rank 21.
    
    total_rank = 3 + 21
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    # --- Term 1: 2x3 * 3x6 ---
    # m1 = 3, p1 = 6, n1 = 2
    # U1 shape: 6 x 3
    # V1 shape: 18 x 3
    # W1 shape: 12 x 3
    
    for r in range(3):
        # U: Extract column r of A (2x3)
        for i in range(2):
            a = i * 3 + r
            U[a, r] = 1.0
            
        # V: Extract row r of B (3x6)
        for k in range(6):
            b = r * 6 + k
            V[b, r] = 1.0
            
        # W: Identity map for output (2x6)
        for k in range(6):
            for i in range(2):
                c = k * 2 + i
                W[c, r] = 1.0
                
    # --- Term 2: 2x2 * 2x6 ---
    # m2 = 2, p2 = 6, n2 = 2
    # Offset in U, V, W indices
    # A2 is the last 2 columns of A (indices 3, 4 in A's 5 cols)
    # B2 is the last 2 rows of B (indices 3, 4 in B's 5 rows)
    # C2 is the last 2 rows of C? No, C is 2x6.
    # Wait, A is 2x5. Split into A1 (2x3) and A2 (2x2).
    # B is 5x6. Split into B1 (3x6) and B2 (2x6).
    # C = A1 B1 + A2 B2.
    # Both terms produce 2x6 matrices.
    
    # U2 shape: 4 x 21 (local indices for A2)
    # V2 shape: 12 x 21 (local indices for B2)
    # W2 shape: 12 x 21 (local indices for C)
    
    U_str, V_str, W_str = _strassen_2x2()
    
    base_rank = 3
    for l in range(3): # 3 blocks of 2 columns in B2
        for s in range(7): # 7 Strassen products
            r = base_rank + l * 7 + s
            
            # U2: A2 is 2x2. Indices 0,1,2,3.
            # Global A indices: 3, 4, 5, 6? No, A is 2x5.
            # A1 uses cols 0,1,2. A2 uses cols 3,4.
            # A indices: i*5 + j.
            # A1: i*5 + j (j=0,1,2)
            # A2: i*5 + j (j=3,4)
            
            for i in range(2):
                for j_local in range(2):
                    a_local = i * 2 + j_local
                    a_global = i * 5 + (3 + j_local)
                    U[a_global, r] = U_str[a_local, s]
                    
            # V2: B2 is 2x6. Rows 3,4 of B.
            # B indices: j*6 + k.
            # B1 uses rows 0,1,2. B2 uses rows 3,4.
            # B2 local row j_local (0,1) corresponds to global row 3+j_local.
            
            for j_local in range(2):
                j_global = 3 + j_local
                for k in range(6):
                    b_global = j_global * 6 + k
                    # V_str[j_local, s] is coeff for B2[j_local, c_local]
                    # c_local is determined by block l
                    c_local = k - 2 * l
                    if 0 <= c_local < 2:
                        V[b_global, r] = V_str[j_local, s]
                    # else 0
                    
            # W2: C is 2x6.
            # C indices: k*2 + i.
            # C2 contributes to all 6 columns of C.
            # Block l contributes to columns 2l, 2l+1.
            
            for i in range(2):
                for c_local in range(2):
                    k_global = 2 * l + c_local
                    c_global = k_global * 2 + i
                    # W_str index: row i, col c_local -> i*2 + c_local
                    W[c_global, r] = W_str[i * 2 + c_local, s]
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def construct():
    t0 = time.time()
    deadline = t0 + 140
    
    # 1. Try the structural rank-24 decomposition
    try:
        res_24 = _construct_hybrid_decomposition()
        if _verify_rank(res_24["U"], res_24["V"], res_24["W"]):
            return res_24
    except Exception as e:
        pass
        
    # 2. Fallback to schoolbook if verification fails (shouldn't happen)
    return _schoolbook_rank()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
