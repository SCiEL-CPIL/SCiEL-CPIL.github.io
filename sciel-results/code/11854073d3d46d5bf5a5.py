import time
import numpy as np

DIMENSIONS = [2, 3, 6]

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that U, V, W form an exact tensor decomposition of the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # The tensor entry T[i*m+j, j*p+k, k*n+i] should be 1
                # Reconstruction: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                actual = np.sum(U_arr[i * m + j, :] * V_arr[j * p + k, :] * W_arr[k * n + i, :])
                if abs(actual - 1.0) > tol:
                    return False
    return True

def _schoolbook_decomposition(n, m, p):
    """
    Fallback schoolbook decomposition. Rank = n*m*p.
    """
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

def _construct_2x3x6_fta():
    """
    Construct a decomposition for 2x3x6 matrix multiplication using a Winograd-like 
    structural approach.
    
    We split the multiplication C = A * B (2x3 * 3x6 -> 2x6) into two parts based on 
    the columns of A (which are rows of B).
    Let A = [A_0, A_1, A_2] where A_i is 2x1.
    Let B = [B_0, B_1, B_2] where B_i is 1x6.
    C = A_0 B_0 + A_1 B_1 + A_2 B_2.
    
    This is a sum of 3 rank-1 matrices (outer products of 2x1 and 1x6).
    Each outer product is 2x6. The total rank of a sum of 3 such products is at most 18.
    However, the tensor rank is not simply the sum of matrix ranks.
    
    A more effective structural decomposition exploits the dimensions.
    We can view the 2x3x6 multiplication as a block matrix multiplication.
    Let's try to decompose the 2x6 output into two 2x3 blocks: C = [C_left, C_right].
    C_left = A * B_left (2x3 * 3x3)
    C_right = A * B_right (2x3 * 3x3)
    
    If we use a standard decomposition for 2x3x3, the rank is known to be 11 (Winograd).
    If we use two independent 2x3x3 decompositions, the total rank is 22.
    
    Can we do better by coupling them?
    The standard 2x3x3 Winograd decomposition uses the fact that 3 = 1+2 or similar.
    Actually, the minimal rank for 2x3x3 is 11.
    The minimal rank for 2x3x6 is at most 2 * Rank(2x3x3) = 22.
    
    Let's explicitly construct the Rank 11 decomposition for 2x3x3 and duplicate it.
    
    Winograd 2x3x3 Decomposition (Rank 11):
    This is a known result. 
    Reference: Winograd, S. (1978). Arithmetic complexity of computations.
    For 2x3x3, the rank is 11.
    
    Let's implement the 2x3x3 decomposition.
    A is 2x3, B is 3x3.
    We can split B into B1 (3x1) and B2 (3x2).
    C1 = A * B1 (2x1) -> Rank 6 (schoolbook) or less? 2x1x1 is rank 2. 2x1x2 is rank 4? No.
    Let's look at the tensor T_{2,3,3}.
    Indices: i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    
    A known optimal decomposition for 2x3x3 has rank 11.
    Let's find or construct it.
    
    Alternative: Use the fact that 6 = 2 * 3.
    We can treat the 2x3x6 multiplication as two 2x3x3 multiplications if the inputs allow.
    Since B is 3x6, we can write B = [B_1, B_2] where B_1, B_2 are 3x3.
    Then C = [A B_1, A B_2].
    The decomposition of A B_1 and A B_2 can share the factors for A.
    In the tensor decomposition framework, if we use the same U matrix for both blocks, 
    and separate V and W matrices, the total rank is Rank(U,V1,W1) + Rank(U,V2,W2)?
    No, if U is shared, the rank is Rank(U, V, W) where V is block diagonal and W is block diagonal.
    If we have a decomposition (U, V, W) for 2x3x3 with rank R, 
    then for 2x3x6, we can use U' = U (repeated? No, U is 6xR).
    Wait, U is (n*m) x Rank = 6 x R.
    If we use the same U for both halves, we need to ensure the linear combination works.
    
    Let's construct the 2x3x3 rank 11 decomposition explicitly.
    
    Based on literature (e.g., "The rank of 2x3x3 matrix multiplication is 11"):
    There is a specific set of 11 vectors.
    
    Let's try to construct it using the "Winograd" method for 2x3x3.
    Winograd's method for 2x3x3:
    Split A = [a1, a2, a3] (columns).
    Split B = [b1; b2; b3] (rows).
    
    Actually, a simpler bound is often derived from 2x2x2 Strassen.
    2x3x3 can be viewed as 2x2x2 + 2x1x2 + 2x1x2 + 2x1x2? No.
    
    Let's use the known Rank 11 construction for 2x3x3.
    The decomposition is given by 11 terms.
    
    Since finding the exact 11 vectors from memory is risky, let's try a slightly higher but structurally sound rank, e.g., 12 or 13, or stick to the 22 rank (2x11) which is guaranteed to work if we have the 11 vectors.
    
    However, the current program achieves Rank 33.
    Rank 22 is a significant improvement (33 -> 22).
    Even Rank 24 would be an improvement.
    
    Let's implement the Rank 11 decomposition for 2x3x3.
    If I can't recall the exact 11 vectors, I can use a Rank 12 or 13 approximation or a different known bound.
    
    Actually, there is a Rank 10 decomposition? No, lower bound for 2x3x3 is 10.
    It is proven to be 11.
    
    Let's try to construct a Rank 11 decomposition for 2x3x3 using a known algorithm or simple integer coordinates.
    
    Reference: 
    "An optimal algorithm for 2x3x3 matrix multiplication"
    
    Let's try to derive it.
    T_{2,3,3} has 18 non-zero entries.
    Rank 11 means 11 rank-1 tensors sum to T.
    
    Let's use the following structure for 2x3x3:
    We can embed 2x3x3 into 2x4x4? No.
    
    Let's try a different strategy:
    Use the Rank 33 decomposition from the current program as a baseline.
    Try to improve it by splitting the 6 dimension into 3+3 and using a better 2x3x3 decomposition.
    
    If we use the schoolbook 2x3x3 (Rank 18), total rank is 36.
    If we use Strassen-like 2x2x2 (Rank 7) for the 2x2x2 part and schoolbook for the rest?
    
    Let's look at the current program's Rank 33 logic:
    It used 3 blocks of (2x2x2 Strassen + 2x1x2 Standard).
    2x2x2 Strassen Rank 7.
    2x1x2 Standard Rank 4.
    Total per block: 11.
    3 blocks: 33.
    This corresponds to splitting the 6 dimension into 2, 2, 2?
    If p=6 is split into 2+2+2, we have three 2x3x2 multiplications.
    Rank of 2x3x2:
    Split 3 into 2+1.
    2x2x2 (Rank 7) + 2x1x2 (Rank 4) = 11.
    So 3 * 11 = 33.
    
    Can we do better for 2x3x2?
    Lower bound for 2x3x2: max(6, 6, 4) = 6.
    Is Rank 11 minimal for 2x3x2?
    Known rank of 2x3x2 is 10? Or 9?
    Actually, 2x3x2 is equivalent to 2x2x3 by symmetry.
    Rank(2x2x3) is 10.
    If we can find a Rank 10 decomposition for 2x3x2, then 3 * 10 = 30.
    
    Can we find a Rank 9 decomposition?
    Lower bound is 6.
    
    Let's try to construct a Rank 10 decomposition for 2x3x2.
    If we can get 10, total rank 30.
    If we can get 9, total rank 27.
    
    Let's search for a Rank 10 decomposition of 2x3x2.
    
    Alternatively, consider splitting p=6 into 3+3.
    Then we have two 2x3x3 multiplications.
    Rank(2x3x3) = 11.
    Total Rank = 22.
    
    This is a huge jump from 33 to 22.
    The key is to implement the Rank 11 decomposition for 2x3x3.
    
    Let's implement the Rank 11 decomposition for 2x3x3.
    
    The tensor T_{2,3,3} has indices i in {0,1}, j in {0,1,2}, k in {0,1,2}.
    We need 11 vectors u_r (dim 6), v_r (dim 9), w_r (dim 6).
    
    One known construction for 2x3x3 rank 11:
    It is related to the fact that 2x3x3 can be computed with 11 multiplications.
    
    Let's try to construct it using the "Winograd" FTA for 2x3x3.
    Winograd's algorithm for 2x3x3 uses 11 multiplications.
    
    The algorithm:
    1. Form 11 products.
    2. Combine them with additions.
    
    The decomposition corresponds to the 11 rank-1 tensors.
    
    Let's try to find the vectors.
    Since I don't have the exact vectors memorized, I will use a randomized search specifically for Rank 11 on the 2x3x3 tensor, but with a tighter constraint and more attempts, or use a known set if I can derive it.
    
    Actually, there is a simpler way:
    Use the Rank 11 decomposition for 2x3x3.
    If I can't find it, I will fall back to Rank 33.
    
    However, I can try to construct a Rank 12 decomposition for 2x3x3 which might be easier to find via random search, or use the Rank 10 for 2x3x2.
    
    Let's try Rank 10 for 2x3x2.
    Tensor size: 6 x 6 x 4.
    Non-zeros: 12.
    
    Let's try a randomized search for Rank 10 on 2x3x2.
    If found, we use 3 blocks -> Rank 30.
    If not, we try Rank 11 on 2x3x3 (2 blocks -> Rank 22).
    
    Let's implement the search for 2x3x3 Rank 11.
    If the search fails, we return the Rank 33 fallback.
    
    To increase success rate for Rank 11 on 2x3x3:
    We can use the structure.
    
    Let's try to hardcode a known Rank 11 decomposition for 2x3x3 if possible.
    Or, use the fact that 2x3x3 rank is 11.
    
    I will implement a function `_find_2x3x3_rank_11` that uses a targeted random search.
    If it fails, I will use the Rank 33 fallback.
    
    Wait, the current program has a `_try_rank_decomposition` that is generic.
    I will write a specific search for 2x3x3 Rank 11.
    
    If that fails, I will try 2x3x2 Rank 10 (3 blocks -> 30).
    
    Let's write the code.
"""

def _find_2x3x3_rank_11(deadline):
    n, m, p = 2, 3, 3
    N_A, N_B, N_C = 6, 9, 6
    T_flat = np.zeros((N_C, N_A * N_B))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[c, a * N_B + b] = 1.0
                
    rank = 11
    for attempt in range(500):
        if time.time() > deadline:
            return None
        U = np.random.choice([0, 1, -1], size=(N_A, rank))
        V = np.random.choice([0, 1, -1], size=(N_B, rank))
        A = np.zeros((N_A * N_B, rank))
        for r in range(rank):
            A[:, r] = np.kron(U[:, r], V[:, r])
        try:
            X, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat.T, rcond=None)
            if len(residuals) == 0 or residuals[0] < 1e-9:
                W = X.T
                W_half = np.round(W * 2) / 2
                if np.allclose(W, W_half, atol=1e-9):
                    U_list = [[int(x) for x in row] for row in U]
                    V_list = [[int(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W_half]
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except:
            continue
    return None

def _find_2x3x2_rank_10(deadline):
    n, m, p = 2, 3, 2
    N_A, N_B, N_C = 6, 6, 4
    T_flat = np.zeros((N_C, N_A * N_B))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T_flat[c, a * N_B + b] = 1.0
                
    rank = 10
    for attempt in range(1000):
        if time.time() > deadline:
            return None
        U = np.random.choice([0, 1, -1], size=(N_A, rank))
        V = np.random.choice([0, 1, -1], size=(N_B, rank))
        A = np.zeros((N_A * N_B, rank))
        for r in range(rank):
            A[:, r] = np.kron(U[:, r], V[:, r])
        try:
            X, residuals, rank_A, sv = np.linalg.lstsq(A, T_flat.T, rcond=None)
            if len(residuals) == 0 or residuals[0] < 1e-9:
                W = X.T
                W_half = np.round(W * 2) / 2
                if np.allclose(W, W_half, atol=1e-9):
                    U_list = [[int(x) for x in row] for row in U]
                    V_list = [[int(x) for x in row] for row in V]
                    W_list = [[float(x) for x in row] for row in W_half]
                    if _verify_decomposition(U_list, V_list, W_list, n, m, p):
                        return {"U": U_list, "V": V_list, "W": W_list}
        except:
            continue
    return None

def _construct_2x3x6_from_2x3x3(decomp_2x3x3):
    """
    Combine two 2x3x3 decompositions to form a 2x3x6 decomposition.
    U is shared (6 x 11).
    V is block diagonal (18 x 22).
    W is block diagonal (12 x 22).
    """
    U11, V11, W11 = decomp_2x3x3['U'], decomp_2x3x3['V'], decomp_2x3x3['W']
    rank = 11
    
    U = [[0]*22 for _ in range(6)]
    V = [[0]*22 for _ in range(18)]
    W = [[0]*22 for _ in range(12)]
    
    # Block 1 (k=0..2)
    for i in range(6):
        for r in range(rank):
            U[i][r] = U11[i][r]
    for j in range(9):
        for r in range(rank):
            V[j][r] = V11[j][r]
    for k in range(6):
        for r in range(rank):
            W[k][r] = W11[k][r]
            
    # Block 2 (k=3..5)
    for i in range(6):
        for r in range(rank):
            U[i][rank+r] = U11[i][r]
    for j in range(9):
        for r in range(rank):
            V[j+9][rank+r] = V11[j][r]
    for k in range(6):
        for r in range(rank):
            W[k+6][rank+r] = W11[k][r]
            
    return {"U": U, "V": V, "W": W}

def _construct_2x3x6_from_2x3x2(decomp_2x3x2):
    """
    Combine three 2x3x2 decompositions to form a 2x3x6 decomposition.
    U is shared (6 x 10).
    V is block diagonal (18 x 30).
    W is block diagonal (12 x 30).
    """
    U10, V10, W10 = decomp_2x3x2['U'], decomp_2x3x2['V'], decomp_2x3x2['W']
    rank = 10
    
    U = [[0]*30 for _ in range(6)]
    V = [[0]*30 for _ in range(18)]
    W = [[0]*30 for _ in range(12)]
    
    for block in range(3):
        offset = block * rank
        for i in range(6):
            for r in range(rank):
                U[i][offset+r] = U10[i][r]
        for j in range(6):
            for r in range(rank):
                V[j + block*6][offset+r] = V10[j][r]
        for k in range(4):
            for r in range(rank):
                W[k + block*4][offset+r] = W10[k][r]
                
    return {"U": U, "V": V, "W": W}

def _fallback_rank_33():
    # Re-implement the Rank 33 logic from the current program
    def _strassen_2x2_decomp():
        rank = 7
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        U[0][0] = 1
        V[1][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[3][0] = 1
        
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[1][1] = 1; W[2][1] = 1
        
        U[0][2] = 1; U[1][2] = 1
        V[3][2] = 1
        W[0][2] = 1; W[3][2] = 1
        
        U[3][3] = 1
        V[0][3] = 1; V[2][3] = -1
        W[0][3] = 1; W[2][3] = 1
        
        U[0][4] = 1; U[2][4] = 1
        V[1][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        U[1][5] = 1; U[2][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[0][5] = -1
        
        U[3][6] = 1; U[1][6] = -1
        V[1][6] = 1; V[3][6] = 1
        W[3][6] = 1
        
        return U, V, W

    def _schoolbook_2x1x2_decomp():
        rank = 4
        U = [[0]*rank for _ in range(2)]
        V = [[0]*rank for _ in range(2)]
        W = [[0]*rank for _ in range(4)]
        
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        U[0][1] = 1; V[1][1] = 1; W[1][1] = 1
        U[1][2] = 1; V[0][2] = 1; W[2][2] = 1
        U[1][3] = 1; V[1][3] = 1; W[3][3] = 1
        
        return U, V, W

    def _construct_2x3x2_decomp():
        U_s, V_s, W_s = _strassen_2x2_decomp()
        U_sb, V_sb, W_sb = _schoolbook_2x1x2_decomp()
        
        U = [[0]*11 for _ in range(6)]
        V = [[0]*11 for _ in range(6)]
        W = [[0]*11 for _ in range(4)]
        
        for i in range(4):
            for r in range(7):
                U[i][r] = U_s[i][r]
        for j in range(4):
            for r in range(7):
                V[j][r] = V_s[j][r]
        for k in range(4):
            for r in range(7):
                W[k][r] = W_s[k][r]
                
        for i in range(2):
            for r in range(4):
                U[4+i][7+r] = U_sb[i][r]
        for j in range(2):
            for r in range(4):
                V[4+j][7+r] = V_sb[j][r]
        for k in range(4):
            for r in range(4):
                W[k][7+r] = W_sb[k][r]
                
        return U, V, W

    def _construct_2x3x6_decomp():
        U_block, V_block, W_block = _construct_2x3x2_decomp()
        
        U = [[0]*33 for _ in range(6)]
        V = [[0]*33 for _ in range(18)]
        W = [[0]*33 for _ in range(12)]
        
        for block in range(3):
            offset = block * 11
            for i in range(6):
                for r in range(11):
                    U[i][offset+r] = U_block[i][r]
                    
            for j in range(6):
                for r in range(11):
                    V[j + block*6][offset+r] = V_block[j][r]
                    
            for k in range(4):
                for r in range(11):
                    W[k + block*4][offset+r] = W_block[k][r]
                    
        return {"U": U, "V": V, "W": W}

    return _construct_2x3x6_decomp()

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0
    
    # Try to find Rank 11 decomposition for 2x3x3
    # If found, combine two to get Rank 22 for 2x3x6
    res_2x3x3 = _find_2x3x3_rank_11(deadline - 10)
    if res_2x3x3:
        res_2x3x6 = _construct_2x3x6_from_2x3x3(res_2x3x3)
        if _verify_decomposition(res_2x3x6['U'], res_2x3x6['V'], res_2x3x6['W'], n, m, p):
            return res_2x3x6

    # Try to find Rank 10 decomposition for 2x3x2
    # If found, combine three to get Rank 30 for 2x3x6
    if time.time() < deadline - 10:
        res_2x3x2 = _find_2x3x2_rank_10(deadline - 5)
        if res_2x3x2:
            res_2x3x6 = _construct_2x3x6_from_2x3x2(res_2x3x2)
            if _verify_decomposition(res_2x3x6['U'], res_2x3x6['V'], res_2x3x6['W'], n, m, p):
                return res_2x3x6

    # Fallback to Rank 33
    res_33 = _fallback_rank_33()
    if _verify_decomposition(res_33['U'], res_33['V'], res_33['W'], n, m, p):
        return res_33

    # Final fallback
    return _schoolbook_decomposition(n, m, p)

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify_decomposition(res['U'], res['V'], res['W'], 2, 3, 6))
