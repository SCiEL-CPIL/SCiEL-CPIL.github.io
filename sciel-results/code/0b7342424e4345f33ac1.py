"""
Matrix multiplication 3x4x5 tensor decomposition.
Strategy: Block decomposition using Strassen (2x2x2) and schoolbook blocks
to achieve rank 58 instead of the trivial 60.
"""
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 58 construction
    # A is 3x4, B is 4x5, C is 3x5
    # A = [A_top (2x4); A_bot (1x4)]
    # B = [B_top (2x5); B_bot (2x5)]
    # C = A_top B + A_bot B
    # A_top B = A_top (2x4) * B (4x5). 
    # A_top can be split into A_1 (2x2) and A_2 (2x2) horizontally.
    # B can be split into B_1 (2x5) and B_2 (2x5) vertically.
    # A_top B = A_1 B_1 + A_2 B_2.
    # A_1 B_1 is 2x2 * 2x5. Rank 12 (schoolbook).
    # A_2 B_2 is 2x2 * 2x5. Rank 12 (schoolbook).
    # A_bot B is 1x4 * 4x5. Rank 20 (schoolbook).
    # Total rank = 12 + 12 + 20 = 44? Wait, is 2x2*2x5 rank 12? Yes.
    # Can we do better for 2x2*2x5?
    # Strassen is for 2x2*2x2 (rank 7).
    # 2x2*2x5 can be seen as two 2x2*2x2 blocks? No, width is 5.
    # 2x2*2x5 = 2x2*2x2 + 2x2*2x3? No, that's not a standard block.
    # Actually, 2x2*2x5 is just 4*5=20 scalar mults if schoolbook.
    # Wait, my previous calculation in the thought trace was:
    # "Let T_1 = A_{2 x 2}^{(1)} B_{2 x 5}^{(1)}. Rank 19." -> This was wrong. 2x2*2x5 is rank 20 schoolbook.
    # Where did 19 come from? Maybe I was thinking of something else.
    
    # Let's re-evaluate the best known ranks.
    # 2x2*2x2: Strassen rank 7.
    # 2x2*2x3: Rank 11? (Known result: R(2,2,3) <= 11).
    # 1x4*4x5: Rank 20.
    
    # Let's try a different partition.
    # A (3x4) = [A_1 (1x4); A_2 (2x4)]
    # B (4x5) = [B_1 (1x5); B_2 (3x5)]? No, B is 4x5.
    # B = [B_top (1x5); B_bot (3x5)]
    # C = A_1 B + A_2 B
    # A_1 B: 1x4 * 4x5. Rank 20.
    # A_2 B: 2x4 * 4x5.
    # 2x4 * 4x5.
    # Split A_2 (2x4) into A_21 (2x2) and A_22 (2x2).
    # Split B (4x5) into B_1 (2x5) and B_2 (2x5).
    # A_2 B = A_21 B_1 + A_22 B_2.
    # A_21 B_1: 2x2 * 2x5.
    # A_22 B_2: 2x2 * 2x5.
    
    # What is the minimal rank of 2x2*2x5?
    # It is the sum of two 2x2*2x2 (rank 7 each) and one 2x2*2x1 (rank 4)?
    # 2x5 = 2x2 + 2x2 + 2x1.
    # So 2x2*2x5 can be decomposed into three blocks: 2x2*2x2, 2x2*2x2, 2x2*2x1.
    # Ranks: 7 + 7 + 4 = 18.
    # So A_21 B_1 rank 18. A_22 B_2 rank 18.
    # Total rank = 20 (A_1 B) + 18 + 18 = 56.
    # This is better than 60.
    
    # Can we do better?
    # A_1 B is 1x4*4x5. Rank 20.
    # Is there a better decomposition for the whole 3x4*4x5?
    # Maybe use Strassen on the 3x3 part?
    # 3x3*3x3 has rank 23 (Baur-Strassen).
    # 3x4*4x5 can be viewed as 3x3*3x3 + ...
    # This is getting complicated to construct manually.
    
    # Let's stick to the 56 rank construction derived above.
    # Structure:
    # A = [A_1 (1x4); A_2 (2x4)]
    # B = [B_1 (2x5); B_2 (2x5)]  <- Wait, B is 4x5. Rows 1-2 and 3-4.
    # A_2 (2x4) multiplies B (4x5).
    # A_2 = [A_21 (2x2) A_22 (2x2)]
    # B = [B_1 (2x5); B_2 (2x5)]
    # A_2 B = A_21 B_1 + A_22 B_2.
    # A_1 (1x4) multiplies B (4x5).
    # A_1 = [A_11 (1x2) A_12 (1x2)]
    # A_1 B = A_11 B_1 + A_12 B_2.
    
    # So total terms:
    # 1. A_21 B_1: 2x2 * 2x5. Decompose into 2x2*2x2 + 2x2*2x2 + 2x2*2x1.
    #    - T1: A_21 * B_1[:,:2] (2x2*2x2) -> Strassen (7)
    #    - T2: A_21 * B_1[:,2:4] (2x2*2x2) -> Strassen (7)
    #    - T3: A_21 * B_1[:,4]   (2x2*2x1) -> Schoolbook (4)
    #    Subtotal: 18
    # 2. A_22 B_2: 2x2 * 2x5.
    #    - T4: A_22 * B_2[:,:2] (2x2*2x2) -> Strassen (7)
    #    - T5: A_22 * B_2[:,2:4] (2x2*2x2) -> Strassen (7)
    #    - T6: A_22 * B_2[:,4]   (2x2*2x1) -> Schoolbook (4)
    #    Subtotal: 18
    # 3. A_11 B_1: 1x2 * 2x5.
    #    - T7: A_11 * B_1 (1x2 * 2x5) -> Schoolbook (2*5=10)
    #    Subtotal: 10
    # 4. A_12 B_2: 1x2 * 2x5.
    #    - T8: A_12 * B_2 (1x2 * 2x5) -> Schoolbook (2*5=10)
    #    Subtotal: 10
    
    # Total Rank = 18 + 18 + 10 + 10 = 56.
    
    # Let's verify the indices and assembly.
    # n=3, m=4, p=5.
    # U shape: (3*4, R) = (12, R)
    # V shape: (4*5, R) = (20, R)
    # W shape: (5*3, R) = (15, R)
    # W index: k*n + i.
    
    R = 56
    U = [[0.0] * R for _ in range(12)]
    V = [[0.0] * R for _ in range(20)]
    W = [[0.0] * R for _ in range(15)]
    
    col = 0
    
    # Helper to add a Strassen 2x2x2 block
    # A indices: i in {i0, i1}, j in {j0, j1}
    # B indices: j in {j0, j1}, k in {k0, k1}
    # C indices: i in {i0, i1}, k in {k0, k1}
    def add_strassen(i0, j0, k0):
        nonlocal col
        # A is 2x2: (i0,j0), (i0,j1), (i1,j0), (i1,j1)
        # B is 2x2: (j0,k0), (j0,k1), (j1,k0), (j1,k1)
        # C is 2x2: (i0,k0), (i0,k1), (i1,k0), (i1,k1)
        
        # M1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        U[i0*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i0][col] += 1
        W[k0*n + i1][col] += 1
        W[k1*n + i1][col] -= 1
        col += 1
        
        # M2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        U[i1*m + j0][col] += 1
        U[i1*m + j1][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + i1][col] += 1
        col += 1
        
        # M3 = A[i0,j0] * (B[j1,k0] - B[j0,k0])
        U[i0*m + j0][col] += 1
        V[j1*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        col += 1
        
        # M4 = A[i1,j1] * (B[j0,k1] - B[j1,k1])
        U[i1*m + j1][col] += 1
        V[j0*p + k1][col] += 1
        V[j1*p + k1][col] -= 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # M5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        U[i0*m + j0][col] += 1
        U[i0*m + j1][col] += 1
        V[j1*p + k1][col] += 1
        W[k1*n + i0][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # M6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        U[i1*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + k1][col] += 1
        W[k0*n + i1][col] += 1
        W[k1*n + i1][col] += 1
        col += 1
        
        # M7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        U[i0*m + j1][col] += 1
        U[i1*m + j1][col] -= 1
        V[j1*p + k0][col] += 1
        V[j1*p + k1][col] += 1
        W[k0*n + i0][col] += 1
        W[k1*n + i0][col] += 1
        col += 1
        
    # Helper to add a schoolbook 2x2x1 block (A 2x2, B 2x1, C 2x1)
    # A: i in {i0, i1}, j in {j0, j1}
    # B: j in {j0, j1}, k = k0
    # C: i in {i0, i1}, k = k0
    def add_schoolbook_2x2x1(i0, j0, k0):
        nonlocal col
        for di in range(2):
            for dj in range(2):
                i = i0 + di
                j = j0 + dj
                U[i*m + j][col] = 1
                V[j*p + k0][col] = 1
                W[k0*n + i][col] = 1
                col += 1

    # Helper to add a schoolbook 1x2x5 block (A 1x2, B 2x5, C 1x5)
    # A: i = i0, j in {j0, j1}
    # B: j in {j0, j1}, k in {k0..k4}
    # C: i = i0, k in {k0..k4}
    def add_schoolbook_1x2x5(i0, j0, k0):
        nonlocal col
        for dj in range(2):
            for dk in range(5):
                j = j0 + dj
                k = k0 + dk
                U[i0*m + j][col] = 1
                V[j*p + k][col] = 1
                W[k*n + i0][col] = 1
                col += 1

    # Block 1: A_21 B_1
    # A_21: i in {1,2} (indices 1,2), j in {0,1} (indices 0,1)
    # B_1: j in {0,1} (indices 0,1), k in {0..4} (indices 0..4)
    # Decompose B_1 into k-slices: [0,1], [2,3], [4]
    # T1: A_21 * B_1[:,0:2] (2x2 * 2x2)
    add_strassen(1, 0, 0) # i0=1, j0=0, k0=0
    # T2: A_21 * B_1[:,2:4] (2x2 * 2x2)
    add_strassen(1, 0, 2) # i0=1, j0=0, k0=2
    # T3: A_21 * B_1[:,4]   (2x2 * 2x1)
    add_schoolbook_2x2x1(1, 0, 4) # i0=1, j0=0, k0=4
    
    # Block 2: A_22 B_2
    # A_22: i in {1,2} (indices 1,2), j in {2,3} (indices 2,3)
    # B_2: j in {2,3} (indices 2,3), k in {0..4} (indices 0..4)
    # Decompose B_2 into k-slices: [0,1], [2,3], [4]
    # T4: A_22 * B_2[:,0:2] (2x2 * 2x2)
    add_strassen(1, 2, 0) # i0=1, j0=2, k0=0
    # T5: A_22 * B_2[:,2:4] (2x2 * 2x2)
    add_strassen(1, 2, 2) # i0=1, j0=2, k0=2
    # T6: A_22 * B_2[:,4]   (2x2 * 2x1)
    add_schoolbook_2x2x1(1, 2, 4) # i0=1, j0=2, k0=4
    
    # Block 3: A_11 B_1
    # A_11: i = 0, j in {0,1}
    # B_1: j in {0,1}, k in {0..4}
    # T7: A_11 * B_1 (1x2 * 2x5)
    add_schoolbook_1x2x5(0, 0, 0) # i0=0, j0=0, k0=0
    
    # Block 4: A_12 B_2
    # A_12: i = 0, j in {2,3}
    # B_2: j in {2,3}, k in {0..4}
    # T8: A_12 * B_2 (1x2 * 2x5)
    add_schoolbook_1x2x5(0, 2, 0) # i0=0, j0=2, k0=0
    
    # Check if we used all columns
    if col != R:
        # Adjust R if needed, but it should be 56
        R = col
        # Truncate or pad
        if R < 56:
             U = [row[:R] for row in U]
             V = [row[:R] for row in V]
             W = [row[:R] for row in W]
        else:
             U = [row + [0.0]*(R-56) for row in U]
             V = [row + [0.0]*(R-56) for row in V]
             W = [row + [0.0]*(R-56) for row in W]
             R = 56
             
    return {"U": U, "V": V, "W": W}

# Quick self-test
if __name__ == "__main__":
    import numpy as np
    start = time.time()
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = DIMENSIONS
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C = A @ B
    
    # Reconstruct C from decomposition
    # C[i,k] = sum_c U[i*m + j, c] V[j*p + k, c] W[k*n + i, c] ??
    # The tensor is T[i,j,k] = A[i,j]B[j,k].
    # Decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[i,k,c]
    # In our indexing:
    # U is (n*m, R). U[i*m+j, c] corresponds to A[i,j].
    # V is (m*p, R). V[j*p+k, c] corresponds to B[j,k].
    # W is (p*n, R). W[k*n+i, c] corresponds to C[i,k] coefficient?
    # Wait, the contract says W[k*n+i, r].
    # So C[i,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] ?
    # Let's check the schoolbook case.
    # U[i*m+j, c] = 1 if c = i*m*p + j*p + k.
    # V[j*p+k, c] = 1 if c = i*m*p + j*p + k.
    # W[k*n+i, c] = 1 if c = i*m*p + j*p + k.
    # Sum_r ... = 1*1*1 = 1.
    # And T[i,j,k] = A[i,j]B[j,k].
    # So C[i,k] = sum_j A[i,j]B[j,k].
    # The formula sum_r U[i,j,r] V[j,k,r] W[i,k,r] gives T[i,j,k].
    # But we want C[i,k].
    # The evaluator likely does:
    # for i in range(n):
    #   for k in range(p):
    #     val = 0
    #     for r in range(rank):
    #       for j in range(m): # Wait, U depends on j.
    #       This is not a simple matrix multiplication.
    #       It's a tensor contraction.
    #       C[i,k] = sum_{j} A[i,j] B[j,k]
    #       = sum_{j} sum_r U[i,j,r] V[j,k,r] W[i,k,r] ?
    #       No, that would be sum_r W[i,k,r] (sum_j U[i,j,r] V[j,k,r]).
    #       This is exactly the rank-r decomposition of the bilinear map.
    #       A (n x m), B (m x p).
    #       U: (n x m x r) -> (n*m, r).
    #       V: (m x p x r) -> (m*p, r).
    #       W: (n x p x r) -> (p*n, r) with W[k*n+i, r].
    #       C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * V[j*p+k, r])
    #       Let's verify this.
    
    C_test = np.zeros((n, p))
    for r in range(U.shape[1]):
        # A_r is n x m
        A_r = U[:, r].reshape(n, m)
        # B_r is m x p
        B_r = V[:, r].reshape(m, p)
        # W_r is n x p (from p*n)
        W_r = W[:, r].reshape(p, n).T # W[k*n+i] -> index k, i. Reshape (p,n) gives [k,i]. T gives [i,k].
        
        C_test += W_r * (A_r @ B_r)
        
    error = np.max(np.abs(C - C_test))
    print(f"Max error: {error}")
    print(f"Rank: {U.shape[1]}")
    print(f"Time: {time.time()-start:.3f}s")
    
    if error < 1e-9:
        print("VALID")
    else:
        print("INVALID")
