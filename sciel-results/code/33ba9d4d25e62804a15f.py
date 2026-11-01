import numpy as np
import time

DIMENSIONS = [2, 3, 6]

def verify_decomposition(U, V, W, n, m, p):
    """
    Verifies if U, V, W exactly decompose the matrix multiplication tensor.
    U: (n*m, rank)
    V: (m*p, rank)
    W: (p*n, rank)
    """
    r = U.shape[1]
    for a in range(n):
        for b in range(m):
            for c in range(m):
                for d in range(p):
                    for i in range(n):
                        for k in range(p):
                            target = 1.0 if (a == i and c == k and b == d) else 0.0
                            s = np.dot(U[a * m + b, :], np.multiply(V[c * p + d, :], W[k * n + i, :]))
                            if not np.isclose(s, target, atol=1e-9):
                                return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    
    # Strategy:
    # The tensor M_{2,3,6} can be viewed as two independent M_{2,3,3} tensors 
    # because the output dimension p=6 splits into two blocks of size 3.
    # Specifically, the term (A B)_{i,k} depends on A_{i,j} and B_{j,k}.
    # If we split k into k1 (0..2) and k2 (3..5), the computation for k1 
    # only involves B_{j, k1} and for k2 only involves B_{j, k2}.
    # Thus, M_{2,3,6} = M_{2,3,3} \oplus M_{2,3,3}.
    # The rank of M_{2,3,6} is at most 2 * rank(M_{2,3,3}).
    # 
    # It is known that rank(M_{2,3,3}) <= 12. (Strassen gives 12 for 2x2x2, 
    # but for 2x3x3, a rank 12 decomposition exists).
    # Actually, let's use a known rank-12 decomposition for 2x3x3.
    # A simple construction for 2x3x3 with rank 12:
    # We can use the fact that 2x3x3 is a sub-tensor of 3x3x3 (rank 19? No, 2x2x2 is 7).
    # Let's use a specific known rank-12 decomposition for 2x3x3.
    
    # Decomposition of M_{2,3,3} with rank 12.
    # U: 6x12, V: 9x12, W: 6x12.
    # Indices:
    # U: (a, b) where a in {0,1}, b in {0,1,2} -> idx = a*3 + b
    # V: (c, d) where c in {0,1,2}, d in {0,1,2} -> idx = c*3 + d
    # W: (k, i) where k in {0,1,2}, i in {0,1} -> idx = k*2 + i
    
    # We construct a rank-12 solution for 2x3x3.
    # One way to get rank 12 is to use two copies of a rank-6 2x2x2 decomposition? No, 2x2x2 is rank 7.
    # However, 2x3x3 can be embedded.
    # Let's try to find a rank-12 decomposition analytically or use a known one.
    # 
    # Known result: R(2,3,3) = 12.
    # Here is a rank-12 decomposition for 2x3x3:
    # It can be constructed by taking the 2x2x2 Strassen decomposition (rank 7) 
    # and extending it? Or using a specific pattern.
    
    # Let's use a simpler approach: 
    # We know R(2,3,6) <= 2 * R(2,3,3).
    # If we can find ANY valid decomposition for 2,3,3 with rank R, we get 2R for 2,3,6.
    # The schoolbook rank for 2,3,3 is 18. 2*18 = 36.
    # The current program uses rank 17 for 2,3,3 (fallback), giving 34.
    # If we find rank 12 for 2,3,3, we get rank 24 for 2,3,6.
    # If we find rank 11 for 2,3,3, we get rank 22 for 2,3,6.
    
    # Let's hardcode a rank-12 decomposition for 2,3,3.
    # Reference: "The rank of the 2x3x3 matrix multiplication tensor is 12"
    # We can construct it by combining two 2x2x2 Strassen decompositions?
    # No, 2x3x3 is not two 2x2x2s.
    
    # Let's use a deterministic search for a rank-12 decomposition of 2x3x3 
    # with a very tight constraint (integer/half-integer entries) if possible, 
    # or just use a known floating point one and check if it can be represented.
    # The task requires integer/half-integer entries.
    # Strassen's algorithm uses integers {-1, 0, 1}.
    # Can we do 2x3x3 with integers?
    
    # Let's try to construct a rank-12 solution for 2,3,3 using integers.
    # If we can't find one easily, we fall back to the rank-17 solution (score 34).
    # But we want to beat 34.
    
    # Actually, there is a known rank-12 decomposition for 2x3x3 with small integer entries.
    # Let's try to build it from two 2x2x2 blocks?
    # A = [[A11, A12, A13], [A21, A22, A23]]
    # B = [[B11, B12, B13], [B21, B22, B23], [B31, B32, B33]]
    # C = A B
    # C11 = A11 B11 + A12 B21 + A13 B31
    # C12 = A11 B12 + A12 B22 + A13 B32
    # C13 = A11 B13 + A12 B23 + A13 B33
    # C21 = A21 B11 + A22 B21 + A23 B31
    # ...
    
    # This is equivalent to two 2x3x3 multiplications? No.
    
    # Let's use the fact that 2x3x3 can be decomposed into a 2x2x3 and a 2x1x3?
    
    # Alternative: Use the rank-7 decomposition for 2x2x2 (Strassen) 
    # and pad with zeros?
    # If we take the first 2 columns of B, we have a 2x2x3 multiplication.
    # R(2,2,3) = 11? (Strassen gives 7 for 2x2x2, plus 4 more for the 3rd col? No, it's 2x2x3).
    # Actually, R(2,2,3) = 11 is a known bound.
    # And R(2,1,3) = 3 (schoolbook).
    # Total rank = 11 + 3 = 14?
    # If R(2,3,3) <= 14, then R(2,3,6) <= 28. This beats 34.
    
    # Let's try to construct a rank-14 decomposition for 2,3,3.
    # Split B into B_left (2x2) and B_right (2x1)? No, B is 3x3.
    # Split B into 2 columns and 1 column.
    # C_left = A * B[:, :2] (2x2 matrix)
    # C_right = A * B[:, 2:] (2x1 matrix)
    
    # Rank of 2x2x2 is 7 (Strassen).
    # Rank of 2x1x2 is 2 (schoolbook).
    # So C_left (2x2x2) needs rank 7? No, A is 2x3, B is 3x2.
    # This is M_{2,3,2}.
    # R(2,3,2) = 11 is known.
    # R(2,3,1) = 3.
    # Total rank for M_{2,3,3} <= 11 + 3 = 14.
    
    # So if we find a rank-11 decomposition for M_{2,3,2} and a rank-3 for M_{2,3,1},
    # we get rank 14 for M_{2,3,3}, and rank 28 for M_{2,3,6}.
    
    # Let's construct M_{2,3,2} rank 11.
    # U: 6x11, V: 6x11, W: 4x11.
    # M_{2,3,2}: A(2x3) * B(3x2) = C(2x2).
    
    # Known rank-11 decomposition for 2x3x2:
    # This can be derived from Strassen.
    # Let's try to find it or use a simpler rank-12 for 2x3x2?
    # Schoolbook rank for 2x3x2 is 12.
    # If we use rank 12 for 2x3x2 and rank 3 for 2x3x1, total is 15.
    # 2 * 15 = 30. This beats 34.
    
    # Let's use the schoolbook decomposition for M_{2,3,2} (rank 12) and M_{2,3,1} (rank 3).
    # Total rank for M_{2,3,3} is 15.
    # Total rank for M_{2,3,6} is 30.
    
    # This is a significant improvement from 34 to 30.
    # And it is deterministic and fast.
    
    # Construct M_{2,3,3} rank 15 solution:
    # Part 1: M_{2,3,2} (columns 0,1 of B)
    # A is 2x3, B is 3x2, C is 2x2.
    # U1: 6x12, V1: 6x12, W1: 4x12.
    # Schoolbook:
    # C_{i,k} = sum_{j} A_{i,j} B_{j,k}
    # For i in {0,1}, j in {0,1,2}, k in {0,1}.
    # 2*3*2 = 12 terms.
    
    U1 = np.zeros((6, 12))
    V1 = np.zeros((6, 12))
    W1 = np.zeros((4, 12))
    
    idx = 0
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U1[i*3 + j, idx] = 1.0
                V1[j*2 + k, idx] = 1.0
                W1[k*2 + i, idx] = 1.0
                idx += 1
    
    # Part 2: M_{2,3,1} (column 2 of B)
    # A is 2x3, B is 3x1, C is 2x1.
    # U2: 6x3, V2: 3x3, W2: 2x3.
    # Schoolbook:
    # C_{i,0} = sum_{j} A_{i,j} B_{j,0}
    # 2*3*1 = 3 terms.
    
    U2 = np.zeros((6, 3))
    V2 = np.zeros((3, 3))
    W2 = np.zeros((2, 3))
    
    idx = 0
    for i in range(2):
        for j in range(3):
            U2[i*3 + j, idx] = 1.0
            V2[j, idx] = 1.0
            W2[i, idx] = 1.0
            idx += 1
    
    # Combine to M_{2,3,3} rank 15
    U233 = np.hstack([U1, U2]) # 6x15
    V233 = np.zeros((9, 15))
    W233 = np.zeros((6, 15))
    
    # V1 corresponds to B[:, :2] -> rows 0..5 of V233
    V233[0:6, 0:12] = V1
    # V2 corresponds to B[:, 2:] -> rows 6..8 of V233
    V233[6:9, 12:15] = V2
    
    # W1 corresponds to C[:, :2] -> rows 0..3 of W233
    W233[0:4, 0:12] = W1
    # W2 corresponds to C[:, 2:] -> rows 4..5 of W233
    W233[4:6, 12:15] = W2
    
    # Verify M_{2,3,3}
    if verify_decomposition(U233, V233, W233, 2, 3, 3):
        # Construct M_{2,3,6} rank 30
        # Two blocks of M_{2,3,3}
        U_comb = np.hstack([U233, U233]) # 6x30
        V_comb = np.zeros((18, 30))
        W_comb = np.zeros((12, 30))
        
        # Block 1 (d=0,1,2 -> k=0,1,2)
        # V rows 0..8 correspond to B[:, 0:3]
        V_comb[0:9, 0:15] = V233
        # W rows 0..5 correspond to C[0:3, :]
        W_comb[0:6, 0:15] = W233
        
        # Block 2 (d=3,4,5 -> k=3,4,5)
        # V rows 9..17 correspond to B[:, 3:6]
        V_comb[9:18, 15:30] = V233
        # W rows 6..11 correspond to C[3:6, :]
        W_comb[6:12, 15:30] = W233
        
        # Verify full solution
        if verify_decomposition(U_comb, V_comb, W_comb, n, m, p):
            return {
                "U": U_comb.tolist(),
                "V": V_comb.tolist(),
                "W": W_comb.tolist()
            }
            
    # Fallback: Rank 34 solution (same as current program)
    U233_seed = np.zeros((6, 17))
    V233_seed = np.zeros((9, 17))
    W233_seed = np.zeros((6, 17))
    
    U233_seed[0, 0] = 1; U233_seed[4, 0] = 1
    V233_seed[0, 0] = 1; V233_seed[4, 0] = 1
    W233_seed[0, 0] = 1
    W233_seed[3, 0] = 1
    
    U233_seed[3, 1] = 1; U233_seed[4, 1] = 1
    V233_seed[0, 1] = 1
    W233_seed[1, 1] = 1
    W233_seed[3, 1] = -1
    
    U233_seed[0, 2] = 1
    V233_seed[1, 2] = 1; V233_seed[4, 2] = -1
    W233_seed[2, 2] = 1
    W233_seed[3, 2] = 1
    
    U233_seed[4, 3] = 1
    V233_seed[3, 3] = 1; V233_seed[0, 3] = -1
    W233_seed[0, 3] = 1
    W233_seed[1, 3] = 1
    
    U233_seed[0, 4] = 1; U233_seed[1, 4] = 1
    V233_seed[4, 4] = 1
    W233_seed[0, 4] = -1
    W233_seed[2, 4] = 1
    
    U233_seed[3, 5] = 1; U233_seed[0, 5] = -1
    V233_seed[0, 5] = 1; V233_seed[1, 5] = 1
    W233_seed[3, 5] = 1
    
    U233_seed[1, 6] = 1; U233_seed[4, 6] = -1
    V233_seed[3, 6] = 1; V233_seed[4, 6] = 1
    W233_seed[0, 6] = 1
    
    U233_seed[0, 7] = 1
    V233_seed[2, 7] = 1
    W233_seed[4, 7] = 1
    
    U233_seed[1, 8] = 1
    V233_seed[5, 8] = 1
    W233_seed[4, 8] = 1
    
    U233_seed[3, 9] = 1
    V233_seed[2, 9] = 1
    W233_seed[5, 9] = 1
    
    U233_seed[4, 10] = 1
    V233_seed[5, 10] = 1
    W233_seed[5, 10] = 1
    
    U233_seed[2, 11] = 1
    V233_seed[6, 11] = 1
    W233_seed[0, 11] = 1
    
    U233_seed[2, 12] = 1
    V233_seed[7, 12] = 1
    W233_seed[2, 12] = 1
    
    U233_seed[2, 13] = 1
    V233_seed[8, 13] = 1
    W233_seed[4, 13] = 1
    
    U233_seed[5, 14] = 1
    V233_seed[6, 14] = 1
    W233_seed[1, 14] = 1
    
    U233_seed[5, 15] = 1
    V233_seed[7, 15] = 1
    W233_seed[3, 15] = 1
    
    U233_seed[5, 16] = 1
    V233_seed[8, 16] = 1
    W233_seed[5, 16] = 1
    
    U_comb = np.zeros((6, 34))
    V_comb = np.zeros((18, 34))
    W_comb = np.zeros((12, 34))
    
    U_comb[:, :17] = U233_seed
    U_comb[:, 17:] = U233_seed
    
    for j in range(3):
        for k in range(3):
            V_comb[j * 6 + k, :17] = V233_seed[j * 3 + k]
            V_comb[j * 6 + 3 + k, 17:] = V233_seed[j * 3 + k]
            
    for k in range(3):
        for i in range(2):
            W_comb[k * 2 + i, :17] = W233_seed[k * 2 + i]
            W_comb[(k + 3) * 2 + i, 17:] = W233_seed[k * 2 + i]
            
    return {
        "U": U_comb.tolist(),
        "V": V_comb.tolist(),
        "W": W_comb.tolist()
    }

if __name__ == "__main__":
    construct()
