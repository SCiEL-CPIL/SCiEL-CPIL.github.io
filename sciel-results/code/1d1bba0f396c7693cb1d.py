import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    # 1. Define Strassen for 2x2x2
    # U, V, W are lists of (n*m, 7), (m*p, 7), (p*n, 7)
    # Indices: U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Strassen factors for 2x2x2
    # r=0: (A00+A11)(B00+B11)
    # r=1: (A10+A11)B00
    # r=2: A00(B10-B11)
    # r=3: A11(B01-B00)
    # r=4: (A00+A01)B11
    # r=5: (A10-A00)(B00+B10)
    # r=6: (A01-A11)(B01+B11)
    
    U_222 = [
        [1, 0, 0, 0, 0, 0, 0],  # A00
        [0, 0, 1, 0, 0, 0, 0],  # A10
        [0, 0, 0, 0, 1, 0, 0],  # A01
        [1, 1, 0, 1, 0, 0, 0]   # A11
    ]
    # V indices: j*p+k -> j*2+k
    # B00=0, B10=1, B01=2, B11=3
    V_222 = [
        [1, 1, 0, 0],  # B00
        [1, 0, 0, 0],  # B10
        [0, 0, 1, 0],  # B01
        [1, 0, 0, 1]   # B11
    ]
    # W indices: k*n+i -> k*2+i
    # C00=0, C10=1, C01=2, C11=3
    W_222 = [
        [1, 0, 0, 0],  # C00
        [0, 1, 0, 0],  # C10
        [0, 0, 1, 0],  # C01
        [1, 0, 0, 1]   # C11
    ]
    # Note: The standard Strassen decomposition is:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B10)
    # M7 = (A01-A11)(B01+B11)
    # C00 = M1 + M4 - M5 + M7
    # C10 = M2 + M3
    # C01 = M5 + M6
    # C11 = M1 - M2 + M3 + M6
    
    # Let's build the matrices properly.
    # U rows: 0:A00, 1:A10, 2:A01, 3:A11
    # V rows: 0:B00, 1:B10, 2:B01, 3:B11
    # W rows: 0:C00, 1:C10, 2:C01, 3:C11
    
    U_s = np.array([
        [1, 0, 1, 0, 1, 0, 0], # A00
        [0, 1, 0, 0, 0, 1, 0], # A10
        [0, 0, 0, 0, 1, 0, 1], # A01
        [1, 1, 0, 1, 0, 0, 0]  # A11
    ])
    V_s = np.array([
        [1, 1, 0, 1, 0, 1, 1], # B00
        [0, 0, 1, 0, 0, 1, 0], # B10
        [1, 0, -1, -1, 0, 0, 1], # B01
        [1, 0, 0, 0, 1, 0, 1]  # B11
    ])
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1], # C00
        [0, 1, 1, 0, 0, 0, 0], # C10
        [0, 0, 0, 0, 1, 1, 0], # C01
        [1, -1, 1, 0, 0, 1, 0]  # C11
    ])
    
    # 2. Define 2x1x2 decomposition (Rank 3)
    # n=2, m=1, p=2
    # U shape (2*1, 3) = (2, 3)
    # V shape (1*2, 3) = (2, 3)
    # W shape (2*2, 3) = (4, 3)
    
    # C00 = A0 B0
    # C10 = A1 B0
    # C01 = A0 B1
    # C11 = A1 B1
    
    U_212 = np.array([
        [1, 1, 0], # A0
        [0, 0, 1]  # A1
    ])
    V_212 = np.array([
        [1, 0, 1], # B0
        [0, 1, 0]  # B1
    ])
    W_212 = np.array([
        [1, 0, 0], # C00 (k=0, i=0)
        [0, 0, 1], # C10 (k=0, i=1)
        [0, 1, 0], # C01 (k=1, i=0)
        [0, 0, 0]  # C11 (k=1, i=1) -- Wait, this is 0.
    ])
    # This rank 3 construction is flawed for C11.
    # C11 = A1 B1.
    # We need a term for A1 B1.
    # Rank 3 is not enough for 2x1x2?
    # Actually, Rank(2x1x2) is 3?
    # Let's use Rank 4 (schoolbook) to be safe if 3 doesn't work easily.
    # Or use Rank 3 with:
    # r0: A0 B0 -> C00
    # r1: A1 B0 -> C10
    # r2: A0 B1 -> C01
    # r3: A1 B1 -> C11
    # This is rank 4.
    
    # Is there a rank 3?
    # C = A B.
    # C00 = A0 B0
    # C10 = A1 B0
    # C01 = A0 B1
    # C11 = A1 B1
    # If we compute X = A0+A1, Y = A0-A1.
    # It is known that Rank(M_{2,1,2}) = 3.
    # Decomposition:
    # r0: U=[1,0], V=[1,0], W=[1,0,0,0] -> A0 B0 at C00
    # r1: U=[0,1], V=[1,0], W=[0,1,0,0] -> A1 B0 at C10
    # r2: U=[1,1], V=[0,1], W=[0,0,1,1] -> (A0+A1) B1 at C01, C11
    # Check C01: A0 B1 + A1 B1. We want A0 B1.
    # So we need to subtract A1 B1.
    # This requires another term.
    
    # Let's stick to Rank 4 for 2x1x2 to ensure correctness, or find the correct Rank 3.
    # Actually, 2x1x2 is equivalent to 1x2x2 transposed?
    # Rank is the same.
    # Let's use Rank 4 for safety.
    # Total rank: 7 * 4 * 2 = 56.
    # 56 < 72.
    
    U_212 = np.eye(2, 4) # 2x4
    V_212 = np.eye(2, 4) # 2x4
    W_212 = np.eye(4, 4) # 4x4
    # Wait, U is 2x4, V is 2x4, W is 4x4.
    # r=0: A0, B0, C00
    # r=1: A1, B0, C10
    # r=2: A0, B1, C01
    # r=3: A1, B1, C11
    
    U_212 = np.array([
        [1, 0, 1, 0],
        [0, 1, 0, 1]
    ])
    V_212 = np.array([
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    W_212 = np.array([
        [1, 0, 0, 0], # C00
        [0, 1, 0, 0], # C10
        [0, 0, 1, 0], # C01
        [0, 0, 0, 1]  # C11
    ])
    
    # 3. Combine 2x2x2 and 2x1x2 to get 4x2x4
    # n=4, m=2, p=4
    # U_424 shape (8, 28)
    # V_424 shape (8, 28)
    # W_424 shape (16, 28)
    
    R1 = 7
    R2 = 4
    R_total_424 = R1 * R2
    
    U_424 = np.zeros((8, R_total_424))
    V_424 = np.zeros((8, R_total_424))
    W_424 = np.zeros((16, R_total_424))
    
    for r1 in range(R1):
        for r2 in range(R2):
            r = r1 * R2 + r2
            # U: Kronecker product of U_s[:, r1] and U_212[:, r2]
            # U_s is 4x7, U_212 is 2x4.
            # Result size 8.
            U_424[:, r] = np.kron(U_s[:, r1], U_212[:, r2])
            
            # V: V_s is 4x7, V_212 is 2x4.
            # Result size 8.
            V_424[:, r] = np.kron(V_s[:, r1], V_212[:, r2])
            
            # W: W_s is 4x7, W_212 is 4x4.
            # Result size 16.
            W_424[:, r] = np.kron(W_s[:, r1], W_212[:, r2])
            
    # 4. Combine 4x2x4 and 1x2x2 to get 4x4x8
    # n=4, m=4, p=8
    # Split 2: 1x2x2.
    # Rank 2?
    # n=1, m=2, p=2.
    # C0 = A0 B00 + A1 B10
    # C1 = A0 B01 + A1 B11
    # Rank 2.
    
    U_122 = np.array([
        [1, 1], # A0
        [1, -1] # A1 -- Wait, this is not schoolbook.
    ])
    # Schoolbook for 1x2x2:
    # r0: A0, B0, C0
    # r1: A1, B0, C1? No.
    # C0 = A0 B00 + A1 B10.
    # C1 = A0 B01 + A1 B11.
    
    U_122 = np.array([
        [1, 1], # A0
        [0, 1]  # A1
    ])
    V_122 = np.array([
        [1, 0], # B00
        [1, 0], # B10 -- No, B is 2x2.
        [0, 1], # B01
        [0, 1]  # B11
    ])
    W_122 = np.array([
        [1, 0], # C0
        [0, 1]  # C1
    ])
    # Check:
    # r0: A0 * (B00 + B10) * C0?
    # V row 0: [1, 0] -> B00? No, V is 2x2.
    # V[0] = [1, 0] means B00=1, B10=0?
    # The term is U[i,r] V[j,r] W[k,r].
    # Sum over j.
    # C_k = sum_j sum_r U[i,r] V[j,r] W[k,r] A[i,j] B[j,k].
    # For n=1, i=0.
    # C_k = sum_j A[j] (sum_r V[j,r] W[k,r]).
    # We want C_0 = A[0] B[0,0] + A[1] B[1,0].
    # C_1 = A[0] B[0,1] + A[1] B[1,1].
    
    # r=0: V[0,0]=1, V[1,0]=0. W[0,0]=1, W[1,0]=0.
    # Term: A[0] B[0,0] * 1.
    # r=1: V[0,1]=0, V[1,1]=1. W[0,1]=0, W[1,1]=1.
    # Term: A[1] B[1,1] * 1.
    # This gives A0 B00 + A1 B11.
    # We need A0 B01 and A1 B10.
    
    # So Rank 2 is not enough for 1x2x2?
    # Rank(1x2x2) = 2?
    # It is known that Rank(M_{1,m,p}) = p.
    # So Rank 2.
    # My construction was wrong.
    
    U_122 = np.array([
        [1, 1],
        [0, 1]
    ])
    V_122 = np.array([
        [1, 0],
        [0, 1],
        [0, 0],
        [0, 0]
    ])
    # This is not working.
    
    # Let's use Rank 4 for 1x2x2 to be safe.
    # Total rank: 28 * 4 = 112.
    # 112 > 72.
    
    # We need a better 1x2x2.
    # Actually, 1x2x2 is just a vector-matrix product.
    # It can be done with rank 2.
    # C0 = A0 B00 + A1 B10.
    # C1 = A0 B01 + A1 B11.
    
    # r0: U=[1,0], V=[1,1,0,0], W=[1,0]
    # r1: U=[0,1], V=[0,0,1,1], W=[0,1]
    
    U_122 = np.array([
        [1, 0],
        [0, 1]
    ])
    V_122 = np.array([
        [1, 0],
        [1, 0],
        [0, 1],
        [0, 1]
    ])
    W_122 = np.array([
        [1, 0],
        [0, 1]
    ])
    
    # Check:
    # C0 = sum_j A[j] (V[j,0] * 1).
    # V[0,0]=1, V[1,0]=1.
    # C0 = A0 + A1.
    # We want A0 B00 + A1 B10.
    # The B values are not in V.
    # V is fixed.
    # So this decomposition is invalid for arbitrary B.
    
    # The tensor rank of 1x2x2 is 2.
    # But the decomposition must work for all A, B.
    # The only way is if V encodes the structure.
    # But V is constant.
    # So Rank(1x2x2) is actually 4?
    # No, Rank(M_{1,m,p}) = p.
    # Wait, if p=2, rank is 2.
    # How?
    # C_k = sum_j A_j B_{j,k}.
    # This is a linear map from A to C for fixed B.
    # The tensor is sum_{j,k} e_0 \otimes e_j \otimes e_k.
    # This is the identity matrix I_2.
    # The rank of the identity matrix as a tensor is 2.
    # But the decomposition must be bilinear in A and B.
    # If we fix B, it's linear in A.
    # The rank is the dimension of the space of linear maps?
    # No, the tensor rank is the minimum number of rank-1 tensors.
    # T = sum_{j,k} e_0 \otimes e_j \otimes e_k.
    # This is e_0 \otimes I_2.
    # The rank of e_0 \otimes I_2 is 2.
    # So Rank 2.
    
    # Why did my construction fail?
    # Because I tried to put B in V.
    # V is part of the tensor.
    # The term is U[i,r] V[j,r] W[k,r].
    # For n=1, i=0.
    # C_k = sum_j sum_r U[0,r] V[j,r] W[k,r] A[j] B[j,k].
    # This must equal sum_j A[j] B[j,k].
    # So sum_r U[0,r] V[j,r] W[k,r] = delta_{j,k}?
    # No, it must be 1 for all j,k?
    # No, it must be such that sum_j (sum_r ...) A[j] B[j,k] = sum_j A[j] B[j,k].
    # This implies sum_r U[0,r] V[j,r] W[k,r] = delta_{j,k} is NOT required.
    # It requires that the map (A, B) -> C is correct.
    
    # If we use r=0,1.
    # r0: U=1, V=[1,0], W=[1,0].
    # r1: U=1, V=[0,1], W=[0,1].
    # C0 = A0 B00 * 1 + A1 B10 * 0 = A0 B00.
    # C1 = A0 B01 * 0 + A1 B11 * 1 = A1 B11.
    # Missing A1 B10 and A0 B01.
    
    # So Rank 2 is not enough?
    # Rank(M_{1,2,2}) = 4?
    # If so, total rank 112.
    
    # Let's return the schoolbook 72 to be safe if we can't find a better one quickly.
    # But 56 was a candidate.
    
    # Let's use the 4x2x4 rank 28 and 1x2x2 rank 4.
    # Total 112.
    
    # Better: Use 4x4x8 schoolbook?
    # 4*4*8 = 128.
    
    # Let's just return the schoolbook 72.
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    construct()
