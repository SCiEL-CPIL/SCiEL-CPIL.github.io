import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    DEADLINE = start_time + 140
    
    n, m, p = DIMENSIONS  # 2, 4, 8
    
    # Rank-32 decomposition for 2x4 * 4x8 = 2x8
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Split: C = A0*B0 + A1*B1
    # where A0 = A[:,0:2] (2x2), A1 = A[:,2:4] (2x2)
    #       B0 = B[0:2,:] (2x8), B1 = B[2:4,:] (2x8)
    # A0*B0: 2x2 * 2x8, rank 16 (schoolbook, optimal by slice bound)
    # A1*B1: 2x2 * 2x8, rank 16
    # Total rank: 32
    
    total_rank = 32
    U = np.zeros((n * m, total_rank), dtype=int)  # shape (8, 32)
    V = np.zeros((m * p, total_rank), dtype=int)  # shape (32, 32)
    W = np.zeros((p * n, total_rank), dtype=int)  # shape (16, 32)
    
    # W index convention: W[k*n + i, r] where k is column index, i is row index
    # C[i, k] -> W[k*2 + i, r]
    
    r = 0
    # A0 * B0: A0[i, a] * B0[b, k] -> C[i, k]
    # A0[i, a]: U index = i*4 + a (i in {0,1}, a in {0,1})
    # B0[b, k]: V index = b*8 + k (b in {0,1}, k in {0..7})
    # C[i, k]: W index = k*2 + i
    for a in range(2):
        for b in range(2):
            for i in range(2):
                for k in range(8):
                    # Only terms where a == b contribute (since B0[b,k] and A0[i,a])
                    # Actually all combinations are valid rank-1 terms
                    # But we can be smarter: A0[i,a]*B0[b,k] only contributes to C[i,k]
                    # The standard rank-16 decomposition:
                    # For each (i,k): C[i,k] = A0[i,0]*B0[0,k] + A0[i,1]*B0[1,k]
                    # So rank-1 terms: (A0[i,a], B0[a,k]) for a=0,1
                    pass
    
    # Simpler: iterate over the 16 terms for A0*B0
    # Term (i, a, k): A0[i,a] * B0[a,k] -> C[i,k]
    # But that's 2*2*8 = 32 terms, not 16.
    # The rank-16 decomposition: for each (i,k), C[i,k] = sum_a A0[i,a]*B0[a,k]
    # This is 16 outputs, each a sum of 2 products = 32 rank-1 terms total? No.
    # Rank of 2x2*2x8: each output C[i,k] depends on A0[i,0],A0[i,1],B0[0,k],B0[1,k]
    # The tensor is sum_{i,k,a} A[i,a] B[a,k] C[i,k]
    # = sum_{i,a} A[i,a] * (sum_k B[a,k] C[i,k])
    # Each term A[i,a] * (B[a,0]*C[i,0] + ... + B[a,7]*C[i,7]) is not rank-1.
    # Rank-1 terms: A[i,a] * B[a,k] * C[i,k] for each (i,a,k).
    # That's 2*2*8 = 32 terms. But slice bound says rank >= 16.
    # Can we do 16? 
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Define: for k=0..7, define S_k = B[0,k] + B[1,k], D_k = B[0,k] - B[1,k]
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #         = (A[i,0]+A[i,1])/2 * S_k + (A[i,0]-A[i,1])/2 * D_k
    # So: C[i,k] = 0.5*(A[i,0]+A[i,1])*S_k + 0.5*(A[i,0]-A[i,1])*D_k
    # Rank-1 terms: (A[i,0]+A[i,1]) * S_k -> C[i,k] and (A[i,0]-A[i,1]) * D_k -> C[i,k]
    # That's 2*8 + 2*8 = 32 terms still. Hmm.
    
    # Actually the tensor rank of 2x2*2xn is 2n. Let me verify with n=1:
    # 2x2*2x1: C[i] = A[i,0]B[0] + A[i,1]B[1]. 2 outputs, 4 inputs.
    # Rank: C[0] = A[0,0]B[0] + A[0,1]B[1], C[1] = A[1,0]B[0] + A[1,1]B[1]
    # 4 rank-1 terms. Slice bound: max(4, 2, 2) = 4. So rank = 4 = 2n.
    # For n=8: rank = 16? Let me check n=2:
    # 2x2*2x2: C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]. 4 outputs.
    # Rank-1 terms: A[i,a]*B[a,k] for i=0,1; a=0,1; k=0,1. That's 8 terms.
    # But 2x2*2x2 has rank 7 (Strassen)! So my "2n" formula is wrong.
    # 2x2*2x2 is the standard 2x2 matrix multiplication, rank 7.
    # So 2x2*2xn for n=2 has rank 7, not 8.
    
    # Let me reconsider. 2x2*2xn: this is multiplying a 2x2 matrix by a 2xn matrix.
    # C[i,k] = A[i,0]B[0,k] + A[i,1]B[1,k]
    # For n=2: standard 2x2*2x2, rank 7.
    # For n=8: can we use Strassen-like tricks?
    # C[:,k] = B[:,k]^T * A^T for each k. A^T is 2x2 fixed.
    # This is like: for each k, compute a 2x1 * 2x2 product... no.
    # C[:,k] = B[:,k] (as row) * A^T. B[:,k] is 2x1, A^T is 2x2, result is 2x1.
    # So C[:,k] = (B[0,k], B[1,k]) * [[A[0,0], A[1,0]], [A[0,1], A[1,1]]]
    # C[0,k] = B[0,k]*A[0,0] + B[1,k]*A[1,0]
    # C[1,k] = B[0,k]*A[0,1] + B[1,k]*A[1,1]
    
    # For all k simultaneously, this is: C (2x8) = A^T^T * B^T... 
    # C = A * B where A is 2x2, B is 2x8.
    # This is a 2x2 * 2x8 multiplication.
    # We can use Strassen on the 2x2 part by padding B to 2x2 four times? No.
    # Alternative: C = A * [B0 B1 B2 B3] where Bj are 2x2.
    # C = [A*B0 A*B1 A*B2 A*B3]
    # Each A*Bj is 2x2*2x2, rank 7. Total: 4*7 = 28.
    # But can we share Strassen products across j?
    # Strassen for A*Bj: M1=(A00+A11)(Bj00+Bj11), etc.
    # The A-part is shared! M1_A = A00+A11, M2_A = A10+A11, etc.
    # M1_j = (A00+A11)*(Bj00+Bj11) -- different B for each j
    # So the products are: (A00+A11)*(Bj00+Bj11) for j=0..3: 4 products
    # (A10+A11)*(Bj00) for j=0..3: 4 products
    # A00*(Bj01-Bj11) for j=0..3: 4 products
    # A11*(Bj10-Bj00) for j=0..3: 4 products
    # (A00+A01)*(Bj11) for j=0..3: 4 products
    # (A10-A00)*(Bj00+Bj01) for j=0..3: 4 products
    # (A01-A11)*(Bj10+Bj11) for j=0..3: 4 products
    # Total: 28 products. Same as 4*7.
    
    # So 2x2*2x8 has rank at most 28. And at least... slice bound: max(4, 16, 16) = 16.
    # Can we do better than 28? Probably not easily.
    
    # So total for 2x4*4x8: 2 * 28 = 56. Same as before!
    
    # Hmm, so my earlier analysis was wrong. 2x2*2x8 has rank 28, not 16.
    # Let me just go with the rank-56 Strassen solution.
    
    # Actually wait - let me reconsider. Maybe I can do better by not splitting into two 2x2*2x8.
    # C[i,k] = sum_{j=0}^{3} A[i,j] B[j,k]
    # = A[i,0]B[0,k] + A[i,1]B[1,k] + A[i,2]B[2,k] + A[i,3]B[3,k]
    
    # What if I use a 4-point transform on the j-index?
    # Define: U0 = A[i,0]+A[i,1]+A[i,2]+A[i,3]
    #         U1 = A[i,0]-A[i,1]+A[i,2]-A[i,3]
    #         U2 = A[i,0]+A[i,1]-A[i,2]-A[i,3]
    #         U3 = A[i,0]-A[i,1]-A[i,2]+A[i,3]
    # Similarly for B: V0[k] = B[0,k]+B[1,k]+B[2,k]+B[3,k], etc.
    # Then C[i,k] = (1/4)(U0*V0 + U1*V1 + U2*V2 + U3*V3)
    # This uses 4 products per (i,k) pair = 4*16 = 64. Worse.
    
    # I'll stick with rank 56.
    
    # Reconstruct the rank-56 Strassen solution
    U = np.zeros((n * m, 56), dtype=int)
    V = np.zeros((m * p, 56), dtype=int)
    W = np.zeros((p * n, 56), dtype=int)
    
    # Strassen 2x2 rank-7
    U_local = np.zeros((4, 7), dtype=int)
    V_local = np.zeros((4, 7), dtype=int)
    W_local = np.zeros((4, 7), dtype=int)
    
    U_local[0,0]=1; U_local[3,0]=1
    V_local[0,0]=1; V_local[3,0]=1
    W_local[0,0]=1; W_local[3,0]=1
    
    U_local[2,1]=1; U_local[3,1]=1
    V_local[0,1]=1
    W_local[2,1]=1; W_local[3,1]=-1
    
    U_local[0,2]=1
    V_local[1,2]=1; V_local[3,2]=-1
    W_local[1,2]=1; W_local[3,2]=1
    
    U_local[3,3]=1
    V_local[2,3]=1; V_local[0,3]=-1
    W_local[0,3]=1; W_local[2,3]=1
    
    U_local[0,4]=1; U_local[1,4]=1
    V_local[3,4]=1
    W_local[0,4]=-1; W_local[1,4]=1
    
    U_local[2,5]=1; U_local[0,5]=-1
    V_local[0,5]=1; V_local[1,5]=1
    W_local[3,5]=1
    
    U_local[1,6]=1; U_local[3,6]=-1
    V_local[2,6]=1; V_local[3,6]=1
    W_local[0,6]=1
    
    for j_group in range(2):
        for k_group in range(4):
            base_r = (j_group * 4 + k_group) * 7
            a_map = [
                0 * 4 + j_group * 2,
                0 * 4 + j_group * 2 + 1,
                1 * 4 + j_group * 2,
                1 * 4 + j_group * 2 + 1
            ]
            b_map = [
                (j_group * 2) * 8 + k_group * 2,
                (j_group * 2) * 8 + k_group * 2 + 1,
                (j_group * 2 + 1) * 8 + k_group * 2,
                (j_group * 2 + 1) * 8 + k_group * 2 + 1
            ]
            c_map = [
                (k_group * 2) * 2 + 0,
                (k_group * 2 + 1) * 2 + 0,
                (k_group * 2) * 2 + 1,
                (k_group * 2 + 1) * 2 + 1
            ]
            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_local[la, l]
                    if val != 0:
                        U[a_map[la], r] += val
                for lb in range(4):
                    val = V_local[lb, l]
                    if val != 0:
                        V[b_map[lb], r] += val
                for lc in range(4):
                    val = W_local[lc, l]
                    if val != 0:
                        W[c_map[lc], r] += val
    
    if time.time() > DEADLINE:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
