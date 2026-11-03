import numpy as np
import time

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    n, m, p = DIMENSIONS  # 4, 4, 8

    # Strassen 2x2x2 factors (rank 7)
    # U2[i, j, r], V2[j, k, r], W2[k, i, r]
    # Such that sum_r U2[i,j,r] * V2[j,k,r] * W2[k,i,r] = delta_{i,k} * (A_{ij} B_{jk})? No.
    # The tensor T_{ijk} = 1 if i==k (for identity) is not right.
    # Matrix multiplication tensor T_{ijk} where C_{ik} = sum_j A_{ij} B_{jk}.
    # T_{i,j,k} is 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    # So T_{i,j,k} = 1 for all i,j,k? No.
    # The tensor is defined by T[i,j,k] = 1 if the index k in C matches i?
    # Let's check the convention: W[k*n+i, r].
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # Decomposition: C_{ik} = sum_r U_{ij,r} V_{jk,r} W_{ik,r}.
    # So we need sum_r U_{ij,r} V_{jk,r} W_{ik,r} = delta_{i,k}? No.
    # We need the sum to be 1 if the indices match the multiplication structure.
    # Actually, the tensor M_{i,j,k} = 1 if we are computing C_{ik} from A_{ij} and B_{jk}.
    # The condition is that the output index is (i,k).
    # So for a fixed r, U_{ij,r} V_{jk,r} W_{ik,r} should sum to 1 for the correct (i,k) pair?
    # No, the decomposition is:
    # M_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}  <-- Note index order in W.
    # The task says W[k*n+i, r]. So W is indexed by (k,i).
    # So we need: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 if the triple (i,j,k) is a valid term in A*B?
    # Wait, A*B has terms A[i,j]*B[j,k] contributing to C[i,k].
    # So for every (i,j,k), the product must be 1? No.
    # The tensor T[i,j,k] is 1 for all i,j,k? No.
    # The tensor T[i,j,k] represents the coefficient of A[i,j]B[j,k] in C[i,k].
    # This coefficient is 1 if the output index is i and k.
    # So T[i,j,k] = 1 for all i,j,k?
    # Let's check: C[i,k] = sum_j A[i,j] B[j,k].
    # If we define T[i,j,k] = 1 for all i,j,k, then sum_r ... = 1.
    # But we only want the term to appear in C[i,k].
    # The decomposition identity is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Where T[i,j,k] = 1 if the indices (i,j,k) form a valid path in the multiplication graph?
    # In matrix multiplication, every (i,j,k) triple is a valid path contributing to C[i,k].
    # So T[i,j,k] = 1 for ALL i,j,k.
    # Wait, if T[i,j,k]=1 for all, then we are decomposing the all-ones tensor?
    # No, the matrix multiplication tensor is NOT the all-ones tensor.
    # The tensor is T[i,j,k] = 1 if the "output" index matches the "input" indices appropriately.
    # For C[i,k] = sum_j A[i,j]B[j,k], the term A[i,j]B[j,k] contributes to C[i,k].
    # So for a fixed (i,k), we sum over j.
    # The tensor T is indexed by (i,j,k).
    # T[i,j,k] = 1.
    # Is that right?
    # If I take random A, B, then C[i,k] = sum_j A[i,j]B[j,k].
    # If I compute sum_r U[i,j,r]V[j,k,r]W[k,i,r] A[i,j]B[j,k], does it equal C?
    # Let's test r=0 in Strassen:
    # U[0,0,0]=1, U[1,1,0]=1.
    # V[0,0,0]=1, V[1,1,0]=1.
    # W[0,0,0]=1, W[1,1,0]=1.
    # Term for i=0,j=0,k=0: U[0,0,0]V[0,0,0]W[0,0,0] = 1.
    # Term for i=0,j=0,k=1: U[0,0,0]V[0,1,0]W[1,0,0] = 1*0*0 = 0.
    # Term for i=0,j=1,k=0: U[0,1,0]V[1,0,0]W[0,0,0] = 0*0*1 = 0.
    # Term for i=1,j=1,k=1: U[1,1,0]V[1,1,0]W[1,1,0] = 1.
    # So it picks out specific (i,j,k) combinations.
    # The sum over r must reconstruct the full multiplication.
    
    # The provided U2, V2, W2 in the prompt are standard Strassen factors.
    # I will reuse them.
    
    U2 = np.zeros((2, 2, 7), dtype=float)
    V2 = np.zeros((2, 2, 7), dtype=float)
    W2 = np.zeros((2, 2, 7), dtype=float)

    # r=0: (A00+A11)(B00+B11)
    U2[0, 0, 0] = 1; U2[1, 1, 0] = 1
    V2[0, 0, 0] = 1; V2[1, 1, 0] = 1
    W2[0, 0, 0] = 1; W2[1, 1, 0] = 1

    # r=1: (A10+A11)B00
    U2[1, 0, 1] = 1; U2[1, 1, 1] = 1
    V2[0, 0, 1] = 1
    W2[0, 1, 1] = 1; W2[1, 1, 1] = -1

    # r=2: A00(B01-B11)
    U2[0, 0, 2] = 1
    V2[0, 1, 2] = 1; V2[1, 1, 2] = -1
    W2[1, 0, 2] = 1; W2[1, 1, 2] = 1

    # r=3: A11(B10-B00)
    U2[1, 1, 3] = 1
    V2[1, 0, 3] = 1; V2[0, 0, 3] = -1
    W2[0, 0, 3] = 1; W2[0, 1, 3] = 1

    # r=4: (A00+A01)B11
    U2[0, 0, 4] = 1; U2[0, 1, 4] = 1
    V2[1, 1, 4] = 1
    W2[0, 0, 4] = -1; W2[1, 0, 4] = 1

    # r=5: (A10-A00)(B00+B01)
    U2[1, 0, 5] = 1; U2[0, 0, 5] = -1
    V2[0, 0, 5] = 1; V2[0, 1, 5] = 1
    W2[1, 1, 5] = 1

    # r=6: (A01-A11)(B10+B11)
    U2[0, 1, 6] = 1; U2[1, 1, 6] = -1
    V2[1, 0, 6] = 1; V2[1, 1, 6] = 1
    W2[0, 0, 6] = 1

    # Construct 4x4x4 factors using Kronecker product of Strassen
    # U4[i,j,R] = sum_{s,r} U2[i1,j1,s] U2[i0,j0,r] where R = 7s+r, i=2i1+i0, j=2j1+j0
    # This is equivalent to np.einsum or kron.
    
    # Reshape U2 to (2,2,7) -> (2, 2, 7)
    # We want U4 of shape (4, 4, 49)
    # Index mapping:
    # i = 2*i1 + i0
    # j = 2*j1 + j0
    # R = 7*s + r
    
    # We can use broadcasting.
    # U4[i1, i0, j1, j0, s, r] = U2[i1, j1, s] * U2[i0, j0, r]
    # Then reshape to (4, 4, 49).
    
    # Let's create the tensor in 6D first.
    # U2 is (2, 2, 7).
    # U4_raw[i1, i0, j1, j0, s, r] = U2[i1, j1, s] * U2[i0, j0, r]
    
    # To do this with numpy:
    # U2_a = U2[:, :, :, np.newaxis, np.newaxis, :] # (2, 2, 7, 1, 1, 7) -> indices i1, j1, s, i0, j0, r? No.
    # U2 is (i, j, r).
    # We need U2[i1, j1, s] and U2[i0, j0, r].
    
    U2_a = U2[:, :, :, np.newaxis, np.newaxis, :] # (2, 2, 7, 1, 1, 7) -> i1, j1, s, i0, j0, r? 
    # Wait, U2_a[i1, j1, s, i0, j0, r] = U2[i1, j1, s] * 1? No.
    # Let's just use loops for clarity and safety, but vectorized where possible.
    # The loop version is fast enough for 49 elements.
    
    U4 = np.zeros((4, 4, 49), dtype=float)
    V4 = np.zeros((4, 4, 49), dtype=float)
    W4 = np.zeros((4, 4, 49), dtype=float)
    
    for i1 in range(2):
        for j1 in range(2):
            for s in range(7):
                u1 = U2[i1, j1, s]
                v1 = V2[j1, k1, s] # Wait, V2 depends on j1, k1.
                # I need to loop over k1 for V4.
                # Let's stick to the explicit loops from the previous program but ensure correctness.
                pass

    # Re-implementing the loop logic cleanly.
    
    for i1 in range(2):
        for j1 in range(2):
            for i0 in range(2):
                for j0 in range(2):
                    i = 2 * i1 + i0
                    j = 2 * j1 + j0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            U4[i, j, R] = U2[i1, j1, s] * U2[i0, j0, r]

    for j1 in range(2):
        for k1 in range(2):
            for j0 in range(2):
                for k0 in range(2):
                    j = 2 * j1 + j0
                    k = 2 * k1 + k0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            V4[j, k, R] = V2[j1, k1, s] * V2[j0, k0, r]

    for k1 in range(2):
        for i1 in range(2):
            for k0 in range(2):
                for i0 in range(2):
                    k = 2 * k1 + k0
                    i = 2 * i1 + i0
                    for s in range(7):
                        for r in range(7):
                            R = 7 * s + r
                            W4[k, i, R] = W2[k1, i1, s] * W2[k0, i0, r]

    # Build 4x4x8 factors
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    # First half: k in 0..3
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * 4 + j, R] = U4[i, j, R]

    for j in range(4):
        for k in range(4):
            for R in range(49):
                V[j * 8 + k, R] = V4[j, k, R]

    for k in range(4):
        for i in range(4):
            for R in range(49):
                W[k * 4 + i, R] = W4[k, i, R]

    # Second half: k in 4..7, mapped to k' = k-4 in 0..3
    for i in range(4):
        for j in range(4):
            for R in range(49):
                U[i * 4 + j, 49 + R] = U4[i, j, R]

    for j in range(4):
        for kp in range(4):
            k = 4 + kp
            for R in range(49):
                V[j * 8 + k, 49 + R] = V4[j, kp, R]

    for kp in range(4):
        k = 4 + kp
        for i in range(4):
            for R in range(49):
                W[k * 4 + i, 49 + R] = W4[kp, i, R]

    # Verification
    if time.time() < deadline - 5.0:
        U3 = U.reshape(4, 4, 98)
        V3 = V.reshape(4, 8, 98)
        W3 = W.reshape(8, 4, 98)
        result = np.einsum('ijr,jkr,kir->ijk', U3, V3, W3)
        # The result should be 1 for all i,j,k?
        # No, result[i,j,k] = sum_r U[i,j,r]V[j,k,r]W[k,i,r]
        # This should equal 1 for all i,j,k?
        # Let's check.
        # If A is identity, B is identity, C is identity.
        # C[i,k] = delta_{i,k}.
        # The decomposition computes C[i,k] = sum_j (sum_r U[i,j,r]V[j,k,r]W[k,i,r]) A[i,j]B[j,k]?
        # No, the tensor T[i,j,k] = sum_r U[i,j,r]V[j,k,r]W[k,i,r].
        # We need T[i,j,k] = 1 for all i,j,k?
        # If T[i,j,k]=1, then sum_j A[i,j]B[j,k] = C[i,k]. This is correct.
        # So we need np.allclose(result, 1.0).
        
        if not np.allclose(result, 1.0, atol=1e-10):
            # Fallback to schoolbook
            rank_school = n * m * p
            U_s = np.zeros((n * m, rank_school))
            V_s = np.zeros((m * p, rank_school))
            W_s = np.zeros((p * n, rank_school))
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_s[i * m + j, col] = 1
                        V_s[j * p + k, col] = 1
                        W_s[k * n + i, col] = 1
                        col += 1
            return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
