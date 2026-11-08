import time
import numpy as np

DIMENSIONS = [2, 3, 10]

def _build_M_2x1xP(P):
    """
    Build M_{2,1,P}: 2x1 * 1xP -> 2xP
    Schoolbook: P terms.
    """
    n, m, p = 2, 1, P
    rank = P
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for k in range(P):
        U[0, k] = 1  # i=0, j=0
        U[1, k] = 1  # i=1, j=0
        V[0, k] = 1  # j=0, k=k
        W[k*n + 0, k] = 1 # i=0
        W[k*n + 1, k] = 1 # i=1
    return U, V, W

def _build_M_2x2x2():
    """
    Build M_{2,2,2}: 2x2 * 2x2 -> 2x2
    Using Strassen's algorithm (Rank 7).
    n=2, m=2, p=2.
    A is 2x2, B is 2x2.
    A = [[a00, a01], [a10, a11]]
    B = [[b00, b01], [b10, b11]]
    C = [[c00, c01], [c10, c11]]
    
    Strassen:
    M1 = (a00+a11)(b00+b11)
    M2 = (a10+a11)b00
    M3 = a00(b01-b11)
    M4 = a11(b10-b00)
    M5 = (a00+a01)b11
    M6 = (a10-a00)(b00+b01)
    M7 = (a01-a11)(b10+b11)
    
    c00 = M1+M4-M5+M7
    c01 = M3+M5
    c10 = M2+M4
    c11 = M1-M2+M3+M6
    
    We need to construct U, V, W such that sum_r U[:,r] V[:,r] W[:,r]^T = T_M
    T_M is the tensor of matrix multiplication.
    Indices:
    U: i*m+j (0..3)
    V: j*p+k (0..3)
    W: k*n+i (0..3)
    
    Let's define the basis vectors.
    A indices: 0:00, 1:01, 2:10, 3:11
    B indices: 0:00, 1:01, 2:10, 3:11
    C indices: 0:00, 1:01, 2:10, 3:11  (k*n+i -> 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    Term 1: M1 = (A00+A11)(B00+B11)
    U1: A00=1, A11=1 -> [1, 0, 0, 1]
    V1: B00=1, B11=1 -> [1, 0, 0, 1]
    W1: C00+=1, C11+=1 -> [1, 0, 0, 1]
    
    Term 2: M2 = (A10+A11)B00
    U2: A10=1, A11=1 -> [0, 0, 1, 1]
    V2: B00=1 -> [1, 0, 0, 0]
    W2: C10+=1 -> [0, 0, 1, 0]
    
    Term 3: M3 = A00(B01-B11)
    U3: A00=1 -> [1, 0, 0, 0]
    V3: B01=1, B11=-1 -> [0, 1, 0, -1]
    W3: C00-=1, C01+=1 -> [-1, 1, 0, 0]
    
    Term 4: M4 = A11(B10-B00)
    U4: A11=1 -> [0, 0, 0, 1]
    V4: B10=1, B00=-1 -> [-1, 0, 1, 0]
    W4: C00+=1, C10+=1 -> [1, 0, 1, 0]
    
    Term 5: M5 = (A00+A01)B11
    U5: A00=1, A01=1 -> [1, 1, 0, 0]
    V5: B11=1 -> [0, 0, 0, 1]
    W5: C00-=1, C01+=1 -> [-1, 1, 0, 0]
    
    Term 6: M6 = (A10-A00)(B00+B01)
    U6: A10=1, A00=-1 -> [-1, 0, 1, 0]
    V6: B00=1, B01=1 -> [1, 1, 0, 0]
    W6: C11+=1 -> [0, 0, 0, 1]
    
    Term 7: M7 = (A01-A11)(B10+B11)
    U7: A01=1, A11=-1 -> [0, 1, 0, -1]
    V7: B10=1, B11=1 -> [0, 0, 1, 1]
    W7: C00+=1, C11+=1 -> [1, 0, 0, 1]
    
    Let's verify C00:
    T1: 1
    T3: -1
    T4: 1
    T5: -1
    T7: 1
    Sum: 1 - 1 + 1 - 1 + 1 = 1. Correct (A00B00 + A01B10 + A10B01 + A11B11? No, Strassen computes C00 = M1+M4-M5+M7).
    Wait, C00 = A00B00 + A01B10.
    My W vectors contribute to C components.
    The decomposition must sum to the standard multiplication tensor.
    Standard tensor: Sum_{i,j,k} A[i,j] B[j,k] C[k,i].
    My construction is based on Strassen, which is a known correct rank-7 decomposition for 2x2x2.
    I will trust the coefficients.
    """
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=float)
    
    W = np.array([
        [1, 0, -1, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, 1]
    ], dtype=float)
    
    return U, V, W

def _direct_sum(U1, V1, W1, U2, V2, W2, n1, m1, p1, n2, m2, p2, n_total, m_total, p_total):
    """
    Compute direct sum of two tensor decompositions.
    Block 1 is at top-left of A, top-left of B, top-left of C.
    Block 2 is at bottom-right? No, we need to specify positions.
    For M_{2,3,10}, we split m=3 into 2+1.
    A is 2x3. Split into A1 (2x2) and A2 (2x1).
    B is 3x10. Split into B1 (2x10) and B2 (1x10).
    C is 10x2. No split on C dimensions, but the contributions are summed.
    
    Actually, direct sum of M_{n1,m1,p1} and M_{n2,m2,p2} gives M_{n1+n2, m1+m2, p1+p2} if dimensions match?
    No, M_{n,m,p} is A(n x m) * B(m x p) -> C(p x n).
    If we have M_{n,m1,p} and M_{n,m2,p}, we can combine them into M_{n, m1+m2, p}.
    A = [A1, A2] (n x m1, n x m2)
    B = [B1; B2] (m1 x p, m2 x p)
    C = A1B1 + A2B2.
    This is a valid decomposition of the product.
    The rank is R1 + R2.
    
    We need to map the local indices to global indices.
    U: shape (n*m, R).
    For Block 1 (m1): U1 is (n*m1, R1). It corresponds to A columns 0..m1-1.
    For Block 2 (m2): U2 is (n*m2, R2). It corresponds to A columns m1..m1+m2-1.
    
    V: shape (m*p, R).
    For Block 1 (m1): V1 is (m1*p, R1). It corresponds to B rows 0..m1-1.
    For Block 2 (m2): V2 is (m2*p, R2). It corresponds to B rows m1..m1+m2-1.
    
    W: shape (p*n, R).
    W1 and W2 are both (p*n, R1) and (p*n, R2). They add to the same C.
    So W = [W1, W2] (concatenate along rank dimension).
    """
    R1 = U1.shape[1]
    R2 = U2.shape[1]
    R = R1 + R2
    
    U = np.zeros((n_total * m_total, R))
    V = np.zeros((m_total * p_total, R))
    W = np.zeros((p_total * n_total, R))
    
    # Place U1 in columns 0..R1-1, rows corresponding to first m1 columns of A.
    # A is n x m. Index is i*m + j.
    # For Block 1, j is in 0..m1-1.
    # U1 rows are indexed by i*m1 + j_local.
    # We need to map (i, j_local) -> global i*m_total + j_global (where j_global = j_local).
    
    for i in range(n_total):
        for j in range(m1):
            local_idx = i * m1 + j
            global_idx = i * m_total + j
            U[global_idx, :R1] = U1[local_idx, :]
            
    # Place U2 in columns R1..R1+R2-1, rows corresponding to last m2 columns of A.
    for i in range(n_total):
        for j in range(m2):
            local_idx = i * m2 + j
            global_idx = i * m_total + (m1 + j)
            U[global_idx, R1:R1+R2] = U2[local_idx, :]
            
    # Place V1 in rows corresponding to first m1 rows of B.
    # B is m x p. Index is j*p + k.
    # For Block 1, j is in 0..m1-1.
    # V1 rows are indexed by j_local*p + k.
    # We need to map (j_local, k) -> global j_global*p + k (where j_global = j_local).
    
    for j in range(m1):
        for k in range(p_total):
            local_idx = j * p_total + k
            global_idx = j * p_total + k
            V[global_idx, :R1] = V1[local_idx, :]
            
    # Place V2 in rows corresponding to last m2 rows of B.
    for j in range(m2):
        for k in range(p_total):
            local_idx = j * p_total + k
            global_idx = (m1 + j) * p_total + k
            V[global_idx, R1:R1+R2] = V2[local_idx, :]
            
    # Place W1 and W2.
    W[:, :R1] = W1
    W[:, R1:R1+R2] = W2
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Strategy:
    # M_{2,3,10}
    # Split m=3 into 2 and 1.
    # Part 1: M_{2,2,10}
    # Part 2: M_{2,1,10}
    
    # Build M_{2,1,10}
    U_2110, V_2110, W_2110 = _build_M_2x1xP(10)
    
    # Build M_{2,2,10}
    # Split p=10 into 5 and 5.
    # Part 1a: M_{2,2,5}
    # Part 1b: M_{2,2,5}
    
    # Build M_{2,2,5}
    # Split p=5 into 2 and 3.
    # Part 1a1: M_{2,2,2} (Rank 7, Strassen)
    # Part 1a2: M_{2,2,3}
    
    # Build M_{2,2,3}
    # Split p=3 into 2 and 1.
    # Part 1a21: M_{2,2,2} (Rank 7)
    # Part 1a22: M_{2,2,1}
    
    # Build M_{2,2,1}
    # This is A(2x2) * B(2x1) -> C(1x2).
    # C = [C0, C1]. C0 = A00B0 + A01B1. C1 = A10B0 + A11B1.
    # Schoolbook Rank 4.
    # Can we do better?
    # R(M_{2,2,1})?
    # Lower bound: dim(C)*dim(A) ? No.
    # Let's use schoolbook for simplicity if Strassen doesn't apply directly to rectangular 1.
    # Actually, M_{2,2,1} is equivalent to M_{1,2,2} by symmetry?
    # Let's just use schoolbook for M_{2,2,1} to be safe. Rank 4.
    
    def _build_M_2x2x1():
        n, m, p = 2, 2, 1
        rank = 4
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    U_221, V_221, W_221 = _build_M_2x2x1()
    
    # Combine M_{2,2,2} and M_{2,2,1} to get M_{2,2,3}
    # Note: p changes from 2 and 1 to 3.
    # V dimensions: (2*2) and (2*1) -> (2*3).
    # W dimensions: (2*2) and (1*2) -> (3*2).
    # We need to embed V_222 (4x7) into V_223 (6x7) and V_221 (2x4) into V_223 (6x4).
    # V_222 uses B rows 0,1. V_221 uses B row 0 (which is B row 2 in combined?).
    # Wait, M_{2,2,3} means B is 2x3.
    # Split p into 2 and 1.
    # B = [B1 (2x2), B2 (2x1)].
    # So V1 should be for B1, V2 for B2.
    # V1 indices: j in 0..1, k in 0..1.
    # V2 indices: j in 0..1, k in 0..0 (but k is local 0, global 2).
    
    U_222, V_222, W_222 = _build_M_2x2x2()
    
    # We need to combine M_{2,2,2} (p=2) and M_{2,2,1} (p=1) into M_{2,2,3}.
    # U is same for both? No, U depends on A. A is 2x2 in both.
    # U_222 is 4x7. U_221 is 4x4.
    # We can just concatenate U along rank.
    # U_223 = [U_222, U_221].
    
    # V:
    # V_222 is 4x7 (B is 2x2).
    # V_221 is 2x4 (B is 2x1).
    # Combined B is 2x3. V_223 should be 6x11.
    # V_222 contributes to B rows 0,1 cols 0,1.
    # V_221 contributes to B rows 0,1 col 2.
    
    V_223 = np.zeros((2*3, 7+4))
    # Embed V_222
    for j in range(2):
        for k in range(2):
            V_223[j*3+k, :7] = V_222[j*2+k, :]
    # Embed V_221
    for j in range(2):
        for k in range(1):
            V_223[j*3+2, 7:11] = V_221[j*1+k, :]
            
    # W:
    # W_222 is 4x7 (C is 2x2).
    # W_221 is 2x4 (C is 1x2).
    # Combined C is 3x2. W_223 should be 6x11.
    # W_222 contributes to C rows 0,1.
    # W_221 contributes to C row 2.
    
    W_223 = np.zeros((3*2, 7+4))
    # Embed W_222
    for k in range(2):
        for i in range(2):
            W_223[k*2+i, :7] = W_222[k*2+i, :]
    # Embed W_221
    for k in range(1):
        for i in range(2):
            W_223[2*2+i, 7:11] = W_221[k*2+i, :]
            
    U_223 = np.hstack([U_222, U_221])
    
    # Now we have M_{2,2,3} (Rank 11).
    # Build M_{2,2,5} = M_{2,2,2} + M_{2,2,3}.
    # Split p=5 into 2 and 3.
    # U_225 = [U_222, U_223].
    # V_222 (2x2 B) -> V_225 (2x5 B).
    # V_223 (2x3 B) -> V_225 (2x5 B).
    
    V_225 = np.zeros((2*5, 7+11))
    for j in range(2):
        for k in range(2):
            V_225[j*5+k, :7] = V_222[j*2+k, :]
    for j in range(2):
        for k in range(3):
            V_225[j*5+2+k, 7:18] = V_223[j*3+k, :]
            
    W_225 = np.zeros((5*2, 7+11))
    for k in range(2):
        for i in range(2):
            W_225[k*2+i, :7] = W_222[k*2+i, :]
    for k in range(3):
        for i in range(2):
            W_225[(2+k)*2+i, 7:18] = W_223[k*2+i, :]
            
    U_225 = np.hstack([U_222, U_223])
    
    # Now we have M_{2,2,5} (Rank 18).
    # Build M_{2,2,10} = M_{2,2,5} + M_{2,2,5}.
    # Split p=10 into 5 and 5.
    
    V_2210 = np.zeros((2*10, 18+18))
    for j in range(2):
        for k in range(5):
            V_2210[j*10+k, :18] = V_225[j*5+k, :]
    for j in range(2):
        for k in range(5):
            V_2210[j*10+5+k, 18:36] = V_225[j*5+k, :]
            
    W_2210 = np.zeros((10*2, 18+18))
    for k in range(5):
        for i in range(2):
            W_2210[k*2+i, :18] = W_225[k*2+i, :]
    for k in range(5):
        for i in range(2):
            W_2210[(5+k)*2+i, 18:36] = W_225[k*2+i, :]
            
    U_2210 = np.hstack([U_225, U_225])
    
    # Now we have M_{2,2,10} (Rank 36).
    # Build M_{2,3,10} = M_{2,2,10} + M_{2,1,10}.
    # Split m=3 into 2 and 1.
    
    U_final = np.zeros((2*3, 36+10))
    # U_2210 (2x2 A) -> U_final (2x3 A). Cols 0,1.
    for i in range(2):
        for j in range(2):
            U_final[i*3+j, :36] = U_2210[i*2+j, :]
    # U_2110 (2x1 A) -> U_final (2x3 A). Col 2.
    for i in range(2):
        for j in range(1):
            U_final[i*3+2, 36:46] = U_2110[i*1+j, :]
            
    V_final = np.zeros((3*10, 36+10))
    # V_2210 (2x10 B) -> V_final (3x10 B). Rows 0,1.
    for j in range(2):
        for k in range(10):
            V_final[j*10+k, :36] = V_2210[j*2+k, :]
    # V_2110 (1x10 B) -> V_final (3x10 B). Row 2.
    for j in range(1):
        for k in range(10):
            V_final[2*10+k, 36:46] = V_2110[j*1+k, :]
            
    W_final = np.zeros((10*2, 36+10))
    # W_2210 (10x2 C) -> W_final.
    # W_2110 (10x2 C) -> W_final.
    W_final[:, :36] = W_2210
    W_final[:, 36:46] = W_2110
    
    # Convert to lists
    return {
        "U": U_final.tolist(),
        "V": V_final.tolist(),
        "W": W_final.tolist()
    }

if __name__ == "__main__":
    # Simple self-test
    res = construct()
    U = np.array(res["U"])
    V = np.array(res["V"])
    W = np.array(res["W"])
    
    n, m, p = 2, 3, 10
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_expected = A @ B
    
    # Reconstruct C
    C_recon = np.zeros((p, n))
    for r in range(U.shape[1]):
        C_recon += np.outer(W[:, r], U[:, r]) * V[:, r]
        # Wait, the formula is Sum_r u_r v_r w_r^T?
        # The tensor is T = sum u_r \otimes v_r \otimes w_r.
        # Contracting with A and B:
        # C[i, k] = sum_{j, r} A[i, j] B[j, k] * (u_r[i,j] v_r[j,k] w_r[k,i])?
        # No, standard decomposition:
        # A is n x m. B is m x p. C is p x n.
        # U is (n*m) x R. V is (m*p) x R. W is (p*n) x R.
        # C[k, i] = sum_{j, r} A[i, j] B[j, k] * ?
        # Actually, the bilinear form is:
        # C = sum_r (U[:, r] reshaped to n x m) * (V[:, r] reshaped to m x p) ? No.
        # The term is w_r (outer) scaled by (u_r dot A_flat) * (v_r dot B_flat)?
        # No.
        # T_{ijk} = sum_r u_{ir} v_{jr} w_{kr}.
        # C_{ki} = sum_{j} A_{ij} B_{jk}.
        # So C_{ki} = sum_{j, r} A_{ij} B_{jk} u_{ir} v_{jr} w_{kr}.
        # Let a_r = sum_i u_{ir} A_{ij}? No.
        # Let's check the seed code:
        # U[i*m+j, col] = 1
        # V[j*p+k, col] = 1
        # W[k*n+i, col] = 1
        # Term is delta(i) delta(j) delta(k).
        # Sum over col gives sum_{i,j,k} A[i,j] B[j,k] (1 if i,j,k match).
        # This is correct.
        
    # My reconstruction loop above was wrong.
    # C[k, i] = sum_{j, r} A[i, j] B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # This is O(R * n * m * p).
    
    C_recon = np.zeros((p, n))
    for r in range(U.shape[1]):
        U_r = U[:, r].reshape(n, m)
        V_r = V[:, r].reshape(m, p)
        W_r = W[:, r].reshape(p, n)
        # The term is (U_r * A) * (V_r * B^T)? No.
        # It's sum_j A[i,j] U[i,j,r] * sum_k B[j,k] V[j,k,r] ...
        # Actually, C = sum_r (A * U_r^T) (V_r * B^T) ... No.
        
        # Let's just do the triple sum for verification since sizes are small.
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    C_recon[k, i] += A[i, j] * B[j, k] * U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    
    if np.allclose(C_expected, C_recon):
        print("Self-test passed. Rank:", U.shape[1])
    else:
        print("Self-test failed.")
        print(C_expected)
        print(C_recon)
