import numpy as np
import time

def _deadline_left(deadline, margin=10.0):
    return deadline - time.time() > margin

def _verify(U, V, W, tol=1e-7):
    """
    Verify exact decomposition:
    sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    """
    n, m, p = 3, 5, 6
    N, M, P = n*m, m*p, p*n
    r = U.shape[1]
    if V.shape[1] != r or W.shape[1] != r:
        return False
    if U.shape[0] != N or V.shape[0] != M or W.shape[0] != P:
        return False
    U3 = np.array(U, dtype=float).reshape(n, m, r)
    V3 = np.array(V, dtype=float).reshape(m, p, r)
    W3 = np.array(W, dtype=float).reshape(p, n, r)
    out = np.einsum('ijc,jkc,kic->ik', U3, V3, W3)
    # The tensor we want is the identity in (i,k) with 1 for each j
    # Actually the target is: for each (i,j,k), the product should equal 1
    # Let's re-read the spec: "sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1 for all i,j,k"
    # This means the target tensor T[i,j,k] = 1 for all i,j,k.
    # So we need sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k.
    # Let's check the einsum: 'ijc,jkc,kic->ik' sums over j and c.
    # Result[i,k] = sum_j sum_c U[i,j,c] V[j,k,c] W[k,i,c].
    # This is not the same as checking each (i,j,k) individually.
    # We need to check the full 3D tensor.
    T_computed = np.einsum('ijc,jkc,kic->ijk', U3, V3, W3)
    T_target = np.ones((n, m, p))
    return np.allclose(T_computed, T_target, atol=tol, rtol=0)

def _strassen():
    """
    Returns rank-7 decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen's 7 multiplications
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B12
    # M3 = A12(B21+B22)
    # M4 = A11(B12-B22)
    # M5 = (A21-A11)(B11+B21)
    # M6 = A22(B21-B11)
    # M7 = (A12+A22)(B11+B12)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M4
    # C21 = M2 + M5
    # C22 = M1 - M2 + M3 + M6
    
    r = 7
    U = np.zeros((4, r))
    V = np.zeros((4, r))
    W = np.zeros((4, r))
    
    # Each row of U corresponds to A[i,j] with index i*2+j
    # U[0, c] for A11, U[1, c] for A12, U[2, c] for A21, U[3, c] for A22
    # V[0, c] for B11, V[1, c] for B12, V[2, c] for B21, V[3, c] for B22
    # W[0, c] for C11 (k=0,i=0 -> 0*2+0), W[1, c] for C12 (k=1,i=0 -> 1*2+0), 
    # W[2, c] for C21 (k=0,i=1 -> 0*2+1), W[3, c] for C22 (k=1,i=1 -> 1*2+1)
    
    # M1
    U[0,0] = 1; U[3,0] = 1
    V[0,0] = 1; V[1,0] = 1
    W[0,0] = 1; W[3,0] = 1
    
    # M2
    U[2,1] = 1; U[3,1] = 1
    V[1,1] = 1
    W[2,1] = 1
    
    # M3
    U[1,2] = 1
    V[2,2] = 1; V[3,2] = 1
    W[1,2] = 1
    
    # M4
    U[0,3] = 1
    V[1,3] = 1; V[3,3] = -1
    W[0,3] = 1; W[1,3] = 1
    
    # M5
    U[2,4] = 1; U[0,4] = -1
    V[0,4] = 1; V[2,4] = 1
    W[0,4] = -1; W[2,4] = 1
    
    # M6
    U[3,5] = 1
    V[2,5] = 1; V[0,5] = -1
    W[3,5] = 1
    
    # M7
    U[1,6] = 1; U[3,6] = 1
    V[0,6] = 1; V[1,6] = 1
    W[0,6] = 1
    
    return U, V, W

def _laderman():
    """
    Returns rank-23 decomposition for 3x3 * 3x3.
    This is Laderman's algorithm.
    Reference: Laderman, 1976. "A non-recursive algorithm for multiplying 3x3 matrices".
    U: (9, 23), V: (9, 23), W: (9, 23)
    U[i*3+j, c], V[j*3+k, c], W[k*3+i, c]
    """
    # The 23 multiplications in Laderman's algorithm
    # I will encode them directly.
    r = 23
    U = np.zeros((9, r))
    V = np.zeros((9, r))
    W = np.zeros((9, r))
    
    # Indices: 0=A11,1=A12,2=A13,3=A21,4=A22,5=A23,6=A31,7=A32,8=A33
    # B: 0=B11,1=B12,2=B13,3=B21,4=B22,5=B23,6=B31,7=B32,8=B33
    # C: 0=C11,1=C12,2=C13,3=C21,4=C22,5=C23,6=C31,7=C32,8=C33
    
    # M1
    U[0,0]=1; U[4,0]=1
    V[1,0]=1; V[5,0]=1
    W[3,0]=1; W[7,0]=1
    
    # M2
    U[1,1]=1; U[4,1]=1
    V[0,1]=1; V[4,1]=1
    W[2,1]=1; W[7,1]=1
    
    # M3
    U[0,2]=1; U[3,2]=1
    V[0,2]=1; V[4,2]=1
    W[1,2]=1; W[4,2]=1
    
    # M4
    U[0,3]=1; U[5,3]=1
    V[1,3]=1; V[2,3]=1
    W[3,3]=1; W[6,3]=1
    
    # M5
    U[1,4]=1; U[5,4]=1
    V[0,4]=1; V[3,4]=1
    W[2,4]=1; W[5,4]=1
    
    # M6
    U[0,5]=1; U[2,5]=1
    V[3,5]=1; V[4,5]=1
    W[0,5]=1; W[4,5]=1
    
    # M7
    U[1,6]=1; U[2,6]=1
    V[3,6]=1; V[4,6]=1
    W[1,6]=1; W[4,6]=1
    
    # M8
    U[0,7]=1; U[4,7]=1
    V[1,7]=1; V[5,7]=1
    W[3,7]=1; W[7,7]=1  # Wait, M1 and M8 are identical? No, Laderman has distinct ones.
    # Let me check Laderman's list carefully.
    
    # Actually, hardcoding Laderman's 23 vectors from scratch is error-prone.
    # A safer approach is to use a known rank-23 decomposition.
    # Since I cannot look it up, I will use a block construction instead.
    # R(3,3,3) can be done by splitting into 2x2 + 1x1 blocks.
    # But R(3,3,3) = 23 is a known bound.
    # Let's stick to the block decomposition for 3x5x6 which gives rank 80.
    # I will remove _laderman and use block sums only.
    # For 3x3x3, I can use a rank-27 schoolbook or find a better one.
    # Actually, for the block 3x3x3, if I use schoolbook rank 27, then:
    # R(3,3,6) = 27 + 27 = 54.
    # R(3,2,3) = 17 (using Strassen for 2x2x2 part).
    # R(3,2,6) = 34.
    # Total = 34 + 54 = 88.
    # This is worse than 90? No, 88 < 90.
    # Can I do better?
    # R(3,3,3) rank 23 is known. If I can get it, total is 80.
    # Let me try to implement Laderman correctly.
    
    # Laderman's 23 terms:
    # m1 = (a11+a22)(b12+b22) -> c21, c23, c24? No.
    
    # I will use a simpler strategy: 
    # For 3x3x3, use Strassen for the top-left 2x2, and handle the rest.
    # This is getting complicated. Let's just use the block sum with Strassen for 2x2x2 parts.
    # And for 3x3x3, use a rank-24 or rank-23 decomposition if available.
    # For now, I'll use a rank-27 schoolbook for 3x3x3 to be safe, giving total rank 88.
    # Wait, 88 is better than 90.
    
    # Let's refine the block decomposition:
    # 3x5 = [3x2, 3x3]
    # 5x6 = [2x6; 3x6]
    # C = A1 * B1 + A2 * B2
    # A1 is 3x2, B1 is 2x6. R(3,2,6) = 2 * R(3,2,3) = 2 * 17 = 34.
    # A2 is 3x3, B2 is 3x6. R(3,3,6) = 2 * R(3,3,3).
    # If R(3,3,3) = 27 (schoolbook), then R(3,3,6) = 54.
    # Total = 34 + 54 = 88.
    
    # Can I do R(3,3,3) better? 
    # R(3,3,3) is known to be 23.
    # I will implement a rank-23 decomposition for 3x3x3.
    # Source: Laderman (1976)
    # The 23 bilinear forms are:
    # (I will use a known list from literature)
    
    # For now, let's return None from _laderman and just use schoolbook for 3x3x3.
    # Rank 88 is a valid improvement.
    
    return None

def _block_decomp():
    """
    Construct rank-88 decomposition using block sums.
    """
    n, m, p = 3, 5, 6
    N, M, P = n*m, m*p, p*n
    
    # Split m=5 into 2 and 3.
    # U: 15 x r_total
    # V: 30 x r_total
    # W: 18 x r_total
    
    U1, V1, W1 = _strassen() # 2x2x2, rank 7
    # We need R(3,2,3).
    # Split n=3 into 1 and 2.
    # R(1,2,3) = 6 (schoolbook)
    # R(2,2,3) = R(2,2,2) + R(2,2,1) = 7 + 4 = 11.
    # Total R(3,2,3) = 17.
    
    # Let's build R(3,2,3) explicitly.
    # A is 3x2, B is 2x3.
    # Split A into A_top (1x2) and A_bot (2x2).
    # C_top = A_top * B (1x3) -> rank 6.
    # C_bot = A_bot * B (2x3) -> split B into B_left (2x2) and B_right (2x1).
    # C_bot_left = A_bot * B_left (2x2) -> Strassen rank 7.
    # C_bot_right = A_bot * B_right (2x1) -> schoolbook rank 4.
    # Total 17.
    
    U_323 = np.zeros((6, 17))
    V_323 = np.zeros((6, 17))
    W_323 = np.zeros((6, 17))
    
    # C_top: 1x3. U_idx: 0,1. V_idx: 0..5. W_idx: 0,1,2.
    # Schoolbook 1x2x3.
    for j in range(2):
        for k in range(3):
            c = j*3 + k
            U_323[j, c] = 1
            V_323[j*3+k, c] = 1
            W_323[k, c] = 1
    
    # C_bot: 2x3. U_idx: 2,3,4,5. V_idx: 0..5. W_idx: 3,4,5,6,7,8.
    # Split B into B_left (2x2, cols 0,1) and B_right (2x1, col 2).
    # C_bot_left: 2x2. U_idx: 2,3,4,5. V_idx: 0,1,2,3. W_idx: 3,4,5,6.
    U_sl, V_sl, W_sl = _strassen()
    U_323[2:6, 6:13] = U_sl
    V_323[0:4, 6:13] = V_sl
    W_323[3:7, 6:13] = W_sl
    
    # C_bot_right: 2x1. U_idx: 2,3,4,5. V_idx: 4,5. W_idx: 5,6.
    # Wait, W indices for C_bot are 3..8. C_bot_right corresponds to col 2 of C_bot.
    # C_bot is rows 2,3 of C (3x3). So C_bot_right is C[2,2] and C[3,2].
    # W indices: C[2,2] is k=2,i=2 -> 2*3+2=8. C[3,2] is k=2,i=3 -> 3*3+2=11? No.
    # W index is k*n+i. n=3.
    # C is 3x3.
    # C_bot is rows 1,2 (0-indexed) of C? No, A is 3x2.
    # A_top is row 0. A_bot is rows 1,2.
    # C_top is row 0 of C (1x3).
    # C_bot is rows 1,2 of C (2x3).
    # C_bot_left is C[1:3, 0:2] (2x2).
    # C_bot_right is C[1:3, 2:3] (2x1).
    
    # W indices for C_bot_left:
    # C[1,0]: k=0,i=1 -> 3. C[1,1]: k=1,i=1 -> 4. C[2,0]: k=0,i=2 -> 6. C[2,1]: k=1,i=2 -> 7.
    # My Strassen W_sl is for 2x2 output. W_sl[0] is C11, W_sl[1] is C12, W_sl[2] is C21, W_sl[3] is C22.
    # So I need to map these to the correct W indices in W_323.
    # W_323 has shape (9, 17).
    # W_sl[0] -> W_323[3, ...]
    # W_sl[1] -> W_323[4, ...]
    # W_sl[2] -> W_323[6, ...]
    # W_sl[3] -> W_323[7, ...]
    
    # I already did W_323[3:7, 6:13] = W_sl. This is wrong.
    # Let's redo the mapping carefully.
    
    U_323 = np.zeros((6, 17))
    V_323 = np.zeros((6, 17))
    W_323 = np.zeros((9, 17)) # 3x3 output
    
    # Part 1: C_top (1x3). Rank 6.
    for j in range(2):
        for k in range(3):
            c = j*3 + k
            U_323[j, c] = 1
            V_323[j*3+k, c] = 1
            W_323[k, c] = 1 # C[0,k] -> W index k*3+0 = k. Correct.
    
    # Part 2: C_bot_left (2x2). Rank 7.
    U_sl, V_sl, W_sl = _strassen()
    # U_sl is 4x7. Corresponds to A[1:3, 0:2].
    # U_323 indices for A[1:3, :] are 2,3,4,5.
    U_323[2:6, 6:13] = U_sl
    # V_sl is 4x7. Corresponds to B[0:2, 0:2].
    # V_323 indices for B[0:2, 0:2] are 0,1,2,3.
    V_323[0:4, 6:13] = V_sl
    # W_sl is 4x7. Corresponds to C[1:3, 0:2].
    # W indices: C[1,0]=3, C[1,1]=4, C[2,0]=6, C[2,1]=7.
    w_indices = [3, 4, 6, 7]
    for i in range(4):
        W_323[w_indices[i], 6:13] = W_sl[i, :]
    
    # Part 3: C_bot_right (2x1). Rank 4.
    # A[1:3, :] * B[:, 2].
    # Schoolbook 2x1x1? No, 2x2 * 2x1.
    # A is 2x2, B is 2x1.
    # C = A * B. C[i,0] = sum_j A[i,j] B[j,0].
    # Rank 4.
    for i in range(2):
        for j in range(2):
            c = 13 + i*2 + j
            U_323[2+i, c] = 1
            V_323[j*3+2, c] = 1 # B[j, 2] -> index j*3+2.
            W_323[3+i, c] = 1 # C[1+i, 2] -> W index (1+i)*3+2? No.
            # C is 3x3. C_bot_right is C[1:3, 2:3].
            # C[1,2] -> k=2,i=1 -> 2*3+1 = 7.
            # C[2,2] -> k=2,i=2 -> 2*3+2 = 8.
            W_323[7+i, c] = 1
    
    return U_323, V_323, W_323

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 3, 5, 6
    N, M, P = n*m, m*p, p*n
    
    # Block decomposition:
    # Split m=5 into m1=2, m2=3.
    # R(3,2,6) + R(3,3,6)
    # R(3,2,6) = 2 * R(3,2,3) = 2 * 17 = 34.
    # R(3,3,6) = 2 * R(3,3,3).
    # If R(3,3,3) = 27, then R(3,3,6) = 54.
    # Total = 88.
    
    U_323, V_323, W_323 = _block_decomp() # Rank 17 for 3x2x3
    
    # We need R(3,2,6).
    # Split p=6 into p1=3, p2=3.
    # R(3,2,3) + R(3,2,3) = 17 + 17 = 34.
    U_326 = np.zeros((6, 34))
    V_326 = np.zeros((12, 34)) # 2x6
    W_326 = np.zeros((18, 34)) # 6x3
    
    # First block: B[:, 0:3]
    U_326[:, 0:17] = U_323
    V_326[:, 0:17] = V_323[:, 0:6] # B[0:2, 0:3] -> indices 0..5
    # W_323 is 3x3. W_326 is 6x3.
    # W_326[0:3, 0:17] = W_323
    W_326[0:3, 0:17] = W_323
    
    # Second block: B[:, 3:6]
    U_326[:, 17:34] = U_323
    V_326[:, 17:34] = V_323[:, 6:12] # B[0:2, 3:6] -> indices 6..11
    W_326[3:6, 17:34] = W_323
    
    # Now R(3,3,6).
    # Use schoolbook for 3x3x3. Rank 27.
    # R(3,3,6) = 2 * 27 = 54.
    U_336 = np.zeros((9, 54))
    V_336 = np.zeros((18, 54)) # 3x6
    W_336 = np.zeros((18, 54)) # 6x3
    
    # First block: B[:, 0:3]
    # Schoolbook 3x3x3.
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = i*9 + j*3 + k
                U_336[i*3+j, c] = 1
                V_336[j*3+k, c] = 1
                W_336[k*3+i, c] = 1
    
    # Second block: B[:, 3:6]
    for i in range(3):
        for j in range(3):
            for k in range(3):
                c = 27 + i*9 + j*3 + k
                U_336[i*3+j, c] = 1
                V_336[j*3+3+k, c] = 1 # B[j, 3+k]
                W_336[(3+k)*3+i, c] = 1 # C[3+k, i] -> W index (3+k)*3+i
    
    # Combine
    r_total = 34 + 54
    U = np.zeros((15, r_total))
    V = np.zeros((30, r_total))
    W = np.zeros((18, r_total))
    
    # A1 is 3x2. U indices 0..5.
    U[0:6, 0:34] = U_326
    # A2 is 3x3. U indices 6..14.
    U[6:15, 34:88] = U_336
    
    # B1 is 2x6. V indices 0..11.
    V[0:12, 0:34] = V_326
    # B2 is 3x6. V indices 12..29.
    V[12:30, 34:88] = V_336
    
    # C is 3x6. W indices 0..17.
    W[0:18, 0:34] = W_326
    W[0:18, 34:88] = W_336
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
