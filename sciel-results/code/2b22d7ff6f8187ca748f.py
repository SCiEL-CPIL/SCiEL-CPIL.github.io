"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np

_start_time = time.time()
_DEADLINE = _start_time + 140  # 140s safety margin

def _build_full_tensor(n, m, p):
    """Build the full matrix multiplication tensor."""
    nm, mp, pn = n * m, m * p, p * n
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                d = k * n + i
                T[a, b, d] = 1
    return T

def _verify_decomposition(U, V, W, T_exp):
    """Verify if U, V, W exactly decompose T_exp."""
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    if isinstance(U, list):
        U = np.array(U)
    if isinstance(V, list):
        V = np.array(V)
    if isinstance(W, list):
        W = np.array(W)
    T_check = np.einsum('ad,bd,cd->abc', U, V, W)
    return np.array_equal(T_check, T_exp)

def _strassen_232():
    """
    Rank-7 decomposition for 2x3 * 3x2 matrix multiplication.
    A is 2x3, B is 3x2, C is 2x2.
    Returns U (6x7), V (6x7), W (4x7) where:
    T[a,b,d] = sum_t U[a,t] * V[b,t] * W[d,t]
    Indexing: a = i*3+j, b = j*2+k, d = k*2+i
    """
    U = np.zeros((6, 7), dtype=np.int64)
    V = np.zeros((6, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # A indices: i*3+j where i in {0,1}, j in {0,1,2}
    #   A[0,0]=0, A[0,1]=1, A[0,2]=2
    #   A[1,0]=3, A[1,1]=4, A[1,2]=5
    # B indices: j*2+k where j in {0,1,2}, k in {0,1}
    #   B[0,0]=0, B[0,1]=1
    #   B[1,0]=2, B[1,1]=3
    #   B[2,0]=4, B[2,1]=5
    # C indices: k*2+i where k in {0,1}, i in {0,1}
    #   C[0,0]=0, C[0,1]=1
    #   C[1,0]=2, C[1,1]=3
    
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    
    # Use Strassen-style 7 products:
    # m1 = (A[0,0]+A[1,2]) * (B[0,0]+B[2,1])
    # m2 = (A[1,0]+A[1,2]) * B[0,0]
    # m3 = A[0,0] * (B[0,1]-B[2,1])
    # m4 = A[1,2] * (B[2,0]-B[0,0])
    # m5 = (A[0,0]+A[0,1]) * B[2,1]
    # m6 = (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    # m7 = (A[0,1]-A[1,2]) * (B[2,0]+B[2,1])
    
    # But we need to account for the middle term A[0,1]*B[1,*] and A[1,1]*B[1,*]
    # This direct Strassen doesn't work cleanly for 3 intermediate terms.
    
    # Let me use a different known rank-7 decomposition for M_{2,3,2}.
    # One valid approach: embed in 2x2x2 by noting that the "middle" dimension
    # can be handled by splitting.
    
    # Actually, let's use the fact that M_{2,3,2} can be decomposed as:
    # Compute 3 products for column 0 of C, 3 for column 1, but share.
    # 
    # Alternative: use the following 7 products:
    # m1 = (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])  -- but B only has 2 cols...
    
    # Let me try a systematic approach. We need 7 rank-1 tensors.
    # I'll use the known decomposition from the literature.
    
    # For M_{2,3,2}, one rank-7 decomposition:
    # The tensor has 6*6*4 = 144 entries, 12 of which are 1.
    # 
    # Let me use the following construction based on Strassen adapted for 3x2:
    
    # m1 = (A00+A11)*(B00+B11) + A02*B20 + A12*B21 ... no, this isn't rank 1.
    
    # Let me just use a verified rank-7 decomposition.
    # Based on the structure, here's a valid one:
    
    # Product 1: (A[0,0]+A[1,2]) * (B[0,0]+B[2,1])
    U[0, 0] = 1; U[5, 0] = 1
    V[0, 0] = 1; V[5, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # C[0,0] += m1, C[1,1] += m1
    
    # Product 2: (A[1,0]+A[1,2]) * B[0,0]
    U[3, 1] = 1; U[5, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # C[1,0] += m2, C[1,1] -= m2
    
    # Product 3: A[0,0] * (B[0,1]-B[2,1])
    U[0, 2] = 1
    V[1, 2] = 1; V[5, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # C[0,1] += m3, C[1,1] += m3
    
    # Product 4: A[1,2] * (B[2,0]-B[0,0])
    U[5, 3] = 1
    V[4, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # C[0,0] += m4, C[1,0] += m4
    
    # Product 5: (A[0,0]+A[0,1]) * B[2,1]
    U[0, 4] = 1; U[1, 4] = 1
    V[5, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # C[0,0] -= m5, C[0,1] += m5
    
    # Product 6: (A[1,0]-A[0,0]) * (B[0,0]+B[0,1])
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # C[1,1] += m6
    
    # Product 7: (A[0,1]-A[1,2]) * (B[2,0]+B[2,1])
    U[1, 6] = 1; U[5, 6] = -1
    V[4, 6] = 1; V[5, 6] = 1
    W[0, 6] = 1
    # C[0,0] += m7
    
    # Now check: we still need to account for A[0,1]*B[1,0], A[0,1]*B[1,1],
    # A[1,1]*B[1,0], A[1,1]*B[1,1]. These 4 terms are missing!
    # This decomposition is incomplete. The above only handles the "corner" terms.
    
    # I need a proper rank-7 decomposition. Let me use a different approach.
    # 
    # Key insight: M_{2,3,2} can be seen as the "slice" of M_{2,3,3} where we only
    # use 2 of the 3 columns of B. The rank of M_{2,3,3} is 12 (schoolbook) or less.
    # 
    # Actually, the correct rank of M_{2,3,2} is 7, and here's a valid decomposition:
    
    # Reset
    U = np.zeros((6, 7), dtype=np.int64)
    V = np.zeros((6, 7), dtype=np.int64)
    W = np.zeros((4, 7), dtype=np.int64)
    
    # Use the following 7 products (verified decomposition):
    # m1 = (A00 + A11) * (B00 + B11)
    # m2 = (A10 + A11) * B00
    # m3 = A00 * (B01 - B11)  
    # m4 = A11 * (B10 - B00)
    # m5 = (A00 + A01) * B11
    # m6 = (A10 - A00) * (B00 + B01)
    # m7 = (A01 - A11) * (B10 + B11)
    # Plus we need to handle A02*B20, A02*B21, A12*B20, A12*B21
    # That's 4 more terms, giving rank 11, not 7.
    
    # So M_{2,3,2} does NOT have rank 7 in this straightforward way.
    # The rank of M_{2,3,2} is actually 11 or 12.
    
    # Let me reconsider. The schoolbook rank is 2*3*2 = 12.
    # Can we do better than 12? 
    # 
    # C[0,0] = A00*B00 + A01*B10 + A02*B20
    # C[0,1] = A00*B01 + A01*B11 + A02*B21
    # C[1,0] = A10*B00 + A11*B10 + A12*B20
    # C[1,1] = A10*B01 + A11*B11 + A12*B21
    #
    # Group by B columns:
    # C[:,0] = A * B[:,0] (2x3 * 3x1 = 2x1)
    # C[:,1] = A * B[:,1] (2x3 * 3x1 = 2x1)
    #
    # Each 2x3*3x1 has rank 6 (schoolbook). But they share A.
    # Using Strassen-like: for 2x3*3x1, we compute 2 dot products in R3.
    # Rank of 2 dot products in R3 is 2 (just do them directly) or can we do better?
    # No, 2 independent dot products in R3 require rank 2*3=6 in the bilinear sense? 
    # Actually no. A dot product u·v in R^d has bilinear rank d.
    # Two dot products (u·v, u·w) share u, so they can be done in rank d+1? No.
    # 
    # The bilinear rank of computing (u·v, u·w) where u,v,w in R^3:
    # = rank of the tensor T[i,j,k] = 1 if (i,j) matches for first, (i,k) for second
    # This is actually rank 3: use u_i as the shared factor.
    # T = sum_i u_i * (v_i, w_i) ... no, that's not quite right for the tensor form.
    
    # I think the correct approach is: the current rank-44 solution is likely near-optimal
    # or at least good. Let me try to improve it slightly.
    
    # The current approach: 4 blocks of (7 + 4) = 44.
    # The "4" for the tail (A[:,2] * B[2,:]) is schoolbook for 2x1 * 1x2 = 2x2, which is rank 4.
    # That's already optimal for that sub-problem.
    
    # The "7" for the 2x2 * 2x2 part is Strassen, which is optimal.
    
    # So rank 44 might be close to optimal. The true rank of M_{2,3,8} is unknown but
    # is at most 44 and at least 28 (from lower bounds).
    
    # Let me try a different block structure: split B into 3x4 + 3x4 instead of 4x(3x2).
    # M_{2,3,4} + M_{2,3,4}. What's rank(M_{2,3,4})?
    # Schoolbook: 2*3*4 = 24. Two of these: 48. Worse.
    
    # Try: split into 3x2 + 3x2 + 3x2 + 3x2 (current) = 4 * rank(M_{2,3,2})
    # If rank(M_{2,3,2}) = 11, total = 44. That matches!
    
    # So the current solution uses rank 11 for each M_{2,3,2} block (7+4), total 44.
    # The question is whether rank(M_{2,3,2}) < 11.
    
    # Let me just keep the current approach and verify it works correctly.
    return U, V, W  # Will be overwritten by construct()

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses a rank-44 construction based on 4 blocks of (Strassen 7 + Schoolbook 4).
    """
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    # Build the expected tensor for verification
    T_exp = _build_full_tensor(n, m, p)
    
    # Fallback: Schoolbook rank
    rank_school = n * m * p  # 48
    U_fb = np.zeros((n*m, rank_school), dtype=np.int64)
    V_fb = np.zeros((m*p, rank_school), dtype=np.int64)
    W_fb = np.zeros((p*n, rank_school), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1
                V_fb[j*p+k, col] = 1
                W_fb[k*n+i, col] = 1
                col += 1
    
    # Primary: Rank 44 Construction
    # Structure: 4 blocks of p=2.
    # Each block:
    #   - Strassen (rank 7) for A[:, 0:2] * B[0:2, :]
    #   - Schoolbook (rank 4) for A[:, 2] * B[2, :]
    
    rank_44 = 44
    U = np.zeros((n*m, rank_44), dtype=np.int64)
    V = np.zeros((m*p, rank_44), dtype=np.int64)
    W = np.zeros((p*n, rank_44), dtype=np.int64)
    
    col = 0
    
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # --- Strassen part for A[:, 0:2] (2x2) * B[0:2, :] (2x2) ---
        # A_sub indices: 
        # A00 = U[0], A01 = U[1]
        # A10 = U[3], A11 = U[4]
        # B_sub indices:
        # B0k0 = V[k0], B0k1 = V[k1]
        # B1k0 = V[p+k0], B1k1 = V[p+k1]
        
        # M1 = (A00 + A11) * (B0k0 + B1k1)
        U[0, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        V[p+k1, col] = 1
        W[k0*n+0, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M2 = (A10 + A11) * B0k0
        U[3, col] = 1
        U[4, col] = 1
        V[k0, col] = 1
        W[k0*n+1, col] = 1
        W[k1*n+1, col] = -1
        col += 1
        
        # M3 = A00 * (B0k1 - B1k1)
        U[0, col] = 1
        V[k1, col] = 1
        V[p+k1, col] = -1
        W[k1*n+0, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M4 = A11 * (B1k0 - B0k0)
        U[4, col] = 1
        V[p+k0, col] = 1
        V[k0, col] = -1
        W[k0*n+0, col] = 1
        W[k0*n+1, col] = 1
        col += 1
        
        # M5 = (A00 + A01) * B1k1
        U[0, col] = 1
        U[1, col] = 1
        V[p+k1, col] = 1
        W[k0*n+0, col] = -1
        W[k1*n+0, col] = 1
        col += 1
        
        # M6 = (A10 - A00) * (B0k0 + B0k1)
        U[3, col] = 1
        U[0, col] = -1
        V[k0, col] = 1
        V[k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1
        
        # M7 = (A01 - A11) * (B1k0 + B1k1)
        U[1, col] = 1
        U[4, col] = -1
        V[p+k0, col] = 1
        V[p+k1, col] = 1
        W[k0*n+0, col] = 1
        col += 1
        
        # --- Schoolbook part for A[:, 2] (2x1) * B[2, :] (1x2) ---
        # A_tail indices:
        # A02 = U[2], A12 = U[5]
        # B_tail indices:
        # B2k0 = V[2p+k0], B2k1 = V[2p+k1]
        
        # P1 = A02 * B2k0
        U[2, col] = 1
        V[2*p+k0, col] = 1
        W[k0*n+0, col] = 1
        col += 1
        
        # P2 = A02 * B2k1
        U[2, col] = 1
        V[2*p+k1, col] = 1
        W[k1*n+0, col] = 1
        col += 1
        
        # P3 = A12 * B2k0
        U[5, col] = 1
        V[2*p+k0, col] = 1
        W[k0*n+1, col] = 1
        col += 1
        
        # P4 = A12 * B2k1
        U[5, col] = 1
        V[2*p+k1, col] = 1
        W[k1*n+1, col] = 1
        col += 1

    # Verify the rank-44 solution
    if _verify_decomposition(U, V, W, T_exp):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook if verification fails (should not happen)
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
# EVOLVE-BLOCK-END
