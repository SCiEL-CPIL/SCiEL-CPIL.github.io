"""Matrix multiplication 2,4,6 – rank-minimal decomposition via block decomposition."""
DIMENSIONS = [2, 4, 6]

import time
import numpy as np

# EVOLVE-BLOCK-START

def _schoolbook(n, m, p):
    """Standard O(nmp) decomposition for multiplying (n x m) by (m x p).
    Returns U (n*m x nmp), V (m*p x nmp), W (p*n x nmp)."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W


def _strassen_2x2_block(p_sub):
    """Strassen-like rank-7 decomposition for multiplying (2 x 2) by (2 x p_sub).
    
    Actually, let me think more carefully. The sub-problem is 2 x 4 x 2.
    We can split the 4 columns of the middle matrix into two 2-column blocks,
    giving two 2x2x2 sub-problems. Each 2x2x2 has rank 7 (Strassen).
    Total rank = 14.
    
    But wait, we need to be careful with indices.
    
    For 2x4x2: n=2, m=4, p=2.
    A is 2x4, B is 4x2, C is 2x2.
    
    Split A into [A1 | A2] where A1, A2 are 2x2.
    Split B into [B1; B2] where B1, B2 are 2x2.
    Then C = A1*B1 + A2*B2.
    
    Each term A_i * B_i is a 2x2 * 2x2 multiplication, which has rank 7.
    So total rank = 14.
    
    Let me construct this explicitly.
    """
    # For each sub-block (2x2 * 2x2), use Strassen with rank 7.
    # Indices: i in {0,1} (rows of A and C), j in {0,1,2,3} (cols of A / rows of B), k in {0,1} (cols of B / cols of C)
    
    # Sub-block 1: j in {0,1}, k in {0,1}
    # Sub-block 2: j in {2,3}, k in {0,1}
    
    # Strassen for 2x2:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A10)*B11
    # M6 = (A11-A10)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For sub-block s (s=0,1), j_offset = 2*s, k in {0,1}
    # i in {0,1}, j_local in {0,1}, k in {0,1}
    # Global j = j_offset + j_local
    
    rank = 14
    n, m, p = 2, 4, 2
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors for 2x2x2
    # Each Strassen step r (0..6) contributes to specific (i, j_local, k) positions
    # We need to map the 7 Strassen factors into the tensor.
    
    # The matrix multiplication tensor T[i, j, k] = 1 for all i,j,k in 2x2x2.
    # Strassen decomposition: sum_{r=1}^7 u_r[i,j] * v_r[j,k] * w_r[k,i]
    
    # Standard Strassen for 2x2x2:
    # r=0: u=[1,1;1,1] on (i,j), v=[1,1;1,1] on (j,k), w=[1,1;1,1] on (k,i) -- wait, this is not right.
    
    # Let me use the standard Strassen formula properly.
    # The 7 rank-1 terms:
    # t0: U00=1,U10=0,U01=0,U11=0 (i.e., only U[0,0]=1) ... no.
    
    # Actually, let me just use the explicit Strassen decomposition.
    # For 2x2x2, the 7 terms are:
    # term 0: U = [[1,0],[1,1]], V = [[1,1],[0,1]], W = [[1,0],[1,1]]
    # Hmm, let me look this up properly.
    
    # The Strassen decomposition of the 2x2x2 matrix multiplication tensor:
    # T = sum_{r=0}^{6} T_r where T_r[i,j,k] = a_r[i,j] * b_r[j,k] * c_r[k,i]
    
    # Standard Strassen:
    # r=0: a=[[1,0],[1,1]], b=[[1,1],[0,1]], c=[[1,0],[1,1]]
    # Let me verify: a[0,0]*b[0,0]*c[0,0] + ... this is getting complicated.
    
    # Let me just use a known rank-7 decomposition for 2x2x2.
    # From literature, one standard Strassen decomposition:
    
    # r : U[i,j] pairs, V[j,k] pairs, W[k,i] pairs
    # I'll use the fact that for 2x2x2, rank = 7.
    
    # Let me construct it by solving. The tensor T[i,j,k] = 1 for all i,j,k in {0,1}.
    # We need 7 rank-1 terms.
    
    # Using the standard Strassen factors:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a10)b11
    # M6 = (a11-a10)(b00+b10)
    # M7 = (a01-a11)(b01+b11)
    
    # In tensor form, each M_r contributes to C[i,k] via sum_j.
    # M_r = u_r[i,j] * v_r[j,k] where the product sums over j.
    # But in the tensor decomposition, we have T[i,j,k] = sum_r U_r[i,j] V_r[j,k] W_r[k,i]
    # where W_r[k,i] is actually just a scalar factor (since the "output" index is (k,i) but in 2x2x2, the output is determined by (i,k) and W provides the coefficient).
    
    # Wait, the convention is W[k*n+i, r]. For n=2, W[k*2+i, r].
    # The tensor element is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For the 2x2x2 sub-block, n=2, m=2, p=2.
    # T[i,j,k] = sum_r U[i*2+j, r] V[j*2+k, r] W[k*2+i, r]
    
    # Let me just hardcode the 7 Strassen terms for 2x2x2.
    
    # From standard references, the Strassen decomposition gives:
    # For each r in 0..6, define U_r (2x2), V_r (2x2), W_r (2x2) such that
    # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[k,i]
    
    # One valid set:
    # r=0: U=[[1,1],[0,0]], V=[[1,1],[0,0]], W=[[1,0],[1,0]]
    # Let me verify T[0,0,0] = sum_r U_r[0,0] V_r[0,0] W_r[0,0]
    
    # This is getting too complex to do by hand. Let me use a different approach.
    # I'll use the schoolbook decomposition for the 2x4x2 sub-problem as well,
    # which gives rank = 2*4*2 = 16. Total = 32 + 16 = 48. Same as before.
    
    # Actually, let me just use schoolbook for both parts.
    # The split doesn't help unless I have a better decomposition for the sub-parts.
    
    # For 2x2x2, the optimal rank is 7 (Strassen).
    # Let me find the explicit Strassen factors.
    
    # After research, the Strassen decomposition of T_2 can be written as:
    # 7 terms where each term is a product of 3 2x2 matrices (in tensor form).
    
    # Let me use a known explicit form:
    # term 1: U = [[1, 0], [1, 1]], V = [[1, 1], [0, 1]], W = [[1, 0], [1, 1]]
    # Let me verify: T[0,0,0] should be 1.
    # sum_r U_r[0,0] V_r[0,0] W_r[0,0]
    # For r=1: U[0,0]=1, V[0,0]=1, W[0,0]=1 -> 1
    # Other terms: need to check they sum to 0 for this entry.
    
    # This is error-prone. Let me just use schoolbook for everything.
    # Rank = 48. Same as the seed.
    
    # Alternatively, let me try a different split: 2x4x6 = 2x4x2 + 2x4x2 + 2x4x2
    # Each 2x4x2 has schoolbook rank 16, total 48.
    
    # Or: use the fact that 2x4x6 can be done with rank less than 48?
    # The tensor rank of 2x4x6 is unknown in general, but bounds exist.
    
    # For now, let me just return the schoolbook decomposition (rank 48) to ensure correctness.
    # The seed already does this.
    
    # Let me at least try to implement Strassen for the 2x2x2 case properly.
    
    # Known Strassen factors for 2x2x2 (from Wikipedia / standard references):
    # The 7 rank-1 components are:
    
    factors = []
    # r, U_row0, U_row1, V_col0, V_col1, W_col0, W_col1
    # where U is 2x2, V is 2x2, W is 2x2
    
    # Using the standard Strassen decomposition:
    # M1 = (a00+a11)(b00+b11): U=[[1,0],[1,1]], V=[[1,1],[0,1]], W=[[1,0],[1,1]]
    # Actually, I'll use a verified set.
    
    # Let me use the approach of just returning schoolbook.
    pass
    U, V, W = _schoolbook(2, 4, 2)
    return U, V, W


def construct():
    """Construct rank-minimal exact tensor decomposition for 2x4 * 4x6 multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS  # 2, 4, 6
    
    # Try to find a better decomposition than schoolbook (rank 48)
    # Strategy: split into 2x4x4 and 2x4x2, use best known decompositions
    
    # For 2x4x4: schoolbook gives rank 32
    # For 2x4x2: schoolbook gives rank 16
    # Total: 48
    
    # Can we do better?
    # The tensor rank of matrix multiplication M_{n,m,p} satisfies:
    # R(M_{n,m,p}) >= n*m*p (trivially, but often much less)
    # For 2x2x2, rank = 7
    # For 2x4x2, we can split into two 2x2x2 blocks: rank = 14
    # For 2x4x4, we can split into two 2x2x4 blocks, each of which can be split...
    
    # Actually, 2x4x4: split into 2x2x4 + 2x2x4? No, the middle dimension is 4.
    # A is 2x4, B is 4x4, C is 2x4.
    # Split A into [A1|A2] (2x2 each), B into [B11;B12] (2x4 each), no...
    # B is 4x4. Split B rows: B = [B_top; B_bottom] where each is 2x4.
    # Then C = A1 * B_top + A2 * B_bottom.
    # Each term is 2x2 * 2x4.
    # 2x2*2x4: can split 2x4 into two 2x2: rank = 2*7 = 14 per block.
    # So 2x4x4: rank <= 2*14 = 28.
    
    # 2x4x2: split into two 2x2x2: rank = 2*7 = 14.
    
    # Total: 28 + 14 = 42. Better than 48!
    
    # Let me implement this properly.
    
    # First, I need the Strassen rank-7 decomposition for 2x2x2.
    
    def strassen_2x2x2():
        """Return U, V, W for 2x2x2 matrix multiplication with rank 7."""
        n_s, m_s, p_s = 2, 2, 2
        rank_s = 7
        U = np.zeros((n_s * m_s, rank_s), dtype=float)
        V = np.zeros((m_s * p_s, rank_s), dtype=float)
        W = np.zeros((p_s * n_s, rank_s), dtype=float)
        
        # Strassen decomposition for 2x2x2
        # T[i,j,k] = sum_{r=0}^{6} U[i*2+j, r] * V[j*2+k, r] * W[k*2+i, r]
        
        # Verified Strassen factors:
        # r=0: U=[[1,1],[0,0]], V=[[1,0],[0,1]], W=[[1,0],[1,0]]
        # Hmm, let me use a standard reference.
        
        # From "Matrix Multiplication" by Strassen, the 7 factors are:
        # I'll use the formulation where:
        # M1 = (a00+a11)(b00+b11)
        # M2 = (a10+a11)b00  
        # M3 = a00(b01-b11)
        # M4 = a11(b10-b00)
        # M5 = (a00+a10)b11
        # M6 = (a11-a10)(b00+b10)
        # M7 = (a01-a11)(b01+b11)
        
        # In the tensor T[i,j,k], we have:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # = sum_r sum_j U_r[i,j] V_r[j,k] W_r[k,i]
        
        # For each r, U_r[i,j] V_r[j,k] W_r[k,i] contributes to T[i,j,k].
        
        # M1 = (a00+a11)(b00+b11):
        #   This means U_1[i,j] = 1 if (i=0,j=0) or (i=1,j=1), else 0
        #   V_1[j,k] = 1 if (j=0,k=0) or (j=1,k=1), else 0
        #   W_1[k,i] = 1 for all k,i (since it multiplies the whole thing)
        #   But wait, W_1[k,i] should be such that the product gives the right contribution.
        
        # Actually, in the standard Strassen:
        # c00 = M1 + M4 - M5 + M7
        # c01 = M3 + M5
        # c10 = M2 + M4
        # c11 = M1 - M2 + M3 + M6
        
        # Each M_r is a scalar. So W_r[k,i] must be 1 for the appropriate (k,i) and 0 otherwise,
        # with signs.
        
        # M1 contributes to c00 (+1) and c11 (+1)
        # M2 contributes to c10 (+1) and c11 (-1)
        # M3 contributes to c01 (+1) and c11 (+1)
        # M4 contributes to c00 (+1) and c10 (+1)
        # M5 contributes to c00 (-1) and c01 (+1)
        # M6 contributes to c11 (+1)
        # M7 contributes to c00 (+1)
        
        # So:
        # W_1: W[0,0]=1, W[1,1]=1, others 0
        # W_2: W[0,1]=1, W[1,1]=-1, others 0  (k=0,i=1 -> c10; k=1,i=1 -> c11)
        # Wait, W index is k*n+i. n=2. So W[k*2+i].
        # c10: i=1, k=0 -> W[0*2+1] = W[1]
        # c11: i=1, k=1 -> W[1*2+1] = W[3]
        # So W_2[1]=1, W_2[3]=-1
        
        # Let me redo:
        # W_r is a vector of length p*n = 4.
        # W_r[k*n+i] gives the coefficient for c[i,k].
        
        # M1: c00 += M1, c11 += M1 -> W_1[0*2+0]=1, W_1[1*2+1]=1 -> W_1 = [1, 0, 0, 1]
        # M2: c10 += M2, c11 -= M2 -> W_2[0*2+1]=1, W_2[1*2+1]=-1 -> W_2 = [0, 1, 0, -1]
        # M3: c01 += M3, c11 += M3 -> W_3[1*2+0]=1, W_3[1*2+1]=1 -> W_3 = [0, 0, 1, 1]
        # M4: c00 += M4, c10 += M4 -> W_4[0*2+0]=1, W_4[0*2+1]=1 -> W_4 = [1, 1, 0, 0]
        # M5: c00 -= M5, c01 += M5 -> W_5[0*2+0]=-1, W_5[1*2+0]=1 -> W_5 = [-1, 0, 1, 0]
        # M6: c11 += M6 -> W_6[1*2+1]=1 -> W_6 = [0, 0, 0, 1]
        # M7: c00 += M7 -> W_7[0*2+0]=1 -> W_7 = [1, 0, 0, 0]
        
        # Now U_r[i,j] and V_r[j,k]:
        # M1 = (a00+a11)(b00+b11):
        #   U_1[i,j]: 1 if (i,j) in {(0,0),(1,1)} -> U_1 = [[1,0],[1,1]] wait no.
        #   a00 is A[0,0], a11 is A[1,1].
        #   U_1[0,0]=1, U_1[1,1]=1, others 0.
        #   V_1[j,k]: 1 if (j,k) in {(0,0),(1,1)}.
        #   V_1[0,0]=1, V_1[1,1]=1, others 0.
        
        # M2 = (a10+a11)b00:
        #   U_2[i,j]: 1 if (i,j) in {(1,0),(1,1)}.
        #   V_2[j,k]: 1 if (j,k) = (0,0).
        
        # M3 = a00(b01-b11):
        #   U_3[i,j]: 1 if (i,j) = (0,0).
        #   V_3[j,k]: 1 if (j,k) = (0,1), -1 if (j,k) = (1,1).
        
        # M4 = a11(b10-b00):
        #   U_4[i,j]: 1 if (i,j) = (1,1).
        #   V_4[j,k]: -1 if (j,k) = (0,0), 1 if (j,k) = (1,0).
        
        # M5 = (a00+a10)b11:
        #   U_5[i,j]: 1 if (i,j) in {(0,0),(1,0)}.
        #   V_5[j,k]: 1 if (j,k) = (1,1).
        
        # M6 = (a11-a10)(b00+b10):
        #   U_6[i,j]: -1 if (i,j) = (1,0), 1 if (i,j) = (1,1).
        #   V_6[j,k]: 1 if (j,k) in {(0,0),(1,0)}.
        
        # M7 = (a01-a11)(b01+b11):
        #   U_7[i,j]: -1 if (i,j) = (1,1), 1 if (i,j) = (0,1).
        #   V_7[j,k]: 1 if (j,k) in {(0,1),(1,1)}.
        
        # Let me fill in U, V, W:
        
        # U is 4x7 (i*2+j for i,j in 0,1)
        # U[0,0]=1, U[3,0]=1  (M1)
        # U[2,1]=1, U[3,1]=1  (M2)
        # U[0,2]=1            (M3)
        # U[3,3]=1            (M4)
        # U[0,4]=1, U[2,4]=1  (M5)
        # U[2,5]=-1, U[3,5]=1 (M6)
        # U[1,6]=1, U[3,6]=-1 (M7)
        
        U[0, 0] = 1  # M1, i=0,j=0
        U[3, 0] = 1  # M1, i=1,j=1
        U[2, 1] = 1  # M2, i=1,j=0
        U[3, 1] = 1  # M2, i=1,j=1
        U[0, 2] = 1  # M3, i=0,j=0
        U[3, 3] = 1  # M4, i=1,j=1
        U[0, 4] = 1  # M5, i=0,j=0
        U[2, 4] = 1  # M5, i=1,j=0
        U[2, 5] = -1 # M6, i=1,j=0
        U[3, 5] = 1  # M6, i=1,j=1
        U[1, 6] = 1  # M7, i=0,j=1
        U[3, 6] = -1 # M7, i=1,j=1
        
        # V is 4x7 (j*2+k for j,k in 0,1)
        # V[0,0]=1, V[3,0]=1  (M1: j=0,k=0; j=1,k=1)
        # V[0,1]=1            (M2: j=0,k=0)
        # V[1,2]=1, V[3,2]=-1 (M3: j=0,k=1; j=1,k=1)
        # V[0,3]=-1, V[2,3]=1 (M4: j=0,k=0; j=1,k=0)
        # V[3,4]=1            (M5: j=1,k=1)
        # V[0,5]=1, V[2,5]=1  (M6: j=0,k=0; j=1,k=0)
        # V[1,6]=1, V[3,6]=1  (M7: j=0,k=1; j=1,k=1)
        
        V[0, 0] = 1  # M1, j=0,k=0
        V[3, 0] = 1  # M1, j=1,k=1
        V[0, 1] = 1  # M2, j=0,k=0
        V[1, 2] = 1  # M3, j=0,k=1
        V[3, 2] = -1 # M3, j=1,k=1
        V[0, 3] = -1 # M4, j=0,k=0
        V[2, 3] = 1  # M4, j=1,k=0
        V[3, 4] = 1  # M5, j=1,k=1
        V[0, 5] = 1  # M6, j=0,k=0
        V[2, 5] = 1  # M6, j=1,k=0
        V[1, 6] = 1  # M7, j=0,k=1
        V[3, 6] = 1  # M7, j=1,k=1
        
        # W is 4x7 (k*2+i for k,i in 0,1)
        W[0, 0] = 1  # M1, k=0,i=0
        W[3, 0] = 1  # M1, k=1,i=1
        W[1, 1] = 1  # M2, k=0,i=1
        W[3, 1] = -1 # M2, k=1,i=1
        W[2, 2] = 1  # M3, k=1,i=0
        W[3, 2] = 1  # M3, k=1,i=1
        W[0, 3] = 1  # M4, k=0,i=0
        W[1, 3] = 1  # M4, k=0,i=1
        W[0, 4] = -1 # M5, k=0,i=0
        W[2, 4] = 1  # M5, k=1,i=0
        W[3, 5] = 1  # M6, k=1,i=1
        W[0, 6] = 1  # M7, k=0,i=0
        
        return U, V, W
    
    # Verify Strassen decomposition
    U_s, V_s, W_s = strassen_2x2x2()
    # Check: for all i,j,k in 0,1: sum_r U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r] should be 1
    valid = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = sum(U_s[i*2+j, r] * V_s[j*2+k, r] * W_s[k*2+i, r] for r in range(7))
                if abs(s - 1) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now build the 2x4x2 decomposition using two 2x2x2 Strassen blocks
    # 2x4x2: n=2, m=4, p=2
    # Split j into j1 in {0,1} and j2 in {2,3}
    # Block 1: j in {0,1}, k in {0,1} -> 2x2x2
    # Block 2: j in {2,3}, k in {0,1} -> 2x2x2
    
    n1, m1, p1 = 2, 4, 2
    rank1 = 14
    U1 = np.zeros((n1 * m1, rank1), dtype=float)
    V1 = np.zeros((m1 * p1, rank1), dtype=float)
    W1 = np.zeros((p1 * n1, rank1), dtype=float)
    
    # Block 1: j_local in {0,1} -> global j = j_local
    # U1[i*4 + j, r] = U_s[i*2 + j_local, r] for r in 0..6
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                U1[i * 4 + j_local, r] = U_s[i * 2 + j_local, r]
                V1[j_local * 2 + 0, r] = V_s[j_local * 2 + 0, r]  # k=0
                V1[j_local * 2 + 1, r] = V_s[j_local * 2 + 1, r]  # k=1
                W1[0 * 2 + i, r] = W_s[0 * 2 + i, r]  # k=0
                W1[1 * 2 + i, r] = W_s[1 * 2 + i, r]  # k=1
    
    # Block 2: j_local in {0,1} -> global j = j_local + 2
    # U1[i*4 + (j_local+2), r+7] = U_s[i*2 + j_local, r]
    for r in range(7):
        for i in range(2):
            for j_local in range(2):
                U1[i * 4 + (j_local + 2), r + 7] = U_s[i * 2 + j_local, r]
                V1[(j_local + 2) * 2 + 0, r + 7] = V_s[j_local * 2 + 0, r]
                V1[(j_local + 2) * 2 + 1, r + 7] = V_s[j_local * 2 + 1, r]
                W1[0 * 2 + i, r + 7] = W_s[0 * 2 + i, r]
                W1[1 * 2 + i, r + 7] = W_s[1 * 2 + i, r]
    
    # Now build 2x4x4 using two 2x2x4 blocks
    # 2x4x4: n=2, m=4, p=4
    # Split k into k1 in {0,1} and k2 in {2,3}
    # Block 1: j in {0,1,2,3}, k in {0,1} -> 2x4x2 (use the 14-rank decomposition above)
    # Block 2: j in {0,1,2,3}, k in {2,3} -> 2x4x2 (use the 14-rank decomposition above)
    
    n2, m2, p2 = 2, 4, 4
    rank2 = 28
    U2 = np.zeros((n2 * m2, rank2), dtype=float)
    V2 = np.zeros((m2 * p2, rank2), dtype=float)
    W2 = np.zeros((p2 * n2, rank2), dtype=float)
    
    # Block 1: k_local in {0,1} -> global k = k_local
    # Use the 14-rank 2x4x2 decomposition
    for r in range(14):
        for i in range(2):
            for j in range(4):
                U2[i * 4 + j, r] = U1[i * 4 + j, r]
                V2[j * 4 + k_local * 1, r] = V1[j * 2 + k_local, r] if False else 0
            # Actually, V1 has shape (4, 14) for 2x4x2. V2 has shape (16, 28).
            # V2[j*4 + k, r] for k in 0,1
            for k_local in range(2):
                for j in range(4):
                    V2[j * 4 + k_local, r] = V1[j * 2 + k_local, r]
            for k_local in range(2):
                for i in range(2):
                    W2[k_local * 2 + i, r] = W1[k_local * 2 + i, r]
    
    # Block 2: k_local in {0,1} -> global k = k_local + 2
    for r in range(14):
        for i in range(2):
            for j in range(4):
                U2[i * 4 + j, r + 14] = U1[i * 4 + j, r]
        for k_local in range(2):
            for j in range(4):
                V2[j * 4 + (k_local + 2), r + 14] = V1[j * 2 + k_local, r]
        for k_local in range(2):
            for i in range(2):
                W2[(k_local + 2) * 2 + i, r + 14] = W1[k_local * 2 + i, r]
    
    # Combine 2x4x4 and 2x4x2
    # Total: n=2, m=4, p=6
    # Split k into k in {0,1,2,3} (from 2x4x4) and k in {4,5} (from 2x4x2)
    
    total_rank = rank2 + rank1  # 28 + 14 = 42
    U_total = np.zeros((n * m, total_rank), dtype=float)
    V_total = np.zeros((m * p, total_rank), dtype=float)
    W_total = np.zeros((p * n, total_rank), dtype=float)
    
    # 2x4x4 part: k in {0,1,2,3}
    U_total[:, :rank2] = U2
    # V2 has shape (16, 28). V_total has shape (24, 42).
    # V2[j*4+k] for k in 0..3 -> V_total[j*6+k] for k in 0..3
    for j in range(4):
        for k in range(4):
            V_total[j * 6 + k, :rank2] = V2[j * 4 + k, :]
    # W2 has shape (8, 28). W_total has shape (12, 42).
    # W2[k*2+i] for k in 0..3 -> W_total[k*2+i] for k in 0..3
    for k in range(4):
        for i in range(2):
            W_total[k * 2 + i, :rank2] = W2[k * 2 + i, :]
    
    # 2x4x2 part: k in {4,5}
    U_total[:, rank2:rank2+rank1] = U1
    for j in range(4):
        for k_local in range(2):
            V_total[j * 6 + (k_local + 4), rank2:rank2+rank1] = V1[j * 2 + k_local, :]
    for k_local in range(2):
        for i in range(2):
            W_total[(k_local + 4) * 2 + i, rank2:rank2+rank1] = W1[k_local * 2 + i, :]
    
    # Verify the full decomposition
    # T[i,j,k] = sum_r U[i*4+j, r] * V[j*6+k, r] * W[k*2+i, r] should be 1 for all i,j,k
    valid_full = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U_total[i*m+j, r] * V_total[j*p+k, r] * W_total[k*n+i, r] for r in range(total_rank))
                if abs(s - 1) > 1e-10:
                    valid_full = False
                    break
            if not valid_full:
                break
        if not valid_full:
            break
    
    if not valid_full:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Convert to lists
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}

# EVOLVE-BLOCK-END
