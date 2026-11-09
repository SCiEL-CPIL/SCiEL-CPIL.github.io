"""Tensor rank minimization for 2x4x5 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 4, 5]
TIME_LIMIT = 120.0
SAFETY_MARGIN = 10.0


def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    # Build the multiplication tensor T[i,j,k] = delta_{j,k} (i.e., (A*B)[i,k] = sum_j A[i,j]*B[j,k])
    # T[i, j, k] = 1 if j == k else 0, but we need the full tensor for n x m x p
    # Actually, the tensor is: T[i,j,k] such that (AB)[i,k] = sum_j A[i,j] B[j,k]
    # The decomposition says: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Wait, let me re-read the convention.
    # U has shape (n*m, rank), indexed by (i*m + j, r)
    # V has shape (m*p, rank), indexed by (j*p + k, r)  
    # W has shape (p*n, rank), indexed by (k*n + i, r)
    # The tensor T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # And T[i,j,k] should equal 1 if the multiplication works out, i.e., T[i,j,k] = delta_{j,k}... 
    # No wait. For matrix multiplication C = AB where A is n x m and B is m x p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T[i,j,k] = delta_{j,k} is NOT right. The tensor is T[i,j,k] = 1 for all valid (i,j,k) 
    # in the sense that the bilinear form is sum_j A[i,j]B[j,k].
    # Actually the multiplication tensor T is defined by T[i,j,k] = 1 if we're computing the (i,k) entry 
    # using A[i,j] and B[j,k]. So T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p).
    # Wait no - the tensor entries are T[i,j,k] = 1 for all valid indices, because each term A[i,j]*B[j,k] 
    # contributes to C[i,k]. So T[i,j,k] = 1 for all i,j,k in their ranges.
    
    # Let me just verify: compute sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] and check it equals 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True


def _snap_to_half_integer(arr, tol=1e-6):
    """Snap values close to half-integers to exact half-integers."""
    snapped = []
    for v in arr:
        # Check if close to a half-integer
        half = round(2 * v) / 2.0
        if abs(v - half) < tol:
            snapped.append(half)
        else:
            snapped.append(v)
    return snapped


def _snap_matrix(M, tol=1e-6):
    """Snap a 2D list to half-integers."""
    return [[_snap_to_half_integer(row)[0] if isinstance(row, list) and len(row)==1 else row for row in M]]


def _schoolbook(n, m, p):
    """Return schoolbook decomposition."""
    rank = n * m * p
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1.0
                V[j*p+k][col] = 1.0
                W[k*n+i][col] = 1.0
                col += 1
    return U, V, W, rank


def _strassen_2x2(n, m, p):
    """
    Try to use Strassen-like decomposition for the 2x2x2 core.
    For 2x4x5, we can think of it as:
    - The m=4 dimension can be split into two 2-dimensions
    - The p=5 dimension can be split into 2+3
    - The n=2 dimension stays
    
    Actually, let's think more carefully. A 2x4x5 multiplication:
    C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    We can group B's columns: B[:, 0:2] (2 cols) and B[:, 2:4] (2 cols) and B[:, 4] (1 col)
    So C = A * B = A[:,:2] * B[:,0:2] + A[:,:2] * B[:,2:4] + A[:,:2] * B[:,4:5]... 
    No, that's not right either. Let me think differently.
    
    Split the m=4 into two groups of 2: m1=2, m2=2
    Split the p=5 into p1=2, p2=2, p3=1
    
    Then C = A * B where:
    C[:, 0:2] = A[:, 0:2] * B[0:2, 0:2] + A[:, 2:4] * B[2:4, 0:2]
    C[:, 2:4] = A[:, 0:2] * B[0:2, 2:4] + A[:, 2:4] * B[2:4, 2:4]  
    C[:, 4:5] = A[:, 0:2] * B[0:2, 4:5] + A[:, 2:4] * B[2:4, 4:5]
    
    For each 2x2x2 block multiplication, we can use Strassen (rank 7 instead of 8).
    For 2x2x1 blocks, we need rank 2 (just two multiplications).
    
    Total: 3 Strassen blocks (each rank 7) + 3 schoolbook 2x2x1 blocks (each rank 2)
    = 3*7 + 3*2 = 21 + 6 = 27
    
    Wait, but the 2x2x1 case: C[i,k] = sum_j A[i,j] B[j,k] where k is fixed.
    That's just 2 multiplications per output entry, so rank 2 per 2x2x1 block.
    
    Actually wait - for 2x2x1: C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0], C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    This has rank 4 (schoolbook) but can we do better? 
    C[0,0] = A00*B00 + A01*B10
    C[1,0] = A10*B00 + A11*B10
    This is a 2x2x1 tensor. Its rank is 4 (it's the sum of 4 rank-1 tensors, and the border rank is also 4 for this shape... actually let me check).
    
    Hmm, for 2x2x1, the tensor is T[i,j,0] = delta_{j,0}... no. T[i,j,k] where k=0 only.
    T[i,j,0] = 1 for all i,j in {0,1}. So it's just a 2x2x1 tensor with all ones.
    Rank of a 2x2x1 tensor with all ones: it's rank 2 (since it's the outer product of [1,1] and [1,1] and [1]).
    Wait no - a 2x2x1 tensor T[i,j,0] = 1 can be written as sum of 2 rank-1 tensors:
    T[i,j,0] = 1*1*1 + 0 + ... hmm.
    Actually T[i,j,0] = u_i * v_j * w_0 where u=[1,1], v=[1,1], w=[1]. That's rank 1!
    But wait, that would mean C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] can be done with rank 1?
    No, that's wrong. The tensor for 2x2x1 matrix multiplication is:
    C[0,0] = A[0,0]B[0,0] + A[0,1]B[1,0]
    C[1,0] = A[1,0]B[0,0] + A[1,1]B[1,0]
    
    The tensor T[i,j,k] has T[i,j,0] = 1 for all i,j. But the decomposition requires:
    T[i,j,0] = sum_r U[i,0,r] * V[j,0,r] * W[0,i,r]
    
    If T[i,j,0] = 1 for all i,j, then we need sum_r U[i,r]*V[j,r]*W[0,i,r] = 1 for all i,j.
    With rank 1: U[i,r]*V[j,r]*W[0,i,r] = 1 for all i,j. This requires U[0,0]*V[0,0]*W[0,0]=1, U[0,0]*V[1,0]*W[0,0]=1, etc.
    So V[0,0]=V[1,0] and U[0,0]*V[0,0]*W[0,0]=1, U[1,0]*V[0,0]*W[1,0]=1, etc.
    We can set W[0,0]=1, W[1,0]=1, U[0,0]=1, U[1,0]=1, V[0,0]=1, V[1,0]=1. Then T[i,j,0]=1 for all i,j. Yes! Rank 1 works for 2x2x1.
    
    Wait, but that can't be right for matrix multiplication. Let me re-examine.
    
    For 2x2x1: A is 2x2, B is 2x1, C is 2x1.
    C[0] = A[0,0]*B[0] + A[0,1]*B[1]
    C[1] = A[1,0]*B[0] + A[1,1]*B[1]
    
    The tensor T[i,j,k] where i indexes rows of A/C, j indexes columns of A / rows of B, k indexes columns of B/C.
    Here k only takes value 0. T[i,j,0] should be such that C[i,k] = sum_j sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] * A[i,j]*B[j,k].
    
    Hmm, I think I'm confusing myself. Let me just use the standard result: the rank of 2x2x1 matrix multiplication is 4 (it's just 4 scalar multiplications and you can't do better since each output depends on different products).
    
    Actually no. C[0] = A00*B0 + A01*B1 and C[1] = A10*B0 + A11*B1. These are 4 distinct products. The rank of this tensor is 4.
    
    OK so for 2x2x1, rank is 4 (same as schoolbook). So my Strassen approach doesn't help for the 2x2x1 parts.
    
    Let me reconsider. For 2x4x5:
    - Split m=4 into m1=2, m2=2
    - Split p=5 into p1=2, p2=2, p3=1
    
    Blocks:
    - (m1, p1): 2x2x2 -> Strassen rank 7
    - (m1, p2): 2x2x2 -> Strassen rank 7  
    - (m1, p3): 2x2x1 -> rank 4
    - (m2, p1): 2x2x2 -> Strassen rank 7
    - (m2, p2): 2x2x2 -> Strassen rank 7
    - (m2, p3): 2x2x1 -> rank 4
    
    Total: 4*7 + 2*4 = 28 + 8 = 36
    
    This matches the known result from the prompt! Score 1.111 = 40/36.
    """
    n, m, p = 2, 4, 5
    # Split m into [0,1] and [2,3], p into [0,1], [2,3], [4]
    m_splits = [(0, 2), (2, 4)]  # (start, end)
    p_splits = [(0, 2), (2, 4), (4, 5)]
    
    U = [[0.0] * 36 for _ in range(n * m)]
    V = [[0.0] * 36 for _ in range(m * p)]
    W = [[0.0] * 36 for _ in range(p * n)]
    
    col = 0
    for m_start, m_end in m_splits:
        for p_start, p_end in p_splits:
            m_size = m_end - m_start
            p_size = p_end - p_start
            
            if m_size == 2 and p_size == 2:
                # Strassen 2x2x2
                # Strassen's 7 multiplications for 2x2x2:
                # M1 = (A00+A11)(B00+B11)
                # M2 = (A10+A11)B00
                # M3 = A00(B10-B11)
                # M4 = A11(B01-B00)
                # M5 = (A00+A01)B11
                # M6 = (A10-A00)(B00+B10)
                # M7 = (A01-A11)(B01+B11)
                # C00 = M1+M4-M5+M7
                # C01 = M3+M5
                # C10 = M2+M4
                # C11 = M1-M2+M3+M6
                
                # For our indexing: i in {0,1}, j in {m_start, m_start+1}, k in {p_start, p_start+1}
                # A[i, j] corresponds to U[i*m + j, col]
                # B[j, k] corresponds to V[j*p + k, col]
                # C[i, k] corresponds to W[k*n + i, col]
                
                # Strassen factors (for 2x2x2 with local indices a,b for A columns, c,d for B rows):
                # We need: for local (i, j, k) where j in {0,1}, k in {0,1}:
                # T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
                
                # The 7 rank-1 terms for Strassen:
                # r=0: (A00+A11)(B00+B11) contributes to C00, C11
                #   U: [1,0,1,0] for (i,j) = (0,0),(0,1),(1,0),(1,1)
                #   V: [1,0,0,1] for (j,k) = (0,0),(0,1),(1,0),(1,1)  
                #   Wait, I need to be more careful.
                
                # Let me define the Strassen decomposition properly.
                # For 2x2x2, the multiplication tensor has entries T[i,j,k] = delta_{j,k}... no.
                # T[i,j,k] = 1 for all i,j,k in {0,1} (since each A[i,j]*B[j,k] contributes to C[i,k]).
                
                # Strassen's 7-term decomposition:
                # T[i,j,k] = sum_{r=0}^{6} U_r[i,j] * V_r[j,k] * W_r[k,i]
                
                # r=0: M1 = (A00+A11)(B00+B11)
                #   U0[i,j]: 1 if (i,j) in {(0,0),(1,1)}, else 0
                #   V0[j,k]: 1 if (j,k) in {(0,0),(1,1)}, else 0
                #   W0[k,i]: 1 if (k,i) in {(0,0),(1,1)}, else 0
                #   Contribution: C00 += 1, C11 += 1, C01 += 0, C10 += 0
                #   Check: T[0,0,0] gets U0[0,0]*V0[0,0]*W0[0,0] = 1*1*1 = 1 ✓
                #   T[1,1,1] gets U0[1,1]*V0[1,1]*W0[1,1] = 1*1*1 = 1 ✓
                #   T[0,0,1] gets U0[0,0]*V0[0,1]*W0[1,0] = 1*0*0 = 0 ✓ (should be 1 for T[0,0,1]!)
                #   Wait, T[0,0,1] should be 1 (A[0,0]*B[0,1] contributes to C[0,1]).
                #   But this term gives 0. That's fine, other terms will contribute.
                
                # Let me just use the known Strassen coefficients.
                # The standard Strassen decomposition of the 2x2x2 tensor:
                
                strassen_U = [
                    [1, 0, 1, 0],  # r=0: A00+A11
                    [0, 0, 1, 1],  # r=1: A10+A11
                    [1, 0, 0, 0],  # r=2: A00
                    [0, 0, 0, 1],  # r=3: A11
                    [1, 1, 0, 0],  # r=4: A00+A01
                    [1, 0, -1, 0], # r=5: A10-A00
                    [0, 1, 0, -1], # r=6: A01-A11
                ]
                strassen_V = [
                    [1, 0, 0, 1],  # r=0: B00+B11
                    [1, 0, 0, 0],  # r=1: B00
                    [0, 1, 0, -1], # r=2: B10-B11
                    [0, 1, 1, 0],  # r=3: B01-B00
                    [0, 0, 0, 1],  # r=4: B11
                    [1, 1, 0, 0],  # r=5: B00+B10
                    [0, 1, 0, 1],  # r=6: B01+B11
                ]
                strassen_W = [
                    [1, 0, 0, 1],  # r=0: contributes to C00, C11
                    [0, 0, 0, 1],  # r=1: contributes to C10
                    [0, 1, 0, 0],  # r=2: contributes to C01
                    [1, 0, 0, 0],  # r=3: contributes to C00
                    [0, 0, 0, -1], # r=4: contributes to C00 (negative)
                    [0, 0, 0, 1],  # r=5: contributes to C11
                    [0, 1, 0, 0],  # r=6: contributes to C00 (wait, need to check)
                ]
                
                # Let me verify this more carefully by checking all 8 entries.
                # T[i,j,k] for i,j,k in {0,1}:
                # T[0,0,0]=1, T[0,0,1]=1, T[0,1,0]=1, T[0,1,1]=1
                # T[1,0,0]=1, T[1,0,1]=1, T[1,1,0]=1, T[1,1,1]=1
                
                # For each (i,j,k), compute sum_r strassen_U[r][i*2+j] * strassen_V[r][j*2+k] * strassen_W[r][k*2+i]
                
                # This is getting complex. Let me just hard-code a verified Strassen decomposition.
                # I'll use the well-known one and verify it programmatically.
                
                # Actually, let me just use the fact that Strassen works and construct it carefully.
                # The 7 products are:
                # p0 = (a00+a11)*(b00+b11)
                # p1 = (a10+a11)*b00
                # p2 = a00*(b10-b11)
                # p3 = a11*(b01-b00)
                # p4 = (a00+a01)*b11
                # p5 = (a10-a00)*(b00+b10)
                # p6 = (a01-a11)*(b01+b11)
                
                # c00 = p0 + p3 - p4 + p6
                # c01 = p2 + p4
                # c10 = p1 + p3
                # c11 = p0 - p1 + p2 + p5
                
                # In tensor form, T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
                # where the "product" p_r = (linear combo of A entries) * (linear combo of B entries)
                # and c[i,k] = sum_r W[k,i,r] * p_r
                
                # So U[i,j,r] is the coefficient of A[i,j] in the first factor of p_r
                # V[j,k,r] is the coefficient of B[j,k] in the second factor of p_r
                # W[k,i,r] is the coefficient of p_r in c[i,k]
                
                # p0 = (a00+a11)*(b00+b11): U col 0 = [1,0,0,1] (for a00,a01,a10,a11), V col 0 = [1,0,0,1] (for b00,b01,b10,b11)
                # p1 = (a10+a11)*b00: U col 1 = [0,0,1,1], V col 1 = [1,0,0,0]
                # p2 = a00*(b10-b11): U col 2 = [1,0,0,0], V col 2 = [0,0,1,-1]
                # p3 = a11*(b01-b00): U col 3 = [0,0,0,1], V col 3 = [-1,1,0,0]
                # p4 = (a00+a01)*b11: U col 4 = [1,1,0,0], V col 4 = [0,0,0,1]
                # p5 = (a10-a00)*(b00+b10): U col 5 = [-1,0,1,0], V col 5 = [1,0,1,0]
                # p6 = (a01-a11)*(b01+b11): U col 6 = [0,1,0,-1], V col 6 = [0,1,0,1]
                
                # W: c00 = p0+p3-p4+p6, c01 = p2+p4, c10 = p1+p3, c11 = p0-p1+p2+p5
                # W[k*n+i, r] where n=2, so W index = k*2+i
                # W[0*2+0, r] = W[0, r] for c00
                # W[1*2+0, r] = W[2, r] for c10
                # W[0*2+1, r] = W[1, r] for c01
                # W[1*2+1, r] = W[3, r] for c11
                
                # W[0] = [1, 0, 0, 1, -1, 0, 1]  (c00)
                # W[1] = [0, 0, 1, 0, 1, 0, 0]   (c01)
                # W[2] = [0, 1, 0, 1, 0, 0, 0]   (c10)
                # W[3] = [1, -1, 1, 0, 0, 1, 0]  (c11)
                
                U_local = [
                    [1, 0, 1, 0, 1, -1, 0],   # i=0,j=0 (a00)
                    [0, 0, 0, 0, 1, 0, 1],    # i=0,j=1 (a01)
                    [0, 1, 0, 0, 0, 1, 0],    # i=1,j=0 (a10)
                    [1, 1, 0, 1, 0, 0, -1],   # i=1,j=1 (a11)
                ]
                V_local = [
                    [1, 1, 0, -1, 0, 1, 0],   # j=0,k=0 (b00)
                    [0, 0, 0, 1, 0, 0, 1],    # j=0,k=1 (b01)
                    [0, 0, 1, 0, 0, 1, 0],    # j=1,k=0 (b10)
                    [1, 0, -1, 0, 1, 0, 1],   # j=1,k=1 (b11)
                ]
                W_local = [
                    [1, 0, 0, 1, -1, 0, 1],   # k=0,i=0 (c00)
                    [0, 0, 1, 0, 1, 0, 0],    # k=0,i=1 (c10)
                    [0, 0, 1, 0, 1, 0, 0],    # k=1,i=0 (c01) -- wait
                    [1, -1, 1, 0, 0, 1, 0],   # k=1,i=1 (c11)
                ]
                
                # Wait, W index is k*n+i. For n=2:
                # k=0,i=0 -> index 0 (c00)
                # k=0,i=1 -> index 1 (c10)
                # k=1,i=0 -> index 2 (c01)
                # k=1,i=1 -> index 3 (c11)
                
                # c01 = p2+p4, so W[2] = [0,0,1,0,1,0,0]
                # c10 = p1+p3, so W[1] = [0,1,0,1,0,0,0]
                
                W_local = [
                    [1, 0, 0, 1, -1, 0, 1],   # c00
                    [0, 1, 0, 1, 0, 0, 0],    # c10
                    [0, 0, 1, 0, 1, 0, 0],    # c01
                    [1, -1, 1, 0, 0, 1, 0],   # c11
                ]
                
                # Verify: for each (i,j,k), sum_r U_local[i*2+j][r] * V_local[j*2+k][r] * W_local[k*2+i][r] should be 1
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(2):
                            s = 0.0
                            for rr in range(7):
                                s += U_local[ii*2+jj][rr] * V_local[jj*2+kk][rr] * W_local[kk*2+ii][rr]
                            if abs(s - 1.0) > 1e-9:
                                # Fallback to schoolbook for this block
                                for rr in range(7):
                                    U[i*m + m_start + jj][col + rr] = 0
                                    V[(m_start + jj)*p + p_start + kk][col + rr] = 0
                                    W[(p_start + kk)*n + ii][col + rr] = 0
                                # Use schoolbook for this 2x2x2 block (rank 8, but we only have 7 slots)
                                # This won't work. Let me just use schoolbook for the whole thing if Strassen fails.
                                pass
                            
                # Fill in the Strassen block
                for ii in range(2):
                    for jj in range(2):
                        for rr in range(7):
                            U[i*m + m_start + jj][col + rr] += U_local[ii*2+jj][rr]
                    for kk in range(2):
                        for rr in range(7):
                            V[(m_start + jj)*p + p_start + kk][col + rr] += V_local[jj*2+kk][rr]
                for kk in range(2):
                    for ii in range(2):
                        for rr in range(7):
                            W[(p_start + kk)*n + ii][col + rr] += W_local[kk*2+ii][rr]
                
                col += 7
                
            elif m_size == 2 and p_size == 1:
                # 2x2x1: schoolbook, rank 4
                for ii in range(2):
                    for jj in range(2):
                        for kk in range(1):
                            U[ii*m + m_start + jj][col] = 1.0
                            V[(m_start + jj)*p + p_start + kk][col] = 1.0
                            W[(p_start + kk)*n + ii][col] = 1.0
                            col += 1
    
    return U, V, W, 36


def _nuclear_norm_relaxation(n, m, p, max_iter=100, tol=1e-4):
    """
    Try nuclear norm minimization approach.
    This is a convex relaxation that may give insights.
    """
    start_time = time.time()
    deadline = start_time + TIME_LIMIT - SAFETY_MARGIN
    
    # Initialize with schoolbook
    U_init, V_init, W_init, rank_init = _schoolbook(n, m, p)
    
    # For now, just return schoolbook as the relaxation is complex to implement correctly
    # in the time available. The Strassen approach is more promising.
    return U_init, V_init, W_init, rank_init


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try Strassen-based decomposition first
    U, V, W, rank = _strassen_2x2(n, m, p)
    
    # Verify
    if _verify(U, V, W, n, m, p):
        # Snap to half-integers
        U_snapped = []
        for row in U:
            U_snapped.append([_snap_to_half_integer([v])[0] for v in row])
        V_snapped = []
        for row in V:
            V_snapped.append([_snap_to_half_integer([v])[0] for v in row])
        W_snapped = []
        for row in W:
            W_snapped.append([_snap_to_half_integer([v])[0] for v in row])
        
        if _verify(U_snapped, V_snapped, W_snapped, n, m, p):
            return {"U": U_snapped, "V": V_snapped, "W": W_snapped}
    
    # Fallback to schoolbook
    U, V, W, rank = _schoolbook(n, m, p)
    return {"U": U, "V": V, "W": W}
