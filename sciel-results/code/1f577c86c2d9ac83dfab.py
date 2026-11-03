"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np

_DEADLINE = time.time() + 110.0  # safety margin before 120s kill

def _is_valid(U, V, W, n, m, p, tol=1e-9):
    """Check that U @ diag @ V^T == W^T (tensor decomposition identity)."""
    rank = len(U[0])
    # Tensor T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We verify by checking: for the multiplication tensor,
    # C[i,k] = sum_j A[i,j]*B[j,k]
    # The tensor is T[i,j,k] = delta (i.e., T[i,j,k]=1 iff the (i,k) entry of A@B uses A[i,j]*B[j,k])
    # Actually the matrix multiplication tensor M_{i,j,k} = 1 for all (i,j,k) in range,
    # and the decomposition says M_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # Wait, that's not right. Let me reconsider.
    #
    # The matrix multiplication tensor: T is a 3-tensor of size n x m x p where
    # T[i,j,k] = 1 (it's the identity in the sense that C[i,k] = sum_j A[i,j] B[j,k])
    # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # with U indexed as U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    
    # Verify: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1 for all i,j,k
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Compute the reconstructed tensor
    # For each c, the outer product U[:,c] (reshaped to n,m) * V[:,c] (reshaped to m,p) * W[:,c] (reshaped to p,n)
    # Actually it's a sum of rank-1 tensors.
    # Reconstruct: R[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # This equals the matmul: (U_reshaped @ diag @ V_reshaped^T) ... no.
    # Simplest: loop over c and accumulate.
    # But for speed, use: for each c, outer = np.einsum('a,b,c->abc', 
    #   U_arr[:,c].reshape(n,m), V_arr[:,c].reshape(m,p), W_arr[:,c].reshape(p,n))
    # That's expensive. Instead, use the fact that:
    # R[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # = sum_c (U_reshaped[i,j,c]) (V_reshaped[j,k,c]) (W_reshaped[k,i,c])
    
    # Use einsum: 'ic,jc,kc->ijk' where i=n, j=m, k=p
    # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
    U_r = U_arr.reshape(n, m, rank)
    V_r = V_arr.reshape(m, p, rank)
    W_r = W_arr.reshape(p, n, rank)
    R = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
    
    # The target tensor is all ones (since T[i,j,k] = 1 for the matmul tensor)
    target = np.ones((n, m, p))
    return np.allclose(R, target, atol=tol)


def _validate_integer(U, V, W, n, m, p, tol=1e-6):
    """Check validity with stricter tolerance and verify entries are near integer/half-integer."""
    if not _is_valid(U, V, W, n, m, p, tol=1e-6):
        return False
    return True


def _round_to_halfint(arr):
    """Round array entries to nearest half-integer."""
    return np.round(np.array(arr) * 2.0) / 2.0


def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 40
    
    # Start with schoolbook as fallback
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Strategy: Use Strassen-like block decomposition.
    # Split 2x4x5 into blocks. 
    # Approach: treat as (2x2)@(2x4) but that doesn't directly help.
    # Better: split the 4 columns of A and 4 rows of B into two 2-column/2-row blocks.
    # A is 2x4, B is 4x5. Split B's 4 rows into two groups of 2: B1 (2x5), B2 (2x5)
    # A's 4 columns into A1 (2x2), A2 (2x2)
    # C = A1@B1 + A2@B2
    # Each 2x2 @ 2x5 can be done with rank... 2x2x5.
    # For 2x2x5: we can use Strassen for the 2x2 part (rank 7) applied to each of 5 columns?
    # No, that gives 7*5 = 35. Not great.
    # 
    # Alternative: 2x2x2 Strassen has rank 7. 2x2x1 is rank 4 (schoolbook).
    # 2x2x5 = 2x2x2 + 2x2x2 + 2x2x1 = 7 + 7 + 4 = 18 per block, times 2 blocks = 36.
    #
    # Let's implement this: 
    # C = A1@B1 + A2@B2 where A1,A2 are 2x2, B1,B2 are 2x5.
    # For each (A_i, B_i) pair (2x2 @ 2x5):
    #   Split B_i into B_i1 (2x2) and B_i2 (2x3)? No, let's split into 2x2 + 2x2 + 2x1.
    #   Actually 2x5 = 2x2 + 2x2 + 2x1 (columns 0-1, 2-3, 4).
    #   A_i @ B_i1: 2x2 @ 2x2 -> Strassen rank 7
    #   A_i @ B_i2: 2x2 @ 2x2 -> Strassen rank 7  
    #   A_i @ B_i3: 2x2 @ 2x1 -> schoolbook rank 4
    #   Total per block: 7+7+4 = 18
    #   Two blocks: 36
    #
    # Let me build this explicitly.
    
    def strassen_2x2x2():
        """Return (U, V, W) for 2x2x2 Strassen decomposition, rank 7.
        U shape (4,7), V shape (4,7), W shape (4,7).
        Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
        """
        # Strassen: M1=(a+d)(e+h), M2=(c+d)e, M3=a(f-h), M4=d(g-h), M5=(a+b)i, M6=(c-d)j, M7=(b+c)(e+h)
        # Wait, standard Strassen for C=A@B where A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
        # C[0,0] = M1 - M4 + M5 - M7
        # C[0,1] = M3 + M4
        # C[1,0] = M2 + M5
        # C[1,1] = M1 - M2 + M3 + M6
        # where:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-h)
        # M5 = (a+b)i  -- wait, B doesn't have i. Let me recheck.
        # B = [[e,f],[g,h]]
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-h)
        # M5 = (a+b)g
        # M6 = (c-d)h
        # M7 = (b+c)(e+f)
        
        # The tensor decomposition: C[i,k] = sum_j A[i,j] B[j,k]
        # T[i,j,k] = 1 for all i,j,k in {0,1}
        # We need: sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        
        # Let me just use the known rank-7 decomposition.
        # Actually, let me construct it from the Strassen formula.
        # C[0,0] = M1 - M4 + M5 - M7
        # C[0,1] = M3 + M4
        # C[1,0] = M2 + M5
        # C[1,1] = M1 - M2 + M3 + M6
        
        # Each M_c is a product of a linear form in A and a linear form in B.
        # M1: A_form = a+d = A[0,0]+A[1,0], B_form = e+h = B[0,0]+B[1,1]
        # M2: A_form = c+d = A[1,0]+A[1,1], B_form = e = B[0,0]
        # M3: A_form = a = A[0,0], B_form = f-h = B[0,1]-B[1,1]
        # M4: A_form = d = A[1,1], B_form = g-h = B[1,0]-B[1,1]
        # M5: A_form = a+b = A[0,0]+A[0,1], B_form = g = B[1,0]
        # M6: A_form = c-d = A[1,0]-A[1,1], B_form = h = B[1,1]
        # M7: A_form = b+c = A[0,1]+A[1,0], B_form = e+f = B[0,0]+B[0,1]
        
        # For the tensor T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
        # We need W[k,i,c] such that the reconstruction works.
        # From Strassen: C[i,k] = sum_c coeff[i,k,c] * M_c
        # where M_c = (A_form_c) * (B_form_c)
        # And A_form_c = sum_j a_c[j] * A[?,j]... 
        # Actually A_form_c is a linear combination of A entries, B_form_c of B entries.
        # A_form_c = sum_j alpha_c[j] * A[row_c, j]? No, it's a specific linear form.
        
        # Let me think differently. The standard approach:
        # U[i,j,c] = coefficient of A[i,j] in A_form_c
        # V[j,k,c] = coefficient of B[j,k] in B_form_c
        # W[k,i,c] = coefficient in C[i,k] expression for M_c
        
        # M1: A_form = A[0,0]+A[1,0] -> U[0,0,0]=1, U[1,0,0]=1
        #       B_form = B[0,0]+B[1,1] -> V[0,0,0]=1, V[1,1,0]=1
        #       C[0,0] has +1, C[1,1] has +1 -> W[0,0,0]=1, W[1,1,0]=1
        # M2: A_form = A[1,0]+A[1,1] -> U[1,0,1]=1, U[1,1,1]=1
        #       B_form = B[0,0] -> V[0,0,1]=1
        #       C[1,0] has +1, C[1,1] has -1 -> W[0,1,1]=1, W[1,1,1]=-1
        # M3: A_form = A[0,0] -> U[0,0,2]=1
        #       B_form = B[0,1]-B[1,1] -> V[0,1,2]=1, V[1,1,2]=-1
        #       C[0,1] has +1, C[1,1] has +1 -> W[1,0,2]=1, W[1,1,2]=1
        # M4: A_form = A[1,1] -> U[1,1,3]=1
        #       B_form = B[1,0]-B[1,1] -> V[1,0,3]=1, V[1,1,3]=-1
        #       C[0,0] has -1, C[0,1] has +1 -> W[0,0,3]=-1, W[1,0,3]=1
        # M5: A_form = A[0,0]+A[0,1] -> U[0,0,4]=1, U[0,1,4]=1
        #       B_form = B[1,0] -> V[1,0,4]=1
        #       C[0,0] has +1, C[1,0] has +1 -> W[0,0,4]=1, W[0,1,4]=1
        # M6: A_form = A[1,0]-A[1,1] -> U[1,0,5]=1, U[1,1,5]=-1
        #       B_form = B[1,1] -> V[1,1,5]=1
        #       C[1,1] has +1 -> W[1,1,5]=1
        # M7: A_form = A[0,1]+A[1,0] -> U[0,1,6]=1, U[1,0,6]=1
        #       B_form = B[0,0]+B[0,1] -> V[0,0,6]=1, V[0,1,6]=1
        #       C[0,0] has -1 -> W[0,0,6]=-1
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # U indexed as U[i*2+j, c]
        # M1: c=0
        U_s[0*2+0][0] = 1  # A[0,0]
        U_s[1*2+0][0] = 1  # A[1,0]
        V_s[0*2+0][0] = 1  # B[0,0]
        V_s[1*2+1][0] = 1  # B[1,1]
        W_s[0*2+0][0] = 1  # C[0,0]
        W_s[1*2+1][0] = 1  # C[1,1]
        # M2: c=1
        U_s[1*2+0][1] = 1  # A[1,0]
        U_s[1*2+1][1] = 1  # A[1,1]
        V_s[0*2+0][1] = 1  # B[0,0]
        W_s[0*2+1][1] = 1  # C[1,0] -> W[k*n+i, c] = W[0*2+1, 1]
        W_s[1*2+1][1] = -1 # C[1,1] -> W[1*2+1, 1]
        # M3: c=2
        U_s[0*2+0][2] = 1  # A[0,0]
        V_s[0*2+1][2] = 1  # B[0,1]
        V_s[1*2+1][2] = -1 # B[1,1]
        W_s[1*2+0][2] = 1  # C[0,1] -> W[1*2+0, 2]
        W_s[1*2+1][2] = 1  # C[1,1] -> W[1*2+1, 2]
        # M4: c=3
        U_s[1*2+1][3] = 1  # A[1,1]
        V_s[1*2+0][3] = 1  # B[1,0]
        V_s[1*2+1][3] = -1 # B[1,1]
        W_s[0*2+0][3] = -1 # C[0,0]
        W_s[1*2+0][3] = 1  # C[0,1]
        # M5: c=4
        U_s[0*2+0][4] = 1  # A[0,0]
        U_s[0*2+1][4] = 1  # A[0,1]
        V_s[1*2+0][4] = 1  # B[1,0]
        W_s[0*2+0][4] = 1  # C[0,0]
        W_s[0*2+1][4] = 1  # C[1,0]
        # M6: c=5
        U_s[1*2+0][5] = 1  # A[1,0]
        U_s[1*2+1][5] = -1 # A[1,1]
        V_s[1*2+1][5] = 1  # B[1,1]
        W_s[1*2+1][5] = 1  # C[1,1]
        # M7: c=6
        U_s[0*2+1][6] = 1  # A[0,1]
        U_s[1*2+0][6] = 1  # A[1,0]
        V_s[0*2+0][6] = 1  # B[0,0]
        V_s[0*2+1][6] = 1  # B[0,1]
        W_s[0*2+0][6] = -1 # C[0,0]
        
        return U_s, V_s, W_s
    
    def schoolbook_2x2x2():
        """Return (U, V, W) for 2x2x2 schoolbook, rank 4."""
        U_s = [[0]*4 for _ in range(4)]
        V_s = [[0]*4 for _ in range(4)]
        W_s = [[0]*4 for _ in range(4)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    U_s[i*2+j][c] = 1
                    V_s[j*2+k][c] = 1
                    W_s[k*2+i][c] = 1
                    c += 1
        return U_s, V_s, W_s
    
    def schoolbook_2x2x1():
        """Return (U, V, W) for 2x2x1 schoolbook, rank 4.
        U shape (4,4), V shape (2,4), W shape (2,4).
        """
        U_s = [[0]*4 for _ in range(4)]
        V_s = [[0]*4 for _ in range(2)]
        W_s = [[0]*4 for _ in range(2)]
        c = 0
        for i in range(2):
            for j in range(2):
                for k in range(1):
                    U_s[i*2+j][c] = 1
                    V_s[j*1+k][c] = 1
                    W_s[k*2+i][c] = 1
                    c += 1
        return U_s, V_s, W_s
    
    # Build the 36-rank decomposition:
    # C = A1@B1 + A2@B2
    # A1 = A[:, 0:2], A2 = A[:, 2:4]
    # B1 = B[0:2, :], B2 = B[2:4, :]
    # Each 2x2 @ 2x5 = 2x2@2x2 (Strassen, rank 7) + 2x2@2x2 (Strassen, rank 7) + 2x2@2x1 (schoolbook, rank 4)
    
    # Total rank = 2 * (7 + 7 + 4) = 36
    rank_total = 36
    U_full = [[0] * rank_total for _ in range(n * m)]  # (8, 36)
    V_full = [[0] * rank_total for _ in range(m * p)]  # (20, 36)
    W_full = [[0] * rank_total for _ in range(p * n)]  # (10, 36)
    
    col = 0
    for block in range(2):
        # A_block = A[:, block*2 : block*2+2], B_block = B[block*2 : block*2+2, :]
        # Split B_block (2x5) into: cols 0-1 (2x2), cols 2-3 (2x2), col 4 (2x1)
        for sub in range(3):
            if sub < 2:
                # 2x2 @ 2x2 Strassen
                Us, Vs, Ws = strassen_2x2x2()
                sub_rank = 7
            else:
                # 2x2 @ 2x1 schoolbook
                Us, Vs, Ws = schoolbook_2x2x1()
                sub_rank = 4
            
            for c in range(sub_rank):
                # U: rows correspond to A_block rows. A_block row i -> full row i (since n=2, A is 2x4, 
                # A_block uses columns block*2+j, so A[i, block*2+j] is at row i*4 + (block*2+j) in U_full)
                # Wait, U has shape (n*m, rank) = (8, rank). U[i*m+j, c] where i in [0,2), j in [0,4).
                # A[i, j] where j = block*2 + j_sub, j_sub in [0,2)
                for i in range(2):
                    for j_sub in range(2):
                        j_full = block * 2 + j_sub
                        U_full[i * m + j_full][col + c] = Us[i * 2 + j_sub][c]
                
                # V: rows correspond to B_block. V[j*p+k, c] where j in [0,4), k in [0,5).
                # B_block = B[block*2 : block*2+2, :], so B[block*2+j_sub, k] 
                # j_full = block*2 + j_sub, k_full = sub_offset + k_sub
                # For sub=0: k in [0,2), sub=1: k in [2,4), sub=2: k in [4,5)
                k_start = sub * 2 if sub < 2 else 4
                k_count = 2 if sub < 2 else 1
                for j_sub in range(2):
                    j_full = block * 2 + j_sub
                    for k_sub in range(k_count):
                        k_full = k_start + k_sub
                        V_full[j_full * p + k_full][col + c] = Vs[j_sub * k_count + k_sub][c] if sub >= 2 else Vs[j_sub * 2 + k_sub][c]
                
                # W: W[k*n+i, c]. k in [0,5), i in [0,2).
                # The sub-problem computes C_sub[i, k_sub] which contributes to C[i, k_full]
                for k_sub in range(k_count):
                    k_full = k_start + k_sub
                    for i in range(2):
                        W_full[k_full * n + i][col + c] = Ws[k_sub * 2 + i][c] if sub >= 2 else Ws[k_sub * 2 + i][c]
            
            col += sub_rank
    
    # Verify
    if _is_valid(U_full, V_full, W_full, n, m, p, tol=1e-6):
        return {"U": U_full, "V": V_full, "W": W_full}
    
    # Fallback to schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
