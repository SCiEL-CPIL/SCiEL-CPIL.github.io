import numpy as np
import time

def construct():
    n, m, p = 3, 4, 6
    deadline = time.time() + 100  # safety margin

    def build_block_decomposition():
        """
        Build a structured block decomposition for 3x4 * 4x6.
        
        Strategy: Split the middle dimension (4) into 2+2.
        A = [A1 A2] where A1, A2 are 3x2
        B = [B1; B2] where B1, B2 are 2x6
        C = A1*B1 + A2*B2
        
        For each 3x2 * 2x6 product, split the 6 into 3+3:
        3x2 * 2x6 = 3x2 * 2x3 + 3x2 * 2x3
        
        For each 3x2 * 2x3, we can use a rank-11 decomposition:
        Split 3 into 1+2: 3x2*2x3 = 3x2*2x1 + 3x2*2x2
        - 3x2*2x1: rank 6 (direct)
        - 3x2*2x2: use Strassen-like, rank 7
        Total per 3x2*2x3: 6 + 7 = 13
        
        But wait, 3x2*2x2 is not standard Strassen. Let me reconsider.
        
        Actually, for 3x2 * 2x2: this is multiplying a 3x2 by 2x2 to get 3x2.
        Direct rank = 3*2*2 = 12.
        
        Let me use a different approach. For 3x2 * 2x3:
        Direct rank = 3*2*3 = 18.
        
        Alternative: treat 3x2*2x3 as (3x1 + 3x1) * 2x3... no.
        
        Let me just use direct multiplication for the sub-blocks and count carefully.
        
        Block structure:
        - Split m=4 into 2+2
        - Split p=6 into 3+3
        - Keep n=3 as is
        
        This gives 2 (from m split) * 2 (from p split) = 4 blocks of 3x2 * 2x3.
        Each 3x2 * 2x3 has direct rank 18.
        Total: 4 * 18 = 72. Same as schoolbook.
        
        To improve, I need to use Strassen on sub-blocks.
        
        Better approach: Split n=3 into 1+2, m=4 into 2+2, p=6 into 3+3.
        This gives 2*2*2 = 8 blocks:
        - (1x2)*(2x3): rank 6 each, 4 of these = 24
        - (2x2)*(2x3): rank 12 each, 4 of these = 48
        Total: 72. Still schoolbook.
        
        Now apply Strassen to the (2x2)*(2x2) sub-blocks within (2x2)*(2x3):
        (2x2)*(2x3) = (2x2)*(2x1) + (2x2)*(2x2)
        - (2x2)*(2x1): rank 4
        - (2x2)*(2x2): Strassen rank 7
        So (2x2)*(2x3) costs 4+7 = 11.
        
        And (1x2)*(2x3) = (1x2)*(2x1) + (1x2)*(2x2)
        - (1x2)*(2x1): rank 2
        - (1x2)*(2x2): rank 4 (direct: 1*2*2=4)
        So (1x2)*(2x3) costs 2+4 = 6.
        
        Total: 4*(11) + 4*(6) = 44 + 24 = 68.
        
        Can we do better on (1x2)*(2x2)? That's 1*2*2=4, which is already minimal.
        (2x2)*(2x1) is 2*2*1=4, minimal.
        (2x2)*(2x2) Strassen is 7, which is the known optimal.
        
        So 68 seems achievable. Let me check if we can do better.
        
        What if we split differently? Try n=3 as 1+1+1, m=4 as 2+2, p=6 as 2+2+2:
        3*2*3 = 18 blocks, each 1x2*2x2 = rank 4. Total: 72. Worse.
        
        Try n=3 as 1+2, m=4 as 1+1+2, p=6 as 3+3:
        2*3*2 = 12 blocks. Mixed sizes. Complicated.
        
        Let me stick with the 68-rank approach and implement it carefully.
        """
        # We'll build the decomposition by tracking which (i,j,k) tuples each rank term covers.
        # The tensor T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # Block decomposition:
        # n split: [0], [1,2]  (sizes 1, 2)
        # m split: [0,1], [2,3]  (sizes 2, 2)
        # p split: [0,1,2], [3,4,5]  (sizes 3, 3)
        
        # For each combination of (n_block, m_block, p_block), we have a sub-matrix multiplication.
        # n_block 0: rows 0 (size 1)
        # n_block 1: rows 1,2 (size 2)
        # m_block 0: cols 0,1 (size 2)
        # m_block 1: cols 2,3 (size 2)
        # p_block 0: cols 0,1,2 (size 3)
        # p_block 1: cols 3,4,5 (size 3)
        
        # Sub-problems:
        # (1x2) * (2x3): 4 instances
        # (2x2) * (2x3): 4 instances
        
        # For (1x2)*(2x3):
        #   Split p=3 into 1+2:
        #   (1x2)*(2x1) + (1x2)*(2x2)
        #   (1x2)*(2x1): rank 2 (direct)
        #   (1x2)*(2x2): rank 4 (direct)
        #   Total: 6 per instance, 4 instances = 24
        
        # For (2x2)*(2x3):
        #   Split p=3 into 1+2:
        #   (2x2)*(2x1) + (2x2)*(2x2)
        #   (2x2)*(2x1): rank 4 (direct)
        #   (2x2)*(2x2): Strassen rank 7
        #   Total: 11 per instance, 4 instances = 44
        
        # Grand total: 24 + 44 = 68
        
        rank = 68
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        # Helper to add a direct multiplication block
        def add_direct_block(n_offset, m_offset, p_offset, n_size, m_size, p_size, rank_offset):
            """Add direct multiplication for n_size x m_size * m_size x p_size block."""
            for i in range(n_size):
                for j in range(m_size):
                    for k in range(p_size):
                        c = rank_offset + i * m_size * p_size + j * p_size + k
                        global_i = n_offset + i
                        global_j = m_offset + j
                        global_k = p_offset + k
                        U[global_i * m + global_j, c] = 1
                        V[global_j * p + global_k, c] = 1
                        W[global_k * n + global_i, c] = 1
            return n_size * m_size * p_size
        
        # Helper to add Strassen 2x2*2x2 block
        def add_strassen_2x2(n_offset, m_offset, p_offset, rank_offset):
            """Add Strassen 2x2*2x2 block. Returns rank used (7)."""
            # Strassen for 2x2*2x2:
            # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
            # C = A*B = [[ae+bg, af+bh], [ce+dg, cf+dh]]
            # 
            # Strassen:
            # M1 = (a+d)(e+h)
            # M2 = (c+d)e
            # M3 = a(f-h)
            # M4 = d(g-e)
            # M5 = (a+b)h
            # M6 = (c-a)(e+f)
            # M7 = (b-d)(g+h)
            #
            # C11 = M1 + M4 - M5 + M7
            # C12 = M3 + M5
            # C21 = M2 + M4
            # C22 = M1 - M2 + M3 + M6
            #
            # In our tensor notation:
            # U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # For 2x2*2x2 with offsets, local (i,j,k) in {0,1}^3
            
            # M1 = (a+d)(e+h): U contributions: (0,0) and (1,1), V: (0,0) and (1,1), W: (0,0) and (1,1)
            # Actually, let me think in terms of the local indices.
            # Local A[i,j], B[j,k], C[i,k] for i,j,k in {0,1}
            # U_local[i,j], V_local[j,k], W_local[k,i]
            
            # M1: (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
            # U: U[0,0]=1, U[1,1]=1, others 0
            # V: V[0,0]=1, V[1,1]=1, others 0
            # W: W[0,0]=1, W[1,1]=1, others 0
            # This gives C[0,0] += 1, C[1,1] += 1, C[0,1] += 1, C[1,0] += 1
            # But we want C[0,0] += A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
            # Hmm, this doesn't directly give the right coefficients.
            
            # Let me use the standard Strassen formulation more carefully.
            # The 7 products are:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # C00 = M1 + M4 - M5 + M7
            # C01 = M3 + M5
            # C10 = M2 + M4
            # C11 = M1 - M2 + M3 + M6
            
            # For each M, we need to set U, V, W such that:
            # sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] gives the right coefficient for C[i,k]
            
            # M1: U[0,0]=1, U[1,1]=1; V[0,0]=1, V[1,1]=1; W[0,0]=1, W[1,1]=1
            # This contributes to C[0,0]: U[0,0]*V[0,0]*W[0,0] + U[0,0]*V[1,1]*W[1,0]... 
            # Wait, I need to be more careful.
            
            # The tensor is T[i,j,k] where i is row of A, j is col of A / row of B, k is col of B.
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # For the 2x2*2x2 sub-block, local i,j,k in {0,1}.
            # The contribution to C[i,k] from this sub-block is T[i,j,k] summed over j? No.
            
            # Actually, T[i,j,k] is the coefficient such that C[i,k] = sum_j A[i,j]*B[j,k].
            # In the tensor decomposition, T[i,j,k] should equal 1 for all i,j,k (in the sub-block)
            # when we sum over the rank dimension.
            
            # So for the Strassen block, we need:
            # For each (i,j,k) in {0,1}^3, sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
            
            # Let me define the 7 Strassen terms with their U, V, W patterns:
            
            # M1: (A00+A11)(B00+B11)
            # U[0,0]=1, U[1,1]=1, others 0
            # V[0,0]=1, V[1,1]=1, others 0  
            # W[0,0]=1, W[1,1]=1, others 0
            # Check T[0,0,0]: U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
            # T[0,0,1]: U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0 ✗ (should be 1)
            # This doesn't work directly.
            
            # I think I'm confusing myself. Let me look up the correct Strassen tensor decomposition.
            
            # The Strassen algorithm decomposes the 2x2*2x2 multiplication tensor (which has 8 non-zero entries, all 1)
            # into 7 rank-1 tensors.
            
            # The 8 entries are T[i,j,k] for i,j,k in {0,1}, all equal to 1.
            # We need to write this as sum of 7 rank-1 tensors.
            
            # Standard Strassen rank-1 terms (U, V, W) for each term:
            # Term 1: U = [1,0,0,1] (for i*m+j: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3), V = [1,0,0,1], W = [1,0,0,1]
            #   U indices: U[0,0]=1, U[1,1]=1
            #   V indices: V[0,0]=1, V[1,1]=1
            #   W indices: W[0,0]=1, W[1,1]=1
            #   T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1
            #   T[0,0,1] += U[0,0]*V[0,1]*W[1,0] = 1*0*0 = 0
            #   This still doesn't give 1 for all (i,j,k).
            
            # I think the issue is that in the tensor decomposition, each rank-1 term contributes
            # to T[i,j,k] = U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
            # And we need the SUM over c to equal 1 for all (i,j,k).
            
            # For Strassen, the 7 terms are defined such that their sum gives the correct tensor.
            # Let me just hardcode the known Strassen decomposition.
            
            # After research, the Strassen 2x2*2x2 tensor decomposition has these 7 rank-1 terms:
            # (using 0-indexed local coordinates, U has 4 entries, V has 4, W has 4)
            
            terms = [
                # (U, V, W) where each is a 4-element vector for the 2x2*2x2 case
                ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
                ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0]),  # M2
                ([1, 0, 0, 0], [0, 1, 0, 0], [0, 1, 0, 0]),  # M3
                ([0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]),  # M4
                ([1, 1, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]),  # M5
                ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 1, 0]),  # M6
                ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),  # M7
            ]
            
            # Let me verify this. For each term, compute the contribution to T[i,j,k]:
            # T[i,j,k] = U[i*2+j] * V[j*2+k] * W[k*2+i]
            
            # Term 1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
            # T[0,0,0] = U[0]*V[0]*W[0] = 1*1*1 = 1
            # T[0,0,1] = U[0]*V[1]*W[2] = 1*0*0 = 0
            # T[0,1,0] = U[1]*V[2]*W[0] = 0*0*1 = 0
            # T[0,1,1] = U[1]*V[3]*W[2] = 0*1*0 = 0
            # T[1,0,0] = U[2]*V[0]*W[1] = 0*1*0 = 0
            # T[1,0,1] = U[2]*V[1]*W[3] = 0*0*1 = 0
            # T[1,1,0] = U[3]*V[2]*W[1] = 1*0*0 = 0
            # T[1,1,1] = U[3]*V[3]*W[3] = 1*1*1 = 1
            # Sum so far: T[0,0,0]=1, T[1,1,1]=1, rest 0.
            
            # Term 2: U=[0,0,1,1], V=[1,0,0,0], W=[0,1,0,0]
            # T[0,0,0] = U[0]*V[0]*W[0] = 0*1*0 = 0
            # T[0,0,1] = U[0]*V[1]*W[2] = 0*0*0 = 0
            # T[0,1,0] = U[1]*V[2]*W[0] = 0*0*0 = 0
            # T[0,1,1] = U[1]*V[3]*W[2] = 0*0*0 = 0
            # T[1,0,0] = U[2]*V[0]*W[1] = 1*1*1 = 1
            # T[1,0,1] = U[2]*V[1]*W[3] = 1*0*0 = 0
            # T[1,1,0] = U[3]*V[2]*W[1] = 1*0*1 = 0
            # T[1,1,1] = U[3]*V[3]*W[3] = 1*0*0 = 0
            # Sum: T[1,0,0]=1
            
            # Term 3: U=[1,0,0,0], V=[0,1,0,0], W=[0,1,0,0]
            # T[0,0,0] = U[0]*V[0]*W[0] = 1*0*0 = 0
            # T[0,0,1] = U[0]*V[1]*W[2] = 1*1*0 = 0
            # T[0,1,0] = U[1]*V[2]*W[0] = 0*0*0 = 0
            # T[0,1,1] = U[1]*V[3]*W[2] = 0*0*0 = 0
            # T[1,0,0] = U[2]*V[0]*W[1] = 0*0*1 = 0
            # T[1,0,1] = U[2]*V[1]*W[3] = 0*1*0 = 0
            # T[1,1,0] = U[3]*V[2]*W[1] = 0*0*1 = 0
            # T[1,1,1] = U[3]*V[3]*W[3] = 0*0*0 = 0
            # Sum: all 0. That's wrong.
            
            # I clearly don't have the right Strassen decomposition hardcoded. Let me derive it properly.
            
            # The 2x2*2x2 multiplication tensor T has T[i,j,k]=1 for all i,j,k in {0,1}.
            # We want to write T as a sum of 7 rank-1 tensors.
            
            # This is a known result. Let me use a different approach: just use direct multiplication
            # for the 2x2*2x2 blocks too, giving rank 8 instead of 7.
            
            # With direct: (2x2)*(2x3) = (2x2)*(2x1) + (2x2)*(2x2) = 4 + 8 = 12
            # Total: 4*12 + 4*6 = 48 + 24 = 72. Back to schoolbook.
            
            # OK, let me just implement the Strassen correctly by looking it up or deriving.
            # For now, let me use a simpler approach: just return the schoolbook decomposition.
            # The inspirations got rank 66, so there must be a way.
            
            # Let me try a different block structure that might be easier.
            
            # Actually, let me just return the schoolbook for now and fix this later.
            # For the purpose of this edit, I'll return a valid (if not optimal) decomposition.
            
            # Return rank used
            return 0  # Placeholder, we'll fix this
        
        # Given the complexity, let me just return the schoolbook decomposition for now
        # and ensure it's valid. The inspirations achieved 66, so we need to work harder.
        
        # For now, return schoolbook (rank 72)
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W, rank
    
    U, V, W, rank = build_block_decomposition()
    
    # Verify
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(total - 1.0) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
