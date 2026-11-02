"""Rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Check if U, V, W correctly decompose the matrix multiplication tensor."""
    r = len(U[0])
    # Build the multiplication tensor T[i,j,k] = 1
    # The decomposition says: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # Let's check a random subset of (i,j,k) triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _try_strassen_block_structure(n, m, p):
    """
    Try to find a rank-28 decomposition using a block-Strassen-like structure.
    
    For 2x4 * 4x4, we can think of the 4x4 matrices as 2x2 blocks of 2x2 matrices.
    A is 2x4 = [A11 A12] where A11, A12 are 2x2
    B is 4x4 = [[B11 B12], [B21 B22]] where each is 2x2
    C = A*B = [A11*B11 + A12*B21,  A11*B12 + A12*B22]
    
    Using Strassen for each 2x2*2x2 product, we get 7 multiplications per block product.
    But we need two such products, so naively 14*2 = 28 scalar multiplications if we can share.
    
    Actually, let's think differently. We can use a bilinear algorithm approach.
    """
    # This is a heuristic attempt - return None if not found
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 by 4x4 matrix multiplication.
    Returns dict with keys 'U', 'V', 'W' of shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 16
    pn = p * n  # 8
    schoolbook_rank = n * m * p  # 32
    
    start_time = time.time()
    deadline = start_time + 120.0  # Safety margin before 150s limit
    
    # Start with schoolbook (guaranteed correct)
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(nm)]
    V = [[0] * rank for _ in range(mp)]
    W = [[0] * rank for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    # Try to reduce rank using a structured approach
    # Strategy: exploit the 2x4 * 4x4 structure
    # A is 2x4, B is 4x4, C is 2x4
    # C[i,k] = sum_j A[i,j] * B[j,k]
    
    # One approach: use the fact that we can group columns of A
    # A = [a0, a1, a2, a3] as column vectors in R^2
    # C[:,k] = a0*B[0,k] + a1*B[1,k] + a2*B[2,k] + a3*B[3,k]
    
    # Try a rank-28 construction based on pairing
    # For each pair of columns, we can use a 2D bilinear trick
    # (x1*x3 + x2*x4) - (x1-x2)*(x3-x4) = 2*x1*x3... no that's not right
    # Actually: (x1+x2)*(x3+x4) - x1*x3 - x2*x4 = x1*x4 + x2*x3
    
    # Let's try to use the identity: for 2D vectors u,v, the outer product has rank 2
    # u*v^T = sum over components
    
    # A structured rank-28 approach:
    # Group the 4 columns of A into 2 pairs, and the 4 rows of B into 2 groups of 2
    # For pair (A[:,2a], A[:,2a+1]) and row-group (B[2b:2b+2,:]):
    # We need A[:,2a]*B[2b,:] + A[:,2a+1]*B[2b+1,:]
    # Using: (A2a+A2a+1)*(B2b+B2b+1) - A2a*B2b - A2a+1*B2b+1... 
    # This doesn't directly reduce rank.
    
    # Alternative: use the 7-term Strassen for 2x2*2x2
    # View A as two 2x2 blocks side by side: A = [A_left | A_right]
    # View B as two 2x2 blocks top and bottom: B = [[B_top], [B_bottom]]
    # C = A_left * B_top + A_right * B_bottom
    # Each 2x2*2x2 product can be done in rank 7 (Strassen)
    # Total: 7 + 7 = 14 block products, each contributing to the 2x4 output
    
    # Wait, that gives rank 14*... no. Each Strassen product is 7 scalar mults.
    # But the Strassen products are on 2x2 matrices, so each "scalar" is really a 2x2 matrix entry.
    # No, Strassen for 2x2*2x2 uses 7 scalar multiplications.
    # So A_left*B_top uses 7 mults, A_right*B_bottom uses 7 mults.
    # Total 14 scalar multiplications? But each produces a 2x2 result that gets added.
    # The issue is that the Strassen intermediates are linear combinations, and when
    # we add the two results, we need to track all the terms.
    
    # Let me think more carefully. 
    # A_left = [[a00, a01], [a10, a11]] (2x2)
    # B_top = [[b00, b01], [b10, b11]] (2x2)
    # C_left = A_left * B_top (2x2)
    # 
    # Strassen for C_left:
    # m1 = (a00+a11)(b00+b11)
    # m2 = (a10+a11)(b00)
    # m3 = (a00)(b01+b11)
    # m4 = (a10)(b10)
    # m5 = (a00+a10)(b10+b11)
    # m6 = (a01+a11)(b00+b10)
    # m7 = (a01+a10)(b11-b10)
    # C_left[0,0] = m1+m4-m5+m7
    # C_left[0,1] = m3+m5
    # C_left[1,0] = m2+m4
    # C_left[1,1] = m1-m2+m3+m6
    
    # Similarly for C_right = A_right * B_bottom with 7 more mults
    # Total 14 scalar multiplications, but they all contribute to different output positions
    # through linear combinations. The rank is the number of scalar multiplications = 14?
    # No wait, the rank of the tensor is the minimum number of rank-1 tensors that sum to it.
    # Each scalar multiplication corresponds to one rank-1 term in the tensor decomposition.
    # So if we can express the whole 2x4*4x4 multiplication using 14 scalar mults,
    # the rank would be 14. But that seems too low.
    
    # The problem is that the Strassen mults for A_left*B_top and A_right*B_bottom
    # are independent, so we just add them: total rank = 7 + 7 = 14? 
    # But each Strassen mult is of the form (linear combo of A entries) * (linear combo of B entries),
    # and these products contribute to specific C entries with coefficients.
    # The full tensor T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,4)
    # A decomposition with rank r means: sum_{c=1}^r U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
    
    # With the Strassen approach, each of the 14 mults gives a term like:
    # (alpha * A[i,j1] + beta * A[i,j2]) * (gamma * B[j1,k1] + delta * B[j2,k2])
    # which expands to multiple (i,j,k) terms.
    # The key question is whether 14 such terms can cover all 32 (i,j,k) positions.
    # Since each term is a product of a 2D (in j) and 2D (in k) linear form,
    # it covers at most 4 (j,k) pairs, times 2 i-values = 8 (i,j,k) positions.
    # 14 terms * 8 = 112 >= 32, so it's possible in principle.
    
    # Let me actually construct this.
    # A[i,j] for i=0,1 and j=0,1,2,3
    # B[j,k] for j=0,1,2,3 and k=0,1,2,3
    # C[i,k] = sum_j A[i,j]*B[j,k]
    
    # A_left = A[:,0:2], A_right = A[:,2:4]
    # B_top = B[0:2,:], B_bottom = B[2:4,:]
    # C[:,0:2] contribution from A_left*B_top... no wait
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] + A[i,3]*B[3,k]
    #         = (A[i,0]*B[0,k] + A[i,1]*B[1,k]) + (A[i,2]*B[2,k] + A[i,3]*B[3,k])
    #         = (A_left[i,0:2] * B_top[:,k]) + (A_right[i,2:4] * B_bottom[:,k])
    
    # So C = A_left @ B_top + A_right @ B_bottom where @ is 2x4 * 4x4... 
    # No, A_left is 2x2, B_top is 2x4, so A_left @ B_top is 2x4.
    # Similarly A_right @ B_bottom is 2x4.
    
    # For A_left (2x2) @ B_top (2x4):
    # This is a 2x2 * 2x4 multiplication.
    # C1[i,k] = A_left[i,0]*B_top[0,k] + A_left[i,1]*B_top[1,k]
    # This has tensor of shape (2,2,4) = 16 entries.
    # Minimum rank for 2x2*2x4?
    # We can use Strassen-like: for each k, it's a 2x2*2x1 which is just 4 mults.
    # For 4 values of k, that's 16 mults if done independently.
    # But we can share: (A[0,0]+A[1,0])*(B[0,k]+B[1,k]) etc.
    
    # Actually, 2x2 * 2x4: think of it as 4 instances of 2x2*2x1.
    # 2x2*2x1 has rank 4 (can't do better, it's just 4 independent products... 
    # wait, 2x2*2x1: C[i] = A[i,0]*B[0] + A[i,1]*B[1] for i=0,1
    # Tensor shape (2,2,1): T[i,j,0] = 1 for all i,j. Rank = 4? 
    # Actually rank of 2x2*2x1 tensor: it's the tensor with T[i,j,k]=1 for i,j in {0,1}, k=0.
    # This is just the all-ones tensor of shape 2x2x1, which has rank 4 (it's the product of 
    # the all-ones 2-vector with itself with the 1-vector).
    # Hmm, actually rank of the all-ones tensor of shape n1 x n2 x n3 is n1*n2*n3 in general,
    # but for 2x2x1 it's 4? Let me check: can we write it with fewer than 4 rank-1 terms?
    # A rank-1 term is u_i * v_j * w_k. For k=0 only, w_0 is just a scalar.
    # So we need sum_c u_c[i] * v_c[j] * w_c = 1 for all i,j.
    # This is: sum_c w_c * (u_c v_c^T) = J (2x2 all-ones).
    # The rank of J as a matrix is 1, so we need rank-1 matrices u_c v_c^T that sum to J.
    # J = [1,1;1,1] = [1,1]^T * [1,1], so rank 1! 
    # So 2x2*2x1 has rank 1? That can't be right for matrix multiplication.
    # 
    # Oh wait, I'm confusing the tensor rank with matrix rank.
    # The matrix multiplication tensor T_{ijk} for m1 x m2 * m2 x m3 has rank = min number of 
    # bilinear multiplications. For 2x2*2x1:
    # C_0 = A_00*B_0 + A_01*B_1
    # C_1 = A_10*B_0 + A_11*B_1
    # This needs 4 multiplications in schoolbook. Can we do it in 3?
    # m1 = (A_00+A_10)*B_0 = A_00*B_0 + A_10*B_0
    # m2 = (A_01+A_11)*B_1 = A_01*B_1 + A_11*B_1
    # m3 = ... we need A_00*B_0, A_01*B_1, A_10*B_0, A_11*B_1
    # C_0 = A_00*B_0 + A_01*B_1
    # C_1 = A_10*B_0 + A_11*B_1
    # With m1, m2, we get two pairs. We still need to separate them.
    # m3 = (A_00-A_10)*(B_0) ... this gives A_00*B_0 - A_10*B_0
    # Then A_00*B_0 = (m1+m3)/2, A_10*B_0 = (m1-m3)/2
    # Similarly we'd need another for B_1. So total 4.
    # Actually the rank of 2x2*2x1 is 4. It's known that rank(m x n x 1) = m*n.
    
    # So 2x2*2x4 has rank 8 (since it's 2x2*2x4 = 4 independent 2x2*2x1? No...)
    # 2x2 * 2x4: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] for i=0,1, k=0,1,2,3
    # Tensor shape (2,2,4). Rank?
    # Schoolbook: 2*2*4 = 16.
    # Can we do better? For each i, it's a 2x4 vector: C[i,:] = A[i,0]*B[0,:] + A[i,1]*B[1,:]
    # This is like 2 separate 1x2*2x4 multiplications... 
    # Actually, the rank of the 2x2*2x4 multiplication tensor is known to be 8.
    # (It's the rank of the "slice" and various results say rank(2,2,4) = 8)
    
    # Hmm, let me just go with a safe approach.
    # Given the complexity, let me try a direct rank-reduction search.
    
    # For now, return the schoolbook rank 32 solution.
    # The knowledge says rank 28 is achievable (score 1.143), so let me try harder.
    
    # Actually, let me try a specific construction for rank 28.
    # Idea: use 7 Strassen-like multiplications for a 2x2*2x2 block,
    # and apply it in a structured way to the 2x4*4x4 problem.
    
    # Alternative: use the fact that 2x4*4x4 can be embedded in a larger problem.
    # Or use a Kronecker product structure.
    
    # Let me try: express the 32 bilinear terms and see if we can merge 4 of them.
    # Each schoolbook term is A[i,j]*B[j,k].
    # We want to find 28 bilinear forms (L_a(A) * L_b(B)) that span all 32 terms.
    
    # One approach: for each j, the terms are A[i,j]*B[j,k] for i=0,1 and k=0,1,2,3.
    # That's 8 terms per j, 4 j-values, total 32.
    # For a fixed j, the 8 terms form a 2x4 "slice".
    # Can we express A[i,j]*B[j,k] for all i,k using fewer than 8 mults?
    # The tensor for fixed j is T[i,k] = A[i,j]*B[j,k], which has rank 1 as a "matrix" 
    # in the (i,k) space (it's outer product of A[:,j] and B[j,:]).
    # But as a bilinear form in (A,B), it's already rank 1: just A[i,j]*B[j,k].
    # So each (i,j,k) term is already rank 1. The question is whether we can combine
    # multiple terms into a single bilinear product.
    
    # Two terms A[i1,j1]*B[j1,k1] and A[i2,j2]*B[j2,k2] can be combined if
    # they share structure. E.g., A[i,j]*B[j,k] + A[i,j']*B[j',k] = A[i,j]*B[j,k] + A[i,j']*B[j',k]
    # = (A[i,j] + A[i,j'])*B[j,k] + A[i,j']*(B[j',k] - B[j,k])... not helpful.
    
    # What about: A[i,j]*B[j,k] + A[i,j+1]*B[j+1,k]?
    # Let X = A[i,j] + A[i,j+1], Y = B[j,k] + B[j+1,k]
    # X*Y = A[i,j]*B[j,k] + A[i,j]*B[j+1,k] + A[i,j+1]*B[j,k] + A[i,j+1]*B[j+1,k]
    # This gives us 2 of our desired terms plus 2 extra.
    # If we also compute A[i,j]*B[j+1,k] and A[i,j+1]*B[j,k] separately or via other means,
    # we can extract the desired terms.
    
    # This is getting complex. Let me just implement a numerical search for rank 28.
    
    if time.time() > deadline - 30:
        return {"U": U, "V": V, "W": W}
    
    # Try to find rank 28 using a structured approach
    # Based on the insight that we can use 7 "Strassen-like" multiplications
    # for each of 4 "blocks"
    
    # Actually, let me try a completely different approach:
    # Use the identity for 2x2 matrix multiplication (rank 7) 
    # applied to 2x4 * 4x4 by viewing it as a block operation.
    
    # A (2x4) * B (4x4) = C (2x4)
    # Partition B into left and right halves: B = [B_L | B_R] where B_L, B_R are 4x2
    # C = A * B = [A*B_L | A*B_R]
    # A*B_L is 2x4 * 4x2 = 2x2
    # A*B_R is 2x4 * 4x2 = 2x2
    
    # For A*B_L (2x4 * 4x2):
    # C_L[i,k] = sum_{j=0}^3 A[i,j]*B_L[j,k] = sum_{j=0}^3 A[i,j]*B[j,k] for k=0,1
    # This is a 2x4 * 4x2 multiplication, tensor shape (2,4,2) = 16 entries.
    # Rank of 2x4*4x2? Schoolbook is 16.
    # 
    # For 2x4 * 4x2, we can view it as:
    # Split A into [A_01 | A_23] (each 2x2)
    # Split B_L into [B_01; B_23] (each 2x2)
    # C_L = A_01 * B_01 + A_23 * B_23 (both 2x2*2x2 = 2x2)
    # Each 2x2*2x2 has rank 7 (Strassen), so total rank = 14 for C_L.
    # Similarly for C_R, another 14. Total = 28!
    
    # So the rank-28 construction is:
    # 1. Split A = [A12 | A34] where A12 = A[:,0:2], A34 = A[:,2:4] (both 2x2)
    # 2. Split B = [[B12]; [B34]] where B12 = B[0:2,:], B34 = B[2:4,:] (both 2x4)
    #    Then further split B12 = [B12_L | B12_R] (both 2x2), B34 = [B34_L | B34_R] (both 2x2)
    # 3. C_L = A12*B12_L + A34*B34_L (each 2x2*2x2, use Strassen rank 7)
    # 4. C_R = A12*B12_R + A34*B34_R (each 2x2*2x2, use Strassen rank 7)
    # Total rank = 4 * 7 = 28.
    
    # Let me construct this explicitly.
    # Strassen for 2x2*2x2:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01+b11)
    # M4 = a10 b10
    # M5 = (a00+a10)(b10+b11)
    # M6 = (a01+a11)(b00+b10)
    # M7 = (a01+a10)(b11-b10)
    # c00 = M1+M4-M5+M7
    # c01 = M3+M5
    # c10 = M2+M4
    # c11 = M1-M2+M3+M6
    
    def strassen_coefficients():
        """Return (U_s, V_s, W_s) for 2x2*2x2 Strassen, rank 7.
        U shape (4,7), V shape (4,7), W shape (4,7)
        Indexing: U[i*2+j, c], V[i*2+j, c], W[i*2+j, c]
        where (i,j) are the 2x2 matrix indices.
        """
        # 7 Strassen multiplications
        # Each mult c has: U entries for A matrix (4 elements), V entries for B matrix (4 elements)
        # W gives the output coefficients.
        
        # A = [[a00, a01], [a10, a11]] -> index: a00=0, a01=1, a10=2, a11=3
        # B = [[b00, b01], [b10, b11]] -> index: b00=0, b01=1, b10=2, b11=3
        # C = [[c00, c01], [c10, c11]] -> index: c00=0, c01=1, c10=2, c11=3
        
        U_s = [[0]*7 for _ in range(4)]
        V_s = [[0]*7 for _ in range(4)]
        W_s = [[0]*7 for _ in range(4)]
        
        # M1 = (a00+a11)(b00+b11)
        U_s[0][0] = 1; U_s[3][0] = 1
        V_s[0][0] = 1; V_s[3][0] = 1
        W_s[0][0] = 1  # c00 += M1
        W_s[3][0] = 1  # c11 += M1
        
        # M2 = (a10+a11)b00
        U_s[2][1] = 1; U_s[3][1] = 1
        V_s[0][1] = 1
        W_s[2][1] = 1  # c10 += M2
        W_s[3][1] = -1 # c11 -= M2
        
        # M3 = a00(b01+b11)
        U_s[0][2] = 1
        V_s[1][2] = 1; V_s[3][2] = 1
        W_s[1][2] = 1  # c01 += M3
        W_s[3][2] = 1  # c11 += M3
        
        # M4 = a10 b10
        U_s[2][3] = 1
        V_s[2][3] = 1
        W_s[0][3] = 1  # c00 += M4
        W_s[2][3] = 1  # c10 += M4
        
        # M5 = (a00+a10)(b10+b11)
        U_s[0][4] = 1; U_s[2][4] = 1
        V_s[2][4] = 1; V_s[3][4] = 1
        W_s[0][4] = -1 # c00 -= M5
        W_s[1][4] = 1  # c01 += M5
        
        # M6 = (a01+a11)(b00+b10)
        U_s[1][5] = 1; U_s[3][5] = 1
        V_s[0][5] = 1; V_s[2][5] = 1
        W_s[3][5] = 1  # c11 += M6
        
        # M7 = (a01+a10)(b11-b10)
        U_s[1][6] = 1; U_s[2][6] = 1
        V_s[3][6] = 1; V_s[2][6] = -1
        W_s[0][6] = 1  # c00 += M7
        
        return U_s, V_s, W_s
    
    # Now build the full rank-28 decomposition.
    # U has shape (8, 28): rows indexed by (i,j) where i in {0,1}, j in {0,1,2,3}
    #   Row index = i*4 + j
    # V has shape (16, 28): rows indexed by (j,k) where j in {0,1,2,3}, k in {0,1,2,3}
    #   Row index = j*4 + k
    # W has shape (8, 28): rows indexed by (k,i) per the trace convention W[k*n+i, r]
    #   Row index = k*2 + i (since n=2)
    
    U28 = [[0]*28 for _ in range(8)]
    V28 = [[0]*28 for _ in range(16)]
    W28 = [[0]*28 for _ in range(8)]
    
    Us, Vs, Ws = strassen_coefficients()
    
    # 4 Strassen instances:
    # Instance 0: A12 * B12_L -> contributes to C[:,0:2] left part
    #   A12 = A[:,0:2], indices j=0,1
    #   B12_L = B[0:2,0:2], indices j=0,1 and k=0,1
    # Instance 1: A34 * B34_L -> contributes to C[:,0:2] right part  
    #   A34 = A[:,2:4], indices j=2,3
    #   B34_L = B[2:4,0:2], indices j=2,3 and k=0,1
    # Instance 2: A12 * B12_R -> contributes to C[:,2:4] left part
    #   B12_R = B[0:2,2:4], indices j=0,1 and k=2,3
    # Instance 3: A34 * B34_R -> contributes to C[:,2:4] right part
    #   B34_R = B[2:4,2:4], indices j=2,3 and k=2,3
    
    # For each instance, the Strassen mult c (c=0..6) contributes:
    # U part: for A matrix entry (ai, aj) in {0,1}x{0,1}, coefficient Us[ai*2+aj][c]
    #   maps to global A index (i, j_global) where j_global = j_offset + aj
    # V part: for B matrix entry (bj, bk) in {0,1}x{0,1}, coefficient Vs[bj*2+bk][c]
    #   maps to global B index (j_global, k_global) where j_global = j_offset + bj, k_global = k_offset + bk
    # W part: for C matrix entry (ci, ck) in {0,1}x{0,1}, coefficient Ws[ci*2+ck][c]
    #   maps to global C output (i, k_global) where k_global = k_offset + ck
    #   W row index = k_global * n + i = k_global * 2 + i
    
    instances = [
        # (j_offset_A, j_offset_B, k_offset_B)
        (0, 0, 0),  # A12 * B12_L: j=0,1 for A and B, k=0,1 for B
        (2, 2, 0),  # A34 * B34_L: j=2,3 for A and B, k=0,1 for B
        (0, 0, 2),  # A12 * B12_R: j=0,1 for A and B, k=2,3 for B
        (2, 2, 2),  # A34 * B34_R: j=2,3 for A and B, k=2,3 for B
    ]
    
    for inst_idx, (joff_a, joff_b, koff) in enumerate(instances):
        for c in range(7):
            col = inst_idx * 7 + c
            # U contributions
            for ai in range(2):
                for aj in range(2):
                    coeff = Us[ai * 2 + aj][c]
                    if coeff != 0:
                        j_global = joff_a + aj
                        row = ai * 4 + j_global
                        U28[row][col] += coeff
            # V contributions
            for bj in range(2):
                for bk in range(2):
                    coeff = Vs[bj * 2 + bk][c]
                    if coeff != 0:
                        j_global = joff_b + bj
                        k_global = koff + bk
                        row = j_global * 4 + k_global
                        V28[row][col] += coeff
            # W contributions
            for ci in range(2):
                for ck in range(2):
                    coeff = Ws[ci * 2 + ck][c]
                    if coeff != 0:
                        k_global = koff + ck
                        row = k_global * 2 + ci  # W[k*n + i, r]
                        W28[row][col] += coeff
    
    # Verify the rank-28 decomposition
    if _verify_decomposition(U28, V28, W28, n, m, p):
        return {"U": U28, "V": V28, "W": W28}
    
    # If verification failed, fall back to schoolbook
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
