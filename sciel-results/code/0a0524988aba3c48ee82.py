"""Matrix multiplication 2,3,8 — rank-minimal exact tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Strategy: Use Strassen-like block decomposition to reduce rank.
    # For 2x3 * 3x8, we can partition the 3x3 middle into blocks and apply
    # Strassen's 7-multiplication trick on 2x2 sub-blocks where applicable.
    #
    # Known result: rank(2x3x8) <= 44 via Strassen on 2x2 blocks.
    # Let's construct this explicitly.
    
    # The 2x3 * 3x8 matrix product:
    # A is 2x3, B is 3x8, C = A*B is 2x8
    #
    # Partition A into columns: A = [A1 | A2 | A3] where each Ai is 2x1
    # Partition B into rows: B = [B1; B2; B3] where each Bi is 1x8
    #
    # C = A1*B1 + A2*B2 + A3*B3 (3 terms, each is 2x8 outer product)
    #
    # This gives rank 3*2*8 = 48 (schoolbook).
    #
    # To use Strassen: group the 3 middle indices into pairs.
    # Let's think of it as: C[i,k] = sum_j A[i,j]*B[j,k]
    #
    # Alternative: view as (2x3)*(3x8). We can split the 3 into 2+1.
    # A = [A_left | A_right] where A_left is 2x2, A_right is 2x1
    # B = [B_top; B_bottom] where B_top is 2x8, B_bottom is 1x8
    #
    # C = A_left * B_top + A_right * B_bottom
    #
    # The first term (2x2)*(2x8) can be done with Strassen: 7 multiplications instead of 4.
    # Each "multiplication" in Strassen for (2x2)*(2x8) involves multiplying a 2x2 matrix
    # by a 2x8 matrix, which itself has rank 2*2*8 = 32... no that's not right.
    #
    # Let me reconsider. The tensor for (n x m) * (m x p) has rank at most n*m*p.
    # Strassen reduces the bilinear complexity of matrix multiplication.
    #
    # For (2x2)*(2x2): rank = 7 (Strassen)
    # For (2x2)*(2x8): we can view this as 8 independent (2x2)*(2x1) products... no.
    # Actually (2x2)*(2x8) = sum over j=0,1 of A[:,j] * B[j,:] where A[:,j] is 2x1 and B[j,:] is 1x8.
    # So rank of (2x2)*(2x8) is at most 2*2*8 = 32 by schoolbook, but we can do better.
    #
    # Actually, let me think about this differently. The bilinear complexity of multiplying
    # an (n x m) matrix by an (m x p) matrix is the rank of the tensor T in U*V*W where
    # T[i,j,k,l] = delta_{j,k} (i from 0..n-1, j from 0..m-1, k from 0..m-1, l from 0..p-1)
    # Wait, the tensor is: C[i,l] = sum_j A[i,j] * B[j,l]
    # So the tensor T has indices (i, j, l) with T[i,j,l] = delta... no.
    # The multiplication tensor M_{n,m,p} is in (R^n * R^p) * (R^m * R^m) * (R^n * R^p)
    # T[(i,l), (j,k), (i',l')] = delta_{i,i'} * delta_{l,l'} * delta_{j,k}
    #
    # Rank of M_{2,3,8}: We need to find the minimum number of rank-1 tensors that sum to it.
    #
    # Known: rank(M_{2,2,2}) = 7 (Strassen)
    # rank(M_{2,3,8}) — let me try to construct a rank-44 decomposition.
    #
    # Approach: Split m=3 into m1=2, m2=1.
    # C = A[:,0:2] @ B[0:2,:] + A[:,2:3] @ B[2:3,:]
    # First term: (2x2)*(2x8). Second term: (2x1)*(1x8).
    #
    # For (2x1)*(1x8): rank = 2*1*8 = 16 (each output C[i,l] = A[i,2]*B[2,l], 16 independent products)
    # Actually this is just 16 rank-1 terms: for each (i,l), A[i,2]*B[2,l].
    #
    # For (2x2)*(2x8): C[i,l] = sum_{j=0}^{1} A[i,j]*B[j,l]
    # This is like doing 8 independent (2x2)*(2x1) multiplications... no, it's (2x2)*(2x8).
    # We can use Strassen on the 2x2 part: for each pair of columns in B, apply Strassen.
    # B is 2x8, so we have 4 pairs of columns. For each pair, we compute a 2x2 * 2x2 product using Strassen (7 mults).
    # But wait, the "multiplications" in Strassen for 2x2*2x2 are scalar multiplications.
    # For 2x2*2x8, if we split B into 4 blocks of 2x2, then C = sum of 4 (2x2)*(2x2) products.
    # Each (2x2)*(2x2) has rank 7, so total rank for first term = 4*7 = 28.
    # Second term: rank 16.
    # Total: 28 + 16 = 44.
    
    # Let's construct this.
    # C[i,l] = sum_{j=0}^{1} A[i,j]*B[j,l] + A[i,2]*B[2,l]
    #
    # For the first part, split l into groups of 2: l = 2*q + r, q=0..3, r=0,1
    # For each q, we compute D_q = A * B[:, 2q:2q+2] where A is 2x2, B[:,2q:2q+2] is 2x2
    # D_q[i,r] = sum_j A[i,j]*B[j, 2q+r]
    #
    # Strassen for 2x2*2x2:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-B[0,0])
    # M5 = (A[0,0]+A[0,1])*B[1,1]
    # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
    # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
    #
    # C[0,0] = M1 - M4 + M6 - M7
    # C[0,1] = M3 + M5 + M7
    # C[1,0] = M2 + M4 - M6
    # C[1,1] = M1 - M3 + M2 + M5
    #
    # Each M is a product of a linear combination of A entries and a linear combination of B entries.
    # In the tensor decomposition, each "multiplication" M corresponds to a rank-1 term.
    #
    # For the full problem:
    # U has dimensions (n*m) x rank = 6 x rank
    # V has dimensions (m*p) x rank = 24 x rank  
    # W has dimensions (p*n) x rank = 16 x rank
    #
    # Wait, let me re-read the interface:
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=2, m=3, p=8
    # U: (6, rank), V: (24, rank), W: (16, rank)
    #
    # The decomposition: C[i,k] = sum_c U[i*m+j... wait.
    # 
    # Standard tensor decomposition for matrix mult:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # = sum_c (sum_j U[i*m+j, c]) * (sum_k V[j*p+k, c]) ... no that's not right.
    #
    # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
    # "output factor follows the official trace convention W[k*n+i,r]"
    #
    # So the tensor is: T[i,j,k] = delta_{j,k} (for the identity in the middle)
    # C[i,k] = sum_j A[i,j] B[j,k]
    #
    # In tensor form: C = A ⊗ T ⊗ B^T (roughly)
    # The decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c is in R^{n*m}, v_c is in R^{m*p}, w_c is in R^{p*n}
    #
    # u_c[i*m + j] = coefficient for A[i,j]
    # v_c[j*p + k] = coefficient for B[j,k]
    # w_c[k*n + i] = coefficient for C[i,k]
    #
    # So: C[i,k] = sum_c (sum_j u_c[i*m+j] * A[i,j]) * (sum_j v_c[j*p+k] * B[j,k])
    # Wait, that's not quite right either. Let me think more carefully.
    #
    # The multiplication tensor M_{n,m,p} is defined by:
    # M_{n,m,p}[(i,j),(j',k),(i',k')] = 1 if i=i' and j=j' and k=k', else 0
    # 
    # Hmm, actually the standard way: the tensor T has T[i,j,k] = delta_{j,k} is not right.
    # 
    # Let me use the convention from the problem:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # If we write A as a vector in R^{n*m} with A_flat[i*m+j] = A[i,j]
    # B as a vector in R^{m*p} with B_flat[j*p+k] = B[j,k]
    # C as a vector in R^{p*n} with C_flat[k*n+i] = C[i,k]
    #
    # Then C_flat[k*n+i] = sum_j A_flat[i*m+j] * B_flat[j*p+k]
    #
    # This is a trilinear form. The tensor T has:
    # T[(i,j), (j',k'), (k'',i'')] = 1 if j=j' and k'=k'' and i=i'', else 0
    #
    # A rank-1 term: u ⊗ v ⊗ w contributes u[i*m+j] * v[j'*p+k'] * w[k''*n+i'']
    # to T[(i,j),(j',k'),(k'',i'')]
    #
    # For the sum over c of u_c[i*m+j]*v_c[j'*p+k']*w_c[k''*n+i''] to equal T:
    # We need: sum_c u_c[i*m+j]*v_c[j'*p+k']*w_c[k''*n+i''] = delta_{j,j'} * delta_{k',k''} * delta_{i,i''}
    #
    # Schoolbook: for each (i,j,k): u has 1 at [i*m+j], v has 1 at [j*p+k], w has 1 at [k*n+i]
    # This gives rank n*m*p = 2*3*8 = 48.
    #
    # Now for Strassen-based rank 44:
    # We split j into j=0,1 (first two) and j=2 (last one).
    #
    # Part 1: j in {0,1}: C[i,k] += A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # Part 2: j=2: C[i,k] += A[i,2]*B[2,k]
    #
    # Part 2 is simple: for each (i,k), one rank-1 term.
    # u has 1 at [i*3+2], v has 1 at [2*8+k], w has 1 at [k*2+i]
    # That's 2*8 = 16 terms.
    #
    # Part 1: For each k, C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    # This is a (2x2)*(2x8) multiplication viewed as 8 independent (2x1)*(2x1) inner products... 
    # no, it's C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] for i=0,1 and k=0..7.
    #
    # Group k into pairs: k = 2q+r, q=0..3, r=0,1
    # For each q, we have a 2x2 * 2x2 sub-problem:
    # C[i,2q+r] = A[i,0]*B[0,2q+r] + A[i,1]*B[1,2q+r] for i=0,1, r=0,1
    #
    # Apply Strassen to each 2x2*2x2:
    # For block q, define:
    # a00=A[0,0], a01=A[0,1], a10=A[1,0], a11=A[1,1]
    # b00=B[0,2q], b01=B[0,2q+1], b10=B[1,2q], b11=B[1,2q+1]
    #
    # M1 = (a00+a11)*(b00+b11)
    # M2 = (a10+a11)*b00
    # M3 = a00*(b01-b11)
    # M4 = a11*(b10-b00)
    # M5 = (a00+a01)*b11
    # M6 = (a10-a00)*(b00+b01)
    # M7 = (a01-a11)*(b10+b11)
    #
    # C[0,2q] = M1 - M4 + M6 - M7
    # C[0,2q+1] = M3 + M5 + M7
    # C[1,2q] = M2 + M4 - M6
    # C[1,2q+1] = M1 - M3 + M2 + M5
    #
    # Each M is a product of a linear combo of A entries (indices j=0,1) and a linear combo of B entries (indices j=0,1, k=2q or 2q+1).
    #
    # For each M, the u vector has support on A[0,0], A[0,1], A[1,0], A[1,1] (indices 0,1,3,4 in flat)
    # The v vector has support on B[0,2q], B[0,2q+1], B[1,2q], B[1,2q+1] (indices 2q, 2q+1, 8+2q, 8+2q+1 in flat)
    # The w vector has support on C[0,2q], C[0,2q+1], C[1,2q], C[1,2q+1] (indices 2q*2+0, 2q*2+1, (2q+1)*2+0, (2q+1)*2+1 in flat)
    #
    # Total rank: 4 blocks * 7 Strassen terms + 16 part-2 terms = 28 + 16 = 44.
    
    rank = 44
    U = [[0]*rank for _ in range(6)]   # n*m = 6
    V = [[0]*rank for _ in range(24)]  # m*p = 24
    W = [[0]*rank for _ in range(16)]  # p*n = 16
    
    col = 0
    # Strassen blocks for q = 0,1,2,3
    for q in range(4):
        k0 = 2*q
        k1 = 2*q+1
        # B indices: B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=8+k0, B[1,k1]=8+k1
        # A indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        # C indices: C[0,k0]=k0*2+0=2k0, C[0,k1]=2k0+1, C[1,k0]=2k0+2, C[1,k1]=2k0+3
        
        # M1 = (a00+a11)*(b00+b11)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = 1; W[2*k0+1][col] = 1; W[2*k0+2][col] = 1; W[2*k0+3][col] = 1
        # C[0,k0] += M1, C[0,k1] += M1, C[1,k0] += M1, C[1,k1] += M1
        # Actually need to be more careful. Let me re-derive.
        # C[0,k0] = M1 - M4 + M6 - M7
        # C[0,k1] = M3 + M5 + M7
        # C[1,k0] = M2 + M4 - M6
        # C[1,k1] = M1 - M3 + M2 + M5
        #
        # So M1 appears in C[0,k0] with +1, C[0,k1] with 0, C[1,k0] with 0, C[1,k1] with +1
        # Wait no:
        # C[0,k0] = M1 - M4 + M6 - M7  => M1 coeff +1
        # C[0,k1] = M3 + M5 + M7       => M1 coeff 0
        # C[1,k0] = M2 + M4 - M6       => M1 coeff 0
        # C[1,k1] = M1 - M3 + M2 + M5  => M1 coeff +1
        #
        # Hmm, that doesn't look right. Let me re-verify Strassen.
        # Standard Strassen for 2x2:
        # C[0,0] = M1 - M4 + M6 - M7
        # C[0,1] = M3 + M5 + M7
        # C[1,0] = M2 + M4 - M6
        # C[1,1] = M1 - M3 + M2 + M5
        #
        # where M1=(a00+a11)(b00+b11), M2=(a10+a11)b00, M3=a00(b01-b11),
        # M4=a11(b10-b00), M5=(a00+a01)b11, M6=(a10-a00)(b00+b01), M7=(a01-a11)(b10+b11)
        
        # Let me redo this properly.
        col = 0  # reset, I'll build all terms fresh
        
    # Let me rebuild cleanly.
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(24)]
    W = [[0]*rank for _ in range(16)]
    
    col = 0
    for q in range(4):
        k0 = 2*q
        k1 = 2*q+1
        # A flat indices: A[0,0]=0, A[0,1]=1, A[1,0]=3, A[1,1]=4
        # B flat indices: B[0,k0]=k0, B[0,k1]=k1, B[1,k0]=8+k0, B[1,k1]=8+k1
        # C flat indices: C[0,k0]=2*k0, C[0,k1]=2*k0+1, C[1,k0]=2*k0+2, C[1,k1]=2*k0+3
        
        # M1 = (a00+a11)*(b00+b11)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = 1    # C[0,k0] += M1
        W[2*k0+3][col] = 1  # C[1,k1] += M1
        col += 1
        
        # M2 = (a10+a11)*b00
        U[3][col] = 1; U[4][col] = 1
        V[k0][col] = 1
        W[2*k0+2][col] = 1  # C[1,k0] += M2
        W[2*k0+3][col] = 1  # C[1,k1] += M2
        col += 1
        
        # M3 = a00*(b01-b11)
        U[0][col] = 1
        V[k1][col] = 1; V[8+k1][col] = -1
        W[2*k0+1][col] = 1  # C[0,k1] += M3
        W[2*k0+3][col] = -1 # C[1,k1] -= M3
        col += 1
        
        # M4 = a11*(b10-b00)
        U[4][col] = 1
        V[8+k0][col] = 1; V[k0][col] = -1
        W[2*k0][col] = -1   # C[0,k0] -= M4
        W[2*k0+2][col] = 1  # C[1,k0] += M4
        col += 1
        
        # M5 = (a00+a01)*b11
        U[0][col] = 1; U[1][col] = 1
        V[8+k1][col] = 1
        W[2*k0+1][col] = 1  # C[0,k1] += M5
        W[2*k0+3][col] = 1  # C[1,k1] += M5
        col += 1
        
        # M6 = (a10-a00)*(b00+b01)
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[k1][col] = 1
        W[2*k0][col] = 1    # C[0,k0] += M6
        W[2*k0+2][col] = -1 # C[1,k0] -= M6
        col += 1
        
        # M7 = (a01-a11)*(b10+b11)
        U[1][col] = 1; U[4][col] = -1
        V[8+k0][col] = 1; V[8+k1][col] = 1
        W[2*k0][col] = -1   # C[0,k0] -= M7
        W[2*k0+1][col] = 1  # C[0,k1] += M7
        col += 1
    
    # Part 2: j=2, for each (i,k): C[i,k] += A[i,2]*B[2,k]
    for i in range(2):
        for k in range(8):
            U[i*3+2][col] = 1
            V[2*8+k][col] = 1
            W[k*2+i][col] = 1
            col += 1
    
    assert col == rank, f"Expected {rank} columns, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
