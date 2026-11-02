import numpy as np
import random
import time

DIMENSIONS = [3, 4, 6]

def _strassen_2x2x2():
    """Return the 7 Strassen multiplication triples for 2x2x2.
    Each triple is (u_vec, v_vec, w_vec) where u_vec is length 4 (2x2 A),
    v_vec is length 4 (2x2 B), w_vec is length 4 (2x2 C).
    Indexing: A[a*2+b], B[b*2+c], C[c*2+a]
    """
    # Strassen's 7 products for 2x2 matrix multiplication
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[i,j],[k,l]]
    # M1 = (a+d)(e+h) -> contributes to C[0][0]+=M1, C[0][1]-=M1, C[1][0]+=M1, C[1][1]+=M1... 
    # Actually standard Strassen:
    # P1 = (a+d)(e+h)
    # P2 = (c+d)(e)
    # P3 = (a-b)(h)
    # P4 = (d)(f-g)
    # P5 = (a)(e+f)
    # P6 = (c)(g+h)
    # P7 = (b-d)(e+h)  -- wait, let me use the standard formulation
    #
    # C[0][0] = P1 - P4 + P5 - P7
    # C[0][1] = P3 + P4
    # C[1][0] = P2 + P5
    # C[1][1] = P1 - P2 + P3 + P6
    #
    # Where:
    # P1 = (A[0][0]+A[1][1])*(B[0][0]+B[1][1])
    # P2 = (A[1][0]+A[1][1])*B[0][0]
    # P3 = (A[0][0]-A[0][1])*B[1][1]
    # P4 = A[1][1]*(B[0][1]-B[1][0])
    # P5 = A[0][0]*(B[0][0]+B[0][1])
    # P6 = A[1][0]*(B[1][0]+B[1][1])
    # P7 = (A[0][1]-A[1][1])*(B[0][0]+B[1][1])
    
    triples = []
    # P1: (A00+A11)*(B00+B11)
    u = [1,0,0,1]  # A00 + A11
    v = [1,0,0,1]  # B00 + B11
    w = [1,0,0,1]  # C00 + C11 (we'll adjust signs later)
    # Actually, let's be more careful. Each triple contributes u·a * v·b to w·c
    # P1 contributes to C[0][0] with +1 and C[1][1] with +1
    # But also C[0][1] with -1 and C[1][0] with +1 from the combination
    # No - each Pi is a scalar, and the final C entries are linear combinations.
    # So each triple is: u (coeffs for A), v (coeffs for B), w (coeffs for C)
    # such that sum over triples of (u·A) * (v·B) * w = C (as a tensor)
    
    # Let me define indices: A[0]=A00, A[1]=A01, A[2]=A10, A[3]=A11
    # B[0]=B00, B[1]=B01, B[2]=B10, B[3]=B11
    # C[0]=C00, C[1]=C01, C[2]=C10, C[3]=C11
    
    # P1 = (A00+A11)(B00+B11)
    # Contribution: +1 to C00, -1 to C01, +1 to C10, +1 to C11? No.
    # From the formulas:
    # C00 = P1 - P4 + P5 - P7
    # C01 = P3 + P4
    # C10 = P2 + P5
    # C11 = P1 - P2 + P3 + P6
    #
    # So P1 appears in C00 with +1 and C11 with +1
    triples.append(([1,0,0,1], [1,0,0,1], [1,0,0,1]))  # P1: C00+1, C11+1
    
    # P2 = (A10+A11)*B00
    # P2 appears in C10 with +1 and C11 with -1
    triples.append(([0,0,1,1], [1,0,0,0], [0,0,1,-1]))  # P2: C10+1, C11-1
    
    # P3 = (A00-A01)*B11
    # P3 appears in C01 with +1 and C11 with +1
    triples.append(([1,-1,0,0], [0,0,0,1], [0,1,0,1]))  # P3: C01+1, C11+1
    
    # P4 = A11*(B01-B10)
    # P4 appears in C00 with -1 and C01 with +1
    triples.append(([0,0,0,1], [0,1,-1,0], [-1,1,0,0]))  # P4: C00-1, C01+1
    
    # P5 = A00*(B00+B01)
    # P5 appears in C00 with +1 and C10 with +1
    triples.append(([1,0,0,0], [1,1,0,0], [1,0,1,0]))  # P5: C00+1, C10+1
    
    # P6 = A10*(B10+B11)
    # P6 appears in C11 with +1
    triples.append(([0,0,1,0], [0,0,1,1], [0,0,0,1]))  # P6: C11+1
    
    # P7 = (A01-A11)*(B00+B11)
    # P7 appears in C00 with -1
    triples.append(([0,1,0,-1], [1,0,0,1], [-1,0,0,0]))  # P7: C00-1
    
    return triples

def _construct_strassen_based():
    n, m, p = 3, 4, 6
    # A is n x m = 3x4, B is m x p = 4x6, C is n x p = 3x6
    # W is indexed as W[k*n+i] for output C[i,k]
    # U is indexed as U[i*m+j] for A[i,j]
    # V is indexed as V[j*p+k] for B[j,k]
    
    # Strategy:
    # Split A rows: row 0 (1 row) and rows 1-2 (2 rows)
    # Split shared dim (4): cols 0-1 (2) and cols 2-3 (2)
    # Split B cols: cols 0-1 (2), cols 2-3 (2), cols 4-5 (2)
    
    # For the top row (A[0,:] is 1x4):
    #   A[0,:] * B[:,k] for each k: this is a 1x4 x 4x1 = just dot products
    #   Split B into top 2 and bottom 2: A[0,0:2]*B[0:2,k] + A[0,2:4]*B[2:4,k]
    #   Each is a 1x2 x 2x1, which has rank 2 (can't reduce below 2)
    #   So for each of the 6 output columns: 2*2 = 4 multiplications? No.
    #   1x2 x 2x1: A[0,0]*B[0,k] + A[0,1]*B[1,k] -> rank 2
    #   A[0,2]*B[2,k] + A[0,3]*B[3,k] -> rank 2
    #   Total for top row: 6 outputs * 4 = 24? No, that's not right either.
    #   Actually for output C[0,k]: sum over j of A[0,j]*B[j,k]
    #   = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    #   This is a 1x4 x 4x1 tensor, rank = 4 (it's just a vector dot product)
    #   But we can split: (A[0,0]*B[0,k] + A[0,1]*B[1,k]) + (A[0,2]*B[2,k] + A[0,3]*B[3,k])
    #   First part: 1x2 x 2x1, rank 2
    #   Second part: 1x2 x 2x1, rank 2
    #   Total rank for top row: 2+2 = 4 per output column, 6 columns = 24
    #   Wait no, the rank is additive across output columns since they're independent.
    #   Actually no - the rank of the whole tensor is what matters.
    
    # Let me think differently. The total tensor T[i,j,k] = delta (i.e., C[i,k] = sum_j A[i,j]B[j,k])
    # We want to write T = sum_r U[:,r] ⊗ V[:,r] ⊗ W[:,r]
    
    # For the top row (i=0): T[0,j,k] = delta (just 1 for each j,k pair in the sum)
    # This is a 1x4x6 tensor that equals the outer product structure of a 4-dim dot product
    # Rank of 1x4x6 "matrix mult" tensor = 4 (it's just vec(A[0,:]) ⊗ something)
    # Actually T[0,j,k] = 1 for all j,k (it's the identity in the tensor sense)
    # No: C[0,k] = sum_j A[0,j] B[j,k], so the tensor entry T[0,j,k] = 1 for all j,k
    # This is the all-ones tensor of shape 1x4x6, which has rank... 
    # It's U ⊗ V ⊗ W where U=[1], V is 4-dim, W is 6-dim, but we need V and W to be fixed.
    # Actually T[0,j,k] = 1 = 1*1*1, so it's rank 1? No, that's not right for matrix mult.
    # 
    # I'm confusing myself. Let me just directly construct the decomposition.
    
    # The tensor T has entries T[i,j,k] where i in [0,n), j in [0,m), k in [0,p)
    # T[i,j,k] = 1 for all i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
    # We need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Schoolbook: one term per (i,j,k) with U=V=W=1. Rank = n*m*p = 72.
    
    # Strassen approach for the 2x2x2 sub-blocks:
    # The 7 Strassen triples give: sum_r u_r[i2] * v_r[j2] * w_r[k2] = delta (for 2x2x2)
    # where i2 in {0,1}, j2 in {0,1}, k2 in {0,1}
    
    # For our 3x4x6 problem:
    # Let's partition:
    #   i: 0 | 1,2  (1 + 2)
    #   j: 0,1 | 2,3  (2 + 2)  
    #   k: 0,1 | 2,3 | 4,5  (2 + 2 + 2)
    
    # For i=0 (top row), j in {0,1,2,3}, k in {0,...,5}:
    #   T[0,j,k] = 1
    #   We can write this using the fact that for fixed i=0, it's a 4x6 "identity" tensor
    #   Split j into j1={0,1} and j2={2,3}, k into k1={0,1}, k2={2,3}, k3={4,5}
    #   T[0,j1,k1] = 1 (2x2 block of ones) - this is rank 2 (since 1x2x2 with all ones)
    #   Actually for 1x2x2 all-ones: T[0,j,k]=1 for j in {0,1}, k in {0,1}
    #   This can be written as: 1*1*1 + 1*1*1? No.
    #   sum_r U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
    #   For j in {0,1}, k in {0,1}: we need 4 equations.
    #   Simplest: 4 terms (schoolbook) or can we do 2?
    #   If we use 2 terms: U[0,0]*V[0,0]*W[0,0] = 1, U[0,1]*V[1,0]*W[1,0] = 1, etc.
    #   This is the rank of the 1x2x2 all-ones tensor, which is 2.
    #   E.g., term1: U[0,0]=1, V[0,0]=1, W[0,0]=1; U[0,1]=1, V[1,0]=1, W[1,0]=1
    #   But we need all 4 entries to be 1. With 2 terms:
    #   T[0,0,0] = U00*V00*W00 + U01*V01*W01 = 1
    #   T[0,0,1] = U00*V02*W00 + U01*V03*W01 = 1  (V index: j*p+k, for j=0: V[0,0],V[0,1])
    #   This is getting complicated. Let me just use the 2x2x2 Strassen structure.
    
    # Key insight: The 2x2x2 matrix mult tensor (rank 7) satisfies:
    # sum_r u_r[a]*v_r[b]*w_r[c] = 1 for all a,b,c in {0,1}
    # So if we can decompose our problem into 2x2x2 sub-problems, we can use Strassen.
    
    # Let's try a different decomposition:
    # For the 2x2x2 part (i in {1,2}, j in {0,1}, k in {0,1}):
    #   This is a 2x2x2 matrix mult: C[i-1,k] = sum_j A[i,j]B[j,k]
    #   Wait, that's not quite right. Let me be more careful.
    
    # Actually, let me just build it directly with the Strassen triples.
    
    U = []
    V = []
    W = []
    
    def add_term(u_vec, v_vec, w_vec):
        """Add a term. u_vec is dict or list for U indices, etc."""
        U.append(u_vec)
        V.append(v_vec)
        W.append(w_vec)
    
    n, m, p = 3, 4, 6
    # U has n*m = 12 rows, V has m*p = 24 rows, W has p*n = 18 rows
    
    # For the top row i=0:
    # C[0,k] = sum_{j=0}^{3} A[0,j] * B[j,k]
    # Split j into {0,1} and {2,3}:
    # C[0,k] = sum_{j=0}^{1} A[0,j]*B[j,k] + sum_{j=2}^{3} A[0,j]*B[j,k]
    # 
    # For each k-group (k in {0,1}, {2,3}, {4,5}):
    #   sum_{j=0}^{1} A[0,j]*B[j,k] is a 1x2x2 sub-tensor (i=0, j in {0,1}, k in 2-group)
    #   This has rank 2 (can't reduce a 1x2x2 identity tensor below rank 2)
    #   Similarly for j in {2,3}: rank 2
    #   So per k-group: rank 4, total for top row: 3*4 = 12
    
    # For the bottom 2 rows i in {1,2}:
    # C[i,k] = sum_{j=0}^{3} A[i,j]*B[j,k]
    # Split j into {0,1} and {2,3}:
    # C[i,k] = sum_{j=0}^{1} A[i,j]*B[j,k] + sum_{j=2}^{3} A[i,j]*B[j,k]
    #
    # For j in {0,1}: this is a 2x2x2 sub-problem for each k-group
    #   A[i,j] for i in {1,2}, j in {0,1} -> 2x2 matrix
    #   B[j,k] for j in {0,1}, k in {0,1} (for k-group 1) -> 2x2 matrix
    #   C[i,k] for i in {1,2}, k in {0,1} -> 2x2 matrix
    #   This is exactly 2x2 matrix mult, rank 7 (Strassen)!
    #   Similarly for k-group 2 and k-group 3: each rank 7
    #   Total for j in {0,1}: 3*7 = 21
    #
    # For j in {2,3}: similarly 3*7 = 21
    # Total for bottom rows: 42
    #
    # Grand total: 12 + 42 = 54
    
    # Let me construct this.
    
    strassen = _strassen_2x2x2()
    
    # Top row (i=0), j in {0,1}, k-group g (k = 2g, 2g+1):
    # Need: for j in {0,1}, k in {2g, 2g+1}: sum_r U[0*m+j, r] * V[j*p+k, r] * W[k*n+0, r] = 1
    # This is a 1x2x2 all-ones tensor. Rank 2.
    # Term 1: U[0,0]=1, V[0,2g]=1, W[2g,0]=1  (covers (j=0,k=2g))
    # Term 2: U[0,1]=1, V[1,2g]=1, W[2g,0]=1  (covers (j=1,k=2g))
    # Wait, that only covers k=2g. I need to cover both k=2g and k=2g+1.
    # 
    # For 1x2x2 all-ones: T[j,k]=1 for j in {0,1}, k in {0,1}
    # Rank 2 decomposition:
    # Term 1: U[j]=1 for j=0,1; V[k]=1 for k=0,1; W=1
    #   But U and V are vectors, so: U=[1,1], V=[1,1], W=[1]
    #   Then T[j,k] = U[j]*V[k]*W = 1*1*1 = 1. Yes! Rank 1!
    # Wait, that's rank 1? The 1x2x2 all-ones tensor IS rank 1!
    # U = [1,1] (length 2), V = [1,1] (length 2), W = [1] (length 1)
    # T[j,k] = U[j]*V[k]*W[0] = 1*1*1 = 1. Correct!
    
    # So for top row, j-group {0,1}, k-group g:
    # U[0*m+0] = 1, U[0*m+1] = 1
    # V[0*p+2g] = 1, V[1*p+2g] = 1, V[0*p+2g+1] = 1, V[1*p+2g+1] = 1
    # W[(2g)*n+0] = 1, W[(2g+1)*n+0] = 1
    # That's 1 term per (j-group, k-group). 2 j-groups * 3 k-groups = 6 terms.
    
    for j_group in range(2):  # j_group 0: j in {0,1}, j_group 1: j in {2,3}
        for k_group in range(3):  # k_group g: k in {2g, 2g+1}
            u = [0]*(n*m)
            v = [0]*(m*p)
            w = [0]*(p*n)
            j0 = 2*j_group
            k0 = 2*k_group
            # U indices: i*m+j = 0*m + j = j
            u[j0] = 1
            u[j0+1] = 1
            # V indices: j*p+k
            v[j0*p + k0] = 1
            v[j0*p + k0+1] = 1
            v[(j0+1)*p + k0] = 1
            v[(j0+1)*p + k0+1] = 1
            # W indices: k*n+i = k*n+0 = k
            w[k0*n + 0] = 1
            w[(k0+1)*n + 0] = 1
            add_term(u, v, w)
    
    # Bottom rows (i in {1,2}), j-group {0,1} or {2,3}, k-group g:
    # This is 2x2x2 matrix mult, use Strassen (rank 7)
    for j_group in range(2):
        for k_group in range(3):
            j0 = 2*j_group
            k0 = 2*k_group
            for (su, sv, sw) in strassen:
                u = [0]*(n*m)
                v = [0]*(m*p)
                w = [0]*(p*n)
                # su is length 2, corresponding to A[i,j] for i in {1,2}, j in {j0, j0+1}
                # U index: i*m + j
                for a in range(2):  # a corresponds to i = 1+a
                    for b in range(2):  # b corresponds to j = j0+b
                        u[(1+a)*m + (j0+b)] = su[a*2+b] if len(su) == 4 else su[b]
                # Wait, su has length 4 for 2x2, indexed as su[a*2+b] where a is row, b is col
                # But in our Strassen triples, su is indexed as [A00, A01, A10, A11]
                # which is su[a*2+b] with a=row, b=col.
                # A[i,j] where i=1+a, j=j0+b
                # U index = i*m + j = (1+a)*m + (j0+b)
                # Let me redo:
                u = [0]*(n*m)
                for a in range(2):
                    for b in range(2):
                        u[(1+a)*m + (j0+b)] = su[a*2+b]
                
                # sv is length 4, indexed as [B00, B01, B10, B11] = [B[j0,k0], B[j0,k0+1], B[j0+1,k0], B[j0+1,k0+1]]
                # B[j,k] where j=j0+b, k=k0+c
                # V index = j*p + k = (j0+b)*p + (k0+c)
                v = [0]*(m*p)
                for b in range(2):
                    for c in range(2):
                        v[(j0+b)*p + (k0+c)] = sv[b*2+c]
                
                # sw is length 4, indexed as [C00, C01, C10, C11]
                # C[i,k] where i=1+a, k=k0+c
                # W index = k*n + i = (k0+c)*n + (1+a)
                w = [0]*(p*n)
                for a in range(2):
                    for c in range(2):
                        w[(k0+c)*n + (1+a)] = sw[c*2+a]
                
                add_term(u, v, w)
    
    # Convert to lists of lists
    U_out = U
    V_out = V
    W_out = W
    
    return {"U": U_out, "V": V_out, "W": W_out}

def _verify(U, V, W):
    """Verify the decomposition."""
    n, m, p = 3, 4, 6
    rank = len(U[0]) if U else 0
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if abs(s - 1) > 1e-9:
                    return False
    return True

def construct():
    deadline = time.time() + 110
    n, m, p = DIMENSIONS
    
    result = _construct_strassen_based()
    
    # Verify
    if _verify(result["U"], result["V"], result["W"]):
        return result
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
