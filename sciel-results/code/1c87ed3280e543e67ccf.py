import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Use a block-based Strassen approach:
    # Partition A (3x3) into:
    #   A11 (2x2), A12 (2x1), A21 (1x2), A22 (1x1)
    # Partition B (3x6) into:
    #   B11 (2x6), B21 (1x6)
    # 
    # C = A*B gives:
    # C11 (2x6) = A11*B11 + A12*B21
    # C21 (1x6) = A21*B11 + A22*B21
    #
    # For A11*B11 (2x2 * 2x6), we can use Strassen on the 2x2 part of B.
    # But B11 is 2x6, so we need to handle it as two 2x2 blocks + one 2x2 block.
    #
    # Actually, let's use a simpler but effective approach:
    # Apply Strassen to each 2x2x2 sub-block and handle cross terms.
    #
    # For 3x3 * 3x6, we can view it as:
    # - 2x2 * 2x6 (top-left block of A times top 2 rows of B)
    # - 1x2 * 2x6 (bottom row of A times top 2 rows of B)  
    # - 2x1 * 1x6 (top 2 rows of A times bottom row of B)
    # - 1x1 * 1x6 (bottom-right of A times bottom row of B)
    #
    # For 2x2 * 2x6: partition B into three 2x2 blocks, apply Strassen to each
    # That gives 7*3 = 21 mults for the 2x2*2x6 part (vs 2*2*6=24 schoolbook)
    # For 1x2 * 2x6: schoolbook = 1*2*6 = 12 mults
    # For 2x1 * 1x6: schoolbook = 2*1*6 = 12 mults
    # For 1x1 * 1x6: schoolbook = 1*1*6 = 6 mults
    # Total: 21 + 12 + 12 + 6 = 51 (vs 54 schoolbook) - small improvement
    #
    # Better: use a more global optimization.
    #
    # Let me try a direct rank-18 construction based on known results.
    # The tensor rank for 3x3*3x6 should be significantly less than 54.
    #
    # I'll use a systematic approach: enumerate all (i,j,k) triples and
    # look for linear dependencies to reduce rank.
    #
    # For now, let's implement a verified Strassen-based decomposition.
    
    # Schoolbook as fallback
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Try to build a lower-rank decomposition using Strassen blocks
    # We'll collect (i, j, k, coeff) tuples and then try to reduce them
    
    terms = []
    
    # Apply Strassen to the top-left 2x2 block of A with each 2x2 block of B
    # A[0:2, 0:2] * B[0:2, 0:2] -> contributes to C[0:2, 0:2]
    # A[0:2, 0:2] * B[0:2, 2:4] -> contributes to C[0:2, 2:4]
    # A[0:2, 0:2] * B[0:2, 4:6] -> contributes to C[0:2, 4:6]
    
    def strassen_2x2x2(i_off, j_off, k_off):
        """Return 7 rank-1 terms for 2x2x2 Strassen."""
        i0, i1 = i_off, i_off + 1
        j0, j1 = j_off, j_off + 1
        k0, k1 = k_off, k_off + 1
        
        # M1 = (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # M2 = (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # M3 = A[i0,j0] * (B[j1,k0]-B[j1,k1])
        # M4 = A[i1,j1] * (B[j0,k1]-B[j0,k0])
        # M5 = (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # M6 = (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # M7 = (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        
        # C[i0,k0] = M1 + M4 - M5 + M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M6
        
        # Each M is a product of a linear form in A and a linear form in B.
        # In the tensor decomposition, each M corresponds to one rank-1 term
        # where U has the A-coefficients, V has the B-coefficients, and W
        # has the output coefficients.
        
        # But wait - in the tensor rank decomposition, we need:
        # sum_r U[r][i*m+j] * V[r][j*p+k] * W[r][k*n+i] = delta(i,j,k)
        # for the matrix multiplication tensor.
        
        # Actually, the tensor T satisfies T[i,j,k] = 1 for all (i,j,k) in the
        # multiplication, and C[i,k] = sum_j T[i,j,k] * A[i,j] * B[j,k].
        
        # A rank-1 term is: u_r[i*m+j] * v_r[j*p+k] * w_r[k*n+i]
        # where u_r, v_r, w_r are vectors.
        
        # For Strassen, each M_r = (sum_{i,j} a_r[i,j] * A[i,j]) * (sum_{j,k} b_r[j,k] * B[j,k])
        # contributes to output entries C[i,k] with coefficient c_r[i,k].
        
        # So the rank-1 term has:
        # U_r[i*m+j] = a_r[i,j]
        # V_r[j*p+k] = b_r[j,k]  
        # W_r[k*n+i] = c_r[i,k]
        
        # And we need: sum_r a_r[i,j] * b_r[j,k] * c_r[i,k] = 1 for all (i,j,k)
        # where the product A[i,j]*B[j,k] appears in C[i,k].
        
        # For the 2x2x2 block at offsets (i_off, j_off, k_off):
        # The Strassen decomposition gives 7 terms that exactly compute
        # the 4 output entries using the 4 A entries and 4 B entries.
        
        terms_list = []
        
        # M1: a = {(i0,j0):1, (i1,j1):1}, b = {(j0,k0):1, (j1,k1):1}
        # c: C[i0,k0]+=1, C[i0,k1]+=0, C[i1,k0]+=0, C[i1,k1]+=1 (from M1 + M4 - M5 + M7 for C[i0,k0], etc.)
        # Actually let me be more careful.
        
        # C[i0,k0] = M1 + M4 - M5 + M7
        # C[i0,k1] = M3 + M5
        # C[i1,k0] = M2 + M4
        # C[i1,k1] = M1 - M2 + M3 + M6
        
        # So for each M_r, the contribution to each C entry is:
        # M1: C[i0,k0] += 1, C[i1,k1] += 1
        # M2: C[i1,k0] += 1, C[i1,k1] += -1
        # M3: C[i0,k1] += 1, C[i1,k1] += 1
        # M4: C[i0,k0] += 1, C[i1,k0] += 1
        # M5: C[i0,k0] += -1, C[i0,k1] += 1
        # M6: C[i1,k1] += 1
        # M7: C[i0,k0] += 1
        
        # M1: a={(i0,j0):1, (i1,j1):1}, b={(j0,k0):1, (j1,k1):1}, c={(i0,k0):1, (i1,k1):1}
        terms_list.append({
            'a': {(i0,j0):1, (i1,j1):1},
            'b': {(j0,k0):1, (j1,k1):1},
            'c': {(i0,k0):1, (i1,k1):1}
        })
        
        # M2: a={(i1,j0):1, (i1,j1):1}, b={(j0,k0):1}, c={(i1,k0):1, (i1,k1):-1}
        terms_list.append({
            'a': {(i1,j0):1, (i1,j1):1},
            'b': {(j0,k0):1},
            'c': {(i1,k0):1, (i1,k1):-1}
        })
        
        # M3: a={(i0,j0):1}, b={(j1,k0):1, (j1,k1):-1}, c={(i0,k1):1, (i1,k1):1}
        terms_list.append({
            'a': {(i0,j0):1},
            'b': {(j1,k0):1, (j1,k1):-1},
            'c': {(i0,k1):1, (i1,k1):1}
        })
        
        # M4: a={(i1,j1):1}, b={(j0,k1):1, (j0,k0):-1}, c={(i0,k0):1, (i1,k0):1}
        terms_list.append({
            'a': {(i1,j1):1},
            'b': {(j0,k1):1, (j0,k0):-1},
            'c': {(i0,k0):1, (i1,k0):1}
        })
        
        # M5: a={(i0,j0):1, (i0,j1):1}, b={(j1,k1):1}, c={(i0,k0):-1, (i0,k1):1}
        terms_list.append({
            'a': {(i0,j0):1, (i0,j1):1},
            'b': {(j1,k1):1},
            'c': {(i0,k0):-1, (i0,k1):1}
        })
        
        # M6: a={(i1,j0):1, (i0,j0):-1}, b={(j0,k0):1, (j0,k1):1}, c={(i1,k1):1}
        terms_list.append({
            'a': {(i1,j0):1, (i0,j0):-1},
            'b': {(j0,k0):1, (j0,k1):1},
            'c': {(i1,k1):1}
        })
        
        # M7: a={(i0,j1):1, (i1,j1):-1}, b={(j1,k0):1, (j1,k1):1}, c={(i0,k0):1}
        terms_list.append({
            'a': {(i0,j1):1, (i1,j1):-1},
            'b': {(j1,k0):1, (j1,k1):1},
            'c': {(i0,k0):1}
        })
        
        return terms_list
    
    # Now, the full 3x3*3x6 multiplication involves all (i,j,k) triples.
    # We can't just apply Strassen to isolated 2x2x2 blocks because the
    # A and B entries are shared across blocks.
    
    # Instead, let's use a different strategy: directly construct a low-rank
    # decomposition by solving for it.
    
    # The tensor T has T[i,j,k] = 1 for all i in [0,3), j in [0,3), k in [0,6).
    # We want to find rank r such that:
    # T[i,j,k] = sum_{r=0}^{R-1} U_r[i*m+j] * V_r[j*p+k] * W_r[k*n+i]
    
    # This is equivalent to finding the tensor rank.
    
    # For a practical approach, let's use the known result that for
    # m x n x p matrix multiplication, the rank is at most m*n*p but can
    # be reduced.
    
    # Let me try a direct construction with rank 18 as suggested.
    # 
    # Actually, let's just return the schoolbook for now and verify it works,
    # then try to improve.
    
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
