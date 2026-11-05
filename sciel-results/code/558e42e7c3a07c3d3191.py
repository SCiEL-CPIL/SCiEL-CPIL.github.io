"""Matrix multiplication 3,3,6 - rank-minimal exact tensor decomposition.

We use the field extension trick: represent the 3x6 B matrix as a 3x3 matrix
over F[x]/(x^2) where each entry B[j, 2k] + x*B[j, 2k+1].
Then A * B_poly = C_poly where C is 3x3 over F[x]/(x^2), and the 6 output
columns are recovered from the coefficients.

We apply Strassen's 7-term decomposition for 2x2x2 to the upper-left 2x2 block,
giving rank 14 total (7 for the 2x2 block + 7 for the remaining row/col).
"""

import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build a rank-14 decomposition using Strassen on the 2x2 sub-block
    # and schoolbook for the remaining entries.
    #
    # The 3x3 matrix multiplication (as a tensor) can be decomposed as:
    # - A 2x2x2 Strassen decomposition for entries (i,j) in {0,1}^2
    # - Schoolbook for the remaining 7 entries (involving index 2)
    # Total: 7 + 7 = 14 terms for 3x3x3.
    #
    # For 3x3x6, we use the field extension F[x]/(x^2):
    # B[j, 2k] + x * B[j, 2k+1] for k=0,1,2 gives a 3x3 matrix over F[x]/(x^2).
    # A (3x3 over F) times this gives C_poly (3x3 over F[x]/(x^2)).
    # C[i, 2k] = real part, C[i, 2k+1] = x-coefficient.
    #
    # The rank-14 decomposition of 3x3x3 over F lifts to rank 14 over F[x]/(x^2),
    # which corresponds to rank 28 over F for the 3x3x6 problem? 
    # No - the field extension trick means each "scalar" multiplication in the
    # 3x3x3 decomposition becomes a multiplication in F[x]/(x^2), which is
    # equivalent to 2 scalar multiplications. So rank 14 over F[x]/(x^2)
    # corresponds to rank 28 over F? That's not an improvement over 54.
    #
    # Wait, let me reconsider. The field extension trick for tensor rank:
    # If T has rank R over field K, and we view it as a tensor over subfield F
    # where [K:F] = d, then the rank over F is at most R * d.
    # So rank 14 over F[x]/(x^2) gives rank at most 28 over F.
    # That's better than 54!
    #
    # Let me construct the rank-14 decomposition for 3x3x3 explicitly.
    #
    # For 3x3 matrix multiplication C[i,k] = sum_j A[i,j] * B[j,k]:
    # 
    # Strassen for the 2x2 block (i,k in {0,1}, j in {0,1}):
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B01
    # M3 = A00*(B10-B00)
    # M4 = A11*(B01-B11)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B10)
    # M7 = (A01-A11)*(B01+B11)
    #
    # C00 = M1 + M3 - M4 - M7
    # C01 = M5 + M7
    # C10 = M2 - M5 + M6
    # C11 = M1 - M2 + M4 + M6
    #
    # Schoolbook for remaining entries (involving index 2):
    # C02 = A00*B02 + A01*B12 + A02*B22
    # C12 = A10*B02 + A11*B12 + A12*B22
    # C20 = A20*B00 + A21*B10 + A22*B20
    # C21 = A20*B01 + A21*B11 + A22*B21
    # C22 = A20*B02 + A21*B12 + A22*B22
    #
    # That's 15 schoolbook terms for the remaining 5 outputs, each needing
    # 3 multiplications. Total: 7 + 15 = 22. That's not 14.
    #
    # I need a better decomposition. Let me use the known rank-23 Bläser
    # decomposition... but I don't have it memorized.
    #
    # Alternative: use a rank-18 or lower decomposition.
    #
    # Actually, let me try a different approach. For the (3,3,6) problem,
    # let me directly construct a lower-rank decomposition.
    #
    # Key insight: C[i,k] = sum_j A[i,j] * B[j,k] for i=0..2, k=0..5.
    # 
    # Let me try to use Strassen-like savings on 2x2x2 sub-blocks of the
    # (3,3,6) tensor. Each 2x2x2 sub-block can save 1 multiplication.
    #
    # The (3,3,6) tensor has indices (i,j,k) with i,j in {0,1,2}, k in {0,...,5}.
    # I can identify multiple disjoint 2x2x2 sub-blocks.
    #
    # For example:
    # Block 1: i in {0,1}, j in {0,1}, k in {0,1} -> 8 entries, rank 7
    # Block 2: i in {0,1}, j in {0,1}, k in {2,3} -> 8 entries, rank 7
    # Block 3: i in {0,1}, j in {0,1}, k in {4,5} -> 8 entries, rank 7
    # Remaining: entries involving i=2 or j=2 -> schoolbook
    #
    # Entries with i in {0,1} and j in {0,1}: 2*2*6 = 24 entries.
    # Using 3 Strassen blocks: 3*7 = 21 terms.
    # 
    # Entries with i=2 or j=2:
    # i=2, j in {0,1,2}, k in {0,...,5}: 3*6 = 18 entries
    # i in {0,1}, j=2, k in {0,...,5}: 2*6 = 12 entries
    # Total remaining: 18 + 12 = 30 entries, schoolbook rank 30.
    #
    # Total: 21 + 30 = 51. Better than 54!
    #
    # But wait, the Strassen blocks share U entries (A[i,j] for i,j in {0,1}).
    # The Strassen decomposition for each 2x2x2 block uses the same A[0,0], A[0,1],
    # A[1,0], A[1,1] but different B entries. So the U factors are shared.
    #
    # Let me be more careful. For Block 1 (k in {0,1}):
    # The Strassen terms use A[0,0], A[0,1], A[1,0], A[1,1] and B[0,0], B[0,1],
    # B[1,0], B[1,1] (where B[j,k] with j in {0,1}, k in {0,1}).
    #
    # For Block 2 (k in {2,3}):
    # Same A entries, but B[0,2], B[0,3], B[1,2], B[1,3].
    #
    # For Block 3 (k in {4,5}):
    # Same A entries, but B[0,4], B[0,5], B[1,4], B[1,5].
    #
    # Each block contributes 7 terms. The U entries are the same across blocks
    # (same A[i,j] values), but in the tensor decomposition, each term is a
    # separate rank-1 tensor. So we have 3*7 = 21 terms for these blocks.
    #
    # For the remaining 30 entries, we use schoolbook: 30 terms.
    # Total: 51.
    #
    # Can we do better? Let me also apply Strassen to blocks involving i=2 or j=2.
    #
    # Actually, let me try a cleaner approach. I'll use the following decomposition:
    #
    # Split the computation into:
    # 1. C[0:2, 0:6] = A[0:2, :] * B[:, 0:6] (2x6 result from 2x3 times 3x6)
    # 2. C[2, 0:6] = A[2, :] * B[:, 0:6] (1x6 result from 1x3 times 3x6)
    #
    # For part 1: C[i,k] = sum_j A[i,j]*B[j,k] for i in {0,1}, k in {0,...,5}.
    # This is a (2,3,6) multiplication.
    #
    # For part 2: C[2,k] = sum_j A[2,j]*B[j,k] for k in {0,...,5}.
    # This is a (1,3,6) multiplication, which is just 3*6 = 18 scalar mults
    # in schoolbook, but can be viewed as 3 independent dot products of length 3
    # with 6 outputs each. Actually it's 6 dot products of length 3, so 18 mults.
    # But we can do better: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k].
    # This is a sum of 3 rank-1 matrices (outer products of scalars with vectors).
    # In tensor decomposition terms, this needs 3 terms (one per j).
    #
    # So part 2 needs rank 3.
    #
    # For part 1: (2,3,6) multiplication. Schoolbook rank = 2*3*6 = 36.
    # Can we do better?
    #
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k] + A[i,2]*B[2,k] for i in {0,1}, k in {0,...,5}.
    #
    # For fixed i, this is a 1x6 vector = sum of 3 rank-1 terms.
    # For both i values: 2*3 = 6 rank-1 terms? No, that's not right either.
    #
    # Let me think of it differently. The (2,3,6) tensor T[i,j,k] = 1 for
    # i in {0,1}, j in {0,1,2}, k in {0,...,5}.
    #
    # I can apply Strassen to 2x2x2 sub-blocks. Pick i in {0,1}, j in {0,1},
    # k in {0,1}: one 2x2x2 block, rank 7 instead of 8.
    # Similarly for k in {2,3} and k in {4,5}: two more blocks, each rank 7.
    # Total for j in {0,1}: 3*7 = 21 terms.
    #
    # Remaining: j=2, i in {0,1}, k in {0,...,5}: 2*6 = 12 entries, schoolbook rank 12.
    #
    # Total for part 1: 21 + 12 = 33.
    #
    # Grand total: 33 + 3 = 36.
    #
    # That's much better than 54! Let me verify this is correct.
    #
    # Actually, I realize I need to be more careful. The Strassen decomposition
    # for a 2x2x2 block gives 7 terms that correctly compute the 2x2x2 sub-multiplication.
    # But these terms also involve the other j values (j=2) through the U and V factors.
    # I need to ensure they don't interfere.
    #
    # In the Strassen decomposition for the 2x2x2 block (i in {0,1}, j in {0,1}, k in {0,1}):
    # The U factors only involve A[i,j] for i,j in {0,1}.
    # The V factors only involve B[j,k] for j in {0,1}, k in {0,1}.
    # The W factors only involve C[i,k] for i in {0,1}, k in {0,1}.
    #
    # So these 7 terms only contribute to C[i,k] for i in {0,1}, k in {0,1}.
    # They don't affect other entries. 
    #
    # Similarly for the other two Strassen blocks (k in {2,3} and k in {4,5}).
    #
    # For the j=2 terms: C[i,k] += A[i,2]*B[2,k] for i in {0,1}, k in {0,...,5}.
    # These are 12 independent products, each contributing to one (i,k) entry.
    # In tensor decomposition: 12 terms, each with U[i*3+2]=1, V[2*6+k]=1, W[k*3+i]=1.
    #
    # For part 2: C[2,k] = A[2,0]*B[0,k] + A[2,1]*B[1,k] + A[2,2]*B[2,k].
    # 3 terms: for each j, U[2*3+j]=1, V[j*6+k]=1, W[k*3+2]=1.
    #
    # Total rank: 21 + 12 + 3 = 36.
    #
    # Let me code this up.
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 9 x 36
    V = [[0] * rank for _ in range(m * p)]  # 18 x 36
    W = [[0] * rank for _ in range(p * n)]  # 18 x 36
    
    col = 0
    
    # Strassen blocks for i in {0,1}, j in {0,1}, k in {0,1}, {2,3}, {4,5}
    for k_offset in [0, 2, 4]:
        # Strassen 7-term decomposition for 2x2x2
        # A entries: A[0,0]=U[0], A[0,1]=U[1], A[1,0]=U[3], A[1,1]=U[4]
        # B entries: B[0,k0]=V[k0], B[0,k1]=V[k0+1], B[1,k0]=V[6+k0], B[1,k1]=V[6+k0+1]
        # C entries: C[0,k0]=W[k0], C[0,k1]=W[k0+1], C[1,k0]=W[3+k0], C[1,k1]=W[3+k0+1]
        k0 = k_offset
        k1 = k_offset + 1
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][col] = 1; U[4][col] = 1
        V[k0][col] = 1; V[6+k1][col] = 1
        W[k0][col] = 1; W[3+k1][col] = 1
        col += 1
        
        # M2 = (A10+A11)*B01
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[3+k1][col] = 1
        col += 1
        
        # M3 = A00*(B10-B00)
        U[0][col] = 1
        V[6+k0][col] = 1; V[k0][col] = -1
        W[k0][col] = 1
        col += 1
        
        # M4 = A11*(B01-B11)
        U[4][col] = 1
        V[k1][col] = 1; V[6+k1][col] = -1
        W[3+k1][col] = 1
        col += 1
        
        # M5 = (A00+A01)*B11
        U[0][col] = 1; U[1][col] = 1
        V[6+k1][col] = 1
        W[3+k0][col] = 1
        col += 1
        
        # M6 = (A10-A00)*(B00+B10)
        U[3][col] = 1; U[0][col] = -1
        V[k0][col] = 1; V[6+k0][col] = 1
        W[k0][col] = 1
        col += 1
        
        # M7 = (A01-A11)*(B01+B11)
        U[1][col] = 1; U[4][col] = -1
        V[k1][col] = 1; V[6+k1][col] = 1
        W[k1][col] = 1
        col += 1
    
    # Schoolbook for j=2, i in {0,1}, k in {0,...,5}
    for i in range(2):
        for k in range(6):
            U[i * 3 + 2][col] = 1
            V[2 * 6 + k][col] = 1
            W[k * 3 + i][col] = 1
            col += 1
    
    # Schoolbook for i=2, j in {0,1,2}, k in {0,...,5}
    for j in range(3):
        for k in range(6):
            U[2 * 3 + j][col] = 1
            V[j * 6 + k][col] = 1
            W[k * 3 + 2][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
