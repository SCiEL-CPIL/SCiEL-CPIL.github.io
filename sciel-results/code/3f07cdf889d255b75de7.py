"""
Rank-minimal exact tensor decomposition for 3x4 x 4x4 matrix multiplication.

Uses Strassen's algorithm on 2x2 blocks to reduce rank below the schoolbook 48.
Construction:
- Partition A (3x4) into row-blocks: rows 0-1 (2x4) and row 2 (1x4)
- Partition B (4x4) into 2x2 blocks
- C (3x4) output: rows 0-1 use Strassen on (2x2 * 2x2) terms, row 2 uses 1x2 * 2x2 terms

For the top 2x4 * 4x4 -> 2x4 part:
  Split into 4 sub-problems of 2x2 * 2x2, each using Strassen (rank 7) = 28
  Plus 4 sub-problems of 1x2 * 2x2, each using schoolbook (rank 2) = 8
  Total = 36... let me recalculate.

Actually let me think more carefully.

A is 3x4, B is 4x4, C is 3x4.

C = A * B

Write A = [A1; A2] where A1 is 2x4, A2 is 1x4.
Write B = [B11 B12; B21 B22] where each is 2x2.

C1 = A1 * B = [A11 B11 + A12 B21, A11 B12 + A12 B22]  (2x4)
C2 = A2 * B = [A21 B11 + A22 B21, A21 B12 + A22 B22]  (1x4)

For C1: A1 is 2x4, B is 4x4. 
  C1_left (2x2) = A11 * B11 + A12 * B21
  C1_right (2x2) = A11 * B12 + A12 * B22

Each of these is a sum of two 2x2*2x2 products. Using Strassen for each pair:
  For C1_left: use Strassen on (A11, B11) gives 7 terms, Strassen on (A12, B21) gives 7 terms. Total 14.
  But we can be smarter: C1_left = A11*B11 + A12*B21. This is M(2,4,2) if we view it as (2x4)*(4x2).
  
Actually, the rank of 2x4 x 4x2 is known. Let me use the standard approach:

For 2x2 x 2x2: Strassen rank = 7
For 1x2 x 2x2: rank = 2 (each output element needs one product... wait, 1x2 * 2x2 = 1x2, so 2 outputs, each is a dot product of length 2, so rank 2)
For 1x4 x 4x2: rank = 2 (1x4 * 4x2 = 1x2, two outputs, each dot product of length 4, rank 2)
For 2x4 x 4x2: This is the sum of two 2x2*2x2 products. Known rank is 14? Or can we do better?

Let me use the straightforward block decomposition:

C1_left (2x2) = A11 * B11 + A12 * B21
  Using Strassen: 7 + 7 = 14 terms
  
C1_right (2x2) = A11 * B12 + A12 * B22
  Using Strassen: 7 + 7 = 14 terms

C2_left (1x2) = A21 * B11 + A22 * B21
  A21 is 1x2, B11 is 2x2, product is 1x2: rank 2
  A22 is 1x2, B21 is 2x2, product is 1x2: rank 2
  Total: 4 terms

C2_right (1x2) = A21 * B12 + A22 * B22
  Total: 4 terms

Grand total: 14 + 14 + 4 + 4 = 36

Wait, but that's not right either. The Strassen decomposition for 2x2*2x2 gives 7 rank-1 terms. But when I say "using Strassen on A11*B11", I mean decomposing that single product into 7 rank-1 terms. Then A12*B21 into another 7. So C1_left uses 14 rank-1 terms.

But actually, there might be a way to share terms between A11*B11 and A12*B21. Let me just go with the simple construction first: rank 36.

Let me implement Strassen's algorithm properly.

Strassen for P = X * Y where X, Y are 2x2:
X = [[x00, x01], [x10, x11]], Y = [[y00, y01], [y10, y11]]

m1 = (x00+x11)*(y00+y11)
m2 = (x10+x11)*y00
m3 = x00*(y01-y11)
m4 = x11*(y10-y00)
m5 = (x00+x01)*y11
m6 = (x10-x00)*(y00+y10)
m7 = (x01-x11)*(y10+y11)

P00 = m1 - m4 + m6 + m7
P01 = m3 + m5
P10 = m2 + m4
P11 = m1 - m3 + m5 + m7

For each mi, I need to express it as a linear combination of the input entries, then the output Pij is a linear combination of the mi's.

In tensor decomposition form:
- U maps from the flattened A (12 entries for 3x4) to rank terms
- V maps from the flattened B (16 entries for 4x4) to rank terms  
- W maps from rank terms to flattened C (12 entries for 3x4), with W[k*n+i, r] convention

Let me define indices:
A[i,j] for i in 0..2, j in 0..3
B[j,k] for j in 0..3, k in 0..3
C[i,k] for i in 0..2, k in 0..3

U has shape (n*m, rank) = (12, rank), indexed by (i*m+j) where i is row of A, j is col of A
V has shape (m*p, rank) = (16, rank), indexed by (j*p+k) where j is row of B, k is col of B
W has shape (p*n, rank) = (12, rank), indexed by (k*n+i) per the trace convention W[k*n+i, r]

So for a rank-1 term: C[i,k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

Let me build the rank-36 decomposition:

Terms 0-13: Strassen for A11*B11 (contributes to C[0:2, 0:2])
  X = A11 = [[A[0,0], A[0,1]], [A[1,0], A[1,1]]]
  Y = B11 = [[B[0,0], B[0,1]], [B[1,0], B[1,1]]]
  
  m1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
  m2 = (A[1,0]+A[1,1])*B[0,0]
  m3 = A[0,0]*(B[0,1]-B[1,1])
  m4 = A[1,1]*(B[1,0]-B[0,0])
  m5 = (A[0,0]+A[0,1])*B[1,1]
  m6 = (A[1,0]-A[0,0])*(B[0,0]+B[1,0])
  m7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])

  C[0,0] += m1 - m4 + m6 + m7
  C[0,1] += m3 + m5
  C[1,0] += m2 + m4
  C[1,1] += m1 - m3 + m5 + m7

For each mk, the U part is the coefficient of A entries, V part is coefficient of B entries, W part is the contribution to C entries.

m1: U: A[0,0]=1, A[1,1]=1; V: B[0,0]=1, B[1,1]=1; W: C[0,0]+=1, C[1,1]+=1
m2: U: A[1,0]=1, A[1,1]=1; V: B[0,0]=1; W: C[1,0]+=1
m3: U: A[0,0]=1; V: B[0,1]=1, B[1,1]=-1; W: C[0,1]+=1, C[1,1]+=-1
m4: U: A[1,1]=1; V: B[1,0]=1, B[0,0]=-1; W: C[0,0]+=-1, C[1,0]+=1
m5: U: A[0,0]=1, A[0,1]=1; V: B[1,1]=1; W: C[0,1]+=1, C[1,1]+=1
m6: U: A[1,0]=1, A[0,0]=-1; V: B[0,0]=1, B[1,0]=1; W: C[0,0]+=1
m7: U: A[0,1]=1, A[1,1]=-1; V: B[1,0]=1, B[1,1]=1; W: C[0,0]+=1, C[1,1]+=1

Terms 14-27: Strassen for A12*B21 (contributes to C[0:2, 0:2])
  X = A12 = [[A[0,2], A[0,3]], [A[1,2], A[1,3]]]
  Y = B21 = [[B[2,0], B[2,1]], [B[3,0], B[3,1]]]
  
  Same pattern with appropriate indices.

Terms 28-41: Wait, I said C1_right uses Strassen for A11*B12 and A12*B22.

Let me redo:
C1_left (C[0:2, 0:2]) = A11*B11 + A12*B21: 7+7 = 14 terms
C1_right (C[0:2, 2:4]) = A11*B12 + A12*B22: 7+7 = 14 terms
C2_left (C[2, 0:2]) = A21*B11 + A22*B21: 2+2 = 4 terms
C2_right (C[2, 2:4]) = A21*B12 + A22*B22: 2+2 = 4 terms

Total: 36 terms.

For the 1x2 * 2x2 products:
A21*B11 where A21=[A[2,0], A[2,1]], B11=[[B[0,0],B[0,1]],[B[1,0],B[1,1]]]
Result: [A[2,0]*B[0,0]+A[2,1]*B[1,0], A[2,0]*B[0,1]+A[2,1]*B[1,1]]
This is 2 rank-1 terms (each output is a dot product of length 2).

Term for C[2,0]: U: A[2,0]=1, A[2,1]=1; V: B[0,0]=1, B[1,0]=1; W: C[2,0]+=1
Wait no, that's not rank-1. A rank-1 term means U has one nonzero per row-block... 

Actually in the tensor decomposition, each rank-1 term has U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]. For a single rank-1 term, U, V, W each have a specific pattern.

For the product of a 1x2 row with a 2x2 matrix:
Result[0] = a[0]*b[0,0] + a[1]*b[1,0]
Result[1] = a[0]*b[0,1] + a[1]*b[1,1]

This requires 2 rank-1 terms:
Term 1: contributes a[0]*b[0,0] to Result[0] and a[0]*b[0,1] to Result[1]
  U: A[2,0]=1 (i=2, j=0)
  V: B[0,0]=1 (j=0,k=0), B[0,1]=1 (j=0,k=1)
  W: C[2,0]=1 (k=0,i=2), C[2,1]=1 (k=1,i=2)

Wait, that's not right either. In a rank-1 term, for fixed r:
C[i,k] gets contribution U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for all i,j,k.

So U[2*4+0, r] * V[0*4+k, r] * W[k*3+2, r] for k=0,1 gives the first term's contribution.

If U[8, r] = 1 (i=2, j=0), V[0, r] = 1 (j=0, k=0), V[1, r] = 1 (j=0, k=1), W[2, r] = 1 (k=0, i=2), W[5, r] = 1 (k=1, i=2):
C[2,0] += U[8,r]*V[0,r]*W[2,r] = 1*1*1 = 1
C[2,1] += U[8,r]*V[1,r]*W[5,r] = 1*1*1 = 1

But we want C[2,0] += a[2,0]*b[0,0] and C[2,1] += a[2,0]*b[0,1]. So we need the U and V to carry the coefficients of a and b.

Hmm, I think I'm overcomplicating this. Let me use the standard approach:

For a term representing a[2,0]*b[0,0] contributing to C[2,0]:
U[2*4+0, r] = 1, V[0*4+0, r] = 1, W[0*3+2, r] = 1, all others 0 for this r.

For a[2,0]*b[0,1] contributing to C[2,1]:
U[2*4+0, r] = 1, V[0*4+1, r] = 1, W[1*3+2, r] = 1

These are separate rank-1 terms. So A21*B11 (1x2 * 2x2 = 1x2) needs 4 rank-1 terms (2 outputs, each is a dot product of length 2, so 2+2=4).

Wait no. A dot product of length 2 is 2 rank-1 terms. So 1x2 * 2x2: the result is 1x2, so 2 outputs. Each output is a sum of 2 products. So total 4 rank-1 terms.

Hmm, that gives:
C2_left: 4 terms
C2_right: 4 terms
C1_left: 14 terms  
C1_right: 14 terms
Total: 36 terms.

OK let me just code this up carefully.
"""

import time

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 4, 4
    
    # Total rank: 14 + 14 + 4 + 4 = 36
    rank = 36
    
    U = [[0.0] * rank for _ in range(n * m)]  # 12 x 36
    V = [[0.0] * rank for _ in range(m * p)]  # 16 x 36
    W = [[0.0] * rank for _ in range(p * n)]  # 12 x 36
    
    def add_strassen(a_row_offset, b_row_offset, c_row_offset, b_col_offset, c_col_offset, start_term):
        """
        Add Strassen terms for (2x2 A block at rows a_row_offset..a_row_offset+1, cols c_col_offset..c_col_offset+1)
        times (2x2 B block at rows b_row_offset..b_row_offset+1, cols c_col_offset..c_col_offset+1)
        contributing to C at rows c_row_offset..c_row_offset+1, cols c_col_offset..c_col_offset+1.
        
        A block: A[a_row_offset:a_row_offset+2, c_col_offset:c_col_offset+2]
        B block: B[b_row_offset:b_row_offset+2, c_col_offset:c_col_offset+2]
        C block: C[c_row_offset:c_row_offset+2, c_col_offset:c_col_offset+2]
        """
        # A indices: i in {a_row_offset, a_row_offset+1}, j in {c_col_offset, c_col_offset+1}
        # B indices: j in {b_row_offset, b_row_offset+1}, k in {c_col_offset, c_col_offset+1}
        # C indices: i in {c_row_offset, c_row_offset+1}, k in {c_col_offset, c_col_offset+1}
        
        ar0 = a_row_offset
        ar1 = a_row_offset + 1
        ac0 = c_col_offset
        ac1 = c_col_offset + 1
        br0 = b_row_offset
        br1 = b_row_offset + 1
        bc0 = c_col_offset
        bc1 = c_col_offset + 1
        cr0 = c_row_offset
        cr1 = c_row_offset + 1
        cc0 = c_col_offset
        cc1 = c_col_offset + 1
        
        # m1 = (A[ar0,ac0]+A[ar1,ac1])*(B[br0,bc0]+B[br1,bc1])
        #   -> C[cr0,cc0] += 1, C[cr1,cc1] += 1
        t = start_term
        U[ar0*m+ac0][t] += 1
        U[ar1*m+ac1][t] += 1
        V[br0*p+bc0][t] += 1
        V[br1*p+bc1][t] += 1
        W[cc0*n+cr0][t] += 1
        W[cc1*n+cr1][t] += 1
        
        # m2 = (A[ar1,ac0]+A[ar1,ac1])*B[br0,bc0]
        #   -> C[cr1,cc0] += 1
        t = start_term + 1
        U[ar1*m+ac0][t] += 1
        U[ar1*m+ac1][t] += 1
        V[br0*p+bc0][t] += 1
        W[cc0*n+cr1][t] += 1
        
        # m3 = A[ar0,ac0]*(B[br0,bc1]-B[br1,bc1])
        #   -> C[cr0,cc1] += 1, C[cr1,cc1] -= 1
        t = start_term + 2
        U[ar0*m+ac0][t] += 1
        V[br0*p+bc1][t] += 1
        V[br1*p+bc1][t] += -1
        W[cc1*n+cr0][t] += 1
        W[cc1*n+cr1][t] += -1
        
        # m4 = A[ar1,ac1]*(B[br1,bc0]-B[br0,bc0])
        #   -> C[cr0,cc0] -= 1, C[cr1,cc0] += 1
        t = start_term + 3
        U[ar1*m+ac1][t] += 1
        V[br1*p+bc0][t] += 1
        V[br0*p+bc0][t] += -1
        W[cc0*n+cr0][t] += -1
        W[cc0*n+cr1][t] += 1
        
        # m5 = (A[ar0,ac0]+A[ar0,ac1])*B[br1,bc1]
        #   -> C[cr0,cc1] += 1, C[cr1,cc1] += 1
        t = start_term + 4
        U[ar0*m+ac0][t] += 1
        U[ar0*m+ac1][t] += 1
        V[br1*p+bc1][t] += 1
        W[cc1*n+cr0][t] += 1
        W[cc1*n+cr1][t] += 1
        
        # m6 = (A[ar1,ac0]-A[ar0,ac0])*(B[br0,bc0]+B[br1,bc0])
        #   -> C[cr0,cc0] += 1
        t = start_term + 5
        U[ar1*m+ac0][t] += 1
        U[ar0*m+ac0][t] += -1
        V[br0*p+bc0][t] += 1
        V[br1*p+bc0][t] += 1
        W[cc0*n+cr0][t] += 1
        
        # m7 = (A[ar0,ac1]-A[ar1,ac1])*(B[br1,bc0]+B[br1,bc1])
        #   -> C[cr0,cc0] += 1, C[cr1,cc1] += 1
        t = start_term + 6
        U[ar0*m+ac1][t] += 1
        U[ar1*m+ac1][t] += -1
        V[br1*p+bc0][t] += 1
        V[br1*p+bc1][t] += 1
        W[cc0*n+cr0][t] += 1
        W[cc1*n+cr1][t] += 1
        
        return start_term + 7
    
    def add_row2x2(a_row, b_row0, b_col0, c_row, c_col0, start_term):
        """
        Add terms for A[a_row, c_col0:c_col0+2] * B[b_row0:b_row0+2, c_col0:c_col0+2]
        A is 1x2 at row a_row, cols c_col0, c_col0+1
        B is 2x2 at rows b_row0, b_row0+1, cols c_col0, c_col0+1
        C is 1x2 at row c_row, cols c_col0, c_col0+1
        
        Result[0] = A[a_row,c_col0]*B[b_row0,c_col0] + A[a_row,c_col0+1]*B[b_row0+1,c_col0]
        Result[1] = A[a_row,c_col0]*B[b_row0,c_col0+1] + A[a_row,c_col0+1]*B[b_row0+1,c_col0+1]
        
        4 rank-1 terms.
        """
        ac0 = c_col0
        ac1 = c_col0 + 1
        br0 = b_row0
        br1 = b_row0 + 1
        bc0 = c_col0
        bc1 = c_col0 + 1
        cr = c_row
        cc0 = c_col0
        cc1 = c_col0 + 1
        
        # Term 1: A[a_row, ac0] * B[br0, bc0] -> C[cr, cc0]
        t = start_term
        U[cr*m+ac0][t] += 1
        V[br0*p+bc0][t] += 1
        W[cc0*n+cr][t] += 1
        
        # Term 2: A[a_row, ac0] * B[br0, bc1] -> C[cr, cc1]
        t = start_term + 1
        U[cr*m+ac0][t] += 1
        V[br0*p+bc1][t] += 1
        W[cc1*n+cr][t] += 1
        
        # Term 3: A[a_row, ac1] * B[br1, bc0] -> C[cr, cc0]
        t = start_term + 2
        U[cr*m+ac1][t] += 1
        V[br1*p+bc0][t] += 1
        W[cc0*n+cr][t] += 1
        
        # Term 4: A[a_row, ac1] * B[br1, bc1] -> C[cr, cc1]
        t = start_term + 3
        U[cr*m+ac1][t] += 1
        V[br1*p+bc1][t] += 1
        W[cc1*n+cr][t] += 1
        
        return start_term + 4
    
    # C[0:2, 0:2] = A[0:2, 0:2]*B[0:2, 0:2] + A[0:2, 2:4]*B[2:4, 0:2]
    # Strassen for A11*B11: A rows 0-1, cols 0-1; B rows 0-1, cols 0-1; C rows 0-1, cols 0-1
    next_t = add_strassen(a_row_offset=0, b_row_offset=0, c_row_offset=0, b_col_offset=0, c_col_offset=0, start_term=0)
    
    # Strassen for A12*B21: A rows 0-1, cols 2-3; B rows 2-3, cols 0-1; C rows 0-1, cols 0-1
    next_t = add_strassen(a_row_offset=0, b_row_offset=2, c_row_offset=0, b_col_offset=0, c_col_offset=2, start_term=next_t)
    
    # C[0:2, 2:4] = A[0:2, 0:2]*B[0:2, 2:4] + A[0:2, 2:4]*B[2:4, 2:4]
    # Strassen for A11*B12: A rows 0-1, cols 0-1; B rows 0-1, cols 2-3; C rows 0-1, cols 2-3
    next_t = add_strassen(a_row_offset=0, b_row_offset=0, c_row_offset=0, b_col_offset=2, c_col_offset=0, start_term=next_t)
    
    # Strassen for A12*B22: A rows 0-1, cols 2-3; B rows 2-3, cols 2-3; C rows 0-1, cols 2-3
    next_t = add_strassen(a_row_offset=0, b_row_offset=2, c_row_offset=0, b_col_offset=2, c_col_offset=2, start_term=next_t)
    
    # C[2, 0:2] = A[2, 0:2]*B[0:2, 0:2] + A[2, 2:4]*B[2:4, 0:2]
    # A21*B11: A row 2, cols 0-1; B rows 0-1, cols 0-1; C row 2, cols 0-1
    next_t = add_row2x2(a_row=2, b_row0=0, b_col0=0, c_row=2, c_col0=0, start_term=next_t)
    
    # A22*B21: A row 2, cols 2-3; B rows 2-3, cols 0-1; C row 2, cols 0-1
    next_t = add_row2x2(a_row=2, b_row0=2, b_col0=0, c_row=2, c_col0=2, start_term=next_t)
    
    # C[2, 2:4] = A[2, 0:2]*B[0:2, 2:4] + A[2, 2:4]*B[2:4, 2:4]
    # A21*B12: A row 2, cols 0-1; B rows 0-1, cols 2-3; C row 2, cols 2-3
    next_t = add_row2x2(a_row=2, b_row0=0, b_col0=2, c_row=2, c_col0=0, start_term=next_t)
    
    # A22*B22: A row 2, cols 2-3; B rows 2-3, cols 2-3; C row 2, cols 2-3
    next_t = add_row2x2(a_row=2, b_row0=2, b_col0=2, c_row=2, c_col0=2, start_term=next_t)
    
    assert next_t == rank, f"Expected {rank} terms, got {next_t}"
    
    return {"U": U, "V": V, "W": W}
