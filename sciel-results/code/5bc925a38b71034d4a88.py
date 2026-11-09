"""Combined approach: block-based 5x5x5 matrix multiplication with Strassen on 2x2 sub-block."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

_T0 = time.time()
_LIMIT = 150.0
_MARGIN = 5.0

def _deadline_exceeded():
    return (time.time() - _T0) > (_LIMIT - _MARGIN)

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-reduced exact tensor decomposition for 5x5x5 matrix multiplication.
    
    Uses a 3+2 block partition. For the 2x2x2 diagonal block (A22*B22),
    uses Strassen's rank-7 decomposition instead of schoolbook rank 8.
    
    Block structure (indices 0-2 are "top", 3-4 are "bottom"):
      A = [[A11 (3x3), A12 (3x2)],
           [A21 (2x3), A22 (2x2)]]
      B = [[B11 (3x3), B12 (3x2)],
           [B21 (2x3), B22 (2x2)]]
      C = A*B = [[C11 (3x3), C12 (3x2)],
                 [C21 (2x3), C22 (2x2)]]
    
    Rank contributions:
      A11*B11: 3*3*3 = 27
      A12*B21: 3*2*3 = 18
      A11*B12: 3*3*2 = 18
      A12*B22: 3*2*2 = 12
      A21*B11: 2*3*3 = 18
      A22*B21: 2*2*3 = 12
      A21*B12: 2*3*2 = 12
      A22*B22: 7 (Strassen)
      Total: 27+18+18+12+18+12+12+7 = 124
    """
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # We'll build the decomposition column by column
    # Each column c has: U[:,c], V[:,c], W[:,c]
    # U has shape (n*m, rank) = (25, rank)
    # V has shape (m*p, rank) = (25, rank)  
    # W has shape (p*n, rank) = (25, rank)
    # Trace convention: W[k*n + i, c]
    
    # Collect all (U_entry, V_entry, W_entry) tuples
    # U_entry: (row_idx, value) where row_idx = i*m + j
    # V_entry: (row_idx, value) where row_idx = j*p + k
    # W_entry: (row_idx, value) where row_idx = k*n + i
    
    columns = []  # list of (U_dict, V_dict, W_dict)
    
    def add_column(u_entries, v_entries, w_entries):
        """Add a rank-1 column. Each is a list of (index, value)."""
        u_d = {}
        for idx, val in u_entries:
            u_d[idx] = u_d.get(idx, 0) + val
        v_d = {}
        for idx, val in v_entries:
            v_d[idx] = v_d.get(idx, 0) + val
        w_d = {}
        for idx, val in w_entries:
            w_d[idx] = w_d.get(idx, 0) + val
        columns.append((u_d, v_d, w_d))
    
    # Block indices
    # Top block: rows/cols 0,1,2
    # Bottom block: rows/cols 3,4
    top = [0, 1, 2]
    bot = [3, 4]
    
    # Schoolbook for a block product A[i1:i2, j1:j2] * B[j1:j2, k1:k2] -> C[i1:i2, k1:k2]
    # For each (i, j, k) in the block, add a column with U[i*m+j]=1, V[j*p+k]=1, W[k*n+i]=1
    def schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_column(
                        [(i * m + j, 1.0)],
                        [(j * p + k, 1.0)],
                        [(k * n + i, 1.0)]
                    )
    
    # Strassen for 2x2x2 block: A22 (rows 3,4; cols 3,4) * B22 (rows 3,4; cols 3,4) -> C22 (rows 3,4; cols 3,4)
    # A22 = [[a00, a01], [a10, a11]] where a00=A[3,3], a01=A[3,4], a10=A[4,3], a11=A[4,4]
    # B22 = [[b00, b01], [b10, b11]] where b00=B[3,3], b01=B[3,4], b10=B[4,3], b11=B[4,4]
    # C22 = [[c00, c01], [c10, c11]] where c00=C[3,3], c01=C[3,4], c10=C[4,3], c11=C[4,4]
    
    # Strassen:
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1+M4-M5+M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M6
    
    # For each Strassen term, we need to express it as a sum of rank-1 tensors.
    # Each term like (a00+a11)(b00+b11) expands to 4 products:
    # a00*b00 + a00*b11 + a11*b00 + a11*b11
    # Each product a_ij * b_kl contributes to c_kl... wait, let me be careful.
    
    # A22[i,j] * B22[j,k] -> C22[i,k]
    # A22 indices: i in {0,1} -> row 3+i, j in {0,1} -> col 3+j
    # B22 indices: j in {0,1} -> row 3+j, k in {0,1} -> col 3+k
    # C22 indices: i in {0,1} -> row 3+i, k in {0,1} -> col 3+k
    
    # So A22[i,j] = A[3+i, 3+j], B22[j,k] = B[3+j, 3+k], C22[i,k] = C[3+i, 3+k]
    
    # For a term a_ij * b_jk, the U entry is at (3+i)*m + (3+j), V at (3+j)*p + (3+k), W at (3+k)*n + (3+i)
    
    def strassen_2x2x2():
        """Add 7 columns for the 2x2x2 Strassen decomposition."""
        # a00=A[3,3], a01=A[3,4], a10=A[4,3], a11=A[4,4]
        # b00=B[3,3], b01=B[3,4], b10=B[4,3], b11=B[4,4]
        # c00=C[3,3], c01=C[3,4], c10=C[4,3], c11=C[4,4]
        
        # M1 = (a00+a11)(b00+b11)
        # = a00*b00 + a00*b11 + a11*b00 + a11*b11
        # Each term a_ij*b_jk: U at (3+i)*5+(3+j), V at (3+j)*5+(3+k), W at (3+k)*5+(3+i)
        m1_terms = [
            # a00*b00: i=0,j=0,k=0
            [((3+0)*m + (3+0), 1.0)],  # U
            [((3+0)*p + (3+0), 1.0)],  # V
            [((3+0)*n + (3+0), 1.0)],  # W
            # a00*b11: i=0,j=0,k=1 (b11 is row 1, col 1 of B22, so j=0 for A col, k=1 for B col? No.)
            # Correction: a_ij * b_jk. 
            # a00: i=0, j=0. b00: j=0, k=0. -> term a00*b00
            # a00: i=0, j=0. b11: j=1, k=1. -> term a00*b11 is INVALID in standard matrix mult unless we are doing something else.
            # Strassen M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11.
            # In standard matrix mult C = AB, c_ik = sum_j a_ij b_jk.
            # The term a00*b11 does NOT appear in standard c_ik for any i,k directly as a single product.
            # However, Strassen computes intermediates.
            # Let's look at the indices again.
            # a00 is A[3,3], a11 is A[4,4].
            # b00 is B[3,3], b11 is B[4,4].
            # M1 = (A[3,3]+A[4,4])(B[3,3]+B[4,4]) = A[3,3]B[3,3] + A[3,3]B[4,4] + A[4,4]B[3,3] + A[4,4]B[4,4].
            # The program seems to be setting up a tensor network or similar where we list terms.
            # The previous code had a list structure `m1_terms = [ [U], [V], [W] ]` for the first term?
            # No, the list `m1_terms` seems to contain entries for each term in the expansion.
            # The error is an unclosed bracket. The list `m1_terms = [` was opened but the content was cut off or malformed.
            # I will close the list properly with a placeholder or complete the first term if that's what was intended, but the prompt says "fix only this error" which is the parse error.
            # The simplest fix to make it parsable is to close the list.
        ]
            # But a00 has j=0, and b11 has j=1 -- these don't match!
            # 
            # Wait, I'm confusing myself. Let me redo this.
            # In Strassen, M1 = (a00+a11)(b00+b11)
            # This is a scalar product of sums. It equals:
            # a00*b00 + a00*b11 + a11*b00 + a11*b11
            # But these are NOT valid matrix multiplication terms because the indices don't align!
            #
            # The issue: Strassen works by creating intermediate products that are NOT 
            # directly matrix multiplication terms. The trick is that when you combine
            # the M's with coefficients, the spurious terms cancel.
            #
            # So each M is a product of a linear combination of A entries and a linear 
            # combination of B entries. To decompose into rank-1 tensors, I need to expand.
            #
            # M1 = (a00+a11)(b00+b11) = a00*b00 + a00*b11 + a11*b00 + a11*b11
            # But a00*b11 is NOT a valid (i,j,k) triple for matrix mult because a00 has j=0 
            # and b11 has j=1. However, in the tensor decomposition, we can still have 
            # U[i*m+j], V[j'*p+k], W[k'*n+i] as long as the product U*V*W gives the right 
            # tensor entries. The key is that the SUM over all columns must give the correct 
            # tensor.
            #
            # Actually, I think I'm overcomplicating this. Let me think differently.
            #
            # The tensor T[i,j,k] for 2x2x2 matrix mult is: T[i,j,k] = 1 if the triple (i,j,k)
            # is a valid index triple, i.e., always 1 for all i,j,k in {0,1}^3. Wait no --
            # for matrix multiplication A*B=C where A is 2x2, B is 2x2, C is 2x2:
            # C[i,k] = sum_j A[i,j]*B[j,k]
            # The tensor is T[i,j,k] = 1 for all i,j,k in {0,1}^3 (it's the full 8-dimensional space).
            # No wait, that's not right either. The tensor has entries T[i,j,k] where
            # i indexes rows of A/C, j indexes cols of A / rows of B, k indexes cols of B/C.
            # T[i,j,k] = 1 for all valid (i,j,k). So all 8 entries are 1.
            #
            # Strassen says this 2x2x2 tensor (all ones in 8 dimensions) can be written as 
            # sum of 7 rank-1 tensors. Each rank-1 tensor is u_i * v_j * w_k for some 
            # vectors u, v, w.
            #
            # The standard Strassen decomposition:
            # Let's use the convention where the tensor T[i,j,k] = 1 for all i,j,k in {0,1}^3.
            # 
            # Actually no! The matrix multiplication tensor is NOT all ones. 
            # T[i,j,k] = 1 iff the triple (i,j,k) corresponds to a term in the matrix product.
            # For 2x2x2: C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for ALL i,j,k in {0,1}^3.
            # Yes, all 8 entries are 1.
            #
            # Strassen's 7 rank-1 decompositions of the all-ones 2x2x2 tensor:
            # Using the standard construction:
            # 
            # r1: u=(1,1), v=(1,1), w=(1,1) -> contributes 1 to all 8 entries... no that's not rank 1 in the right sense.
            
            # Let me just use the explicit Strassen formulas and expand them.
            # 
            # The 7 products are:
            # M1 = (A00+A11)(B00+B11)
            # M2 = (A10+A11)B00
            # M3 = A00(B01-B11)
            # M4 = A11(B10-B00)
            # M5 = (A00+A01)B11
            # M6 = (A10-A00)(B00+B01)
            # M7 = (A01-A11)(B10+B11)
            #
            # Where Ai0 = A[i,0], Ai1 = A[i,1], Bj0 = B[j,0], Bj1 = B[j,1]
            #
            # And:
            # C00 = M1+M4-M5+M7
            # C01 = M3+M5
            # C10 = M2+M4
            # C11 = M1-M2+M3+M6
            #
            # Now, each M is a product of a linear form in A and a linear form in B.
            # To write as rank-1 tensors, I expand each M into its constituent products.
            #
            # M1 = (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
            # Each term A[i,j]*B[j',k] where j and j' may differ.
            # As a tensor contribution: the term A[i,j]*B[j',k] contributes to T[i,j,j',k]... 
            # but our tensor is 3-indexed T[i,j,k] where the middle index is shared.
            #
            # I think the correct way to view this: each "product" A[i,j]*B[j',k] where j≠j' 
            # is NOT a valid rank-1 tensor by itself. But when combined with the right 
            # coefficients, the sum works out.
            #
            # Actually, I think the cleaner way: Strassen gives 7 rank-1 decompositions 
            # of the 2x2x2 matrix multiplication tensor. Let me look up the explicit 
            # rank-1 vectors.
            #
            # The 2x2x2 matrix multiplication tensor T has T[i,j,k]=1 for all i,j,k in {0,1}^3.
            # A rank-1 decomposition is T[i,j,k] = sum_{r=1}^{7} u_r[i] * v_r[j] * w_r[k]
            #
            # One known decomposition (from Strassen):
            # r=1: u=(1,1), v=(1,1), w=(1,1)
            # r=2: u=(0,1), v=(1,0), w=(1,0)
            # r=3: u=(1,0), v=(0,1), w=(0,-1)
            # r=4: u=(0,1), v=(0,1), w=(1,-1)
            # r=5: u=(1,1), v=(0,1), w=(0,1)
            # r=6: u=(1,-1), v=(1,1), w=(1,0)
            # r=7: u=(1,-1), v=(0,1), w=(1,1)
            #
            # Let me verify r=1: u[0]*v[0]*w[0]=1*1*1=1, u[0]*v[0]*w[1]=1*1*1=1, etc. All 8 entries get +1.
            # r=2: u=(0,1),v=(1,0),w=(1,0): only u[1]=1,v[0]=1,w[0]=1 -> T[1,0,0]+=1
            # r=3: u=(1,0),v=(0,1),w=(0,-1): only u[0]=1,v[1]=1,w[1]=-1 -> T[0,1,1]+=-1
            # r=4: u=(0,1),v=(0,1),w=(1,-1): u[1]=1,v[1]=1,w[0]=1 -> T[1,1,0]+=1; u[1]=1,v[1]=1,w[1]=-1 -> T[1,1,1]+=-1
            # r=5: u=(1,1),v=(0,1),w=(0,1): u[0]=1,v[1]=1,w[1]=1 -> T[0,1,1]+=1; u[1]=1,v[1]=1,w[1]=1 -> T[1,1,1]+=1
            # r=6: u=(1,-1),v=(1,1),w=(1,0): 
            #   u[0]=1,v[0]=1,w[0]=1 -> T[0,0,0]+=1
            #   u[0]=1,v[1]=1,w[0]=1 -> T[0,1,0]+=1
            #   u[1]=-1,v[0]=1,w[0]=1 -> T[1,0,0]+=-1
            #   u[1]=-1,v[1]=1,w[0]=1 -> T[1,1,0]+=-1
            # r=7: u=(1,-1),v=(0,1),w=(1,1):
            #   u[0]=1,v[1]=1,w[0]=1 -> T[0,1,0]+=1
            #   u[0]=1,v[1]=1,w[1]=1 -> T[0,1,1]+=1
            #   u[1]=-1,v[1]=1,w[0]=1 -> T[1,1,0]+=-1
            #   u[1]=-1,v[1]=1,w[1]=1 -> T[1,1,1]+=-1
            #
            # Let me sum all contributions to each T[i,j,k]:
            # T[0,0,0]: r1=1, r6=1 => 2? That's wrong, should be 1.
            #
            # Hmm, my decomposition is wrong. Let me try a different known decomposition.
            #
            # Actually, I recall that the standard way to express Strassen is different.
            # Let me just directly expand the Strassen formulas into rank-1 tensor terms.
            #
            # Key insight: Each M_r is a product (linear_form_in_A)(linear_form_in_B).
            # If linear_form_in_A = sum_i sum_j a_ij * A[i,j] and linear_form_in_B = sum_j sum_k b_jk * B[j,k],
            # then M_r = sum_{i,j,j',k} a_ij * b_jk * A[i,j]*B[j',k]
            #
            # But A[i,j]*B[j',k] is only a valid "tensor entry" when j=j'. When j≠j', 
            # it's a "cross term" that doesn't correspond to any T[i,j,k] entry.
            #
            # The magic of Strassen is that these cross terms cancel when you compute 
            # C[i,k] = sum_r coeff_{r,ik} * M_r.
            #
            # For the tensor decomposition, I need: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
            # where U_r[i,j] is the coefficient of A[i,j] in the r-th A-linear-form,
            # V_r[j,k] is the coefficient of B[j,k] in the r-th B-linear-form,
            # W_r[i,k] is the coefficient in C[i,k] from M_r.
            #
            # Wait, that's not quite right either. The tensor decomposition is:
            # T[i,j,k] = sum_r u_r[i] * v_r[j] * w_r[k]
            # where u_r is a vector of length n*m (or rather, indexed by (i,j) pairs),
            # v_r indexed by (j,k) pairs, w_r indexed by (i,k) pairs.
            #
            # Actually for matrix multiplication, the natural tensor is:
            # T[i,j,k] = 1 for all valid (i,j,k)
            # And a rank-1 decomposition is: T[i,j,k] = sum_r U[r][i,j] * V[r][j,k] * W[r][i,k]
            # where U[r] is a matrix (n x m), V[r] is a matrix (m x p), W[r] is a matrix (p x n)
            # with the trace convention W[k*n+i].
            #
            # For Strassen, each M_r corresponds to one rank-1 term where:
            # U_r[i,j] = coefficient of A[i,j] in the A-part of M_r
            # V_r[j,k] = coefficient of B[j,k] in the B-part of M_r  
            # W_r[i,k] = coefficient of C[i,k] in the C-part (how M_r contributes to C)
            #
            # But wait, that's not how it works. In a rank-1 tensor decomposition:
            # T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
            #
            # For the full matrix multiplication tensor, T[i,j,k]=1 for all i,j,k.
            # So we need: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k.
            #
            # In Strassen, we have 7 products M_r. Each M_r is used to compute multiple C[i,k] entries.
            # The relationship is: C[i,k] = sum_r c_{r,ik} * M_r
            # where c_{r,ik} is the coefficient.
            #
            # And M_r = (sum_{i',j'} a_{r,i'j'} A[i',j']) * (sum_{j',k'} b_{r,j'k'} B[j',k'])
            #
            # So C[i,k] = sum_r c_{r,ik} * sum_{i',j'} a_{r,i'j'} A[i',j'] * sum_{j',k'} b_{r,j'k'} B[j',k']
            #
            # = sum_{i',j',j',k'} (sum_r c_{r,ik} * a_{r,i'j'} * b_{r,j'k'}) A[i',j'] B[j',k']
            #
            # For this to equal sum_j A[i,j] B[j,k], we need:
            # sum_r c_{r,ik} * a_{r,i'j'} * b_{r,j'k'} = delta_{i',i} * delta_{j',j} * delta_{j',k'}... 
            # 
            # This is getting complicated. Let me just directly construct the rank-1 vectors.
            #
            # For each Strassen term r, define:
            # U_r[i,j] = a_{r,i,j} (coefficient of A[i,j] in M_r's A-part)
            # V_r[j,k] = b_{r,j,k} (coefficient of B[j,k] in M_r's B-part)
            # W_r[i,k] = c_{r,i,k} (coefficient of M_r in C[i,k])
            #
            # Then the decomposition is: T[i,j,k] = sum_r U_r[i,j] * V_r[j,k] * W_r[i,k]
            #
            # Wait, but T[i,j,k] should be 1 for all i,j,k. Let's check:
            # sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] should equal 1 for all i,j,k.
            #
            # Hmm, but that's not quite the right tensor. The matrix multiplication tensor is:
            # T[i,j,k] = 1 for all i,j,k (for the 2x2x2 case, all 8 triples are valid).
            #
            # So yes, we need sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] = 1 for all i,j,k in {0,1}^3.
            #
            # Let me define the 7 Strassen terms explicitly:
            #
            # M1 = (A00+A11)(B00+B11)
            #   A-part: A00 coeff=1, A11 coeff=1, others 0
            #   B-part: B00 coeff=1, B11 coeff=1, others 0
            #   C-part: C00 += 1, C11 += 1 (from C00=M1+M4-M5+M7, C11=M1-M2+M3+M6)
            #
            # M2 = (A10+A11)B00
            #   A-part: A10 coeff=1, A11 coeff=1, others 0
            #   B-part: B00 coeff=1, others 0
            #   C-part: C10 += 1, C11 -= 1
            #
            # M3 = A00(B01-B11)
            #   A-part: A00 coeff=1, others 0
            #   B-part: B01 coeff=1, B11 coeff=-1, others 0
            #   C-part: C01 += 1, C11 += 1
            #
            # M4 = A11(B10-B00)
            #   A-part: A11 coeff=1, others 0
            #   B-part: B10 coeff=1, B00 coeff=-1, others 0
            #   C-part: C00 += 1, C10 += 1
            #
            # M5 = (A00+A01)B11
            #   A-part: A00 coeff=1, A01 coeff=1, others 0
            #   B-part: B11 coeff=1, others 0
            #   C-part: C01 += 1, C00 -= 1
            #
            # M6 = (A10-A00)(B00+B01)
            #   A-part: A10 coeff=1, A00 coeff=-1, others 0
            #   B-part: B00 coeff=1, B01 coeff=1, others 0
            #   C-part: C11 += 1
            #
            # M7 = (A01-A11)(B10+B11)
            #   A-part: A01 coeff=1, A11 coeff=-1, others 0
            #   B-part: B10 coeff=1, B11 coeff=1, others 0
            #   C-part: C00 += 1
            #
            # Now let me verify: sum_r U_r[i,j] * V_r[j,k] * W_r[i,k] should = 1 for all i,j,k.
            #
            # Let's check T[0,0,0]: need sum_r U_r[0,0]*V_r[0,0]*W_r[0,0] = 1
            # r=1: U=1, V=1, W=1 (C00 coeff for M1 is 1) -> 1
            # r=2: U=0 (A00 not in M2) -> 0
            # r=3: U=1 (A00 in M3), V=0 (B00 not in M3's B-part: B01,B11) -> 0
            # r=4: U=0 (A00 not in M4) -> 0
            # r=5: U=1 (A00 in M5), V=0 (B00 not in M5: only B11) -> 0
            # r=6: U=-1 (A00 coeff in M6 is -1), V=1 (B00 in M6), W=0 (M6 doesn't contribute to C00) -> 0
            # r=7: U=0 (A00 not in M7) -> 0
            # Total: 1. Correct!
            #
            # T[0,0,1]: need sum_r U_r[0,0]*V_r[0,1]*W_r[0,1] = 1
            # r=1: U=1, V=0 (B01 not in M1: B00,B11) -> 0
            # r=2: U=0 -> 0
            # r=3: U=1, V=1 (B01 in M3), W=1 (M3 contributes +1 to C01) -> 1
            # r=4: U=0 -> 0
            # r=5: U=1, V=0 (B01 not in M5: only B11) -> 0
            # r=6: U=-1, V=1 (B01 in M6), W=0 (M6 doesn't contribute to C01) -> 0
            # r=7: U=0 -> 0
            # Total: 1. Correct!
            #
            # T[0,1,0]: need sum_r U_r[0,1]*V_r[1,0]*W_r[0,0] = 1
            # r=1: U=0 (A01 not in M1) -> 0
            # r=2: U=0 -> 0
            # r=3: U=0 -> 0
            # r=4: U=0 -> 0
            # r=5: U=1 (A01 in M5), V=0 (B10 not in M5: only B11) -> 0
            # r=6: U=0 (A01 not in M6) -> 0
            # r=7: U=1 (A01 in M7), V=1 (B10 in M7), W=1 (M7 contributes +1 to C00) -> 1
            # Total: 1. Correct!
            #
            # T[0,1,1]: need sum_r U_r[0,1]*V_r[1,1]*W_r[0,1] = 1
            # r=1: U=0 -> 0
            # r=2: U=0 -> 0
            # r=3: U=0 -> 0
            # r=4: U=0 -> 0
            # r=5: U=1 (A01 in M5), V=1 (B11 in M5), W=1 (M5 contributes +1 to C01) -> 1
            # r=6: U=0 -> 0
            # r=7: U=1 (A01 in M7), V=1 (B11 in M7), W=0 (M7 doesn't contribute to C01) -> 0
            # Total: 1. Correct!
            #
            # Great, the decomposition works! Now let me map to the global indices.
            #
            # For the 2x2 block at positions (3,4) x (3,4):
            # A[i,j] global: A[3+i, 3+j], so U index = (3+i)*m + (3+j) = (3+i)*5 + (3+j)
            # B[j,k] global: B[3+j, 3+k], so V index = (3+j)*p + (3+k) = (3+j)*5 + (3+k)
            # C[i,k] global: C[3+i, 3+k], so W index = (3+k)*n + (3+i) = (3+k)*5 + (3+i)
            #
            # Now I can construct the 7 columns.
            
        # Define the 7 Strassen terms with their (i,j), (j,k), (i,k) coefficient maps
        # Using local 2x2 indices (0,1)
        
        strassen_terms = [
            # (A_coeffs: {(i,j): coeff}, B_coeffs: {(j,k): coeff}, C_coeffs: {(i,k): coeff})
            # M1
            ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
            # M2
            ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
            # M3
            ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
            # M4
            ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
            # M5
            ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,1): 1, (0,0): -1}),
            # M6
            ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
            # M7
            ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ]
        
        for a_coeffs, b_coeffs, c_coeffs in strassen_terms:
            u_entries = []
            v_entries = []
            w_entries = []
            for (li, lj), a_val in a_coeffs.items():
                gi, gj = 3 + li, 3 + lj
                u_entries.append((gi * m + gj, a_val))
            for (lj, lk), b_val in b_coeffs.items():
                gj, gk = 3 + lj, 3 + lk
                v_entries.append((gj * p + gk, b_val))
            for (li, lk), c_val in c_coeffs.items():
                gi, gk = 3 + li, 3 + lk
                w_entries.append((gk * n + gi, c_val))
            add_column(u_entries, v_entries, w_entries)
    
    # Now build the full decomposition
    # Schoolbook blocks:
    # A11*B11: i in top, j in top, k in top
    schoolbook_block(top, top, top)  # 27 columns
    
    # A12*B21: i in top, j in bot (A's cols), k in top (B's cols)
    # A12 is A[top, bot], B21 is B[bot, top]
    # A[i,j] with i in top, j in bot; B[j,k] with j in bot, k in top
    schoolbook_block(top, bot, top)  # 18 columns
    
    # A11*B12: i in top, j in top, k in bot
    schoolbook_block(top, top, bot)  # 18 columns
    
    # A12*B22: i in top, j in bot, k in bot
    schoolbook_block(top, bot, bot)  # 12 columns
    
    # A21*B11: i in bot, j in top, k in top
    schoolbook_block(bot, top, top)  # 18 columns
    
    # A22*B21: i in bot, j in bot, k in top
    schoolbook_block(bot, bot, top)  # 12 columns
    
    # A21*B12: i in bot, j in top, k in bot
    schoolbook_block(bot, top, bot)  # 12 columns
    
    # A22*B22: Strassen, 7 columns
    strassen_2x2x2()
    
    # Total columns: 27+18+18+12+18+12+12+7 = 124
    rank = len(columns)
    
    # Build the matrices
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u_d, v_d, w_d) in enumerate(columns):
        for idx, val in u_d.items():
            U[idx][c] = val
        for idx, val in v_d.items():
            V[idx][c] = val
        for idx, val in w_d.items():
            W[idx][c] = val
    
    if _deadline_exceeded():
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
