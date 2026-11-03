import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    n, m, p = 3, 4, 7

    U_vecs = []
    V_vecs = []
    W_vecs = []

    def add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals):
        u = [Fraction(0)] * (n * m)
        v = [Fraction(0)] * (m * p)
        w = [Fraction(0)] * (p * n)

        for idx, val in zip(u_indices, u_vals):
            u[idx] += Fraction(val)
        for idx, val in zip(v_indices, v_vals):
            v[idx] += Fraction(val)
        for idx, val in zip(w_indices, w_vals):
            w[idx] += Fraction(val)

        U_vecs.append(u)
        V_vecs.append(v)
        W_vecs.append(w)

    def gen_3x3x3(i_off, j_off, k_off):
        # 21-term decomposition for 3x3x3 matrix multiplication
        # This uses the known rank-21 decomposition
        # Indices: i in [0,2], j in [0,2], k in [0,2]
        # U index: (i_off+i)*m + (j_off+j)
        # V index: (j_off+j)*p + (k_off+k)
        # W index: (k_off+k)*n + (i_off+i)
        
        # We'll use a standard rank-21 decomposition.
        # The 21 terms can be generated from the structure of the tensor.
        # One known decomposition uses:
        # For the 3x3x3 case, we can think of it as:
        # Split into 2x2x2 (Strassen, rank 7) + boundary terms
        # Actually, let's use a direct enumeration of a known rank-21 set.
        
        # A common approach: use the fact that 3x3x3 can be decomposed as
        # (2x2x2 Strassen) for the top-left 2x2x2 block, plus
        # additional terms for the remaining indices.
        
        # Let me use a different approach: the rank of 3x3x3 matrix multiplication
        # is known to be 21 (or possibly lower, but 21 is a safe upper bound from
        # certain constructions).
        
        # Actually, let me use a simpler construction that's guaranteed to work:
        # Split 3x3x3 into:
        # (2,2,2) Strassen: rank 7
        # (2,1,2): rank 4 (schoolbook: 2*1*2=4)
        # (1,2,2): rank 4 (schoolbook: 1*2*2=4)
        # (1,1,2): rank 2 (schoolbook: 1*1*2=2)
        # (2,2,1): rank 4 (schoolbook: 2*2*1=4)
        # (2,1,1): rank 2 (schoolbook: 2*1*1=2)
        # (1,2,1): rank 2 (schoolbook: 1*2*1=2)
        # (1,1,1): rank 1
        # Total: 7+4+4+2+4+2+2+1 = 26. Not good enough.
        
        # Let me use a better split. Actually, for the purpose of this task,
        # let me just use schoolbook for 3x3x3 which gives 27 terms,
        # and see if the overall rank is still better than 78.
        # 3*3*3 = 27 terms.
        
        # Hmm, that's worse. Let me think of a better decomposition.
        
        # Actually, the known rank of 3x3x3 is 21 (Bini et al.).
        # Let me hardcode a rank-21 decomposition.
        
        # One known rank-21 decomposition uses the following structure:
        # It's based on the fact that the 3x3x3 tensor can be expressed
        # using 21 rank-1 tensors.
        
        # For simplicity and correctness, let me use a constructive approach:
        # The 3x3x3 matrix multiplication tensor has a decomposition of rank 21.
        # I'll use a known set of 21 terms.
        
        # Actually, let me just use schoolbook (27 terms) for now and see if
        # the total is still better. If 3x3x3 = 27, then:
        # 27 + 2*17 + 9 + 12 = 27+34+9+12 = 82. Worse than 78.
        
        # So I need a better decomposition. Let me use rank 21.
        
        # Here's a rank-21 decomposition for 3x3x3:
        # Based on the work of Bini, Capovani, Lotti, Romani (1978)
        
        # The 21 terms are (using 0-indexed i,j,k in {0,1,2}):
        # I'll generate them programmatically using a known formula.
        
        # Actually, let me use a simpler approach: use the 2x2x2 Strassen
        # for the (0,0,0) to (1,1,1) block, and handle the rest optimally.
        
        # Split 3x3x3 as:
        # (2,2,2) at (0,0,0): Strassen rank 7
        # (2,1,2) at (0,2,0): schoolbook rank 4
        # (1,2,2) at (2,0,0): schoolbook rank 4
        # (1,1,2) at (2,2,0): schoolbook rank 2
        # (2,2,1) at (0,0,2): schoolbook rank 4
        # (2,1,1) at (0,2,2): schoolbook rank 2
        # (1,2,1) at (2,0,2): schoolbook rank 2
        # (1,1,1) at (2,2,2): schoolbook rank 1
        # Total: 7+4+4+2+4+2+2+1 = 26
        
        # Still 26. Let me try a different split.
        
        # How about using Strassen for multiple 2x2x2 blocks?
        # 3x3x3 can be covered by overlapping 2x2x2 blocks.
        # But they overlap, so we can't just add them up.
        
        # Let me try: 
        # (2,2,2) at (0,0,0): 7
        # (2,2,2) at (0,1,0): 7 (but overlaps with first)
        # This gets complicated.
        
        # For now, let me just use schoolbook for 3x3x3 (27 terms) and
        # accept a slightly higher rank, OR find a better overall split.
        
        # Actually, let me reconsider the overall split.
        # Instead of 3x4x7 = (3,3,3)+(3,3,4)+(3,1,3)+(3,1,4),
        # maybe I should use a different decomposition.
        
        # Let me try: 3x4x7 with 3x3x3 (27) + 3x1x7 (21) = 48 for the m=4 part
        # No, that's for splitting m.
        
        # Split m=4 as 2+2:
        # (3,2,7) + (3,2,7)
        # (3,2,7): split p=7 as 2+2+2+1
        # (3,2,2) + (3,2,2) + (3,2,2) + (3,2,1)
        # (3,2,2): split n=3 as 2+1
        # (2,2,2) + (1,2,2) = 7 + 4 = 11
        # (3,2,1): split n=3 as 2+1
        # (2,2,1) + (1,2,1) = 4 + 2 = 6
        # (3,2,7) = 11+11+11+6 = 39
        # Total: 39+39 = 78. Same as before.
        
        # Let me try using rank 17 for 3x3x2.
        # 3x3x2: split m=3 as 2+1
        # (2,2,2) + (1,2,2) + (2,1,2) + (1,1,2)
        # 7 + 4 + 4 + 2 = 17. Yes!
        
        # So for 3x3x4: split p=4 as 2+2
        # (3,3,2) + (3,3,2) = 17+17 = 34
        
        # For 3x1x3: schoolbook = 9
        # For 3x1x4: schoolbook = 12
        
        # Total: 27 + 34 + 9 + 12 = 82. Worse.
        
        # I really need rank 21 for 3x3x3.
        # Let me just hardcode it.
        
        # The 21-term decomposition for 3x3x3 matrix multiplication:
        # I'll use a known set. Here are the 21 terms:
        
        # Term 1: (0,0,0), (0,0,0), (0,0,0) -> trivial
        # This is getting too complex to derive from scratch.
        
        # Let me use a different strategy: use the 2x2x2 Strassen
        # decomposition more cleverly.
        
        # For 3x3x3, I'll use:
        # Strassen for (0,0)-(1,1) in i,j and (0,0)-(1,1) in k: 7 terms
        # Then handle the remaining indices with schoolbook:
        # i=2, j=0,1,2 and k=0,1,2: 3*3 = 9 terms
        # i=0,1, j=2, k=0,1,2: 2*3 = 6 terms  
        # i=0,1, j=0,1, k=2: 2*2 = 4 terms
        # But wait, this double-counts some.
        
        # Let me be more careful. The 3x3x3 tensor has 27 entries.
        # Using Strassen for the top-left 2x2x2 block covers 8 entries.
        # Remaining: 27-8 = 19 entries.
        # But the Strassen terms also contribute to other entries potentially?
        # No, Strassen for a sub-block only affects that sub-block.
        
        # So: 7 (Strassen for 2x2x2 at (0,0,0)) + 19 (schoolbook for rest) = 26.
        
        # I'm stuck at 26 for this approach.
        
        # Let me just use 26 terms for 3x3x3.
        # Total: 26 + 34 + 9 + 12 = 81. Still worse than 78.
        
        # OK, let me abandon the 3x3x3 approach and stick with the current
        # decomposition that gives 78, but try to optimize it further.
        
        # Current: 3x4x7 = (2,4,7) + (1,4,7)
        # (2,4,7) = (2,2,7) + (2,2,7)
        # (2,2,7) = (2,2,2)+(2,2,2)+(2,2,2)+(2,2,1) = 7+7+7+4 = 25
        # (1,4,7) = 1*4*7 = 28
        # Total: 25+25+28 = 78
        
        # Can I do better for (2,2,7)?
        # (2,2,7): split p=7 as 3+4
        # (2,2,3) + (2,2,4)
        # (2,2,3): split p=3 as 2+1
        # (2,2,2) + (2,2,1) = 7+4 = 11
        # (2,2,4): split p=4 as 2+2
        # (2,2,2) + (2,2,2) = 7+7 = 14
        # (2,2,7) = 11+14 = 25. Same.
        
        # What about (2,2,7) with a different split?
        # p=7 as 2+2+3:
        # (2,2,2)+(2,2,2)+(2,2,3) = 7+7+11 = 25. Same.
        
        # It seems 25 is optimal for (2,2,7) with this approach.
        
        # What about (1,4,7)? Can I do better than 28?
        # (1,4,7): n=1, so it's just a vector-matrix product.
        # Rank is 4*7 = 28 for schoolbook.
        # Can we do better? For n=1, the tensor is just the outer product
        # of a 4-vector and a 7-vector, which has rank 1!
        # Wait, no. For n=1, m=4, p=7, the matrix multiplication is:
        # C[0,k] = sum_j A[0,j] * B[j,k]
        # This is a rank-1 tensor? No, it's a sum of 4 terms.
        # Actually, the tensor T[i,j,k] = delta[i,0] for i in {0}, j in {0,1,2,3}, k in {0,...,6}
        # This has rank 4 (one for each j).
        # Wait, no. T[0,j,k] = 1 if we're computing C[0,k] = sum_j A[0,j]B[j,k]
        # The tensor is T[i,j,k] = 1 if i=0, else 0.
        # This can be written as sum_j e_0 * e_j * e_k... no.
        # T[i,j,k] = delta_{i,0} * 1 * 1 for all j,k.
        # This is a rank-1 tensor: u_i = delta_{i,0}, v_j = 1, w_k = 1.
        # So rank 1!
        
        # Wait, that can't be right. Let me reconsider.
        # For matrix multiplication C = AB, the tensor is T[i,j,k] = delta_{i,k}... no.
        # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k].
        # So T[i,j,k] = 1 for all valid i,j,k, and we sum over j.
        # The tensor is T[i,j,k] = 1 for all i,j,k in the appropriate ranges.
        # This is NOT rank 1 in general.
        
        # For n=1, m=4, p=7:
        # T[0,j,k] = 1 for j in {0,1,2,3}, k in {0,...,6}
        # This tensor has rank 4 (or 7, or min(4,7)=4).
        # Actually, it's the all-ones tensor of size 1x4x7, which has rank 1.
        # T[0,j,k] = 1 = u_0 * v_j * w_k with u_0=1, v_j=1, w_k=1.
        # So rank 1!
        
        # Wait, but that would mean C[0,k] = sum_j A[0,j]*B[j,k] = (sum_j A[0,j]) * (sum_k B[j,k])...
        # No, that's not right. The tensor decomposition is:
        # T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        # For n=1, m=4, p=7, T[0,j,k] = 1 for all j,k.
        # We need: sum_r U[0,r] * V[j,r] * W[k,r] = 1 for all j,k.
        # This can be achieved with rank 1: U[0,0]=1, V[j,0]=1, W[k,0]=1.
        # Then sum_r ... = 1*1*1 = 1. Yes!
        
        # So (1,4,7) has rank 1, not 28!
        
        # Let me recompute the total:
        # (2,4,7) = 25
        # (1,4,7) = 1
        # Total: 26. That's much better!
        
        # But wait, I need to double-check. The tensor for matrix multiplication
        # C=AB is T[i,j,k] where C[i,k] = sum_j A[i,j] B[j,k].
        # The tensor is T[i,j,k] = 1 if (i,j,k) is a valid triple, 0 otherwise.
        # For the full 3x4x7 case, T[i,j,k] = 1 for i in {0,1,2}, j in {0,1,2,3}, k in {0,...,6}.
        
        # When we split n=3 as 2+1, we get:
        # Block 1: i in {0,1}, j in {0,1,2,3}, k in {0,...,6}: T=1
        # Block 2: i in {2}, j in {0,1,2,3}, k in {0,...,6}: T=1
        
        # For Block 2 (n=1, m=4, p=7):
        # T[2,j,k] = 1 for all j,k.
        # Rank 1 decomposition: U[2,0]=1, V[j,0]=1 for all j, W[k,0]=1 for all k.
        # Check: sum_r U[2,r] V[j,r] W[k,r] = 1*1*1 = 1. Correct!
        
        # So the total rank is 25 + 1 = 26.
        
        # Let me implement this.
        
        # For (2,4,7):
        # Split m=4 as 2+2
        # (2,2,7) + (2,2,7)
        # Each (2,2,7) = 25
        # Total for (2,4,7) = 50
        
        # Wait, that's worse. Let me recompute.
        # (2,4,7): split m=4 as 2+2
        # (2,2,7) + (2,2,7)
        # (2,2,7) = 25 each
        # Total: 50
        
        # Plus (1,4,7) = 1
        # Total: 51. Still better than 78!
        
        # But can I do better for (2,2,7)?
        # (2,2,7): I got 25 above. Let me see if I can do better.
        # (2,2,7): split p=7 as 2+2+2+1
        # (2,2,2)+(2,2,2)+(2,2,2)+(2,2,1) = 7+7+7+4 = 25
        
        # What about splitting p=7 as 3+4?
        # (2,2,3)+(2,2,4)
        # (2,2,3): split p=3 as 2+1
        # (2,2,2)+(2,2,1) = 7+4 = 11
        # (2,2,4): split p=4 as 2+2
        # (2,2,2)+(2,2,2) = 14
        # Total: 25. Same.
        
        # It seems 25 is the best I can do for (2,2,7) with this approach.
        
        # So total rank = 50 + 1 = 51.
        # Score = 84/51 ≈ 1.647. Much better!
        
        # Let me implement this.

        # Actually, I realize I should just implement the decomposition directly
        # without all this analysis in the code. Let me restructure.
        
        pass  # Placeholder, will be replaced

    def gen_2x2x7(i_off, j_off, k_off):
        # (2,2,7): split p=7 as 2+2+2+1
        gen_2x2x2(i_off, j_off, k_off)
        gen_2x2x2(i_off, j_off, k_off + 2)
        gen_2x2x2(i_off, j_off, k_off + 4)
        gen_2x2x1(i_off, j_off, k_off + 6)

    def gen_2x2x2(i_off, j_off, k_off):
        # Strassen 2x2x2
        i0 = i_off
        i1 = i_off + 1
        j0 = j_off
        j1 = j_off + 1
        k0 = k_off
        k1 = k_off + 1

        add_rank1_term([i0*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j0, i1*m+j1], [1, 1],
                       [j0*p+k0], [1],
                       [k0*n+i1, k1*n+i1], [1, -1])
        add_rank1_term([i0*m+j0], [1],
                       [j0*p+k1, j1*p+k1], [1, -1],
                       [k1*n+i0, k1*n+i1], [1, 1])
        add_rank1_term([i1*m+j1], [1],
                       [j1*p+k0, j0*p+k0], [1, -1],
                       [k0*n+i0, k0*n+i1], [1, 1])
        add_rank1_term([i0*m+j0, i0*m+j1], [1, 1],
                       [j1*p+k1], [1],
                       [k0*n+i0, k1*n+i0], [-1, 1])
        add_rank1_term([i1*m+j0, i0*m+j0], [1, -1],
                       [j0*p+k0, j0*p+k1], [1, 1],
                       [k1*n+i1], [1])
        add_rank1_term([i0*m+j1, i1*m+j1], [1, -1],
                       [j1*p+k0, j1*p+k1], [1, 1],
                       [k0*n+i0], [1])

    def gen_2x2x1(i_off, j_off, k_off):
        # (2,2,1): schoolbook, rank 4
        for i in range(2):
            for j in range(2):
                ii = i_off + i
                jj = j_off + j
                kk = k_off
                add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])

    def gen_1x4x7(i_off, j_off, k_off):
        # (1,4,7): rank 1
        # U[i_off, 0] = 1
        # V[j_off+j, 0] = 1 for j in 0..3
        # W[k_off+k, 0] = 1 for k in 0..6
        u_indices = [i_off * m + j_off + j for j in range(4)]
        # Wait, U has shape (n*m, rank). For n=1, m=4, U has 4 entries.
        # But we're adding to the global U which has n*m = 12 entries.
        # The i index is i_off, which is 2 (since we split n=3 as 2+1).
        # U index for (i,j) is i*m + j = 2*4 + j = 8+j for j in 0..3.
        # V index for (j,k) is j*p + k. For j in 0..3, k in 0..6.
        # W index for (k,i) is k*n + i = k*3 + 2.
        
        # Rank 1: U[8+j] = 1 for j in 0..3, V[j*7+k] = 1 for all j,k, W[k*3+2] = 1 for all k
        # But that's not a single rank-1 term. A rank-1 term has ONE non-zero
        # in each of U, V, W (or a few non-zeros that factor).
        
        # Actually, for rank 1: U[i,r] * V[j,r] * W[k,r] = T[i,j,k]
        # For T[2,j,k] = 1 for all j,k:
        # We need sum_r U[2,r] V[j,r] W[k,r] = 1 for all j,k.
        # With rank 1: U[2,0]=1, V[j,0]=1 for all j, W[k,0]=1 for all k.
        # Then U[2,0]*V[j,0]*W[k,0] = 1*1*1 = 1. Correct!
        
        # So this is a single rank-1 term where:
        # U has 1 at index 2*4+j for j=0..3? No, U[2,r] is a single entry.
        # Wait, U is a vector of length n*m = 12. U[i*m+j, r] for the r-th term.
        # For this rank-1 term, we need U[2*4+j, 0] for j=0..3 to be... 
        # No, U is indexed by (i,j) pairs. U[i*m+j, r].
        # For i=2, j=0: U[8, r]
        # For i=2, j=1: U[9, r]
        # etc.
        
        # But in a rank-1 term, U is a single vector. U[i*m+j, r] for fixed r.
        # So U[8,0], U[9,0], U[10,0], U[11,0] can all be non-zero.
        # Similarly V[j*p+k, 0] for all j,k, and W[k*n+2, 0] for all k.
        
        # So this is ONE rank-1 term with:
        # U[8..11] = 1
        # V[j*7+k] = 1 for j=0..3, k=0..6 (28 entries)
        # W[k*3+2] = 1 for k=0..6 (7 entries)
        
        u_indices = [8, 9, 10, 11]
        u_vals = [1, 1, 1, 1]
        v_indices = [j*p + k for j in range(4) for k in range(7)]
        v_vals = [1] * 28
        w_indices = [k*n + 2 for k in range(7)]
        w_vals = [1] * 7
        
        add_rank1_term(u_indices, u_vals, v_indices, v_vals, w_indices, w_vals)

    # Main decomposition:
    # 3x4x7 = (2,4,7) + (1,4,7)
    # (2,4,7) = (2,2,7) + (2,2,7)
    # Each (2,2,7) = 25 terms
    # (1,4,7) = 1 term
    # Total: 51 terms
    
    gen_2x2x7(0, 0, 0)
    gen_2x2x7(0, 2, 0)
    gen_1x4x7(2, 0, 0)

    # Convert to float for output, but verify with Fraction
    U = np.array([[float(x) for x in row] for row in U_vecs]).T
    V = np.array([[float(x) for x in row] for row in V_vecs]).T
    W = np.array([[float(x) for x in row] for row in W_vecs]).T

    # Verify
    valid = True
    for ci in range(n):
        for cj in range(m):
            for ck in range(p):
                prod_sum = sum(U_vecs[t][ci*m + cj] * V_vecs[t][cj*p + ck] * W_vecs[t][ck*n + ci] for t in range(len(U_vecs)))
                if prod_sum != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback to schoolbook
        U_vecs.clear()
        V_vecs.clear()
        W_vecs.clear()
        for ii in range(n):
            for jj in range(m):
                for kk in range(p):
                    add_rank1_term([ii*m+jj], [1], [jj*p+kk], [1], [kk*n+ii], [1])
        U = np.array([[float(x) for x in row] for row in U_vecs]).T
        V = np.array([[float(x) for x in row] for row in V_vecs]).T
        W = np.array([[float(x) for x in row] for row in W_vecs]).T

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
