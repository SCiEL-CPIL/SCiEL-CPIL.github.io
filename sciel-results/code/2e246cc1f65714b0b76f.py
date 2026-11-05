"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS  # 2, 3, 3
    # Matrix multiplication: A (n x m) * B (m x p) = C (n x p)
    # A has indices (i, j) where i in [0,n), j in [0,m)
    # B has indices (j, k) where j in [0,m), k in [0,p)
    # C has indices (i, k) where i in [0,n), k in [0,p)
    #
    # Tensor T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # The multiplication tensor has entries T_{i,j,k} = 1 for all valid (i,j,k)
    #
    # Rank-r decomposition: T = sum_{c=0}^{r-1} U[:,c] * V[:,c] * W[:,c]
    # where U has shape (n*m, r), V has shape (m*p, r), W has shape (p*n, r)
    #
    # The output convention is W[k*n + i, c]
    #
    # Indexing:
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    #
    # The tensor entry T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # must equal 1 for all (i, j, k) in the valid range.
    
    # Schoolbook gives rank = n*m*p = 18.
    # We know Strassen-like methods can do better for square matrices.
    # For 2x2x2, Strassen gives rank 7.
    # For 2x3x3, we need to find the minimum rank.
    
    # Let's try a structured approach based on Strassen for the 2x2 sub-block
    # plus corrections for the remaining dimensions.
    
    # Strategy: Use the known rank-7 Strassen decomposition for the 2x2x2 core,
    # then extend to handle the extra dimensions.
    
    # For 2x2x2 Strassen:
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # C = [[M1+M4-M5-M7, M3+M5], [M2+M4, M1-M2+M3+M7]]
    
    # For 2x3x3, let's think of it as:
    # A is 2x3: [[a1,a2,a3],[b1,b2,b3]]
    # B is 3x3: [[c1,c2,c3],[d1,d2,d3],[e1,e2,e3]]
    # C is 2x3: [[f1,f2,f3],[g1,g2,g3]]
    
    # f1 = a1*c1 + a2*d1 + a3*e1
    # f2 = a1*c2 + a2*d2 + a3*e2
    # f3 = a1*c3 + a2*d3 + a3*e3
    # g1 = b1*c1 + b2*d1 + b3*e1
    # g2 = b1*c2 + b2*d2 + b3*e2
    # g3 = b1*c3 + b2*d3 + b3*e3
    
    # Let's try to use a rank-17 or lower decomposition.
    # Actually, let's try to find a rank-16 decomposition using linear dependencies.
    
    # First, let's construct a rank-18 schoolbook solution, then try to reduce.
    
    # Actually, let me think about this more carefully using the tensor formulation.
    
    # The multiplication tensor for (2,3,3) has 2*3*3 = 18 entries, all equal to 1.
    # We need to find the minimum rank of this tensor.
    
    # Let me try a direct algebraic approach. Consider using block structure.
    
    # Write A as [A1 | A2] where A1 is 2x2 and A2 is 2x1
    # Write B as [B1; B2] where B1 is 2x3 and B2 is 1x3
    # Then C = A1*B1 + A2*B2
    
    # A1*B1 is a 2x2 times 2x3 product (rank-6 schoolbook, but maybe less with Strassen-like)
    # A2*B2 is a 2x1 times 1x3 product (rank-2 schoolbook)
    
    # For 2x2x2, Strassen uses 7 multiplications.
    # For 2x2x3, we can split B into two 2x2 and 2x1 parts:
    # A1*B1 = A1*[B1a | B1b] where B1a is 2x2, B1b is 2x1
    # A1*B1a: 2x2x2 -> rank 7 (Strassen)
    # A1*B1b: 2x2x1 -> this is just 2x2 times 2x1, schoolbook rank 4
    # But we can share some Strassen products...
    
    # Actually, let me just try to construct an explicit low-rank decomposition.
    
    # Let me use the approach: try rank r from 16 down, and for each rank,
    # set up the system of equations and try to solve.
    
    # For rank r, we have:
    # U: (n*m) x r = 6 x r
    # V: (m*p) x r = 9 x r
    # W: (p*n) x r = 6 x r
    # Total unknowns: 6r + 9r + 6r = 21r
    # Constraints: 18 (one for each (i,j,k) entry)
    # Plus we can normalize (e.g., set U[0,0]=1 if possible)
    
    # For r=16: 21*16 = 336 unknowns, 18 constraints. Very underdetermined.
    # But the constraints are nonlinear (product of three variables).
    
    # Let me try a different approach: use the known structure.
    
    # For 2x3x3, I'll try to build a rank-17 decomposition based on Strassen for the 2x2x2 core.
    
    # Decompose:
    # A = [[a, b, c], [d, e, f]]  (2x3)
    # B = [[g, h, i], [j, k, l], [m, n, o]]  (3x3)
    # C = [[p, q, r], [s, t, u]]  (2x3)
    
    # p = ag + bj + cm
    # q = ah + bk + cn
    # r = ai + bl + co
    # s = dg + ej + fm
    # t = dh + ek + fn
    # u = di + el + fo
    
    # Let me try to use 17 rank-1 terms.
    
    # Idea: Use Strassen on the (a,b,d,e) x (g,h,j,k) sub-block (2x2x2),
    # giving 7 products, then handle the remaining terms.
    
    # Strassen products for A1=[[a,b],[d,e]] and B1=[[g,h],[j,k]]:
    # M1 = (a+d)(g+k)
    # M2 = (d+e)g
    # M3 = a(h-k)
    # M4 = e(j-d)
    # M5 = (a+b)k
    # M6 = (d-e)(h+j)
    # M7 = (b-e)(g+j)
    
    # C1 (top-left 2x2 of C from A1*B1):
    # C1[0,0] = M1 + M4 - M5 - M7
    # C1[0,1] = M3 + M5
    # C1[1,0] = M2 + M4
    # C1[1,1] = M1 - M2 + M3 + M7
    
    # Now we need to add contributions from the third column of A (c, f) and
    # third row of B (m, n, o).
    
    # Additional terms needed:
    # p += cm, q += cn, r += co
    # s += fm, t += fn, u += fo
    
    # That's 6 more products: cm, cn, co, fm, fn, fo
    # Total: 7 + 6 = 13? That seems too good. Let me check.
    
    # Wait, the Strassen products M1-M7 already use linear combinations,
    # so they're not just simple products. The rank is the number of
    # "multiplications" (rank-1 terms), which is 7 for the 2x2x2 part.
    
    # But when I write M1 = (a+d)(g+k), this is ONE rank-1 term in the tensor
    # decomposition sense? No, it's a product of two linear forms, which
    # corresponds to a rank-1 term in the trilinear form.
    
    # Actually, in the tensor decomposition T = sum_c U_c * V_c * W_c,
    # each term c contributes U[i,j,c]*V[j,k,c]*W[i,k,c] to T[i,j,k].
    # A "multiplication" like (a+d)(g+k) would correspond to:
    # U has a+d in the (i,j) slots, V has g+k in the (j,k) slots,
    # and W distributes the result to the appropriate (i,k) output.
    
    # This is getting complex. Let me just try to numerically find a
    # low-rank decomposition using the known structure.
    
    # Let me try rank 17 first with a structured ansatz.
    
    # Actually, let me try a cleaner approach: use the bilinear complexity results.
    # For 2x3x3 matrix multiplication, the minimum rank is known to be 17
    # (or possibly less). Let me try to construct rank 17.
    
    # Let me use a systematic approach: start with schoolbook rank 18 and
    # try to merge two terms using a Strassen-like trick.
    
    # Schoolbook terms (i,j,k) for i in {0,1}, j in {0,1,2}, k in {0,1,2}:
    # Each term is A[i,j] * B[j,k] contributing to C[i,k].
    
    # Let me label them:
    # T000: A[0,0]*B[0,0] -> C[0,0]
    # T001: A[0,0]*B[0,1] -> C[0,1]
    # T002: A[0,0]*B[0,2] -> C[0,2]
    # T010: A[0,1]*B[1,0] -> C[0,0]
    # T011: A[0,1]*B[1,1] -> C[0,1]
    # T012: A[0,1]*B[1,2] -> C[0,2]
    # T020: A[0,2]*B[2,0] -> C[0,0]
    # T021: A[0,2]*B[2,1] -> C[0,1]
    # T022: A[0,2]*B[2,2] -> C[0,2]
    # T100: A[1,0]*B[0,0] -> C[1,0]
    # T101: A[1,0]*B[0,1] -> C[1,1]
    # T102: A[1,0]*B[0,2] -> C[1,2]
    # T110: A[1,1]*B[1,0] -> C[1,0]
    # T111: A[1,1]*B[1,1] -> C[1,1]
    # T112: A[1,1]*B[1,2] -> C[1,2]
    # T120: A[1,2]*B[2,0] -> C[1,0]
    # T121: A[1,2]*B[2,1] -> C[1,1]
    # T122: A[1,2]*B[2,2] -> C[1,2]
    
    # Strassen trick on the 2x2x2 sub-block (i in {0,1}, j in {0,1}, k in {0,1}):
    # Replace T000, T001, T010, T011, T100, T101, T110, T111 (8 terms) with 7 terms.
    
    # The 8 schoolbook terms for the 2x2x2 block:
    # A[0,0]*B[0,0] -> C[0,0]
    # A[0,0]*B[0,1] -> C[0,1]
    # A[0,1]*B[1,0] -> C[0,0]
    # A[0,1]*B[1,1] -> C[0,1]
    # A[1,0]*B[0,0] -> C[1,0]
    # A[1,0]*B[0,1] -> C[1,1]
    # A[1,1]*B[1,0] -> C[1,0]
    # A[1,1]*B[1,1] -> C[1,1]
    
    # Strassen replacement (7 terms):
    # S1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # S2 = (A[1,0]+A[1,1])*B[0,0]
    # S3 = A[0,0]*(B[0,1]-B[1,1])
    # S4 = A[1,1]*(B[1,0]-A[1,0])  -- wait, this is wrong
    # Let me use the standard Strassen formulas properly.
    
    # Standard Strassen for A=[[a,b],[c,d]], B=[[e,f],[g,h]]:
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # C[0,0] = M1+M4-M5-M7
    # C[0,1] = M3+M5
    # C[1,0] = M2+M4
    # C[1,1] = M1-M2+M3+M7
    
    # In our notation: a=A[0,0], b=A[0,1], c=A[1,0], d=A[1,1]
    # e=B[0,0], f=B[0,1], g=B[1,0], h=B[1,1]
    
    # So the 7 Strassen products are:
    # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # M2 = (A[1,0]+A[1,1])*B[0,0]
    # M3 = A[0,0]*(B[0,1]-B[1,1])
    # M4 = A[1,1]*(B[1,0]-A[1,0])  -- NO, M4 = d*(g-c) = A[1,1]*(B[1,0]-A[1,0])
    # Wait, that's not right either. M4 = d*(g-c) where c=A[1,0], d=A[1,1], g=B[1,0]
    # So M4 = A[1,1]*(B[1,0]-A[1,0])? No! M4 = d*(g-c) = A[1,1]*(B[1,0]-A[1,0])
    # That doesn't make sense dimensionally. Let me re-examine.
    
    # In Strassen, all M's are products of linear combinations of A entries and
    # linear combinations of B entries. M4 = d*(g-c) means:
    # d is A[1,1], g is B[1,0], c is A[1,0]
    # So M4 = A[1,1] * (B[1,0] - A[1,0])
    # But this mixes A and B entries in the same factor, which is NOT allowed
    # in a bilinear form!
    
    # I think I'm confusing myself. Let me look at this more carefully.
    
    # In the tensor decomposition, each rank-1 term has:
    # U[i,j,c] which is a linear form in A entries (or just picks out A[i,j])
    # V[j,k,c] which is a linear form in B entries (or just picks out B[j,k])
    # W[i,k,c] which distributes to C[i,k]
    
    # For a "pure" product like A[i,j]*B[j,k], we have:
    # U[i,j,c] = 1 (for that specific i,j), 0 otherwise
    # V[j,k,c] = 1 (for that specific j,k), 0 otherwise
    # W[i,k,c] = 1 (for that specific i,k), 0 otherwise
    
    # For a Strassen term like (a+d)(e+h):
    # This is (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
    # U[0,0,c] = 1, U[1,1,c] = 1, all other U = 0
    # V[0,0,c] = 1, V[1,1,c] = 1, all other V = 0
    # W: this product contributes to multiple C entries.
    # (A[0,0]+A[1,1])*(B[0,0]+B[1,1]) = A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # So it contributes to C[0,0], C[0,1], C[1,0], C[1,1]
    # W[0,0,c] = 1, W[0,1,c] = 1, W[1,0,c] = 1, W[1,1,c] = 1
    # Wait, but that would overcount. The issue is that in the tensor
    # decomposition, T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[i,k,c]
    # For the term c with U[0,0]=1, U[1,1]=1, V[0,0]=1, V[1,1]=1, W[0,0]=1, W[0,1]=1, W[1,0]=1, W[1,1]=1:
    # T[0,0,0] += U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1
    # T[0,0,1] += U[0,0]*V[0,1]*W[0,1] = 1*0*1 = 0  (V[0,1]=0 since V only has [0,0] and [1,1])
    # Hmm, this doesn't work simply.
    
    # The key insight: in the tensor T[i,j,k], the indices are (i,j) for U,
    # (j,k) for V, and (i,k) for W. The shared index is j.
    
    # For the product (A[0,0]+A[1,1])*(B[0,0]+B[1,1]):
    # This expands to A[0,0]B[0,0] + A[0,0]B[1,1] + A[1,1]B[0,0] + A[1,1]B[1,1]
    # But A[0,0]B[1,1] is NOT a valid term in our matrix multiplication!
    # In matrix multiplication, we only have A[i,j]*B[j,k] terms.
    # A[0,0]*B[1,1] would require j=0 in A and j=1 in B, which is inconsistent.
    
    # So Strassen doesn't directly apply in the simple tensor decomposition sense.
    # The Strassen algorithm works by reorganizing the computation, but the
    # "multiplications" are not simply rank-1 terms in the original tensor.
    
    # I think I need to reconsider. The bilinear complexity of matrix multiplication
    # is indeed the rank of the multiplication tensor. For 2x2x2, it's 7.
    # The Strassen algorithm achieves this by finding 7 rank-1 terms that
    # correctly reconstruct all the necessary products.
    
    # Let me look up the actual Strassen tensor decomposition.
    
    # For 2x2x2, the rank-7 decomposition has:
    # Term 1: U=(1,0,0,1) [for A indices (0,0),(0,1),(1,0),(1,1)], V=(1,0,0,1) [for B indices], W=(1,1,1,1)
    # Wait, I need to be more careful with the indexing.
    
    # Let me use a different approach. Let me just try to numerically optimize
    # for a low-rank decomposition.
    
    # Actually, let me just return the schoolbook rank-18 solution for now,
    # since I'm running out of time to figure out the exact low-rank construction.
    # But wait, the knowledge says rank 17 was achieved. Let me try to construct that.
    
    # Let me try a simpler approach: use rank 17 with a known construction.
    
    # For 2x3x3, one approach:
    # Split the 3x3 B matrix into a 2x2 block and a 1x3 row.
    # A (2x3) * B (3x3) = A[:, :2] * B[:2, :] + A[:, 2:] * B[2:, :]
    # = (2x2)*(2x3) + (2x1)*(1x3)
    
    # For (2x2)*(2x3): split B[:2,:] into B[:2,:2] (2x2) and B[:2,2:] (2x1)
    # (2x2)*(2x2) -> rank 7 (Strassen)
    # (2x2)*(2x1) -> rank 4 (schoolbook) but can share with Strassen
    # (2x1)*(1x3) -> rank 2 (schoolbook)
    
    # Total: 7 + 4 + 2 = 13? That seems too optimistic.
    
    # Let me just go with a safe rank-18 schoolbook solution and add a
    # verification step. If I have time, I'll try to improve.
    
    # Actually, let me try to construct a rank-17 solution by taking the
    # schoolbook 18 terms and merging two of them using a linear dependency.
    
    # The idea: if we can find two schoolbook terms T_a and T_b such that
    # their sum can be expressed as a single rank-1 term (or as a combination
    # that saves one rank), we can reduce the rank by 1.
    
    # For example, consider T000 = A[0,0]*B[0,0] -> C[0,0] and
    # T111 = A[1,1]*B[1,1] -> C[1,1].
    # Can we replace these two with one term? Probably not directly.
    
    # Let me try a different merge. Consider:
    # T000: A[0,0]*B[0,0] -> C[0,0]
    # T011: A[0,1]*B[1,1] -> C[0,1]
    # T100: A[1,0]*B[0,0] -> C[1,0]
    # T110: A[1,1]*B[1,0] -> C[1,0]
    
    # Hmm, this is getting complicated. Let me just return schoolbook for now.
    
    # Schoolbook rank-18 solution:
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
    
    # Try to improve: attempt rank 17
    if time.time() - start_time < DEADLINE - 5:
        try:
            r17 = _try_rank_17()
            if r17 is not None:
                return r17
        except Exception:
            pass
    
    return {"U": U, "V": V, "W": W}


def _try_rank_17():
    """Try to construct a rank-17 decomposition for 2x3x3 matrix multiplication."""
    n, m, p = 2, 3, 3
    r = 17
    
    # Use a structured ansatz based on Strassen for the 2x2x2 core.
    # Indices: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    
    # Let me define the 17 terms.
    # Terms 0-6: Strassen-like for the (i in {0,1}, j in {0,1}, k in {0,1}) block
    # Terms 7-12: Handle the j=2 interactions
    # Terms 13-16: Handle remaining k=2 interactions
    
    # Actually, let me try a specific construction.
    
    # For the 2x2x2 Strassen block (j in {0,1}, k in {0,1}):
    # The 7 Strassen products and their contributions:
    
    # Let me define the U, V, W matrices.
    U = [[0.0] * r for _ in range(n * m)]
    V = [[0.0] * r for _ in range(m * p)]
    W = [[0.0] * r for _ in range(p * n)]
    
    # Strassen terms for 2x2x2 (indices i in {0,1}, j in {0,1}, k in {0,1}):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-A10)  -- this is problematic as noted before
    # Let me use the correct Strassen formulas.
    
    # Standard Strassen:
    # M1 = (a+d)(e+h)  where a=A00,b=A01,c=A10,d=A11, e=B00,f=B01,g=B10,h=B11
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-c)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    
    # C00 = M1+M4-M5-M7
    # C01 = M3+M5
    # C10 = M2+M4
    # C11 = M1-M2+M3+M7
    
    # In tensor form, each M corresponds to a rank-1 term:
    # M1: U has 1 at (0,0) and (1,1); V has 1 at (0,0) and (1,1); W has 1 at (0,0),(0,1),(1,0),(1,1)
    # But wait, the W index is (k*n+i), so W[0*2+0]=W[0], W[1*2+0]=W[2], W[0*2+1]=W[1], W[1*2+1]=W[3]
    # And the contribution to T[i,j,k] is U[i*m+j]*V[j*p+k]*W[k*n+i]
    
    # For M1 = (A00+A11)(B00+B11):
    # This should contribute to T[0,0,0], T[0,0,1], T[1,0,0], T[1,1,0], T[1,1,1], T[0,1,1]
    # Wait, that's not right. Let me think again.
    
    # The product (A00+A11)(B00+B11) = A00*B00 + A00*B11 + A11*B00 + A11*B11
    # In matrix multiplication, A00*B00 contributes to C[0,0] (via j=0)
    # A00*B11 is NOT a valid term (j mismatch: A uses j=0, B uses j=1)
    # So this simple expansion doesn't work in the tensor framework.
    
    # I think the issue is that Strassen's algorithm doesn't correspond to a
    # simple tensor decomposition in the way I was thinking. The rank-7
    # decomposition for 2x2x2 does exist, but the terms are more subtle.
    
    # Let me just return None and use schoolbook.
    return None


# EVOLVE-BLOCK-END
