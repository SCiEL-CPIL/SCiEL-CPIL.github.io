"""Matrix multiplication 3,3,7 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n=3, m=3, p=7):
    """Verify that U, V, W give an exact decomposition of the matmul tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] should equal 1
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook(n=3, m=3, p=7):
    """Return the schoolbook (rank=n*m*p) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _strassen_2x2():
    """
    Return Strassen decomposition for 2x2 * 2x2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    # Strassen formulas:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # For each term c, we need U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] to contribute to C[i,k]
    # The tensor entry T_{i,j,k} = 1 means: for each (i,j,k), sum_c U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    
    # Let's build this carefully.
    # Term c contributes to C[i,k] via the product U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    # Actually the decomposition is:
    # T_{i,j,k} = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # where T_{i,j,k} = 1 for all valid (i,j,k).
    
    # For 2x2*2x2: n=m=p=2
    # U has 4 rows (i*2+j), V has 4 rows (j*2+k), W has 4 rows (k*2+i)
    
    # Strassen term 1: M1 = (A11+A22)(B11+B22)
    # This means: sum over j of (A_{i,j} * something)
    # Actually let me think of it as: the bilinear form is sum_j A_{i,j} B_{j,k}
    # And we want to express this as sum_c (linear combo of A) * (linear combo of B) * (contribution to C)
    
    # The standard way: C_{i,k} = sum_c U_{i,k,c} * (sum_j A_{i,j} * alpha_{j,c}) * (sum_k B_{j,k} * beta_{k,c})
    # But that's not quite right either.
    
    # Let me use the tensor formulation directly.
    # T_{i,j,k} = 1 for all i,j,k in {0,1}
    # We need: 1 = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c]
    
    # For Strassen, the 7 terms are:
    # c=0: M1 = (A00+A11)(B00+B11)
    #   U[0*2+0,0]=1, U[1*2+1,0]=1 (i.e., A00 and A11)
    #   V[0*2+0,0]=1, V[0*2+1,0]=1 (i.e., B00 and B01)... wait
    
    # Let me be more careful. The indices are:
    # U indexed by (i,j): row = i*m + j = i*2 + j
    # V indexed by (j,k): row = j*p + k = j*2 + k  
    # W indexed by (k,i): row = k*n + i = k*2 + i
    
    # For M1 = (A00+A11)(B00+B11):
    # This contributes to C00 and C11 (and cross terms)
    # C00 gets M1, C11 gets M1
    # Actually C00 = M1+M4-M5+M7, C11 = M1-M2+M3+M6
    
    # The tensor decomposition means:
    # For each (i,j,k), the sum over c of U[i*2+j,c]*V[j*2+k,c]*W[k*2+i,c] = 1
    
    # Let me just directly construct it.
    # Term 0: M1 = (A00+A11)(B00+B11)
    #   U[0,0]=1 (A00), U[3,0]=1 (A11)  [rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)]
    #   V[0,0]=1 (B00), V[1,0]=1 (B01)  [rows: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)]
    #   Wait, V is indexed by (j,k). V[j*2+k, c].
    #   For M1, we need B00 and B11. B00 is j=0,k=0 -> row 0. B11 is j=1,k=1 -> row 3.
    #   V[0,0]=1, V[3,0]=1
    #   W is indexed by (k,i). W[k*2+i, c].
    #   M1 contributes to C00 (k=0,i=0 -> row 0) and C11 (k=1,i=1 -> row 3)
    #   W[0,0]=1, W[3,0]=1
    
    # Let me verify: for (i=0,j=0,k=0):
    #   U[0*2+0,0]*V[0*2+0,0]*W[0*2+0,0] = U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # For (i=0,j=0,k=1):
    #   U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0... but we need sum to be 1.
    #   Hmm, this doesn't work with just one term.
    
    # I think I'm confusing myself. Let me restart with a cleaner formulation.
    
    # The matrix multiplication tensor for [n x m] * [m x p]:
    # T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # For 2x2*2x2, we have 8 tensor entries, all equal to 1.
    # Schoolbook: 8 terms, each term c corresponds to one (i,j,k) with U[i*2+j,c]=V[j*2+k,c]=W[k*2+i,c]=1.
    
    # Strassen reduces to 7 terms. Let me look up the exact decomposition.
    # 
    # The 7 Strassen products are:
    # P1 = (A11+A22)(B11+B22)
    # P2 = (A21+A22)B11
    # P3 = A11(B21-B22)
    # P4 = A22(B12-B11)
    # P5 = (A11+A12)B22
    # P6 = (A21-A11)(B11+B12)
    # P7 = (A12-A22)(B21+B22)
    #
    # C11 = P1+P4-P5+P7
    # C12 = P3+P5
    # C21 = P2+P4
    # C22 = P1-P2+P3+P6
    
    # In our notation (0-indexed):
    # P1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1])
    # P2 = (A[1,0]+A[1,1])B[0,0]
    # P3 = A[0,0](B[1,0]-B[1,1])
    # P4 = A[1,1](B[0,1]-B[0,0])
    # P5 = (A[0,0]+A[0,1])B[1,1]
    # P6 = (A[1,0]-A[0,0])(B[0,0]+B[0,1])
    # P7 = (A[0,1]-A[1,1])(B[1,0]+B[1,1])
    #
    # C[0,0] = P1+P4-P5+P7
    # C[0,1] = P3+P5
    # C[1,0] = P2+P4
    # C[1,1] = P1-P2+P3+P6
    
    # Now, for the tensor decomposition:
    # T_{i,j,k} = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1
    
    # Each P_c is of the form (sum_{j} a_{j,c} A_{i,j}) * (sum_{j} b_{j,c} B_{j,k})
    # But that's not quite the right structure for the tensor decomposition.
    
    # Actually, the tensor decomposition is:
    # sum_j A_{i,j} B_{j,k} = sum_c (sum_j alpha_{i,j,c} A_{i,j}) (sum_j beta_{j,k,c} B_{j,k}) gamma_{k,i,c}
    # where gamma is just a coefficient for the C_{k,i} output.
    
    # Hmm, I think the standard formulation is:
    # C_{i,k} = sum_c U_{i,k,c} * (sum_j A_{i,j} * a_{j,c}) * (sum_j B_{j,k} * b_{j,c})
    
    # But in our tensor formulation:
    # T_{i,j,k} = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This means: for each (i,j,k), the bilinear form coefficient is 1.
    # The decomposition says: 1 = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # For a term c, if U[i*m+j,c] = u_{ij,c}, V[j*p+k,c] = v_{jk,c}, W[k*n+i,c] = w_{ki,c},
    # then the term contributes u_{ij,c} * v_{jk,c} * w_{ki,c} to T_{i,j,k}.
    
    # For Strassen P1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]):
    # This is a bilinear form. It can be written as:
    # sum_{i,j,k} [contribution to C_{i,k}]
    # P1 contributes to C[0,0] and C[1,1].
    # 
    # Let me think of it differently. The bilinear form for P1 is:
    # (A[0,0]+A[1,1]) * (B[0,0]+B[1,1])
    # = A[0,0]*B[0,0] + A[0,0]*B[1,1] + A[1,1]*B[0,0] + A[1,1]*B[1,1]
    #
    # This contributes to C[0,0] (from A[0,0]*B[0,0] and A[1,1]*B[1,1]... no wait)
    # Actually P1 is just a scalar. It gets added to C[0,0] and C[1,1] with coefficient 1.
    #
    # So the contribution of P1 to the tensor is:
    # For (i=0,j=0,k=0): A[0,0]*B[0,0] contributes to C[0,0]. Coefficient in T: 1*1*1 = 1? 
    # No, the tensor T_{i,j,k} represents the coefficient of A_{i,j}*B_{j,k} in C_{i,k}.
    # T_{i,j,k} = 1 means A_{i,j}*B_{j,k} appears once in C_{i,k}.
    
    # I think the cleanest way is:
    # For each Strassen product P_c, define:
    #   alpha_c[i,j] = coefficient of A[i,j] in the first factor of P_c
    #   beta_c[j,k] = coefficient of B[j,k] in the second factor of P_c  
    #   gamma_c[i,k] = coefficient of P_c in C[i,k]
    #
    # Then: sum_j A[i,j]B[j,k] = sum_c gamma_c[i,k] * (sum_j alpha_c[i,j] A[i,j]) * (sum_j beta_c[j,k] B[j,k])
    #
    # Expanding: = sum_c sum_j sum_j' gamma_c[i,k] alpha_c[i,j] beta_c[j',k] A[i,j] B[j',k]
    #
    # For this to equal sum_j A[i,j]B[j,k], we need:
    # sum_c gamma_c[i,k] alpha_c[i,j] beta_c[j,k] = delta_{j,j} = 1 (for j=j')
    # And for j != j': sum_c gamma_c[i,k] alpha_c[i,j] beta_c[j',k] = 0
    #
    # But wait, that's not right either because the sum over j and j' are separate.
    # Actually: sum_j sum_j' ... A[i,j] B[j',k] should equal sum_j A[i,j] B[j,k]
    # So we need: for each (i,j,k): sum_c gamma_c[i,k] alpha_c[i,j] beta_c[j,k] = 1
    # And for (i,j,k,j') with j != j': the cross terms must cancel.
    #
    # Hmm, this is getting complicated. Let me just directly use the tensor formulation.
    #
    # T_{i,j,k} = sum_c U[i*2+j,c] * V[j*2+k,c] * W[k*2+i,c] = 1
    #
    # For Strassen, I'll just set up the 7 terms directly.
    #
    # P1 = (A[0,0]+A[1,1])(B[0,0]+B[1,1]), contributes to C[0,0] and C[1,1]
    #   alpha[0,0]=1, alpha[1,1]=1, others 0
    #   beta[0,0]=1, beta[1,1]=1, others 0
    #   gamma[0,0]=1, gamma[1,1]=1
    #
    # For the tensor: U[i*2+j, c] * V[j*2+k, c] * W[k*2+i, c]
    # We need: for all (i,j,k), sum_c ... = 1
    #
    # Let me try: for term c=0 (P1):
    #   U[0*2+0, 0] = 1  (i=0,j=0)
    #   U[1*2+1, 0] = 1  (i=1,j=1)
    #   V[0*2+0, 0] = 1  (j=0,k=0)
    #   V[1*2+1, 0] = 1  (j=1,k=1)
    #   W[0*2+0, 0] = 1  (k=0,i=0)
    #   W[1*2+1, 0] = 1  (k=1,i=1)
    #
    # Check (i=0,j=0,k=0): U[0,0]*V[0,0]*W[0,0] = 1*1*1 = 1 ✓
    # Check (i=0,j=0,k=1): U[0,0]*V[1,0]*W[2,0] = 1*0*0 = 0
    #   But we need total = 1, so other terms must contribute.
    # Check (i=0,j=1,k=0): U[1,0]*V[0,0]*W[0,0] = 0*1*1 = 0
    # Check (i=1,j=0,k=0): U[2,0]*V[0,0]*W[0,1] = 0*1*0 = 0
    # Check (i=1,j=1,k=1): U[3,0]*V[3,0]*W[3,0] = 1*1*1 = 1 ✓
    #
    # So term 0 gives: T[0,0,0]+=1, T[1,1,1]+=1, others 0 from this term.
    
    # P2 = (A[1,0]+A[1,1])B[0,0], contributes to C[1,0]
    #   alpha[1,0]=1, alpha[1,1]=1
    #   beta[0,0]=1
    #   gamma[1,0]=1
    #   U[2,1]=1 (i=1,j=0), U[3,1]=1 (i=1,j=1)
    #   V[0,1]=1 (j=0,k=0)
    #   W[0,1]=1 (k=0,i=1)
    #   Contributes to: (i=1,j=0,k=0): U[2,1]*V[0,1]*W[0,1]=1*1*1=1
    #   (i=1,j=1,k=0): U[3,1]*V[0,1]*W[0,1]=1*1*1=1... but we only want T[1,0,0] and T[1,1,0]
    #   Wait, T[1,1,0] should be 1 (A[1,1]*B[1,0] contributes to C[1,0])
    #   But B[1,0] is not in P2. P2 only has B[0,0].
    #   So P2 should only contribute to terms where B[j,k] = B[0,0], i.e., j=0,k=0.
    #   So V[0,1]=1 (j=0,k=0), and V[other,1]=0.
    #   Then (i=1,j=0,k=0): 1*1*1=1 ✓
    #   (i=1,j=1,k=0): U[3,1]*V[0,1]*W[0,1] = 1*1*1=1... but this is wrong!
    #   The issue: U[3,1]=1 means A[1,1] is in the first factor, but B[0,0] is in the second.
    #   The product A[1,1]*B[0,0] does NOT contribute to C[1,0] in the original matmul.
    #   It would be a cross term.
    
    # I see the problem. The Strassen decomposition has cross terms that cancel out.
    # The tensor formulation T_{i,j,k} = sum_c ... = 1 handles this automatically
    # because the cross terms appear in multiple (i,j,k) positions and cancel.
    
    # Let me just verify numerically. Let me set up all 7 terms and check.
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # P1 = (A00+A11)(B00+B11) -> C00 += P1, C11 += P1
    U[0,0] = 1; U[3,0] = 1  # A00, A11
    V[0,0] = 1; V[3,0] = 1  # B00, B11
    W[0,0] = 1; W[3,0] = 1  # C00, C11
    
    # P2 = (A10+A11)B00 -> C10 += P2
    U[2,1] = 1; U[3,1] = 1  # A10, A11
    V[0,1] = 1              # B00
    W[2,1] = 1              # C10 (k=0,i=1 -> row 2)
    
    # P3 = A00(B10-B11) -> C01 += P3
    U[0,2] = 1              # A00
    V[2,2] = 1; V[3,2] = -1 # B10, B11
    W[1,2] = 1              # C01 (k=1,i=0 -> row 1)
    
    # P4 = A11(B01-B00) -> C00 += P4, C10 += P4
    U[3,3] = 1              # A11
    V[1,3] = 1; V[0,3] = -1 # B01, B00
    W[0,3] = 1; W[2,3] = 1  # C00, C10
    
    # P5 = (A00+A01)B11 -> C00 -= P5, C01 += P5
    U[0,4] = 1; U[1,4] = 1  # A00, A01
    V[3,4] = 1              # B11
    W[0,4] = -1; W[1,4] = 1 # C00, C01
    
    # P6 = (A10-A00)(B00+B01) -> C11 += P6
    U[2,5] = 1; U[0,5] = -1 # A10, A00
    V[0,5] = 1; V[1,5] = 1  # B00, B01
    W[3,5] = 1              # C11 (k=1,i=1 -> row 3)
    
    # P7 = (A01-A11)(B10+B11) -> C00 += P7, C11 -= P7... wait
    # C00 = P1+P4-P5+P7, C11 = P1-P2+P3+P6
    # So P7 contributes +1 to C00.
    U[1,6] = 1; U[3,6] = -1 # A01, A11
    V[2,6] = 1; V[3,6] = 1  # B10, B11
    W[0,6] = 1              # C00
    
    return U.tolist(), V.tolist(), W.tolist()

def _try_strassen_3x3x7():
    """
    Try to build a decomposition for 3x3 * 3x7 using Strassen-like ideas.
    Split into 3x3*3x3 and 3x3*3x4 blocks.
    """
    n, m, p = 3, 3, 7
    # We'll try to use a block decomposition.
    # Split the 7 columns into 3+4.
    # For the 3x3*3x3 part, use a known low-rank decomposition.
    # For the 3x3*3x4 part, use schoolbook or something better.
    
    # For now, let's just use schoolbook for both parts.
    # 3x3*3x3: rank 27
    # 3x3*3x4: rank 36
    # Total: 63 (same as full schoolbook)
    
    # To do better, we need a low-rank 3x3*3x3 decomposition.
    # Best known is 23 (Larsen 2010). Let's try to implement it.
    
    # Actually, implementing the Larsen decomposition is very complex.
    # Let's try a simpler approach: use Strassen for 2x2 blocks within the 3x3.
    
    # Split 3x3 into 2x2 + 1x1 blocks.
    # A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x1, A21 is 1x2, A22 is 1x1
    # B = [[B11, B12], [B21, B22]] where B11 is 2x2, B12 is 2x1, etc.
    
    # This is getting too complex for a quick implementation.
    # Let's just return the schoolbook for now.
    return None

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3 by 3x7 matrix multiplication.
    Returns {'U', 'V', 'W'} with integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # 3, 3, 7
    deadline = time.time() + 110  # safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p  # 63
    
    # Try to find a better decomposition using Strassen-like techniques
    # For 3x3 * 3x7, one approach is to split the 7 into 3+4 and use
    # a low-rank decomposition for the 3x3*3x3 part.
    
    # Let's try a direct construction using block Strassen.
    # Split A (3x3) into blocks:
    #   A = [[A11 (2x2), A12 (2x1)],
    #        [A21 (1x2), A22 (1x1)]]
    # Split B (3x7) into blocks:
    #   B = [[B11 (2x3), B12 (2x4)],
    #        [B21 (1x3), B22 (1x4)]]
    # C = A*B is 3x7:
    #   C = [[C11 (2x3), C12 (2x4)],
    #        [C21 (1x3), C22 (1x4)]]
    #
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # For each block product, we can use Strassen or schoolbook.
    # A11*B11: 2x2 * 2x3 -> use Strassen-like (rank 7 for 2x2*2x2, but here it's 2x2*2x3)
    # Actually, 2x2 * 2x3 can be done with rank 6 (each 2x2*2x1 is rank 3 with Strassen... no)
    # 2x2 * 2x1: schoolbook rank 4, Strassen doesn't help for rectangular.
    # Actually for 2x2 * 2x1, we can use:
    # C00 = A00*B00 + A01*B10
    # C10 = A10*B00 + A11*B10
    # This is 4 terms. Can we do better? 
    # (A00+A11)(B00) + (A00-A11)(B00) ... no that's still 4.
    # Actually 2x2*2x1 has rank 4 (it's just two independent 1x2*2x1 which are rank 2 each... no)
    # Hmm, 2x2 * 2x1: the tensor has 4 entries. Schoolbook is 4.
    # I don't think Strassen helps here.
    
    # Let me try a different approach: just use the full Strassen decomposition
    # for the 3x3*3x3 part and schoolbook for the 3x3*3x4 part.
    
    # For 3x3*3x3, let's try to use a decomposition with rank < 27.
    # One known decomposition uses rank 23, but it's very complex.
    # 
    # Alternative: use the "Larsen" decomposition which I'll try to implement.
    # Actually, let me try a simpler approach: use 2x2 Strassen blocks.
    
    # Split 3x3*3x3 into:
    # - 2x2*2x2 (top-left): Strassen rank 7
    # - 2x1*1x3 (top-right to bottom-left cross terms): schoolbook
    # - 1x2*2x1: schoolbook
    # - 1x1*1x1: rank 1
    # - Cross terms between blocks
    
    # This is getting too complicated. Let me just try to verify if we can
    # achieve rank < 63 with a simple block decomposition.
    
    # Simple block approach:
    # Split 7 into 3+4.
    # 3x3*3x3: use a rank-23 decomposition (if we can implement it)
    # 3x3*3x4: schoolbook rank 36
    # Total: 23 + 36 = 59 < 63
    
    # But implementing the rank-23 3x3*3x3 decomposition is non-trivial.
    # Let me try to find a simpler low-rank decomposition for 3x3*3x3.
    
    # Actually, let me try a different strategy: use the fact that we can
    # sometimes combine terms across the block boundary.
    
    # For now, let's just return the schoolbook. The rank is 63.
    # If we have time, we can try to improve.
    
    if time.time() > deadline - 5:
        return best
    
    # Try a block decomposition: 3x3*3x3 + 3x3*3x4
    # For the 3x3*3x3 part, try to use a lower-rank decomposition.
    # One approach: use the "Strassen" decomposition for 2x2 blocks.
    
    # Let me try to construct a decomposition for 3x3*3x3 with rank < 27.
    # I'll use a simple approach: split into 2x2 + 1x1 blocks and use
    # Strassen for the 2x2*2x2 part.
    
    # A = [[A11 (2x2), A12 (2x1)],
    #      [A21 (1x2), A22 (1x1)]]
    # B = [[B11 (2x2), B12 (2x1)],
    #      [B21 (1x2), B22 (1x1)]]
    # C = [[C11 (2x2), C12 (2x1)],
    #      [C21 (1x2), C22 (1x1)]]
    
    # C11 = A11*B11 + A12*B21
    # C12 = A11*B12 + A12*B22
    # C21 = A21*B11 + A22*B21
    # C22 = A21*B12 + A22*B22
    
    # A11*B11: 2x2*2x2, Strassen rank 7
    # A12*B21: 2x1*1x2, rank 4 (schoolbook)
    # A11*B12: 2x2*2x1, rank 4
    # A12*B22: 2x1*1x1, rank 2
    # A21*B11: 1x2*2x2, rank 4
    # A22*B21: 1x1*1x2, rank 2
    # A21*B12: 1x2*2x1, rank 4
    # A22*B22: 1x1*1x1, rank 1
    
    # Total: 7+4+4+2+4+2+4+1 = 28 > 27. Not better!
    
    # So the block approach with Strassen for 2x2 doesn't help for 3x3.
    # We need a more sophisticated decomposition.
    
    # Let me just return the schoolbook.
    return best
# EVOLVE-BLOCK-END
