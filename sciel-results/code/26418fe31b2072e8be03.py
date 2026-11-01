"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

import time
import random
import itertools
import numpy as np
from fractions import Fraction

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Shapes: U (n*m, rank), V (m*p, rank), W (p*n, rank)
    # Tensor T[i, j, k] = 1 for all i,j,k
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]

    def schoolbook():
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
        return {"U": U, "V": V, "W": W}, rank

    best_result, best_rank = schoolbook()

    def check_decomposition(U, V, W):
        """Check if the decomposition is exact."""
        r = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for rr in range(r):
                        s += U[i*m+j][rr] * V[j*p+k][rr] * W[k*n+i][rr]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    # Strassen rank-7 decomposition for 2x2x2
    # (A11 A12; A21 A22) * (B11 B12; B21 B22) = (C11 C12; C21 C22)
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6

    def strassen_2x2x2():
        """Return the 7 triples (U_idx, V_idx, W_idx) for 2x2x2 Strassen."""
        # Indices: U is 2x2 -> (0,1,2,3) for (0,0),(0,1),(1,0),(1,1)
        # V is 2x2 -> (0,1,2,3) for (0,0),(0,1),(1,0),(1,1)
        # W is 2x2 -> (0,1,2,3) for (0,0),(0,1),(1,0),(1,1)
        # 
        # Each term M_r contributes to C_ij = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
        # Actually, for 2x2x2:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # Decomposition: C_{i,k} = sum_r U_{i,k}^{(r)} * V_{j}^{(r)} ... 
        # Wait, the tensor is T[i,j,k] where i=input row, j=inner, k=output col
        # For 2x2x2: i in {0,1}, j in {0,1}, k in {0,1}
        # T[i,j,k] = 1 if the (i,j) of A times (j,k) of B contributes to (i,k) of C
        # So the tensor entries are T[i,j,k] = 1 for all i,j,k in {0,1}
        # 
        # Strassen gives: C_{i,k} = sum_r U_{i,k}^{(r)} * (linear combo of A entries) * (linear combo of B entries)
        # But our decomposition is: T[i,j,k] = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
        # This is different! The tensor is over (i,j,k) not (i,k) with j as inner.
        #
        # Actually, the matrix multiplication tensor M_{n,m,p} has entries M[i,j,k] = 1 for all i,j,k
        # and the decomposition is M[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Wait no, the standard tensor decomposition of matrix multiplication is:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # The tensor is T[i,j,k] = delta (it's just 1 for all valid indices)
        # And the decomposition is: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Hmm, that doesn't look right dimensionally. Let me re-read the task.
        #
        # "output factor follows the official trace convention W[k*n+i,r]"
        # So W is indexed by (k*n+i, r) where k is output col, i is input row
        # U is indexed by (i*m+j, r) where i is input row, j is inner index
        # V is indexed by (j*p+k, r) where j is inner index, k is output col
        #
        # The tensor equation is: for all i,j,k:
        # 1 = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        #
        # This is the standard matrix multiplication tensor where the "trace" convention
        # means W depends on (k,i) not (i,k). Let's verify with schoolbook:
        # U[i*m+j, col] = 1, V[j*p+k, col] = 1, W[k*n+i, col] = 1
        # Product = 1*1*1 = 1. Sum over r gives 1. Correct.
        #
        # Now for Strassen on 2x2x2:
        # The 7 terms are:
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        # M2 = (A10+A11)B00 -> contributes to C10
        # M3 = A00(B01-B11) -> contributes to C01
        # M4 = A11(B10-B00) -> contributes to C10
        # M5 = (A00+A01)B11 -> contributes to C00
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        # M7 = (A01-A11)(B10+B11) -> contributes to C11
        #
        # In our tensor notation for 2x2x2 (i,j,k all in {0,1}):
        # T[i,j,k] = 1 for all i,j,k
        # We need: 1 = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        #
        # For M1: (A00+A11)(B00+B11)
        # This term appears in C00 = M1+M4-M5+M7 and C11 = M1-M2+M3+M6
        # But how does it map to our tensor?
        #
        # Actually, let me think differently. The Strassen decomposition says:
        # C_{i,k} = sum_r M_r * (coeff for C_{i,k})
        # where M_r = (linear combo of A) * (linear combo of B)
        #
        # In our tensor form, each "term" r has:
        # U[i,j,r] = coefficient of A_{i,j} in the linear combo for M_r
        # V[j,k,r] = coefficient of B_{j,k} in the linear combo for M_r
        # W[k,i,r] = coefficient of C_{i,k} that M_r contributes to
        #
        # Wait, that's not quite right either. Let me think about it as:
        # C_{i,k} = sum_j A_{i,j} B_{j,k}
        # Strassen: C_{i,k} = sum_r U_{i,k}^{(r)} * M_r
        # where M_r = (sum_{i',j'} alpha_{i',j'}^{(r)} A_{i',j'}) * (sum_{j',k'} beta_{j',k'}^{(r)} B_{j',k'})
        #
        # Expanding: C_{i,k} = sum_r U_{i,k}^{(r)} * (sum_{i',j'} alpha_{i',j'}^{(r)} A_{i',j'}) * (sum_{j',k'} beta_{j',k'}^{(r)} B_{j',k'})
        #
        # But we need: C_{i,k} = sum_j A_{i,j} B_{j,k}
        #
        # So the tensor equation is:
        # delta_{i,k} (as a function of the multiplication) ... 
        #
        # I think the cleanest way is: the tensor T has entries T[i,j,k] = 1
        # and we want T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        #
        # For Strassen 2x2x2, the 7 terms are:
        # Term 1: A_part = A00+A11, B_part = B00+B11, C_part = C00+C11
        #   U[0,0,1]=1, U[1,1,1]=1, others 0
        #   V[0,0,1]=1, V[1,1,1]=1, others 0
        #   W[0,0,1]=1, W[1,1,1]=1, others 0
        #   Check: sum over r=1: U[i,j,1]*V[j,k,1]*W[k,i,1]
        #   For (i,j,k)=(0,0,0): U[0,0,1]*V[0,0,1]*W[0,0,1] = 1*1*1 = 1
        #   For (i,j,k)=(0,1,0): U[0,1,1]*V[1,0,1]*W[0,0,1] = 0*0*1 = 0
        #   For (i,j,k)=(1,1,1): U[1,1,1]*V[1,1,1]*W[1,1,1] = 1*1*1 = 1
        #   But we need T[i,j,k]=1 for ALL i,j,k, not just diagonal!
        #
        # I see the issue. The Strassen decomposition is for the BILINEAR form, not the trilinear tensor.
        # The tensor T[i,j,k] = 1 for all i,j,k means that every (i,j,k) triple must be covered.
        #
        # Let me reconsider. The standard matrix multiplication tensor of size n x m x p has:
        # T[i,j,k] = 1 for all i in [n], j in [m], k in [p]
        # and rank is the minimum r such that T[i,j,k] = sum_{r=1}^r U[i,j,r] * V[j,k,r] * W[k,i,r]
        #
        # For 2x2x2, the schoolbook rank is 8 (one term per (i,j,k)).
        # Strassen shows that the BILINEAR complexity is 7, but that's for a different tensor structure.
        #
        # Actually, I think I was confusing two things. Let me look at this more carefully.
        #
        # The matrix multiplication tensor M_{n,m,p} is defined by:
        # M[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to (i,k) entry of C
        # In the standard convention, M[i,j,k] = 1 for all valid i,j,k.
        #
        # The Strassen algorithm shows that 2x2 matrix multiplication can be done with 7 multiplications.
        # This means the tensor M_{2,2,2} has rank 7 (or less).
        #
        # The 7 terms for Strassen 2x2x2 (using the trace convention W[k*n+i]):
        # Let me derive them properly.
        #
        # We want: for all i,j,k in {0,1}: 1 = sum_{r=1}^7 U[i,j,r] * V[j,k,r] * W[k,i,r]
        #
        # Strassen's 7 multiplications:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # Now, each M_r is a product of a linear combo of A entries and a linear combo of B entries.
        # M_r = (sum_{i,j} alpha_{i,j}^{(r)} A_{i,j}) * (sum_{j,k} beta_{j,k}^{(r)} B_{j,k})
        #
        # And C_{i,k} = sum_r gamma_{i,k}^{(r)} M_r
        #
        # In our tensor form:
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        #
        # Matching: U[i,j,r] = alpha_{i,j}^{(r)}, V[j,k,r] = beta_{j,k}^{(r)}, W[k,i,r] = gamma_{i,k}^{(r)}
        # Note the trace convention: W is indexed by (k,i) not (i,k).
        #
        # Let's write out the 7 terms:
        # r=1: M1 = (A00+A11)(B00+B11)
        #   alpha: (0,0)->1, (1,1)->1
        #   beta: (0,0)->1, (1,1)->1
        #   gamma: (0,0)->1, (1,1)->1
        #   U[0,0,1]=1, U[1,1,1]=1
        #   V[0,0,1]=1, V[1,1,1]=1
        #   W[0,0,1]=1, W[1,1,1]=1
        #
        # r=2: M2 = (A10+A11)B00
        #   alpha: (1,0)->1, (1,1)->1
        #   beta: (0,0)->1
        #   gamma: (1,0)->1
        #   U[1,0,2]=1, U[1,1,2]=1
        #   V[0,0,2]=1
        #   W[0,1,2]=1  (k=0, i=1)
        #
        # r=3: M3 = A00(B01-B11)
        #   alpha: (0,0)->1
        #   beta: (0,1)->1, (1,1)->-1
        #   gamma: (0,1)->1
        #   U[0,0,3]=1
        #   V[0,1,3]=1, V[1,1,3]=-1
        #   W[1,0,3]=1  (k=1, i=0)
        #
        # r=4: M4 = A11(B10-B00)
        #   alpha: (1,1)->1
        #   beta: (1,0)->1, (0,0)->-1
        #   gamma: (0,0)->1, (1,0)->1
        #   U[1,1,4]=1
        #   V[1,0,4]=1, V[0,0,4]=-1
        #   W[0,0,4]=1, W[0,1,4]=1
        #
        # r=5: M5 = (A00+A01)B11
        #   alpha: (0,0)->1, (0,1)->1
        #   beta: (1,1)->1
        #   gamma: (0,0)->-1, (0,1)->1
        #   U[0,0,5]=1, U[0,1,5]=1
        #   V[1,1,5]=1
        #   W[0,0,5]=-1, W[1,0,5]=1
        #
        # r=6: M6 = (A10-A00)(B00+B01)
        #   alpha: (1,0)->1, (0,0)->-1
        #   beta: (0,0)->1, (0,1)->1
        #   gamma: (1,1)->1
        #   U[1,0,6]=1, U[0,0,6]=-1
        #   V[0,0,6]=1, V[0,1,6]=1
        #   W[1,1,6]=1  (k=1, i=1)
        #
        # r=7: M7 = (A01-A11)(B10+B11)
        #   alpha: (0,1)->1, (1,1)->-1
        #   beta: (1,0)->1, (1,1)->1
        #   gamma: (0,0)->1, (1,1)->1
        #   U[0,1,7]=1, U[1,1,7]=-1
        #   V[1,0,7]=1, V[1,1,7]=1
        #   W[0,0,7]=1, W[1,1,7]=1
        #
        # Let's verify for (i,j,k) = (0,0,0):
        # r=1: U[0,0,1]*V[0,0,1]*W[0,0,1] = 1*1*1 = 1
        # r=2: U[0,0,2]*V[0,0,2]*W[0,0,2] = 0*1*0 = 0
        # r=3: U[0,0,3]*V[0,0,3]*W[0,0,3] = 1*0*0 = 0
        # r=4: U[0,0,4]*V[0,0,4]*W[0,0,4] = 0*(-1)*1 = 0
        # r=5: U[0,0,5]*V[0,0,5]*W[0,0,5] = 1*0*(-1) = 0
        # r=6: U[0,0,6]*V[0,0,6]*W[0,0,6] = -1*1*0 = 0
        # r=7: U[0,0,7]*V[0,0,7]*W[0,0,7] = 0*0*1 = 0
        # Sum = 1. Correct!
        #
        # Let's verify for (i,j,k) = (0,1,0):
        # r=1: U[0,1,1]*V[1,0,1]*W[0,0,1] = 0*0*1 = 0
        # r=2: U[0,1,2]*V[1,0,2]*W[0,0,2] = 0*0*0 = 0
        # r=3: U[0,1,3]*V[1,0,3]*W[0,0,3] = 0*(-1)*0 = 0
        # r=4: U[0,1,4]*V[1,0,4]*W[0,0,4] = 0*1*1 = 0
        # r=5: U[0,1,5]*V[1,0,5]*W[0,0,5] = 1*1*(-1) = -1
        # r=6: U[0,1,6]*V[1,0,6]*W[0,0,6] = 0*0*0 = 0
        # r=7: U[0,1,7]*V[1,0,7]*W[0,0,7] = 1*1*1 = 1
        # Sum = -1 + 1 = 0. But we need 1!
        #
        # Hmm, that's wrong. Let me recheck.
        #
        # Wait, I think I made an error in the gamma coefficients. Let me re-derive.
        #
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        #
        # So gamma_{i,k}^{(r)} is the coefficient of M_r in C_{i,k}:
        # gamma[0,0] = [1, 0, 0, 1, -1, 0, 1]
        # gamma[0,1] = [0, 0, 1, 0, 1, 0, 0]
        # gamma[1,0] = [0, 1, 0, 1, 0, 0, 0]
        # gamma[1,1] = [1, -1, 1, 0, 0, 1, 0]
        #
        # With trace convention W[k,i,r] = gamma[i,k,r]:
        # W[0,0,r] = gamma[0,0,r]
        # W[1,0,r] = gamma[0,1,r]
        # W[0,1,r] = gamma[1,0,r]
        # W[1,1,r] = gamma[1,1,r]
        #
        # Let me redo the verification for (i,j,k) = (0,1,0):
        # We need sum_r U[0,1,r] * V[1,0,r] * W[0,0,r] = 1
        #
        # U[0,1,r]: r=1:0, r=2:0, r=3:0, r=4:0, r=5:1, r=6:0, r=7:1
        # V[1,0,r]: r=1:0, r=2:0, r=3:-1, r=4:1, r=5:0, r=6:0, r=7:1
        # W[0,0,r] = gamma[0,0,r]: r=1:1, r=2:0, r=3:0, r=4:1, r=5:-1, r=6:0, r=7:1
        #
        # r=1: 0*0*1 = 0
        # r=2: 0*0*0 = 0
        # r=3: 0*(-1)*0 = 0
        # r=4: 0*1*1 = 0
        # r=5: 1*0*(-1) = 0
        # r=6: 0*0*0 = 0
        # r=7: 1*1*1 = 1
        # Sum = 1. Correct!
        #
        # I made an arithmetic error before. Let me continue with the full derivation.

    def strassen_2x2x2_terms():
        """Return list of 7 tuples (U_dict, V_dict, W_dict) for 2x2x2 Strassen."""
        terms = []
        
        # r=1: M1 = (A00+A11)(B00+B11), C: C00+=1, C11+=1
        U = {(0,0):1, (1,1):1}
        V = {(0,0):1, (1,1):1}
        W = {(0,0):1, (1,1):1}  # W[k,i]: (0,0)->C00, (1,1)->C11
        terms.append((U, V, W))
        
        # r=2: M2 = (A10+A11)B00, C: C10+=1
        U = {(1,0):1, (1,1):1}
        V = {(0,0):1}
        W = {(0,1):1}  # W[0,1] -> C[1,0]
        terms.append((U, V, W))
        
        # r=3: M3 = A00(B01-B11), C: C01+=1
        U = {(0,0):1}
        V = {(0,1):1, (1,1):-1}
        W = {(1,0):1}  # W[1,0] -> C[0,1]
        terms.append((U, V, W))
        
        # r=4: M4 = A11(B10-B00), C: C00+=1, C10+=1
        U = {(1,1):1}
        V = {(1,0):1, (0,0):-1}
        W = {(0,0):1, (0,1):1}  # W[0,0]->C00, W[0,1]->C10
        terms.append((U, V, W))
        
        # r=5: M5 = (A00+A01)B11, C: C00-=1, C01+=1
        U = {(0,0):1, (0,1):1}
        V = {(1,1):1}
        W = {(0,0):-1, (1,0):1}  # W[0,0]->C00, W[1,0]->C01
        terms.append((U, V, W))
        
        # r=6: M6 = (A10-A00)(B00+B01), C: C11+=1
        U = {(1,0):1, (0,0):-1}
        V = {(0,0):1, (0,1):1}
        W = {(1,1):1}  # W[1,1] -> C[1,1]
        terms.append((U, V, W))
        
        # r=7: M7 = (A01-A11)(B10+B11), C: C00+=1, C11+=1
        U = {(0,1):1, (1,1):-1}
        V = {(1,0):1, (1,1):1}
        W = {(0,0):1, (1,1):1}  # W[0,0]->C00, W[1,1]->C11
        terms.append((U, V, W))
        
        return terms

    def build_block_decomposition():
        """
        Build a block-structured decomposition using Strassen for 2x2x2 sub-blocks.
        Split m=4 into 2 blocks of size 2, p=6 into 3 blocks of size 2.
        For each pair (block_j, block_k), use Strassen 2x2x2 with n=2.
        Total rank = 2 * 3 * 7 = 42.
        """
        n, m, p = DIMENSIONS
        # m split into m_blocks blocks of size m_block
        # p split into p_blocks blocks of size p_block
        m_block = 2
        p_block = 2
        m_blocks = m // m_block  # 2
        p_blocks = p // p_block  # 3
        
        rank = m_blocks * p_blocks * 7
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        strassen_terms = strassen_2x2x2_terms()
        
        col = 0
        for jb in range(m_blocks):
            for kb in range(p_blocks):
                for (U_dict, V_dict, W_dict) in strassen_terms:
                    # U_dict keys: (i, j_local) where i in {0,1}, j_local in {0,1}
                    # Map to global: i_global = i, j_global = jb * m_block + j_local
                    for (i, j_local), val in U_dict.items():
                        U[i * m + (jb * m_block + j_local)][col] = val
                    
                    # V_dict keys: (j_local, k_local) where j_local in {0,1}, k_local in {0,1}
                    # Map to global: j_global = jb * m_block + j_local, k_global = kb * p_block + k_local
                    for (j_local, k_local), val in V_dict.items():
                        V[(jb * m_block + j_local) * p + (kb * p_block + k_local)][col] = val
                    
                    # W_dict keys: (k_local, i) where k_local in {0,1}, i in {0,1}
                    # Map to global: k_global = kb * p_block + k_local, i_global = i
                    for (k_local, i), val in W_dict.items():
                        W[(kb * p_block + k_local) * n + i][col] = val
                    
                    col += 1
        
        return {"U": U, "V": V, "W": W}

    # Try block decomposition
    if time.time() < deadline - 20:
        block_decomp = build_block_decomposition()
        if check_decomposition(block_decomp["U"], block_decomp["V"], block_decomp["W"]):
            best_result = block_decomp
            best_rank = len(block_decomp["U"][0])
            print(f"Block decomposition found with rank {best_rank}")
    
    # Try to reduce rank further using linear algebra
    # The idea: if we have a valid decomposition of rank r, we might be able to
    # find a lower rank one by looking for linear dependencies in the columns.
    # This is a hard problem in general, but for small ranks we can try.
    
    def try_reduce_rank(decomp, target_rank, time_limit=30.0):
        """Try to reduce the rank of a decomposition to target_rank."""
        start = time.time()
        U = np.array(decomp["U"], dtype=float)
        V = np.array(decomp["V"], dtype=float)
        W = np.array(decomp["W"], dtype=float)
        
        r = U.shape[1]
        
        # Check if we can find a lower rank decomposition by solving a system
        # This is a non-convex problem, so we'll use a simple approach:
        # Try random projections and see if we can maintain validity.
        
        # For now, just return the original if we can't find something better
        return decomp

    # Return the best result found
    if time.time() > deadline - 10:
        return best_result
    
    return best_result

# EVOLVE-BLOCK-END
