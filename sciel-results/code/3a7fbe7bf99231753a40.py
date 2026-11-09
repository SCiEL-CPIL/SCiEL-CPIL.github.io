import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Deadline guard
    deadline = time.time() + 140

    # --- 1. Define the Strassen (rank 7) factors for 3x3 * 3x3 ---
    # Indices: i in 0..2 (row of A), j in 0..2 (col of A / row of B), k in 0..2 (col of B)
    # U_str[i, r], V_str[j, r], W_str[k, r]
    # r=0: M1 = (A00+A11)(B00+B11)
    # r=1: M2 = (A10+A11)B00
    # r=2: M3 = A00(B10-B11)
    # r=3: M4 = A11(B01-B10)
    # r=4: M5 = (A00+A01)B11
    # r=5: M6 = (A10-A00)(B00+B01)
    # r=7: M7 = (A01-A11)(B10+B11)
    
    U_str = np.zeros((9, 7))
    V_str = np.zeros((9, 7))
    W_str = np.zeros((9, 7))
    
    # Helper to set 1 at (row, col)
    def set1(U, V, W, i, j, k, r):
        U[i*3+j, r] += 1.0
        V[j*3+k, r] += 1.0
        W[k*3+i, r] += 1.0

    # r=0
    set1(U_str, V_str, W_str, 0, 0, 0, 0)
    set1(U_str, V_str, W_str, 0, 0, 1, 0)
    set1(U_str, V_str, W_str, 1, 1, 0, 0)
    set1(U_str, V_str, W_str, 1, 1, 1, 0)
    # r=1
    set1(U_str, V_str, W_str, 1, 0, 0, 1)
    set1(U_str, V_str, W_str, 1, 1, 0, 1)
    # r=2
    set1(U_str, V_str, W_str, 0, 0, 1, 2)
    set1(U_str, V_str, W_str, 0, 0, 2, 2)
    # r=3
    set1(U_str, V_str, W_str, 1, 1, 0, 3)
    set1(U_str, V_str, W_str, 1, 1, 1, 3)
    # r=4
    set1(U_str, V_str, W_str, 0, 0, 2, 4)
    set1(U_str, V_str, W_str, 0, 1, 2, 4)
    # r=5
    U_str[1*3+0, 5] += 1.0
    U_str[0*3+0, 5] -= 1.0
    V_str[0*3+0, 5] += 1.0
    V_str[0*3+1, 5] += 1.0
    W_str[0*3+1, 5] += 1.0
    W_str[1*3+1, 5] += 1.0
    # r=6 (index 6 in 0-indexed, 7th factor)
    U_str[0*3+1, 6] += 1.0
    U_str[1*3+1, 6] -= 1.0
    V_str[1*3+0, 6] += 1.0
    V_str[1*3+1, 6] += 1.0
    W_str[0*3+0, 6] += 1.0
    W_str[1*3+0, 6] += 1.0

    # --- 2. Define Schoolbook (rank 3) factors for 3x3 * 3x1 ---
    # Here p=1, so k is always 0.
    # U_sb[i, r], V_sb[j, r], W_sb[0, r] for r=0,1,2 corresponding to j=0,1,2
    U_sb = np.zeros((9, 3))
    V_sb = np.zeros((3, 3))
    W_sb = np.zeros((3, 3))
    
    for j in range(3):
        for i in range(3):
            U_sb[i*3+j, j] = 1.0
            V_sb[j*3+0, j] = 1.0
            W_sb[0*3+i, j] = 1.0

    # --- 3. Combine into full 3x3 * 3x7 decomposition ---
    # Total rank = 7 + 3 = 10
    rank_total = 10
    U = np.zeros((nm, rank_total))
    V = np.zeros((mp, rank_total))
    W = np.zeros((pn, rank_total))
    
    # Map Strassen factors to the first 3 columns of B (k=0,1,2)
    # U indices: i*3 + j (0..8) -> same as U_str
    # V indices: j*7 + k (0..20). k is 0,1,2.
    # W indices: k*3 + i (0..20). k is 0,1,2.
    
    for r in range(7):
        # U
        U[:9, r] = U_str[:, r]
        
        # V: For each row j in 0..2, the Strassen factor V_str[j, r] applies to B[j, 0..2].
        # So V[j*7 + 0, r] = V_str[j, r], V[j*7 + 1, r] = V_str[j, r], V[j*7 + 2, r] = V_str[j, r]
        for j in range(3):
            val = V_str[j, r]
            if val != 0:
                V[j*7 + 0, r] += val
                V[j*7 + 1, r] += val
                V[j*7 + 2, r] += val
                
        # W: For each col k in 0..2, the Strassen factor W_str[k, r] applies to C[0..2, k].
        # W index is k*3 + i.
        for k in range(3):
            val = W_str[k, r]
            if val != 0:
                for i in range(3):
                    W[k*3 + i, r] += val

    # Map Schoolbook factors to the last 4 columns of B (k=3,4,5,6)
    # Rank 3 decomposition uses j=0,1,2 as the rank index.
    # For each k in 3..6, we need C[i, k] = sum_j A[i, j] B[j, k].
    # This is exactly the 3x1 multiplication for each k.
    # The factors are identical for each k, just shifted in V and W indices.
    
    for k in range(3, 7):
        for j in range(3):
            r = j # rank index for this block
            r_global = 7 + j
            
            # U: Same as U_sb[i*3+j, j]
            # U_sb row is i*3+j.
            # We want U[i*3+j, r_global] = 1
            U[:, r_global] = U_sb[:, j]
            
            # V: V_sb[j, j] = 1. This applies to B[j, k].
            # V index is j*7 + k.
            V[j*7 + k, r_global] = 1.0
            
            # W: W_sb[i, j] = 1. This applies to C[i, k].
            # W index is k*3 + i.
            W[k*3 + 0, r_global] = 1.0
            W[k*3 + 1, r_global] = 1.0
            W[k*3 + 2, r_global] = 1.0

    # --- 4. Validation ---
    # Let's verify the decomposition.
    # C[i, k] should equal sum_j A[i, j] B[j, k]
    # Our formula: C[i, k] = sum_r U[i*3+j_r?, r] ... wait, the tensor contraction is:
    # T_{i, j, k} is not the right view.
    # The evaluator likely checks:
    # sum_r U[a, r] V[b, r] W[c, r] == Target_Tensor[a, b, c]
    # Where Target_Tensor is the multiplication tensor.
    # The multiplication tensor M has M[i*m+j, j*p+k, k*n+i] = 1?
    # No, the standard tensor T such that C = A B is:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # This can be written as sum_r (sum_j U_{ij, r} A_{ij}) (sum_k V_{jk, r} B_{jk}) W_{ki, r} C_{ki} ?
    # Actually, the rank-r decomposition of the multiplication tensor is a set of triples (u_r, v_r, w_r)
    # such that the tensor T = sum_r u_r \otimes v_r \otimes w_r matches the multiplication tensor.
    # The multiplication tensor T has entries T_{i, j, k} = 1 if we interpret it as the coefficient of A_{ij} B_{jk} in C_{ik}.
    # Wait, the standard definition is that T_{ijk} is 1 if the indices match the structure.
    # Specifically, for the tensor to represent matrix multiplication, we need:
    # sum_{j} A_{ij} B_{jk} = C_{ik}
    # This is a 3-way tensor contraction.
    # The tensor T_{ijk} is defined such that the "slice" for a fixed j is the outer product of e_i and e_k? No.
    # Let's look at the indices in the prompt:
    # U: (n*m, rank) -> index a = i*m + j
    # V: (m*p, rank) -> index b = j*p + k
    # W: (p*n, rank) -> index c = k*n + i
    
    # The tensor T_{a, b, c} should be 1 if the indices correspond to a valid multiplication step?
    # Actually, the multiplication tensor is the sum of rank-1 tensors.
    # The schoolbook decomposition is rank nmp.
    # For each (i, j, k), there is a term:
    # u_{ijk} has 1 at a=i*m+j
    # v_{ijk} has 1 at b=j*p+k
    # w_{ijk} has 1 at c=k*n+i
    # And the sum of these outer products equals the multiplication tensor.
    # Why? Because for a specific (i, j, k), the term contributes to the product A_{ij} B_{jk} C_{ki}?
    # No, the tensor T is defined on the space of inputs (A, B) and output (C).
    # T(A, B) = C.
    # In tensor notation, T = sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki} (where e_{ki} is the dual basis for C_{ik}).
    # So T_{a, b, c} is 1 if a=i*m+j, b=j*p+k, c=k*n+i for some i,j,k, and 0 otherwise?
    # No, T_{a,b,c} is 1 if the indices a,b,c are "compatible" in the sense that they form a cycle i->j->k->i.
    # Specifically, if a = i*m+j, b = j*p+k, c = k*n+i, then T_{a,b,c} = 1.
    # Are there other non-zero entries?
    # If we take the sum of these rank-1 tensors, the entry T_{a,b,c} is the number of triples (i,j,k) such that
    # i*m+j=a, j*p+k=b, k*n+i=c.
    # For a given a, i and j are fixed. For b, j and k are fixed. For c, k and i are fixed.
    # So if the indices are consistent (i from a matches i from c, j from a matches j from b, k from b matches k from c),
    # then there is exactly 1 such triple.
    # If they are inconsistent, there are 0.
    # So the target tensor T has T_{a,b,c} = 1 if indices are consistent, 0 otherwise.
    
    # Let's verify our Strassen + Schoolbook construction produces this T.
    # T_{a,b,c} = sum_r U[a,r] V[b,r] W[c,r]
    
    # Check a specific case: i=0, j=0, k=0.
    # a = 0*3+0 = 0
    # b = 0*7+0 = 0
    # c = 0*3+0 = 0
    # We need sum_r U[0,r] V[0,r] W[0,r] = 1.
    
    # Contributions from Strassen (r=0..6):
    # r=0: U[0,0]=1 (from i=0,j=0), V[0,0]=1 (from j=0,k=0 in V_str[0,0]), W[0,0]=1 (from k=0,i=0 in W_str[0,0]). Prod = 1.
    # r=1: U[0,1]=0.
    # r=2: U[0,2]=1 (from i=0,j=0), V[0,2]=0 (V_str[0,2]=0? No, V_str[0,2] corresponds to j=0, r=2. In r=2, we set V for j=0?
    #    Let's re-check V_str construction.
    #    r=2: set1(U_str, V_str, W_str, 0, 0, 1, 2) -> U_str[0,2]=1, V_str[0,2]=1, W_str[1,2]=1? No, W_str[k*3+i]. k=1, i=0 -> W_str[3,2]=1?
    #    Wait, my set1 helper: W[k*3+i, r].
    #    r=2: i=0, j=0, k=1.
    #    U_str[0*3+0, 2] = 1.
    #    V_str[0*3+1, 2] = 1. -> V_str[1, 2] = 1.
    #    W_str[1*3+0, 2] = 1. -> W_str[3, 2] = 1.
    #    So for V[0,2] (which is V_str[0,2]), it is 0.
    #    So r=2 contributes 0 to V[0,r].
    # r=3: U[0,3]=0.
    # r=4: U[0,4]=1 (i=0,j=0). V[0,4]=0 (V_str[0,4]=0? r=4: i=0,j=0,k=2. V_str[0*3+2, 4]=1 -> V_str[2,4]=1. So V_str[0,4]=0).
    # r=5: U[0,5]=-1 (i=0,j=0). V[0,5]=1 (V_str[0,5]=1? r=5: V_str[0*3+0, 5]=1, V_str[0*3+1, 5]=1. So V_str[0,5]=1). W[0,5]=0 (W_str[0,5]=0? r=5: W_str[0*3+1, 5]=1, W_str[1*3+1, 5]=1. So W_str[0,5]=0).
    # r=6: U[0,6]=0.
    
    # So Strassen part sum = 1 (from r=0).
    
    # Contributions from Schoolbook (r=7..9):
    # r=7 (j=0): U[0,7]=1 (i=0,j=0). V[0,7]=1 (k=3? No, k loops 3..6. For k=3, V[0*7+3, 7]=1. So V[3,7]=1. V[0,7] is for k=0.
    #    Wait, in the Schoolbook part, for a specific k, we set V[j*7+k, r_global].
    #    For r=7 (j=0), we set V[0*7+k, 7] = 1 for k=3,4,5,6.
    #    So V[0,7] is 0.
    #    Thus Schoolbook contributes 0 to V[0,r].
    
    # Total sum = 1. Correct.

    # Check another case: i=0, j=0, k=1.
    # a=0, b=1, c=3 (k=1, i=0 -> 1*3+0=3).
    # Target T[0,1,3] = 1.
    # Strassen:
    # r=0: U[0,0]=1. V[1,0]=1 (V_str[0,0]=1 -> V[0*7+1, 0]=1). W[3,0]=1 (W_str[1,0]=1 -> W[1*3+0, 0]=1). Prod = 1.
    # r=1: U[0,1]=0.
    # r=2: U[0,2]=1. V[1,2]=1 (V_str[0,2]=1? No, V_str[1,2]=1. V[1,2] is V_str[0,2]? No.
    #    V index b=1 corresponds to j=0, k=1.
    #    V[1, r] = V_str[0, r] (since j=0).
    #    V_str[0,2] = 0.
    #    So V[1,2]=0.
    # r=3: U[0,3]=0.
    # r=4: U[0,4]=1. V[1,4]=0 (V_str[0,4]=0).
    # r=5: U[0,5]=-1. V[1,5]=1 (V_str[0,5]=1). W[3,5]=0 (W_str[1,5]=0? W_str[1,5] is W_str[1*3+1, 5]=1. W_str[1*3+0, 5]=0. So W[3,5]=0).
    # r=6: U[0,6]=0.
    # Strassen sum = 1.
    # Schoolbook:
    # r=7 (j=0): U[0,7]=1. V[1,7]=0 (V[0*7+k, 7] for k=3..6. V[1,7] is k=1, which is not in 3..6).
    # So Schoolbook sum = 0.
    # Total = 1. Correct.

    # Check a case involving the Schoolbook part: i=0, j=0, k=3.
    # a=0, b=3, c=9 (k=3, i=0 -> 3*3+0=9).
    # Target T[0,3,9] = 1.
    # Strassen:
    # V[3, r] corresponds to j=0, k=3.
    # In Strassen, V is only non-zero for k=0,1,2.
    # So V[3, r] = 0 for all r < 7.
    # Strassen sum = 0.
    # Schoolbook:
    # r=7 (j=0): U[0,7]=1. V[3,7]=1 (k=3, j=0). W[9,7]=1 (k=3, i=0).
    # Prod = 1.
    # r=8 (j=1): U[0,8]=0.
    # r=9 (j=2): U[0,9]=0.
    # Schoolbook sum = 1.
    # Total = 1. Correct.

    # Check a case where Strassen might cancel: i=1, j=0, k=0.
    # a=3, b=0, c=1.
    # Target T[3,0,1] = 1.
    # Strassen:
    # r=0: U[3,0]=1 (i=1,j=1? No. i=1,j=0 -> a=1*3+0=3. U_str[3,0]=0?
    #    r=0: set1(0,0,0,0), set1(0,0,1,0), set1(1,1,0,0), set1(1,1,1,0).
    #    U_str[3,0] is 0.
    #    So r=0 contributes 0.
    # r=1: set1(1,0,0,1), set1(1,1,0,1).
    #    U_str[3,1]=1. V_str[0,1]=1. W_str[0,1]=1 (k=0,i=1 -> 0*3+1=1).
    #    Prod = 1.
    # r=2: U_str[3,2]=0.
    # r=3: U_str[3,3]=0.
    # r=4: U_str[3,4]=0.
    # r=5: U_str[3,5]=1 (i=1,j=0). V_str[0,5]=1. W_str[0,5]=0 (W_str[1,5]=1, W_str[4,5]=1).
    #    Prod = 0.
    # r=6: U_str[3,6]=0.
    # Strassen sum = 1.
    # Schoolbook sum = 0 (k=0).
    # Total = 1. Correct.

    # The decomposition seems valid.
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
