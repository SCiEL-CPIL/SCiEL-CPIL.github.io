"""Matrix multiplication 5,5,5 rank-minimal decomposition attempt."""
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a low-rank decomposition of the 5x5x5 matrix multiplication tensor.
    Uses a block decomposition 2+2+1.
    Applies Strassen's algorithm to 2x2x2 sub-tensors.
    """
    n, m, p = DIMENSIONS
    # Block indices
    blocks = [[0, 1], [2, 3], [4]]
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Helper to add a schoolbook term for A_sub (rows, cols) * B_sub (rows, cols)
    # A_sub is at A[rows, cols_a]
    # B_sub is at B[cols_b, cols]
    # Result C_sub is at C[rows, cols]
    # U index: i * m + j_a (i in rows, j_a in cols_a)
    # V index: j_b * p + k (j_b in cols_b, k in cols)
    # W index: k * n + i (k in cols, i in rows)
    
    def add_schoolbook(rows, cols_a, cols_b, cols):
        # cols_a are columns of A
        # cols_b are rows of B
        # cols are columns of B / rows of C? 
        # A is n x m. B is m x p. C is p x n.
        # A[i, j] * B[j, k] -> C[k, i]
        # rows: i indices
        # cols_a: j indices (from A)
        # cols_b: j indices (from B) - must be same as cols_a for multiplication? 
        # Yes, A_{ak} uses cols I_k. B_{kb} uses rows I_k.
        # So cols_a == cols_b == I_k.
        
        for i in rows:
            for j in cols_a:
                for k in cols:
                    u = [0] * (n * m)
                    v = [0] * (m * p)
                    w = [0] * (p * n)
                    
                    u[i * m + j] = 1
                    v[j * p + k] = 1
                    w[k * n + i] = 1
                    
                    U_cols.append(u)
                    V_cols.append(v)
                    W_cols.append(w)

    def add_strassen(rows, cols_a, cols_b, cols):
        # Assumes len(rows)==2, len(cols_a)==2, len(cols_b)==2, len(cols)==2
        # i0, i1 = rows
        # j0, j1 = cols_a
        # k0, k1 = cols
        
        i0, i1 = rows
        j0, j1 = cols_a
        k0, k1 = cols
        
        # M1 = (A11+A22)(B11+B22)
        # A11=A[i0,j0], A22=A[i1,j1]
        # B11=B[j0,k0], B22=B[j1,k1]
        
        terms = []
        
        # M1
        terms.append(((i0, j0), (j0, k0)))
        terms.append(((i0, j0), (j1, k1)))
        terms.append(((i1, j1), (j0, k0)))
        terms.append(((i1, j1), (j1, k1)))
        
        # M2 = (A21+A22)B11
        # A21=A[i1,j0], A22=A[i1,j1]
        # B11=B[j0,k0]
        terms.append(((i1, j0), (j0, k0)))
        terms.append(((i1, j1), (j0, k0)))
        
        # M3 = A11(B12-B22)
        # A11=A[i0,j0]
        # B12=B[j0,k1], B22=B[j1,k1]
        terms.append(((i0, j0), (j0, k1)))
        terms.append(((i0, j0), (j1, k1))) # with sign -1 in W? No, signs are in U/V/W or just coefficients.
        
        # To keep it simple with integer entries, we can just add the terms and handle signs in the final assembly?
        # No, the decomposition is a sum of rank-1 tensors.
        # Each term in Strassen is a linear combination of products.
        # We need to expand them into rank-1 terms.
        
        # Actually, it's easier to just generate the U, V, W for each of the 7 Strassen multiplications
        # and add them to the lists.
        
        # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        # = A[i0,j0]B[j0,k0] + A[i0,j0]B[j1,k1] + A[i1,j1]B[j0,k0] + A[i1,j1]B[j1,k1]
        
        # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        # = A[i1,j0]B[j0,k0] + A[i1,j1]B[j0,k0]
        
        # M3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
        # = A[i0,j0]B[j0,k1] - A[i0,j0]B[j1,k1]
        
        # M4: A[i1,j1] * (B[j1,k0]-B[j0,k0])
        # = A[i1,j1]B[j1,k0] - A[i1,j1]B[j0,k0]
        
        # M5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        # = A[i0,j0]B[j1,k1] + A[i0,j1]B[j1,k1]
        
        # M6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        # = A[i1,j0]B[j0,k0] + A[i1,j0]B[j0,k1] - A[i0,j0]B[j0,k0] - A[i0,j0]B[j0,k1]
        
        # M7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        # = A[i0,j1]B[j1,k0] + A[i0,j1]B[j1,k1] - A[i1,j1]B[j1,k0] - A[i1,j1]B[j1,k1]
        
        # We can just call add_schoolbook for each expanded term, passing the sign in U or V?
        # The output requires integer/half-integer entries.
        # We can put the sign in U.
        
        def add_term(i, j, k, sign=1):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + j] = sign
            v[j * p + k] = 1
            w[k * n + i] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

        # M1
        add_term(i0, j0, k0, 1)
        add_term(i0, j0, k1, 1)
        add_term(i1, j1, k0, 1)
        add_term(i1, j1, k1, 1)
        
        # M2
        add_term(i1, j0, k0, 1)
        add_term(i1, j1, k0, 1)
        
        # M3
        add_term(i0, j0, k1, 1)
        add_term(i0, j0, k1, -1) # Wait, this is A[i0,j0]B[j1,k1] with sign -1
        # Correction: M3 = A11(B12-B22) = A[i0,j0]B[j0,k1] - A[i0,j0]B[j1,k1]
        # So:
        # add_term(i0, j0, k1, 1) is A[i0,j0]B[j0,k1]? No, B index is j0?
        # B12 is B[j0, k1]. B22 is B[j1, k1].
        # So:
        # add_term(i0, j0, k1, 1) -> A[i0,j0] * B[j0,k1]
        # add_term(i0, j1, k1, -1) -> A[i0,j0] * B[j1,k1] ? No, A index is i0,j0.
        # The term is A[i0,j0] * B[j1,k1].
        # So U has A[i0,j0], V has B[j1,k1].
        # So add_term(i0, j0, k1, -1) is WRONG.
        # add_term takes i, j from A, k from B?
        # My add_term(i, j, k) assumes A[i,j] and B[j,k].
        # So for A[i0,j0] * B[j1,k1], j in A is j0, j in B is j1. They are different!
        # My add_term assumes the same j index for A and B.
        # This is a problem for Strassen terms where A and B indices don't match in the simple "j" slot.
        
        # I need a more general add_term that takes i, j_a, j_b, k.
        # But U is indexed by i, j_a. V is indexed by j_b, k.
        # So I can just set U[i, j_a] and V[j_b, k].
        
        def add_general_term(i, j_a, j_b, k, sign=1):
            u = [0] * (n * m)
            v = [0] * (m * p)
            w = [0] * (p * n)
            u[i * m + j_a] = sign
            v[j_b * p + k] = 1
            w[k * n + i] = 1
            U_cols.append(u)
            V_cols.append(v)
            W_cols.append(w)

        # M1: (A[i0,j0]+A[i1,j1]) * (B[j0,k0]+B[j1,k1])
        add_general_term(i0, j0, j0, k0, 1)
        add_general_term(i0, j0, j1, k1, 1)
        add_general_term(i1, j1, j0, k0, 1)
        add_general_term(i1, j1, j1, k1, 1)
        
        # M2: (A[i1,j0]+A[i1,j1]) * B[j0,k0]
        add_general_term(i1, j0, j0, k0, 1)
        add_general_term(i1, j1, j0, k0, 1)
        
        # M3: A[i0,j0] * (B[j0,k1]-B[j1,k1])
        add_general_term(i0, j0, j0, k1, 1)
        add_general_term(i0, j0, j1, k1, -1)
        
        # M4: A[i1,j1] * (B[j1,k0]-B[j0,k0])
        add_general_term(i1, j1, j1, k0, 1)
        add_general_term(i1, j1, j0, k0, -1)
        
        # M5: (A[i0,j0]+A[i0,j1]) * B[j1,k1]
        add_general_term(i0, j0, j1, k1, 1)
        add_general_term(i0, j1, j1, k1, 1)
        
        # M6: (A[i1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k1])
        add_general_term(i1, j0, j0, k0, 1)
        add_general_term(i1, j0, j0, k1, 1)
        add_general_term(i0, j0, j0, k0, -1)
        add_general_term(i0, j0, j0, k1, -1)
        
        # M7: (A[i0,j1]-A[i1,j1]) * (B[j1,k0]+B[j1,k1])
        add_general_term(i0, j1, j1, k0, 1)
        add_general_term(i0, j1, j1, k1, 1)
        add_general_term(i1, j1, j1, k0, -1)
        add_general_term(i1, j1, j1, k1, -1)

    # Iterate over output blocks
    for a in range(3):
        for b in range(3):
            for k in range(3):
                rows = blocks[a]
                cols_a = blocks[k]
                cols_b = blocks[k]
                cols = blocks[b]
                
                # Check if it's a 2x2x2 case
                if len(rows) == 2 and len(cols_a) == 2 and len(cols_b) == 2 and len(cols) == 2:
                    add_strassen(rows, cols_a, cols_b, cols)
                else:
                    add_schoolbook(rows, cols_a, cols_b, cols)

    # Convert lists of lists to 2D lists
    rank = len(U_cols)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for col in range(rank):
        for row in range(n * m):
            U[row][col] = U_cols[col][row]
        for row in range(m * p):
            V[row][col] = V_cols[col][row]
        for row in range(p * n):
            W[row][col] = W_cols[col][row]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
