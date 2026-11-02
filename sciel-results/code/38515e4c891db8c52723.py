import numpy as np

def construct():
    n, m, p = 3, 3, 8
    rank = 0
    # We will build the decomposition term by term.
    # To avoid dynamic sizing issues, we can pre-allocate or use lists.
    # Max rank is 72 (schoolbook). We expect ~68.
    # Let's use lists of tuples (u_indices, v_indices, w_indices, coeff)
    # and then convert to matrices.
    
    terms = []
    
    def add_term(u_idx, v_idx, w_idx, coeff=1):
        # u_idx: list of (i, j)
        # v_idx: list of (j, k)
        # w_idx: list of (k, i)
        terms.append((u_idx, v_idx, w_idx, coeff))
        
    # Strassen blocks for i in {0,1}, j in {0,1}
    # k blocks: (0,1), (2,3), (4,5), (6,7)
    k_blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    for k0, k1 in k_blocks:
        # M1 = (A00 + A11)(B0k0 + B1k1)
        # Note: Strassen B indices are relative.
        # B00 -> B[0, k0], B01 -> B[0, k1]
        # B10 -> B[1, k0], B11 -> B[1, k1]
        
        # Term 1: (A00 + A11)(B0k0 + B1k1)
        # Wait, standard Strassen M1 is (A00+A11)(B00+B11).
        # Here B00 is B[0, k0], B11 is B[1, k1].
        add_term([(0,0), (1,1)], [(0,k0), (1,k1)], [(k0,0), (k1,1)], 1)
        
        # Term 2: (A10 + A11)B0k0
        add_term([(1,0), (1,1)], [(0,k0)], [(k0,1)], 1)
        
        # Term 3: A00(B0k1 - B1k1)
        add_term([(0,0)], [(0,k1), (1,k1)], [(k1,0)], 1)
        # Coeff for B1k1 is -1, so term is A00*B0k1*C0k1 - A00*B1k1*C0k1?
        # No, W is the output.
        # The term is U * V * W.
        # If V has (0,k1) with 1 and (1,k1) with -1, it works.
        
        # Term 4: A11(B1k0 - B0k0)
        add_term([(1,1)], [(1,k0), (0,k0)], [(k0,1)], 1)
        # V: (1,k0) coeff 1, (0,k0) coeff -1.
        
        # Term 5: (A00 + A01)B1k1
        add_term([(0,0), (0,1)], [(1,k1)], [(k1,0)], 1)
        
        # Term 6: (A10 - A00)(B0k0 + B0k1)
        add_term([(1,0), (0,0)], [(0,k0), (0,k1)], [(k0,1), (k1,1)], 1)
        # U: (1,0) coeff 1, (0,0) coeff -1.
        
        # Term 7: (A01 - A11)(B1k0 + B1k1)
        add_term([(0,1), (1,1)], [(1,k0), (1,k1)], [(k0,0), (k1,0)], 1)
        # U: (0,1) coeff 1, (1,1) coeff -1.
        
        # Now we need to distribute these terms to the correct C entries.
        # The 'add_term' above added a single rank-1 tensor.
        # But Strassen terms contribute to multiple C entries.
        # My add_term function adds a full rank-1 term.
        # I need to add 7 rank-1 terms, but each term's W vector has multiple non-zeros.
        # The W vector for Term 1 should be such that it contributes to C0k0, C1k1, etc?
        # No, in the tensor decomposition, each rank-1 term is $u \otimes v \otimes w$.
        # The sum of these terms must equal the multiplication tensor.
        # The Strassen algorithm defines 7 products $M_r$.
        # $C_{i,k} = \sum_r \alpha_{i,k,r} M_r$.
        # This means we can write the decomposition as:
        # $\sum_r ( \sum_{i,j} \beta_{i,j,r} A_{i,j} ) ( \sum_{j,k} \gamma_{j,k,r} B_{j,k} ) ( \sum_{k,i} \delta_{k,i,r} C_{k,i} )$?
        # No, the tensor is $\sum T_{i,j,k} A_{i,j} B_{j,k} C_{k,i}$.
        # We want $\sum_{r=1}^R U_r \otimes V_r \otimes W_r = T$.
        # This means $T_{i,j,k} = \sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        
        # So for each Strassen term $r$, we define $U_r, V_r, W_r$.
        # The $W_r$ vector will have non-zero entries at the $C$ positions it affects.
        
        # Let's re-evaluate.
        # Term 1: $M_1 = (A_{0,0}+A_{1,1})(B_{0,k0}+B_{1,k1})$.
        # This product is used in:
        # $C_{0,k0} += M_1$
        # $C_{1,k1} += M_1$
        # So $W_1$ should have 1 at $(k0, 0)$ and 1 at $(k1, 1)$.
        # Wait, $W$ is indexed by $(k, i)$.
        # $C_{i,k}$ corresponds to $W_{k,i}$.
        # So $W_1[k0, 0] = 1, W_1[k1, 1] = 1$.
        
        # Let's redo the terms with correct W vectors.

    # Clear terms
    terms = []
    
    for k0, k1 in k_blocks:
        # T1: (A00+A11)(B0k0+B1k1) -> C0k0 + C1k1
        add_term([(0,0), (1,1)], [(0,k0), (1,k1)], [(k0,0), (k1,1)], 1)
        
        # T2: (A10+A11)B0k0 -> C1k0
        add_term([(1,0), (1,1)], [(0,k0)], [(k0,1)], 1)
        
        # T3: A00(B0k1-B1k1) -> C0k1
        add_term([(0,0)], [(0,k1), (1,k1)], [(k1,0)], 1)
        
        # T4: A11(B1k0-B0k0) -> C1k0
        add_term([(1,1)], [(1,k0), (0,k0)], [(k0,1)], 1)
        
        # T5: (A00+A01)B1k1 -> C0k1
        add_term([(0,0), (0,1)], [(1,k1)], [(k1,0)], 1)
        
        # T6: (A10-A00)(B0k0+B0k1) -> C1k0 + C1k1
        add_term([(1,0), (0,0)], [(0,k0), (0,k1)], [(k0,1), (k1,1)], 1)
        
        # T7: (A01-A11)(B1k0+B1k1) -> C0k0 + C0k1
        add_term([(0,1), (1,1)], [(1,k0), (1,k1)], [(k0,0), (k1,0)], 1)
        
    # Schoolbook for remaining terms
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Check if this term is covered by Strassen
                if i in [0, 1] and j in [0, 1] and k in [0, 1, 2, 3, 4, 5, 6, 7]:
                    # All k are covered by the 4 blocks
                    continue
                add_term([(i, j)], [(j, k)], [(k, i)], 1)
                
    # Convert terms to matrices
    R = len(terms)
    U = [[0]*R for _ in range(n*m)]
    V = [[0]*R for _ in range(m*p)]
    W = [[0]*R for _ in range(p*n)]
    
    for r, (u_idx, v_idx, w_idx, coeff) in enumerate(terms):
        for (i, j) in u_idx:
            U[i*m + j][r] += coeff
        for (j, k) in v_idx:
            V[j*p + k][r] += coeff
        for (k, i) in w_idx:
            W[k*n + i][r] += coeff
            
    return {"U": U, "V": V, "W": W}
