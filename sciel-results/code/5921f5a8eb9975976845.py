import numpy as np

def construct():
    n, m, p = 3, 5, 6
    # We will build the U, V, W matrices
    # Start with empty lists
    U_list = []
    V_list = []
    W_list = []
    
    def add_term(u, v, w):
        U_list.append(u)
        V_list.append(v)
        W_list.append(w)
        
    # Helper to create basis vectors
    def e_vec(dim, idx):
        vec = [0] * dim
        vec[idx] = 1
        return vec
        
    # Helper to create linear combination of basis vectors
    def comb_vec(dim, indices_coeffs):
        vec = [0] * dim
        for idx, coeff in indices_coeffs:
            vec[idx] += coeff
        return vec

    # Define Strassen terms for a 2x2 block
    # Local indices: i in {0,1}, j in {0,1}, k in {0,1}
    # Global mapping will be applied later
    
    # We define the 7 Strassen products in terms of local A, B indices
    # A is 2x2, B is 2x2
    # m1 = (A00+A11)(B00+B11)
    # m2 = (A10+A11)B00
    # m3 = A00(B10-B11)
    # m4 = A11(B00-B01)
    # m5 = (A00+A01)B11
    # m6 = (A10-A00)(B00+B01)
    # m7 = (A01-A11)(B10+B11)
    
    # U vectors (for A)
    # A indices are (i, j) -> i*m + j
    # For 2x2 block, A size is 2x2.
    # But in our global system, A is 3x5.
    # We need to embed the 2x2 block into the 3x5 matrix.
    
    # Let's define the blocks explicitly
    blocks = [
        # (i_start, j_start, k_start)
        (0, 0, 0),
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
        (0, 0, 4),
        (0, 2, 4),
    ]
    
    # For each block, add 7 Strassen terms
    for (i_s, j_s, k_s) in blocks:
        # Local A indices (li, lj) -> Global (i_s+li, j_s+lj)
        # Global A index: (i_s+li) * m + (j_s+lj)
        
        # Local B indices (lj, lk) -> Global (j_s+lj, k_s+lk)
        # Global B index: (j_s+lj) * p + (k_s+lk)
        
        # Local C indices (li, lk) -> Global (i_s+li, k_s+lk)
        # Global C index: (k_s+lk) * n + (i_s+li)  <-- Note W index convention: k*n+i
        
        # m1: (A00+A11)(B00+B11)
        u1 = comb_vec(n*m, [(0*m + 0, 1), (1*m + 1, 1)]) # A00, A11
        v1 = comb_vec(m*p, [(0*p + 0, 1), (1*p + 1, 1)]) # B00, B11
        w1 = e_vec(p*n, 0*n + 0) # C00? No, W is for C.
        # Wait, W vector corresponds to the output C.
        # The term contributes to C_{li, lk}.
        # In Strassen, m1 contributes to C00, C11, C22?
        # C00 = m1 + m4 - m5 + m7
        # C11 (local 1,1) = m1 - m2 + m3 + m6
        # C01 (local 0,1) = m3 + m5
        # C10 (local 1,0) = m2 + m4
        
        # So m1 contributes to C00, C11.
        # W vector for m1 should have 1s at C00 and C11?
        # No, the decomposition is T = sum U_r V_r W_r.
        # The tensor entry T_{i,j,k} is sum U[i,r] V[j,r] W[k,r].
        # For a specific block, the local tensor is T_local.
        # We are adding terms to the global tensor.
        # So for each Strassen term, we define U, V, W such that
        # sum U[i,r] V[j,r] W[k,r] equals the local tensor contribution.
        
        # Let's re-evaluate.
        # The Strassen algorithm gives linear combinations of products.
        # m1 = (A00+A11)(B00+B11).
        # This is a product of two linear forms.
        # So U is the linear form for A, V is the linear form for B.
        # W is the linear form for C?
        # No, the product m1 is added to specific C entries.
        # C00 += m1
        # C11 += m1
        # So the term m1 contributes to C00 and C11.
        # This means the W vector for this term must have non-zero entries at C00 and C11.
        # Specifically, W[global_C_00] = 1, W[global_C_11] = 1.
        # And U, V are the coefficients for A and B.
        
        # So for m1:
        # U has 1 at A00, 1 at A11.
        # V has 1 at B00, 1 at B11.
        # W has 1 at C00, 1 at C11.
        
        # Let's verify:
        # Term = U V W.
        # Contribution to T_{i,j,k} is U[i] V[j] W[k].
        # For i=0, j=0, k=0 (local):
        # U[0]=1, V[0]=1, W[0]=1 -> 1.
        # For i=1, j=1, k=1 (local):
        # U[1]=1, V[1]=1, W[1]=1 -> 1.
        # For i=0, j=1, k=0:
        # U[0]=1, V[1]=1, W[0]=1 -> 1.
        # But m1 should not contribute to C01?
        # C01 = m3 + m5.
        # So m1 should NOT contribute to C01.
        # In my W vector, if I put 1 at C00 and C11, then:
        # T_{0,1,0} (i=0, j=1, k=0) gets U[0] V[1] W[0] = 1*1*1 = 1.
        # This is wrong. m1 should not affect C01.
        
        # The issue is that a single rank-1 term U V W adds to ALL combinations of i,j,k where U[i], V[j], W[k] are non-zero.
        # Strassen terms are not rank-1 in the tensor sense of "one product added to one output".
        # They are "one product added to MULTIPLE outputs".
        # This requires the W vector to be a sum of basis vectors?
        # No, W is a vector.
        # If W has 1 at C00 and 1 at C11, then the term adds to C00 and C11.
        # But it also adds to C01 and C10 if U and V have non-zeros there.
        # In m1, U has non-zeros at A00, A11. V has non-zeros at B00, B11.
        # So the term adds to:
        # (i=0, j=0, k=0) -> C00. Correct.
        # (i=0, j=0, k=1) -> C01. U[0]=1, V[0]=1, W[1]=1 (if W has 1 at C11? No, W index is k*n+i).
        # Let's check W indices.
        # C00 -> index 0*n + 0 = 0.
        # C11 -> index 1*n + 1 = 4 (if n=3? No, local n=2).
        
        # The problem is that the rank-1 term U V W creates a "box" of contributions.
        # Strassen's m1 is NOT a rank-1 term in the tensor decomposition sense if it adds to multiple outputs with the same coefficient?
        # Actually, it IS a rank-1 term if we define W appropriately?
        # No.
        # A rank-1 term $u \otimes v \otimes w$ adds $u_i v_j w_k$ to $T_{ijk}$.
        # If $w$ has support on $\{0, 1\}$, then it adds to $k=0$ and $k=1$.
        # If $u$ has support on $\{0, 1\}$ and $v$ has support on $\{0, 1\}$, it adds to all 4 combinations of $(i,j)$ for each $k$.
        # So m1 would add to C00, C01, C10, C11.
        # But Strassen m1 only adds to C00 and C11.
        # Therefore, m1 CANNOT be represented as a single rank-1 term $u \otimes v \otimes w$ with constant coefficients?
        # Wait.
        # $T_{ijk} = u_i v_j w_k$.
        # If $u=[1,1], v=[1,1], w=[1,0]$, it adds to C00, C01.
        # If we want to add to C00 and C11 only, we need $w$ to be different for i=0 and i=1?
        # But $w$ is a vector in the C-space. It cannot depend on i.
        # So a single rank-1 term cannot add to C00 and C11 but not C01 and C10.
        # Thus, Strassen's 7 terms are NOT rank-1 terms in the standard tensor decomposition?
        
        # CORRECTION:
        # The Strassen algorithm reduces the number of *multiplications*.
        # The tensor decomposition rank is the number of multiplications.
        # Each multiplication is a rank-1 term.
        # In Strassen, we compute 7 products.
        # Then we form linear combinations of these products to get the outputs.
        # This means the outputs are linear combinations of the products.
        # $C = M \cdot P$ where P is the vector of 7 products.
        # This is equivalent to a tensor decomposition where the W vectors are the rows of M?
        # Yes.
        # We have 7 products $p_r = u_r \cdot A \cdot v_r \cdot B$.
        # $C_{ik} = \sum_r M_{ik, r} p_r$.
        # So $T_{ijk} = \sum_r u_{r,i} v_{r,j} M_{r,ik}$.
        # Here $W_r$ is the vector corresponding to $M_{r, \cdot}$.
        # So $W_r$ has entries $M_{r, ik}$.
        # This works!
        # The W vector for term r is the linear combination of C outputs that term r contributes to.
        
        # So for m1:
        # U: [1, 1, 0, 0] (A00, A11)
        # V: [1, 1, 0, 0] (B00, B11)
        # W: [1, 0, 0, 1] (C00, C11) -> Index 0 and 3 (if 2x2).
        # Check:
        # T_{0,0,0} = 1*1*1 = 1.
        # T_{0,1,0} = 1*1*0 = 0.
        # T_{1,0,0} = 1*1*0 = 0.
        # T_{1,1,0} = 1*1*1 = 1.
        # This matches m1 contributing to C00 and C11.
        
        # So we can implement Strassen this way.

        # Local indices for 2x2 block:
        # A: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
        # B: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
        # C: 0,1,2,3 (0:00, 1:01, 2:10, 3:11)
        
        # m1: U=[1,0,0,1], V=[1,0,0,1], W=[1,0,0,1]
        # m2: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0] (C10)
        # m3: U=[1,0,0,0], V=[0,1,1,0], W=[0,1,0,0] (C01) -- Wait, m3 = A00(B10-B11).
        # B10 is index 2, B11 is index 3.
        # V=[0,0,1,-1].
        # m3 contributes to C01.
        # W=[0,1,0,0].
        
        # Let's list them properly.
        # m1: (A00+A11)(B00+B11) -> C00, C11
        # m2: (A10+A11)B00 -> C10
        # m3: A00(B10-B11) -> C01
        # m4: A11(B00-B01) -> C00, C10
        # m5: (A00+A01)B11 -> C01, C11
        # m6: (A10-A00)(B00+B01) -> C11
        # m7: (A01-A11)(B10+B11) -> C00, C11

        # Check C00: m1 + m4 - m5 + m7
        # C11: m1 - m2 + m3 + m6
        # C01: m3 + m5
        # C10: m2 + m4

        # Let's construct U, V, W for each.
        
        def add_strassen_term(i_s, j_s, k_s, u_local, v_local, w_local):
            # Map local to global
            u_global = [0] * (n*m)
            for idx, val in enumerate(u_local):
                if val != 0:
                    li = idx // 2
                    lj = idx % 2
                    gi = i_s + li
                    gj = j_s + lj
                    u_global[gi*m + gj] = val
            
            v_global = [0] * (m*p)
            for idx, val in enumerate(v_local):
                if val != 0:
                    lj = idx // 2
                    lk = idx % 2
                    gj = j_s + lj
                    gk = k_s + lk
                    v_global[gj*p + gk] = val
            
            w_global = [0] * (p*n)
            for idx, val in enumerate(w_local):
                if val != 0:
                    li = idx // 2
                    lk = idx % 2
                    gi = i_s + li
                    gk = k_s + lk
                    # W index is k*n + i
                    w_global[gk*n + gi] = val
            
            add_term(u_global, v_global, w_global)

        # m1
        add_strassen_term(i_s, j_s, k_s, [1,0,0,1], [1,0,0,1], [1,0,0,1])
        # m2
        add_strassen_term(i_s, j_s, k_s, [0,0,1,1], [1,0,0,0], [0,0,1,0])
        # m3
        add_strassen_term(i_s, j_s, k_s, [1,0,0,0], [0,0,1,-1], [0,1,0,0])
        # m4
        add_strassen_term(i_s, j_s, k_s, [0,0,0,1], [1,-1,0,0], [1,0,1,0]) # C00, C10
        # m5
        add_strassen_term(i_s, j_s, k_s, [1,1,0,0], [0,0,0,1], [0,1,0,1]) # C01, C11
        # m6
        add_strassen_term(i_s, j_s, k_s, [0,0,1,-1], [1,1,0,0], [0,0,0,1]) # C11
        # m7
        add_strassen_term(i_s, j_s, k_s, [0,1,0,-1], [0,0,1,1], [1,0,0,1]) # C00, C11

    # Remainders
    # 1. i=2, j in 0..4, k in 0..5
    for j in range(5):
        for k in range(6):
            u = [0] * (n*m)
            u[2*m + j] = 1
            v = [0] * (m*p)
            v[j*p + k] = 1
            w = [0] * (p*n)
            w[k*n + 2] = 1
            add_term(u, v, w)
            
    # 2. i in 0..1, j=4, k in 0..5
    for i in range(2):
        for k in range(6):
            u = [0] * (n*m)
            u[i*m + 4] = 1
            v = [0] * (m*p)
            v[4*p + k] = 1
            w = [0] * (p*n)
            w[k*n + i] = 1
            add_term(u, v, w)

    U = U_list
    V = V_list
    W = W_list
    
    return {"U": U, "V": V, "W": W}
