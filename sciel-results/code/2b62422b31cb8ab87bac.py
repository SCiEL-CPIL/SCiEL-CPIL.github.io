"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Partition: n=[2,1], m=[2,2,1], p=[2,2,2]
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    n_offsets = [0, 2]
    m_offsets = [0, 2, 4]
    p_offsets = [0, 2, 4]
    
    rank = 0
    U = []
    V = []
    W = []
    
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal rank
        U.append([0] * n * m)
        V.append([0] * m * p)
        W.append([0] * p * n)
        U[rank][u_idx] = u_val
        V[rank][v_idx] = v_val
        W[rank][w_idx] = w_val
        rank += 1

    def add_schoolbook(i0, j0, k0, ni, mj, pk):
        for i in range(ni):
            for j in range(mj):
                for k in range(pk):
                    u_idx = (i0 + i) * m + (j0 + j)
                    v_idx = (j0 + j) * p + (k0 + k)
                    w_idx = (k0 + k) * n + (i0 + i)
                    add_term(u_idx, v_idx, w_idx, 1, 1, 1)

    def add_strassen(i0, j0, k0):
        # 2x2x2 block at offset i0, j0, k0
        # A indices: (i0, j0), (i0, j0+1), (i0+1, j0), (i0+1, j0+1)
        # B indices: (j0, k0), (j0, k0+1), (j0+1, k0), (j0+1, k0+1)
        # C indices: (i0, k0), (i0, k0+1), (i0+1, k0), (i0+1, k0+1)
        
        # Helper to get global indices
        def u(i, j): return (i0 + i) * m + (j0 + j)
        def v(j, k): return (j0 + j) * p + (k0 + k)
        def w(k, i): return (k0 + k) * n + (i0 + i)
        
        # M1 = (A00 + A11)(B00 + B11)
        # U: A00, A11. V: B00, B11. W: C00, C11, C10, C01 (with signs)
        # We need to decompose the sum of products into rank-1 terms.
        # This is not just 1 term, it's the Strassen combination.
        # Actually, Strassen's algorithm computes 7 intermediate matrices M1..M7.
        # Each M is a product of linear combinations of A and B.
        # M1 = (A00+A11)(B00+B11). This is a rank-1 tensor in the space of linear combinations?
        # No, M1 is a scalar (if we view 1x1 blocks) or a matrix.
        # Here, the "blocks" are 1x1 scalars in the sub-problem.
        # So M1 is a scalar value computed as (A00+A11)*(B00+B11).
        # This corresponds to 1 rank-1 term in the decomposition of the sub-tensor?
        # No, the product of two linear forms is not rank 1.
        # (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
        # This is a sum of 4 rank-1 terms.
        # Strassen replaces the 8 terms of schoolbook with 7 terms of this type?
        # No, Strassen computes 7 products of linear combinations.
        # Each product is a bilinear form.
        # The decomposition of the tensor is a sum of rank-1 tensors.
        # The Strassen algorithm provides a decomposition of the multiplication tensor into 7 rank-1 tensors?
        # No, it provides an algorithm. The rank is 7.
        # This means the tensor can be written as a sum of 7 rank-1 tensors.
        # The 7 terms correspond to the 7 Strassen products.
        # Each Strassen product $M_k = L_k(A) R_k(B)$ is a scalar.
        # The output $C$ is a linear combination of $M_k$.
        # So the decomposition is:
        # $T = \sum_{k=1}^7 u_k \otimes v_k \otimes w_k$
        # where $u_k$ is the vector of coefficients for $L_k(A)$, $v_k$ for $R_k(B)$, and $w_k$ for the output.
        
        # Let's define the 7 terms for 2x2x2.
        # Indices 0,1 for rows/cols.
        
        # Term 1: M1 = (A00 + A11)(B00 + B11)
        # u: A00:1, A11:1. v: B00:1, B11:1. w: C00:1, C11:1, C10:-1, C01:1 (from C formulas)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to add 7 terms to the global matrices.
        # Each term has specific non-zero entries in U, V, W.
        
        # T1: M1
        # u: u(0,0)=1, u(1,1)=1
        # v: v(0,0)=1, v(1,1)=1
        # w: w(0,0)=1, w(1,1)=1, w(0,1)=-1, w(1,0)=1  <- Wait, C01 is (0,1), C10 is (1,0).
        # C00 (0,0): +M1
        # C11 (1,1): +M1
        # C10 (1,0): -M1? No, C10 = M2+M4. M1 is not in C10.
        # C01 (0,1): M3+M5. M1 is not in C01.
        # So w for T1: w(0,0)=1, w(1,1)=1.
        
        add_term(u(0,0), v(0,0), w(0,0), 1, 1, 1) # Part of M1? No, this is just one component.
        # This approach of adding individual components is wrong.
        # A rank-1 term is a single column in U, V, W.
        # The term $u \otimes v \otimes w$ means $C_{ik} += u_i v_j w_k$? No.
        # $C_{ik} = \sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r}$?
        # The convention is $W[k n + i, r]$.
        # So $C_{ik} = \sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r}$ is NOT correct.
        # The tensor is $T_{ijk} = \sum_r U_{i m + j, r} V_{j p + k, r} W_{k n + i, r}$.
        # Yes, this is the standard decomposition.
        
        # So for M1 = (A00+A11)(B00+B11), the contribution to C is:
        # C00 += M1
        # C11 += M1
        # This means we need a term where:
        # U has 1 at (0,0) and (1,1).
        # V has 1 at (0,0) and (1,1).
        # W has 1 at (0,0) and (1,1).
        # But U, V, W are vectors. A single rank-1 term can only have one non-zero in U, one in V, one in W?
        # No, U is a vector of size $n \cdot m$. It can have multiple non-zeros.
        # So one column in U can have 1 at $u(0,0)$ and 1 at $u(1,1)$.
        # One column in V can have 1 at $v(0,0)$ and 1 at $v(1,1)$.
        # One column in W can have 1 at $w(0,0)$ and 1 at $w(1,1)$.
        # This single term contributes $1 \cdot 1 \cdot 1$ to $C_{00}$ (via $i=0,j=0,k=0$) and $1 \cdot 1 \cdot 1$ to $C_{11}$ (via $i=1,j=1,k=1$).
        # Does it contribute to $C_{01}$? $i=0,j=1,k=1$. U(0,1)=0. So no.
        # Does it contribute to $C_{10}$? $i=1,j=0,k=0$. U(1,0)=0. So no.
        # So this works!
        
        # So for each Strassen product $M_k = L_k(A) R_k(B)$, we create ONE rank-1 term.
        # The U vector has the coefficients of $L_k(A)$.
        # The V vector has the coefficients of $R_k(B)$.
        # The W vector has the coefficients of how $M_k$ enters $C$.
        
        # Let's define the 7 terms.
        # Indices: 0,1.
        
        # 1. M1 = (A00+A11)(B00+B11)
        # U: (0,0):1, (1,1):1
        # V: (0,0):1, (1,1):1
        # W: (0,0):1, (1,1):1
        _add_strassen_term(
            [(0,0,1), (1,1,1)], # U
            [(0,0,1), (1,1,1)], # V
            [(0,0,1), (1,1,1)]  # W
        )
        
        # 2. M2 = (A10+A11)B00
        # U: (1,0):1, (1,1):1
        # V: (0,0):1
        # W: (0,1):1, (1,1):-1
        _add_strassen_term(
            [(1,0,1), (1,1,1)],
            [(0,0,1)],
            [(0,1,1), (1,1,-1)]
        )
        
        # 3. M3 = A00(B10-B11)
        # U: (0,0):1
        # V: (1,0):1, (1,1):-1
        # W: (0,1):1, (1,1):1
        _add_strassen_term(
            [(0,0,1)],
            [(1,0,1), (1,1,-1)],
            [(0,1,1), (1,1,1)]
        )
        
        # 4. M4 = A11(B01-B00)
        # U: (1,1):1
        # V: (0,1):1, (0,0):-1
        # W: (0,0):1, (1,0):1
        _add_strassen_term(
            [(1,1,1)],
            [(0,1,1), (0,0,-1)],
            [(0,0,1), (1,0,1)]
        )
        
        # 5. M5 = (A00+A01)B11
        # U: (0,0):1, (0,1):1
        # V: (1,1):1
        # W: (0,0):-1, (0,1):1
        _add_strassen_term(
            [(0,0,1), (0,1,1)],
            [(1,1,1)],
            [(0,0,-1), (0,1,1)]
        )
        
        # 6. M6 = (A10-A00)(B00+B01)
        # U: (1,0):1, (0,0):-1
        # V: (0,0):1, (0,1):1
        # W: (1,1):1
        _add_strassen_term(
            [(1,0,1), (0,0,-1)],
            [(0,0,1), (0,1,1)],
            [(1,1,1)]
        )
        
        # 7. M7 = (A01-A11)(B10+B11)
        # U: (0,1):1, (1,1):-1
        # V: (1,0):1, (1,1):1
        # W: (0,0):1
        _add_strassen_term(
            [(0,1,1), (1,1,-1)],
            [(1,0,1), (1,1,1)],
            [(0,0,1)]
        )

    def _add_strassen_term(u_list, v_list, w_list):
        nonlocal rank
        U.append([0] * n * m)
        V.append([0] * m * p)
        W.append([0] * p * n)
        for (i, j, val) in u_list:
            U[rank][(i0 + i) * m + (j0 + j)] = val
        for (j, k, val) in v_list:
            V[rank][(j0 + j) * p + (k0 + k)] = val
        for (k, i, val) in w_list:
            W[rank][(k0 + k) * n + (i0 + i)] = val
        rank += 1

    # Apply blocks
    # 4 Strassen blocks
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    
    # Schoolbook blocks
    # (0, 4, 0) 2x1x2
    add_schoolbook(0, 4, 0, 2, 1, 2)
    # (0, 4, 2) 2x1x2
    add_schoolbook(0, 4, 2, 2, 1, 2)
    # (0, 4, 4) 2x1x2
    add_schoolbook(0, 4, 4, 2, 1, 2)
    
    # (2, 0, 0) 1x2x2
    add_schoolbook(2, 0, 0, 1, 2, 2)
    # (2, 0, 2) 1x2x2
    add_schoolbook(2, 0, 2, 1, 2, 2)
    # (2, 0, 4) 1x2x2
    add_schoolbook(2, 0, 4, 1, 2, 2)
    
    # (2, 2, 0) 1x2x2
    add_schoolbook(2, 2, 0, 1, 2, 2)
    # (2, 2, 2) 1x2x2
    add_schoolbook(2, 2, 2, 1, 2, 2)
    # (2, 2, 4) 1x2x2
    add_schoolbook(2, 2, 4, 1, 2, 2)
    
    # (2, 4, 0) 1x1x2
    add_schoolbook(2, 4, 0, 1, 1, 2)
    # (2, 4, 2) 1x1x2
    add_schoolbook(2, 4, 2, 1, 1, 2)
    # (2, 4, 4) 1x1x2
    add_schoolbook(2, 4, 4, 1, 1, 2)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
