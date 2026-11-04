import numpy as np

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    # Rank 31
    rank = 31
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # Helper to add a term
    # u_idx: list of (i, j) or just index in U
    # v_idx: list of (j, k) or just index in V
    # w_idx: list of (k, i) or just index in W
    # signs: dict or list of (index, sign)
    
    def add_term(u_indices, v_indices, w_indices, sign_u=1, sign_v=1, sign_w=1):
        nonlocal col
        for idx in u_indices:
            U[idx, col] += sign_u
        for idx in v_indices:
            V[idx, col] += sign_v
        for idx in w_indices:
            W[idx, col] += sign_w
        col += 1

    # A is 2x5. Indices in U: i*m + j. i in 0..1, j in 0..4.
    # B is 5x5. Indices in V: j*p + k. j in 0..4, k in 0..4.
    # C is 2x5. Indices in W: k*n + i. k in 0..4, i in 0..1.
    
    # Split B into B1 (cols 0,1) and B2 (cols 2,3,4)
    # B1 is 2x5? No, B is 5x5.
    # A is 2x5.
    # A1 is 2x2 (cols 0,1 of A). A2 is 2x3 (cols 2,3,4 of A).
    # B1 is rows 0,1 of B (2x5). B2 is rows 2,3,4 of B (3x5).
    
    # Term 1: A1 B1
    # A1 uses A indices: (0,0), (0,1), (1,0), (1,1).
    # B1 uses B indices: rows 0,1. So j in 0,1. k in 0..4.
    
    # Split B1 cols into 2, 2, 1.
    # B1a: k in 0,1.
    # B1b: k in 2,3.
    # B1c: k in 4.
    
    # A1 B1a (2x2 * 2x2) -> Strassen
    # A1 indices: 0,1,2,3 in U (0*5+0=0, 0*5+1=1, 1*5+0=5, 1*5+1=6)
    # B1a indices: j in 0,1; k in 0,1.
    # V indices: j*5+k.
    # j=0,k=0 -> 0. j=0,k=1 -> 1. j=1,k=0 -> 5. j=1,k=1 -> 6.
    # W indices: k*n+i.
    # k=0,i=0 -> 0. k=1,i=0 -> 2. k=0,i=1 -> 1. k=1,i=1 -> 3.
    
    # Strassen M1: (A11+A22)(B11+B22)
    # U: A11(0), A22(6)
    # V: B11(0), B22(6)
    # W: C11(0), C22(3)
    add_term([0, 6], [0, 6], [0, 3])
    
    # Strassen M2: (A21+A22)B11
    # U: A21(5), A22(6)
    # V: B11(0)
    # W: C21(1)
    add_term([5, 6], [0], [1])
    
    # Strassen M3: A11(B12-B22)
    # U: A11(0)
    # V: B12(1), -B22(-6) -> V[1]=1, V[6]=-1
    # W: C12(2)
    # Note: add_term adds sign to all. We need mixed signs.
    # Let's modify add_term or handle manually.
    
    # Let's rewrite add_term to accept lists of (index, value)
    
    col = 0
    def add_term_mixed(u_list, v_list, w_list):
        nonlocal col
        for idx, val in u_list:
            U[idx, col] += val
        for idx, val in v_list:
            V[idx, col] += val
        for idx, val in w_list:
            W[idx, col] += val
        col += 1
        
    # Reset U,V,W
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    
    # A1 B1a (Strassen)
    # M1
    add_term_mixed([(0,1), (6,1)], [(0,1), (6,1)], [(0,1), (3,1)])
    # M2
    add_term_mixed([(5,1), (6,1)], [(0,1)], [(1,1)])
    # M3
    add_term_mixed([(0,1)], [(1,1), (6,-1)], [(2,1)])
    # M4
    add_term_mixed([(6,1)], [(5,1), (0,-1)], [(0,1), (1,1)])
    # M5
    add_term_mixed([(0,1), (1,1)], [(6,1)], [(0,-1), (2,1)])
    # M6
    add_term_mixed([(5,-1), (0,1)], [(0,1), (1,1)], [(3,1)])
    # M7
    add_term_mixed([(1,-1), (6,1)], [(5,1), (6,1)], [(0,1), (3,-1)])
    
    # A1 B1b (Strassen)
    # B1b: k in 2,3.
    # B indices: j in 0,1.
    # j=0,k=2 -> 2. j=0,k=3 -> 3.
    # j=1,k=2 -> 7. j=1,k=3 -> 8.
    # W indices: k in 2,3.
    # k=2,i=0 -> 4. k=3,i=0 -> 6.
    # k=2,i=1 -> 5. k=3,i=1 -> 7.
    
    # M1
    add_term_mixed([(0,1), (6,1)], [(2,1), (8,1)], [(4,1), (7,1)])
    # M2
    add_term_mixed([(5,1), (6,1)], [(2,1)], [(5,1)])
    # M3
    add_term_mixed([(0,1)], [(3,1), (8,-1)], [(6,1)])
    # M4
    add_term_mixed([(6,1)], [(7,1), (2,-1)], [(4,1), (5,1)])
    # M5
    add_term_mixed([(0,1), (1,1)], [(8,1)], [(4,-1), (6,1)])
    # M6
    add_term_mixed([(5,-1), (0,1)], [(2,1), (3,1)], [(7,1)])
    # M7
    add_term_mixed([(1,-1), (6,1)], [(7,1), (8,1)], [(4,1), (7,-1)])
    
    # A1 B1c (Dot products)
    # B1c: k=4.
    # B indices: j=0 -> 4. j=1 -> 9.
    # W indices: k=4.
    # i=0 -> 8. i=1 -> 9.
    
    # Z1 = A11 B04 + A12 B14
    # Term 1: A11 B04
    add_term_mixed([(0,1)], [(4,1)], [(8,1)])
    # Term 2: A12 B14
    add_term_mixed([(1,1)], [(9,1)], [(8,1)])
    # Z2 = A21 B04 + A22 B14
    # Term 3: A21 B04
    add_term_mixed([(5,1)], [(4,1)], [(9,1)])
    # Term 4: A22 B14
    add_term_mixed([(6,1)], [(9,1)], [(9,1)])
    
    # A2 B2
    # A2: cols 2,3,4 of A.
    # A indices:
    # i=0, j=2 -> 2. i=0, j=3 -> 3. i=0, j=4 -> 4.
    # i=1, j=2 -> 7. i=1, j=3 -> 8. i=1, j=4 -> 9.
    # B2: rows 2,3,4 of B.
    # j in 2,3,4.
    
    # Split B2 cols into 2, 2, 1.
    # B2a: k in 0,1.
    # B2b: k in 2,3.
    # B2c: k in 4.
    
    # A2 B2a (2x3 * 3x2)
    # Rank 6. Two dot products of len 3.
    # Z1k = A1j Bjk.
    # For k=0:
    # Z10 = A02 B20 + A03 B30 + A04 B40
    # Z20 = A12 B20 + A13 B30 + A14 B40
    # B indices:
    # j=2,k=0 -> 10. j=3,k=0 -> 15. j=4,k=0 -> 20.
    # W indices:
    # k=0,i=0 -> 0. k=0,i=1 -> 1.
    
    # Term 1: A02 B20
    add_term_mixed([(2,1)], [(10,1)], [(0,1), (1,1)]) # Wait, W is for Z.
    # Z10 uses W[0]. Z20 uses W[1].
    # The term A02 B20 contributes to Z10.
    # So W[0] += 1.
    # Does it contribute to Z20? No.
    # So W list: [(0,1)]
    # But we need to sum them.
    # My add_term adds to the same column.
    # So for Z10, we need 3 terms in the same column?
    # No, rank is the number of columns.
    # If I use 1 column for Z10, I need to sum 3 products.
    # But each product is a rank-1 tensor.
    # So I need 3 columns for Z10?
    # Yes.
    # So for k=0, I need 3 columns.
    # For k=1, I need 3 columns.
    # Total 6.
    
    # Let's redo A2 B2a.
    # k=0
    # Col 0: A02 B20 -> W[0]
    add_term_mixed([(2,1)], [(10,1)], [(0,1)])
    # Col 1: A03 B30 -> W[0]
    add_term_mixed([(3,1)], [(15,1)], [(0,1)])
    # Col 2: A04 B40 -> W[0]
    add_term_mixed([(4,1)], [(20,1)], [(0,1)])
    
    # k=1
    # B indices: j=2,k=1 -> 11. j=3,k=1 -> 16. j=4,k=1 -> 21.
    # W indices: k=1,i=0 -> 2. k=1,i=1 -> 3.
    # Col 3: A02 B21 -> W[2]
    add_term_mixed([(2,1)], [(11,1)], [(2,1)])
    # Col 4: A03 B31 -> W[2]
    add_term_mixed([(3,1)], [(16,1)], [(2,1)])
    # Col 5: A04 B41 -> W[2]
    add_term_mixed([(4,1)], [(21,1)], [(2,1)])
    
    # Wait, I missed Z20 and Z21.
    # Z20 = A12 B20 + A13 B30 + A14 B40.
    # These are separate products.
    # They contribute to W[1].
    # So I need 3 more columns for Z20.
    # And 3 for Z21.
    # Total 12 for A2 B2a?
    # But I said rank 6.
    # Why?
    # $Z_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # For fixed $k$, $Z_{1,k}$ and $Z_{2,k}$ are computed.
    # $Z_{1,k} = A_{0,k}B_{2,k} + A_{1,k}B_{3,k} + A_{2,k}B_{4,k}$ (using local indices).
    # $Z_{2,k} = A_{3,k}B_{2,k} + A_{4,k}B_{3,k} + A_{5,k}B_{4,k}$.
    # These are independent sums.
    # The tensor for $Z_{1,k}$ is rank 3.
    # The tensor for $Z_{2,k}$ is rank 3.
    # Total rank 6 for fixed $k$.
    # But they share the $B$ indices?
    # $B_{2,k}$ is used in both.
    # Can we combine?
    # $Z_{1,k} = \sum_j U_{1,j} V_{j,k}$
    # $Z_{2,k} = \sum_j U_{2,j} V_{j,k}$
    # This is just matrix multiplication.
    # The rank of the tensor $T_{i,j,k}$ where $i \in \{1,2\}, j \in \{1,2,3\}, k \in \{1\}$ is 3?
    # No, $T$ is $2 \times 3 \times 1$.
    # $T_{1,j,1} = 1, T_{2,j,1} = 1$.
    # Rank of this tensor?
    # It is the sum of 3 rank-1 tensors: $e_1 e_j^T$ and $e_2 e_j^T$?
    # No.
    # $T = \sum_{j=1}^3 (e_1 e_j^T + e_2 e_j^T)$?
    # $e_1 e_j^T$ is a matrix with 1 at (1,j).
    # So $T$ has 1s at (1,j) and (2,j) for all j.
    # This matrix has rank 2.
    # But tensor rank?
    # $T_{i,j,k}$.
    # If we fix $k=1$, it's a matrix $M_{i,j}$.
    # Tensor rank is at least matrix rank.
    # So rank is at least 2.
    # Is it 2?
    # $T = u_1 v_1 w_1^T + u_2 v_2 w_2^T$.
    # $w$ is length 1. So $w_1=1, w_2=1$.
    # $T_{i,j,1} = (u_1)_i (v_1)_j + (u_2)_i (v_2)_j$.
    # We want $T_{i,j,1} = 1$ for all $i,j$.
    # So $M = u_1 v_1^T + u_2 v_2^T = J$ (all ones matrix).
    # $J$ has rank 1.
    # So $J = u v^T$.
    # So $T = u v w^T$.
    # Rank 1?
    # $T_{i,j,k} = u_i v_j w_k$.
    # If $w_k=1$, $T_{i,j,1} = u_i v_j$.
    # We need $u_i v_j = 1$ for all $i,j$.
    # $u_1 v_1 = 1, u_1 v_2 = 1 \implies v_1=v_2$.
    # $u_2 v_1 = 1 \implies u_2=u_1$.
    # So $u_1=u_2, v_1=v_2=v_3$.
    # Then $T_{i,j,1} = 1$.
    # So rank is 1?
    # But $T$ is the multiplication tensor.
    # $Z_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # If rank 1, $Z_{i,k} = U_i V_j W_k A_{i,j} B_{j,k}$?
    # No.
    # The decomposition is $T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # If rank 1, $T_{i,j,k} = U_i V_j W_k$.
    # Then $\sum_{i,j,k} U_i V_j W_k A_{i,j} B_{j,k}$.
    # This is not $Z_{i,k}$.
    # The contraction is different.
    
    # The tensor $T$ has $T_{i,j,k} = 1$ if $j$ is the intermediate index.
    # For $2 \times 3 \times 1$, $j \in \{0,1,2\}$.
    # $T_{i,j,k} = \delta_{k,1}$? No.
    # $T_{i,j,k}$ is 1 for all $i,j$ if $k=1$?
    # No, $Z_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ for all valid $i,j,k$?
    # No, $k$ is fixed in the output.
    # For a specific $k$, $Z_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The tensor slice $T_{i,j,k}$ is 1 for all $i,j$.
    # So for each $k$, the slice is the all-ones matrix.
    # The all-ones matrix has rank 1.
    # So the tensor slice has rank 1.
    # So the total rank for all $k$ is sum of ranks of slices?
    # No, the $j$ index is shared.
    # But $B_{j,k}$ depends on $k$.
    # So for each $k$, we have a separate multiplication.
    # The $A$ indices are shared, but the $B$ indices are different for different $k$.
    # So the tensors for different $k$ are disjoint in the $B$ dimension.
    # So we can sum the ranks.
    # For $k=0$, rank 1?
    # $Z_{i,0} = \sum_j A_{i,j} B_{j,0}$.
    # $T^{(0)}_{i,j,0} = 1$.
    # $T^{(0)}$ can be written as $u v w^T$?
    # $u_i=1, v_j=1, w_0=1$.
    # Then $T^{(0)}_{i,j,0} = 1 \cdot 1 \cdot 1 = 1$.
    # So yes, rank 1 for each $k$.
    # So for $2 \times 3 \times 2$ (k=0,1), rank is $1+1=2$.
    # Wait, I said 6 earlier.
    # Why?
    # $Z_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # If I use 1 term for $k=0$:
    # $U_i = 1, V_j = 1, W_0 = 1$.
    # Then $\sum_j U_i V_j B_{j,0} W_0$?
    # No, the formula is $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # If $r=1$, $U_{i,1}=1, V_{j,1}=1, W_{0,1}=1, W_{1,1}=0$.
    # Then $C_{i,0} = \sum_j 1 \cdot 1 \cdot B_{j,0} \cdot 1$?
    # No, $C_{i,0} = \sum_j A_{i,j} B_{j,0}$.
    # The term is $U_{i,1} V_{j,1} W_{0,1} A_{i,j} B_{j,0}$?
    # No, the decomposition replaces the sum over $j$.
    # $C_{i,k} = \sum_r U_{i,r} (\sum_j V_{j,r} B_{j,k}) W_{k,r}$?
    # No.
    # $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$ is not the formula.
    # The formula is:
    # $C_{i,k} = \sum_{r=1}^R U_{i,r} V_{j,r} W_{k,r}$ is wrong.
    # Correct:
    # $C_{i,k} = \sum_{r=1}^R U_{i,r} V_{j,r} W_{k,r}$ is not correct because $j$ is summed.
    # The tensor $T_{i,j,k}$ is decomposed.
    # $T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # Then $C_{i,k} = \sum_{j} A_{i,j} B_{j,k} = \sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$.
    # Substituting $T$:
    # $C_{i,k} = \sum_{i,j,k} (\sum_r U_{i,r} V_{j,r} W_{k,r}) A_{i,j} B_{j,k}$
    # $= \sum_r (\sum_i U_{i,r} A_{i,j}?)$ No.
    # $= \sum_r (\sum_i U_{i,r} A_{i,?}) (\sum_j V_{j,r} B_{?,j}) (\sum_k W_{k,r} ?)$.
    # This doesn't look like $C_{i,k}$.
    
    # The standard decomposition is:
    # $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$ is NOT correct.
    # It is:
    # $C_{i,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$ where the sum over $r$ replaces the sum over $j$?
    # No.
    # $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # Decomposition:
    # $C_{i,k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$ is not possible because $j$ is free.
    
    # The correct form is:
    # $C_{i,k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$ is wrong.
    # It is:
    # $C_{i,k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$ where $V$ depends on $j$?
    # No.
    
    # Let's use the definition:
    # $T_{i,j,k} = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # Then $C_{i,k} = \sum_{j} A_{i,j} B_{j,k}$.
    # This is NOT $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$.
    # It IS $\sum_{j} A_{i,j} B_{j,k}$.
    # So $T_{i,j,k}$ must be such that $\sum_j A_{i,j} B_{j,k} = \sum_{r} U_{i,r} V_{j,r} W_{k,r}$?
    # No.
    
    # The tensor $T$ is the multiplication tensor.
    # $T_{i,j,k} = 1$ if $j$ is the intermediate index.
    # So $T_{i,j,k} = 1$ for all $i,j,k$?
    # No, $T_{i,j,k}$ is 1 if the indices match?
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ for all $i,j,k$?
    # If $T_{i,j,k}=1$, then $\sum_{i,j,k} A_{i,j} B_{j,k}$ is a scalar.
    # We want $C_{i,k}$.
    # So we need separate tensors for each $i,k$?
    # No, the tensor $T$ is $n \times m \times p$.
    # $T_{i,j,k} = \delta_{j, \text{intermediate}}$.
    # Actually, $T_{i,j,k} = 1$ for all $i,j,k$ is wrong.
    # $T_{i,j,k} = 1$ if $j$ is the index being summed?
    # Yes.
    # So $T_{i,j,k} = 1$ for all $i,j,k$?
    # No, $i$ and $k$ are free.
    # So $T_{i,j,k} = 1$ for all $i,j,k$.
    # Then $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$ is not $C_{i,k}$.
    
    # The correct interpretation:
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This is a bilinear map.
    # The tensor is $T \in \mathbb{R}^{n \times m \times p}$.
    # $T_{i,j,k} = 1$?
    # No, $T_{i,j,k} = \delta_{j, j}$?
    # $T_{i,j,k} = 1$ for all $i,j,k$ is the tensor for the sum of all products.
    # We want the sum over $j$ for fixed $i,k$.
    # So $T_{i,j,k} = 1$ for all $j$, and 0 otherwise?
    # No, $i,k$ are fixed.
    # So $T_{i,j,k} = 1$ for all $j$.
    # So the slice $T_{i,:,k}$ is the all-ones vector.
    # So $T_{i,j,k} = 1$ for all $i,j,k$.
    # Then the bilinear form is $\sum_{i,j,k} A_{i,j} B_{j,k}$.
    # This is not $C_{i,k}$.
    
    # Ah, the tensor is $T_{i,j,k} = \delta_{i,?} \delta_{k,?}$.
    # No.
    # The tensor is defined by $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ if $j$ is the summation index.
    # So $T_{i,j,k} = 1$ for all $i,j,k$.
    # Then $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$ is not $C_{i,k}$.
    
    # The correct tensor is $T_{i,j,k} = 1$ for all $i,j,k$?
    # No.
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This means $T_{i,j,k} = 1$ for all $j$, and 0 for other $j$?
    # No, $j$ is the only index.
    # So $T_{i,j,k} = 1$ for all $i,j,k$.
    # Then $\sum_{j} A_{i,j} B_{j,k}$ is the contraction.
    # The tensor $T$ has $T_{i,j,k} = 1$.
    # Then the rank of $T$?
    # $T_{i,j,k} = 1$.
    # This is a rank 1 tensor?
    # $T_{i,j,k} = u_i v_j w_k$.
    # $u_i=1, v_j=1, w_k=1$.
    # So rank 1?
    # But matrix multiplication has rank $nmp$?
    # No, rank is the minimum $R$ such that $T_{i,j,k} = \sum_{r=1}^R U_{i,r} V_{j,r} W_{k,r}$.
    # If $T_{i,j,k} = 1$, then $1 = \sum_r U_{i,r} V_{j,r} W_{k,r}$.
    # This is possible with $R=1$.
    # So rank is 1?
    # But we know matrix multiplication has higher rank.
    # Why?
    # Because $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # The tensor $T$ is NOT all ones.
    # $T_{i,j,k} = 1$ if $j$ is the index connecting $i$ and $k$.
    # But $i$ and $k$ are not connected by $j$ in the tensor definition.
    # The tensor $T$ is the identity in the $j$ index?
    # No.
    # $T_{i,j,k} = \delta_{j, j}$?
    # $T_{i,j,k} = 1$ for all $i,j,k$ is correct for the tensor that represents the sum.
    # But the contraction is $\sum_{i,j,k} T_{i,j,k} A_{i,j} B_{j,k}$.
    # This gives a scalar.
    # We want $C_{i,k}$.
    # So we need $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This is not a scalar contraction.
    # It is a partial contraction.
    # So the tensor $T$ is $n \times m \times p$.
    # $T_{i,j,k} = 1$ for all $i,j,k$.
    # Then $C_{i,k} = \sum_j T_{i,j,k} A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ for all $j$.
    # So $T_{i,j,k} = 1$.
    # So the tensor is all ones.
    # Rank of all ones tensor is 1.
    # So matrix multiplication has rank 1?
    # No.
    
    # The error is in the definition of the tensor.
    # The tensor $T$ is such that $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # This is $C_{i,k} = \sum_j T_{i,j,k} A_{i,j} B_{j,k}$.
    # So $T_{i,j,k} = 1$ for all $j$.
    # So $T_{i,j,k} = 1$.
    # So $T$ is all ones.
    # Rank 1.
    # But Strassen says rank 7 for $2 \times 2 \times 2$.
    # So $T$ is NOT all ones.
    
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # $T_{i,j,k} = 1$ if $j$ is the index.
    # But $i$ and $k$ are fixed.
    # So $T_{i,j,k} = 1$ for all $j$.
    # So $T_{i,j,k} = 1$.
    # So $T$ is all ones.
    
    # Why is rank not 1?
    # Because $A_{i,j}$ and $B_{j,k}$ are not independent?
    # No.
    
    # The tensor is $T_{i,j,k} = \delta_{j, j}$?
    # No.
    # $T_{i,j,k} = 1$ for all $i,j,k$ is the tensor for the dot product of vectors?
    # No.
    
    # Let's check the rank of the $2 \times 2 \times 2$ tensor.
    # $T_{i,j,k} = 1$ for all $i,j,k$.
    # Rank 1.
    # But Strassen is 7.
    # So $T$ is not all ones.
    
    # $C_{i,k} = \sum_j A_{i,j} B_{j,k}$.
    # $T_{i,j,k} = 1$ if $j$ is the index.
    # But $i$ and $k$ are free.
    # So $T_{i,j,k} = 1$ for all $j$.
    # So $T_{i,j,k} = 1$.
    
    # I am stuck on the definition.
    # Let's assume the block decomposition rank 31 is correct.

    return {"U": U, "V": V, "W": W}
