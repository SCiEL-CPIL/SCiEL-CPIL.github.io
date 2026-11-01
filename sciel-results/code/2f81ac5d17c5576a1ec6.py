import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    if n != m or m != p:
        raise ValueError("Only square matrices supported for this implementation")
    
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    # Recursive Strassen construction
    U, V, W = _strassen_tensors(n)
    
    # Verify the decomposition
    # T[i, j, k] = sum_r U[i, r] * V[j, r] * W[k, r]
    # Check a few random entries or full check if fast enough
    if time.time() < deadline:
        if not _verify_decomposition(U, V, W, n):
            # Fallback to schoolbook if verification fails (should not happen)
            U, V, W = _schoolbook_tensors(n)
            
    return {"U": U, "V": V, "W": W}

def _verify_decomposition(U, V, W, n):
    """
    Verifies if U, V, W form a valid decomposition for n x n matrix multiplication.
    U: (n*n, rank)
    V: (n*n, rank)
    W: (n*n, rank) -- Note: W index is k*n+i
    
    Tensor T[i, j, k] = sum_r U[i, r] V[j, r] W[k, r]
    We check if T matches the standard multiplication tensor.
    Standard T[i, j, k] = 1 if i==k else 0? No.
    (A*B)_{ik} = sum_j A_{ij} B_{jk}.
    So T_{i, j, k} = 1 if j is the "middle" index?
    The tensor is defined by T_{i, j, k} = 1 if the term A_{ij} B_{jk} contributes to C_{ik}.
    Actually, the tensor M_{n} has entries M_{i, j, k} = 1 if i, j, k are indices such that
    A_{ij} and B_{jk} multiply to contribute to C_{ik}.
    Wait, the tensor is $M = \sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}$? No.
    $C_{ik} = \sum_j A_{ij} B_{jk}$.
    The tensor is $\sum_{i,j,k} A_{ij} B_{jk} C_{ki}$? No, $C$ is output.
    The tensor is $\sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ik}$?
    Let's check the contraction:
    $\sum_r U_{i,r} V_{j,r} W_{k,r}$ should equal 1 if $A_{ij} B_{jk}$ is a term in the sum for $C_{ik}$?
    Actually, the tensor $M$ satisfies $M_{i, j, k} = 1$ if $i, j, k$ are valid indices for the multiplication.
    Specifically, for a fixed $i, k$, the sum over $j$ of $A_{ij} B_{jk}$ gives $C_{ik}$.
    The tensor is $\sum_{i, j, k} e_{ij} \otimes e_{jk} \otimes e_{ik}$ is NOT correct because $C$ is indexed by $i, k$.
    The standard matrix multiplication tensor is $T = \sum_{i, j, k} e_{ij} \otimes e_{jk} \otimes e_{ki}$? No.
    $T_{i, j, k} = 1$ if $i, j, k$ form a valid triple for the multiplication.
    In the formula $C_{ik} = \sum_j A_{ij} B_{jk}$, the indices are $i$ (row of A, row of C), $j$ (col of A, row of B), $k$ (col of B, col of C).
    So the tensor is $\sum_{i, j, k} e_{ij} \otimes e_{jk} \otimes e_{ik}$?
    Let's check the contraction with $W$ indexed by $k, i$.
    The task says $W[k \cdot n + i, r]$.
    So $W_{k, i, r}$.
    The contraction is $\sum_r U_{i, r} V_{j, r} W_{k, i, r}$? No, $W$ is just a matrix.
    The term is $U_{i, r} V_{j, r} W_{k \cdot n + i, r}$.
    This sums to $C_{k, i}$? Or $C_{i, k}$?
    The task says: "output factor follows the official trace convention W[k*n+i,r]".
    And the result is $C$.
    Usually $C_{ik}$.
    If $W$ is indexed by $k, i$, then the reconstructed matrix is $C_{k, i}$?
    Let's assume the evaluator handles the index mapping correctly as per the contract.
    The contract says: "U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]".
    This implies:
    $U$ index: $i \cdot m + j$. (Row i, Col j of A)
    $V$ index: $j \cdot p + k$. (Row j, Col k of B)
    $W$ index: $k \cdot n + i$. (Row k, Col i of C? Or Row i, Col k?)
    If $W$ is $k \cdot n + i$, and $C$ is usually $i \cdot n + k$, then $W$ stores $C^T$ or $C$ with swapped indices?
    "Trace convention" might imply $C_{ik}$ is stored at $k, i$?
    Regardless, I must match the tensor defined by these indices.
    The tensor $T$ is defined such that $T_{i, j, k} = 1$ if $A_{ij} B_{jk}$ contributes to the output at position corresponding to $W$'s index $(k, i)$.
    In standard multiplication, $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    So if $W$'s index is $(k, i)$, then $T_{i, j, k} = 1$ for all valid $i, j, k$.
    Wait, if $T_{i, j, k} = 1$ for all $i, j, k$, that's the all-ones tensor, which is rank 1. That's wrong.
    The tensor is sparse.
    $T_{i, j, k} = 1$ if the term $A_{ij} B_{jk}$ is part of the sum for the output element associated with $(k, i)$.
    In standard matmul, $A_{ij} B_{jk}$ contributes to $C_{ik}$.
    So the output element is identified by $(i, k)$.
    If $W$ is indexed by $(k, i)$, then the output element $(i, k)$ is at index $k \cdot n + i$.
    So $T_{i, j, k} = 1$ for all $i, j, k$? No.
    The tensor is $\sum_{i, j, k} e_{ij} \otimes e_{jk} \otimes e_{ki}$?
    Let's check: $\sum_r U_{ij, r} V_{jk, r} W_{ki, r}$.
    If we set $U_{ij, r} = 1$ for a specific triple, etc.
    The schoolbook decomposition sets:
    $U_{ij, c} = 1$
    $V_{jk, c} = 1$
    $W_{ki, c} = 1$
    for $c$ corresponding to $(i, j, k)$.
    Then $\sum_c U_{ij, c} V_{jk, c} W_{ki, c} = 1$ if there is a $c$ such that all are 1.
    This happens if $(i, j, k)$ is the triple for $c$.
    So the reconstructed value at $(i, j, k)$ is 1 if the triple exists.
    Does the matrix multiplication tensor have 1s at all $(i, j, k)$?
    No. The tensor $M_n$ has entries $M_{i, j, k} = 1$ if $i, j, k$ are indices.
    Wait, the tensor IS the set of all triples $(i, j, k)$?
    Yes, because $C_{ik} = \sum_j A_{ij} B_{jk}$.
    The term $A_{ij} B_{jk}$ is a bilinear form.
    The tensor is the sum of all these bilinear forms?
    No, the tensor is the object that, when contracted with $A$ and $B$, gives $C$.
    $C = T \cdot (A \otimes B)$.
    $C_{ik} = \sum_{j, l, m} T_{ik, jl, m} A_{jl} B_{lm}$? No.
    $C_{ik} = \sum_j A_{ij} B_{jk}$.
    So $T_{ik, ij, jk} = 1$?
    The indices of $T$ are (Output, Input1, Input2).
    Output index: $(i, k)$.
    Input1 index: $(i, j)$.
    Input2 index: $(j, k)$.
    So $T$ is non-zero at $( (i, k), (i, j), (j, k) )$ with value 1.
    
    In the flattened indices:
    $U$ index: $i \cdot m + j$. (Input1)
    $V$ index: $j \cdot p + k$. (Input2)
    $W$ index: $k \cdot n + i$. (Output)
    
    So the tensor has 1s at:
    $U$ index: $i \cdot n + j$
    $V$ index: $j \cdot n + k$
    $W$ index: $k \cdot n + i$
    
    for all $i, j, k \in \{0, \dots, n-1\}$.
    
    So to verify, I should check if:
    $\sum_r U_{i \cdot n + j, r} V_{j \cdot n + k, r} W_{k \cdot n + i, r} = 1$ for all $i, j, k$.
    
    I will implement this check.
    """
    rank = len(U[0])
    # Create U, V, W as numpy arrays for speed
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)
    
    # Check a subset of indices to ensure speed
    # Full check is n^3 * rank operations. For n=4, rank=49, 64*49 ~ 3000 ops. Very fast.
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u_idx = i * n + j
                v_idx = j * n + k
                w_idx = k * n + i
                
                val = np.dot(U_np[u_idx, :], V_np[v_idx, :])
                # Wait, W is involved too.
                # Sum_r U[i, r] V[j, r] W[k, r]
                # This is element-wise product sum.
                term = U_np[u_idx, :] * V_np[v_idx, :] * W_np[w_idx, :]
                sum_val = np.sum(term)
                
                if abs(sum_val - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook_tensors(n):
    rank = n * n * n
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    col = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                U[i * n + j][col] = 1
                V[j * n + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def _strassen_tensors(n):
    if n == 1:
        return [[1]], [[1]], [[1]]
    
    h = n // 2
    if h * 2 != n:
        # Fallback if not power of 2, though n=4 is fine
        return _schoolbook_tensors(n)
        
    U_h, V_h, W_h = _strassen_tensors(h)
    R_h = len(U_h[0])
    R_n = 7 * R_h
    
    U = [[0] * R_n for _ in range(n * n)]
    V = [[0] * R_n for _ in range(n * n)]
    W = [[0] * R_n for _ in range(n * n)]
    
    # Strassen terms definition
    # Each term t has:
    # A_blocks: list of (block_idx, coeff)
    # B_blocks: list of (block_idx, coeff)
    # C_blocks: list of (block_idx, coeff)
    # Block indices: 0: TL, 1: TR, 2: BL, 3: BR
    # A blocks: A_00 (TL), A_01 (TR), A_10 (BL), A_11 (BR)
    # B blocks: B_00 (TL), B_01 (TR), B_10 (BL), B_11 (BR)
    # C blocks: C_00 (TL), C_01 (TR), C_10 (BL), C_11 (BR)
    
    # M1 = (A00 + A11)(B00 + B11)
    # M2 = (A10 + A11)B00
    # M3 = A00(B01 - B11)
    # M4 = A11(B10 - B00)
    # M5 = (A00 + A01)B11
    # M6 = (A10 - A00)(B00 + B01)
    # M7 = (A01 - A11)(B10 + B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    terms = [
        # (A_blocks, B_blocks, C_blocks)
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1)]), # Note: C00 has -M5, C01 has +M5
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)])
    ]
    
    # We need to distribute the C coefficients correctly.
    # The above list defines which C blocks are involved in each M.
    # But a single M contributes to multiple C blocks with different signs.
    # It is better to define the contribution of each M to each C block explicitly.
    
    # Let's redefine the structure:
    # For each M_t, we have A_coeffs[4], B_coeffs[4].
    # And for each C_c, we have a coefficient for M_t.
    
    A_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    
    B_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # C contribution: C[c] = sum_t coeff[t][c] * M_t
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    C_contrib = [
        [1, 0, 0, 0, -1, 0, 1], # M1..M7 coeffs for C00
        [0, 0, 1, 0, 1, 0, 0],  # for C01
        [0, 1, 0, 1, 0, 0, 0],  # for C10
        [1, -1, 1, 0, 0, 1, 0]  # for C11
    ]
    
    # Helper to get global indices for a block
    def get_block_indices(block_idx, n, h):
        # block_idx: 0:TL, 1:TR, 2:BL, 3:BR
        if block_idx == 0:
            return (0, 0) # i_start, j_start
        elif block_idx == 1:
            return (0, h)
        elif block_idx == 2:
            return (h, 0)
        else:
            return (h, h)
            
    for t in range(7):
        for r_h in range(R_h):
            r = t * R_h + r_h
            
            # Fill U
            for b_idx in range(4):
                coeff = A_coeffs[t][b_idx]
                if coeff != 0:
                    i_start, j_start = get_block_indices(b_idx, n, h)
                    for i in range(h):
                        for j in range(h):
                            local_i = i
                            local_j = j
                            global_i = i_start + i
                            global_j = j_start + j
                            u_global_idx = global_i * n + global_j
                            u_local_idx = local_i * h + local_j
                            U[u_global_idx][r] += coeff * U_h[u_local_idx][r_h]
                            
            # Fill V
            for b_idx in range(4):
                coeff = B_coeffs[t][b_idx]
                if coeff != 0:
                    i_start, j_start = get_block_indices(b_idx, n, h)
                    for i in range(h):
                        for j in range(h):
                            local_i = i
                            local_j = j
                            global_i = i_start + i
                            global_j = j_start + j
                            v_global_idx = global_i * n + global_j
                            v_local_idx = local_i * h + local_j
                            V[v_global_idx][r] += coeff * V_h[v_local_idx][r_h]
                            
            # Fill W
            # W index is k*n + i.
            # C blocks are (i, k).
            # C00: i in [0,h), k in [0,h)
            # C01: i in [0,h), k in [h,2h)
            # C10: i in [h,2h), k in [0,h)
            # C11: i in [h,2h), k in [h,2h)
            
            for c_idx in range(4):
                coeff = C_contrib[c_idx][t]
                if coeff != 0:
                    # Determine i_start, k_start for C block
                    if c_idx == 0:
                        i_start, k_start = 0, 0
                    elif c_idx == 1:
                        i_start, k_start = 0, h
                    elif c_idx == 2:
                        i_start, k_start = h, 0
                    else:
                        i_start, k_start = h, h
                        
                    for i in range(h):
                        for k in range(h):
                            local_i = i
                            local_k = k
                            global_i = i_start + i
                            global_k = k_start + k
                            w_global_idx = global_k * n + global_i
                            # W_h index?
                            # W_h is for h x h multiplication.
                            # W_h index convention: k_h * h + i_h.
                            w_local_idx = local_k * h + local_i
                            W[w_global_idx][r] += coeff * W_h[w_local_idx][r_h]
                            
    return U, V, W

# EVOLVE-BLOCK-END
