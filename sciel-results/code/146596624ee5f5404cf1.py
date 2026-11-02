import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    # U shape (n*m, rank) -> (9, rank)
    # V shape (m*p, rank) -> (12, rank)
    # W shape (p*n, rank) -> (12, rank)
    
    # We build 2 blocks of 17 factors each.
    total_rank = 34
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    def add_factor(col, a_idx, b_idx, c_idx, a_val=1, b_val=1, c_val=1):
        # a_idx: (i, j)
        # b_idx: (j, k)
        # c_idx: (k, i)
        U[a_idx[0] * m + a_idx[1]][col] += a_val
        V[b_idx[0] * p + b_idx[1]][col] += b_val
        W[c_idx[0] * n + c_idx[1]][col] += c_val

    # Function to add the 17 factors for a specific k-range [k_start, k_end]
    def add_block(k_start, base_col):
        # Strassen M1-M7
        # A_TL: (0,0), (0,1), (1,0), (1,1)
        # B_TL: (0, k_start), (0, k_start+1), (1, k_start), (1, k_start+1)
        # C_TL: (k_start, 0), (k_start, 1), (k_start+1, 0), (k_start+1, 1)
        
        # M1 = (A00+A11)(B00+B11)
        add_factor(base_col + 0, (0,0), (0,k_start), (k_start,0), 1, 1, 1)
        add_factor(base_col + 0, (0,0), (1,k_start), (k_start,0), 1, 1, 1)
        add_factor(base_col + 0, (1,0), (0,k_start), (k_start,0), 1, 1, 1)
        add_factor(base_col + 0, (1,0), (1,k_start), (k_start,0), 1, 1, 1)
        
        # Wait, Strassen M1 contributes to C00, C11.
        # C00 += M1. C11 += M1.
        # My add_factor adds to a single C cell.
        # I need to add the contribution to each C cell separately or handle multiple C cells per factor.
        # The tensor decomposition requires each factor to be a rank-1 tensor.
        # A rank-1 tensor is $u \otimes v \otimes w$.
        # $u$ is a vector in A space, $v$ in B space, $w$ in C space.
        # If a product $P$ contributes to multiple C cells, it is not a single rank-1 term in the tensor sum unless the C part is a linear combination.
        # But the decomposition is $\sum T_r$.
        # Each $T_r$ has entries $U_{i,j,r} V_{j,k,r} W_{k,i,r}$.
        # If $M_1$ contributes to $C_{00}$ and $C_{11}$, we can have:
        # $T_r$ such that $W$ has 1 at $C_{00}$ and 1 at $C_{11}$?
        # No, $W$ is a column vector.
        # If $W_{c0, r} = 1$ and $W_{c1, r} = 1$, then the term is $1 \cdot A_{\dots} B_{\dots} \cdot 1$ for $C_{00}$ AND $1 \cdot A_{\dots} B_{\dots} \cdot 1$ for $C_{11}$.
        # This is valid. The term is added to both outputs.
        
        # So for M1:
        # A part: A00 + A11.
        # B part: B00 + B11 (where B indices are (0,k_start) and (1,k_start)).
        # C part: C(k_start, 0) + C(k_start+1, 1).
        
        # Let's re-implement add_factor to allow multiple C indices.
        
    # Re-define add_factor
    def add_factor_multi(col, a_indices, b_indices, c_indices):
        # a_indices: list of (i, j, coeff)
        # b_indices: list of (j, k, coeff)
        # c_indices: list of (k, i, coeff)
        
        # This logic is flawed. A rank-1 tensor is outer product of 3 vectors.
        # $u \otimes v \otimes w$.
        # $u$ has non-zeros at specific A indices.
        # $v$ has non-zeros at specific B indices.
        # $w$ has non-zeros at specific C indices.
        # The product is $\sum_{i,j} u_{ij} A_{ij} \sum_{j,k} v_{jk} B_{jk} \sum_{k,i} w_{ki} C_{ki}$.
        # This expands to $\sum u_{ij} v_{jk} w_{ki} A_{ij} B_{jk} C_{ki}$.
        # This matches the tensor entry $T_{ij, jk, ki}$.
        
        # So I just need to set the non-zero entries in U, V, W for that column.
        
        for (i, j, coeff) in a_indices:
            U[i * m + j][col] += coeff
        for (j, k, coeff) in b_indices:
            V[j * p + k][col] += coeff
        for (k, i, coeff) in c_indices:
            W[k * n + i][col] += coeff

    # Now define the 17 factors for a block k_start, k_start+1
    
    def build_block(k_start, base_col):
        # M1
        add_factor_multi(base_col + 0, 
                         [(0,0,1), (1,1,1)], 
                         [(0,k_start,1), (1,k_start,1)], 
                         [(k_start,0,1), (k_start+1,1,1)])
        # M2
        add_factor_multi(base_col + 1, 
                         [(1,0,1), (1,1,1)], 
                         [(0,k_start,1)], 
                         [(k_start+1,0,1)])
        # M3
        add_factor_multi(base_col + 2, 
                         [(0,0,1)], 
                         [(0,k_start+1,1), (1,k_start+1,-1)], 
                         [(k_start,1,1)])
        # M4
        add_factor_multi(base_col + 3, 
                         [(1,1,1)], 
                         [(1,k_start,1), (0,k_start,-1)], 
                         [(k_start,0,1)])
        # M5
        add_factor_multi(base_col + 4, 
                         [(0,0,1), (0,1,1)], 
                         [(1,k_start+1,1)], 
                         [(k_start,0,1)])
        # M6
        add_factor_multi(base_col + 5, 
                         [(1,0,1), (0,0,-1)], 
                         [(0,k_start,1), (0,k_start+1,1)], 
                         [(k_start+1,1,1)])
        # M7
        add_factor_multi(base_col + 6, 
                         [(0,1,1), (1,1,-1)], 
                         [(1,k_start,1), (1,k_start+1,1)], 
                         [(k_start,1,1)])
        
        # P1: A02 B20 -> C00
        add_factor_multi(base_col + 7, 
                         [(0,2,1)], 
                         [(2,k_start,1)], 
                         [(k_start,0,1)])
        # P2: A02 B21 -> C01
        add_factor_multi(base_col + 8, 
                         [(0,2,1)], 
                         [(2,k_start+1,1)], 
                         [(k_start,1,1)])
        # P3: A12 B20 -> C10
        add_factor_multi(base_col + 9, 
                         [(1,2,1)], 
                         [(2,k_start,1)], 
                         [(k_start+1,0,1)])
        # P4: A12 B21 -> C11
        add_factor_multi(base_col + 10, 
                         [(1,2,1)], 
                         [(2,k_start+1,1)], 
                         [(k_start+1,1,1)])
        
        # Q1: A20 B00 -> C20
        add_factor_multi(base_col + 11, 
                         [(2,0,1)], 
                         [(0,k_start,1)], 
                         [(2,k_start,1)])
        # Q2: A20 B01 -> C21
        add_factor_multi(base_col + 12, 
                         [(2,0,1)], 
                         [(0,k_start+1,1)], 
                         [(2,k_start+1,1)])
        # Q3: A21 B10 -> C20
        add_factor_multi(base_col + 13, 
                         [(2,1,1)], 
                         [(1,k_start,1)], 
                         [(2,k_start,1)])
        # Q4: A21 B11 -> C21
        add_factor_multi(base_col + 14, 
                         [(2,1,1)], 
                         [(1,k_start+1,1)], 
                         [(2,k_start+1,1)])
        
        # R1: A22 B20 -> C20
        add_factor_multi(base_col + 15, 
                         [(2,2,1)], 
                         [(2,k_start,1)], 
                         [(2,k_start,1)])
        # R2: A22 B21 -> C21
        add_factor_multi(base_col + 16, 
                         [(2,2,1)], 
                         [(2,k_start+1,1)], 
                         [(2,k_start+1,1)])

    build_block(0, 0)
    build_block(2, 17)
    
    return {"U": U, "V": V, "W": W}
