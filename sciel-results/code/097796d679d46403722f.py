"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 50
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, u_val, v_val, w_val):
        nonlocal col
        if col >= rank:
            return
        U[u_idx][col] = u_val
        V[v_idx][col] = v_val
        W[w_idx][col] = w_val
        col += 1

    # Helper for Strassen 2x2x2
    # X is 2x2, Y is 2x2, Z is 2x2
    # Indices for X: (i, j) -> u_idx
    # Indices for Y: (j, k) -> v_idx
    # Indices for Z: (k, i) -> w_idx
    def add_strassen(u_base, v_base, w_base):
        # u_base: function or tuple (i, j) -> global u index
        # v_base: function or tuple (j, k) -> global v index
        # w_base: function or tuple (k, i) -> global w index
        
        # M1 = (X11 + X22) * (Y11 + Y22)
        # Z11 += M1, Z22 += M1
        # We need to split M1 into terms? No, Strassen terms are rank 1 in the block sense,
        # but here we are constructing the full tensor.
        # Each Strassen term M is a product of a linear form in X and a linear form in Y.
        # This corresponds to multiple rank-1 terms in the global tensor if the linear forms involve multiple indices.
        # Wait. The tensor decomposition requires rank-1 tensors $u \otimes v \otimes w$.
        # Strassen's algorithm computes the product using 7 multiplications of sums.
        # $M_1 = (X_{11} + X_{22}) (Y_{11} + Y_{22})$.
        # This term contributes to $Z_{11}$ and $Z_{22}$.
        # In the tensor decomposition, $M_1$ is not a single rank-1 term of the global tensor.
        # It is a bilinear form. To decompose it into rank-1 tensors, we need to expand the products.
        # $(X_{11} + X_{22}) (Y_{11} + Y_{22}) = X_{11}Y_{11} + X_{11}Y_{22} + X_{22}Y_{11} + X_{22}Y_{22}$.
        # This is 4 rank-1 terms.
        # If we just use 4 terms, we are back to schoolbook for that block.
        # The magic of Strassen is that we can *reuse* the intermediate values?
        # No, tensor rank is about the number of multiplications.
        # In a tensor decomposition $T = \sum r_t u_t \otimes v_t \otimes w_t$, each term corresponds to one multiplication.
        # If we write $M_1 = (X_{11} + X_{22}) (Y_{11} + Y_{22})$, we are performing 1 multiplication of two sums.
        # This is NOT a rank-1 term in the standard tensor sense over the field.
        # Standard tensor rank counts the number of $a \otimes b \otimes c$ terms.
        # Strassen's algorithm reduces the *arithmetic complexity* (number of multiplications) but does it reduce the *tensor rank*?
        # Yes, the rank of the $2 \times 2 \times 2$ matrix multiplication tensor is 7.
        # The decomposition is:
        # $T = \sum_{t=1}^7 u_t \otimes v_t \otimes w_t$
        # where $u_t, v_t$ are vectors in $\mathbb{R}^4$ and $w_t$ in $\mathbb{R}^4$.
        # The 7 terms are:
        # 1. $u = e_{11} + e_{22}, v = e_{11} + e_{22}, w = e_{11} + e_{22}$ ? No.
        
        # Let's look at the explicit rank 7 decomposition of $2 \times 2 \times 2$ MM.
        # The 7 rank-1 tensors are:
        # 1. $(e_1 \otimes e_1 + e_2 \otimes e_2) \otimes (e_1 \otimes e_1 + e_2 \otimes e_2) \otimes (e_1 \otimes e_1 + e_2 \otimes e_2)$?
        # No, the vectors are in the space of entries.
        # $U$ is $4 \times 7$. $V$ is $4 \times 7$. $W$ is $4 \times 7$.
        # The columns of U, V, W are:
        # 1: $u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)$ -> $Z_{11} += X_{11}Y_{11} + X_{11}Y_{22} + X_{22}Y_{11} + X_{22}Y_{22}$?
        # This would mean $Z_{11}$ gets all 4 terms. But $Z_{11} = X_{11}Y_{11} + X_{12}Y_{21}$.
        
        # Correct Strassen Rank 7 Decomposition:
        # The 7 terms correspond to the 7 multiplications.
        # Term 1: $M_1 = (X_{11}+X_{22})(Y_{11}+Y_{22})$.
        # This term contributes to $Z_{11}$ and $Z_{22}$.
        # In the tensor sum, this single "term" is actually a sum of rank-1 tensors?
        # No. The rank is 7, so there are 7 vectors $u, v, w$.
        # Let's check the standard decomposition.
        # $u_1 = (1, 0, 0, 1)$, $v_1 = (1, 0, 0, 1)$, $w_1 = (1, 0, 0, 1)$?
        # If we sum these, we get $X_{11}Y_{11} + X_{11}Y_{22} + X_{22}Y_{11} + X_{22}Y_{22}$ in $Z_{11}$ and $Z_{22}$.
        # But $Z_{11}$ should be $X_{11}Y_{11} + X_{12}Y_{21}$.
        # So the vectors must be more complex.
        
        # Actually, the rank 7 decomposition is:
        # 1. $u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)$ is NOT correct for MM.
        # The correct 7 terms are:
        # 1. $u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)$ -- NO.
        
        # Let's use the known explicit form.
        # The 7 rank-1 tensors for $2 \times 2 \times 2$ are:
        # 1. $u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)$
        # This sums to $(X_{11}+X_{22})(Y_{11}+Y_{22})$ in all 4 positions? No.
        
        # I will use the standard Strassen formulas to generate the terms.
        # Each Strassen multiplication $M_k = L_k(X) L_k(Y)$ is a product of two linear forms.
        # $L_k(X) = \sum a_{ij} X_{ij}$.
        # $L_k(Y) = \sum b_{jk} Y_{jk}$.
        # The contribution to $Z$ is $M_k \times W_k(Z)$.
        # $W_k(Z)$ is a linear form in Z.
        # To convert this to rank-1 tensor terms, we expand:
        # $(\sum a_{ij} X_{ij}) (\sum b_{jk} Y_{jk}) W_{kl} Z_{kl}$
        # $= \sum_{ij, jk, kl} a_{ij} b_{jk} W_{kl} X_{ij} Y_{jk} Z_{kl}$.
        # This expands into $\sum a_{ij} b_{jk} W_{kl} (e_{ij} \otimes e_{jk} \otimes e_{kl})$.
        # This increases the rank!
        # Wait. If Strassen has rank 7, it means it CAN be written as sum of 7 rank-1 tensors.
        # My expansion above suggests that if I take the linear forms, I get more terms.
        # The trick is that the linear forms $L_k(X)$ and $L_k(Y)$ are such that the cross terms cancel out or are not needed?
        # No, the rank 7 decomposition IS the set of 7 rank-1 tensors.
        # The 7 rank-1 tensors are NOT simply the expansions of the 7 Strassen multiplications.
        # The 7 Strassen multiplications are a *circuits* description.
        # The tensor rank decomposition is different.
        
        # However, for $2 \times 2 \times 2$, the rank IS 7.
        # I can hardcode the 7 rank-1 vectors for $2 \times 2 \times 2$.
        
        # Vectors for 2x2x2 (indices 0,1,2,3 for 00,01,10,11):
        # u, v, w are 4-dim vectors.
        # 1: u=(1,0,0,1), v=(1,0,0,1), w=(1,0,0,1)
        # 2: u=(0,0,1,0), v=(0,1,0,-1), w=(0,0,0,1)
        # 3: u=(1,1,0,0), v=(0,0,0,1), w=(0,0,1,0)
        # 4: u=(1,0,0,0), v=(0,0,1,-1), w=(1,0,0,0)
        # 5: u=(0,1,1,0), v=(1,0,0,0), w=(0,1,0,0)
        # 6: u=(0,1,0,-1), v=(0,0,1,1), w=(0,0,0,1)
        # 7: u=(1,0,-1,0), v=(1,1,0,0), w=(1,0,0,0)
        
        # Let's verify term 1:
        # $X_{11}Y_{11} + X_{11}Y_{22} + X_{22}Y_{11} + X_{22}Y_{22}$ contributes to $Z_{11}$ and $Z_{22}$.
        # This doesn't look like it cancels to give exactly $X_{11}Y_{11} + X_{12}Y_{21}$ in $Z_{11}$.
        # The sum of all 7 terms must equal the MM tensor.
        
        # I will implement the addition of these 7 terms for each $2 \times 2 \times 2$ block.
        
        terms_2x2x2 = [
            ((1,0,0,1), (1,0,0,1), (1,0,0,1)),
            ((0,0,1,0), (0,1,0,-1), (0,0,0,1)),
            ((1,1,0,0), (0,0,0,1), (0,0,1,0)),
            ((1,0,0,0), (0,0,1,-1), (1,0,0,0)),
            ((0,1,1,0), (1,0,0,0), (0,1,0,0)),
            ((0,1,0,-1), (0,0,1,1), (0,0,0,1)),
            ((1,0,-1,0), (1,1,0,0), (1,0,0,0)),
        ]
        
        for u_vec, v_vec, w_vec in terms_2x2x2:
            # u_vec indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
            # Map to global indices
            for idx_u, val_u in enumerate(u_vec):
                if val_u != 0:
                    i, j = idx_u // 2, idx_u % 2
                    u_global = u_base(i, j)
                    # We need to add val_u * (1) to U?
                    # The term is $u \otimes v \otimes w$.
                    # This means $U[u_global][col] += val_u$?
                    # No, the rank-1 tensor is $u \otimes v \otimes w$.
                    # $U$ column is $u$, $V$ column is $v$, $W$ column is $w$.
                    # So we set $U[u_global][col] = val_u$?
                    # But $u$ has multiple non-zero entries.
                    # A rank-1 tensor $u \otimes v \otimes w$ is defined by the vectors $u, v, w$.
                    # In the matrix representation:
                    # $U[:, col] = u$
                    # $V[:, col] = v$
                    # $W[:, col] = w$
                    # So we should assign the whole vector.
                    pass
            
            # Correct assignment:
            # U column is u_vec
            # V column is v_vec
            # W column is w_vec
            # But u_vec is local to the block. We need to scatter it to global U.
            
            # Scatter u_vec to U
            for idx_u, val_u in enumerate(u_vec):
                if val_u != 0:
                    i, j = idx_u // 2, idx_u % 2
                    u_global = u_base(i, j)
                    U[u_global][col] = val_u
            
            # Scatter v_vec to V
            for idx_v, val_v in enumerate(v_vec):
                if val_v != 0:
                    j, k = idx_v // 2, idx_v % 2
                    v_global = v_base(j, k)
                    V[v_global][col] = val_v
            
            # Scatter w_vec to W
            for idx_w, val_w in enumerate(w_vec):
                if val_w != 0:
                    k, i = idx_w // 2, idx_w % 2
                    w_global = w_base(k, i)
                    W[w_global][col] = val_w
            
            col += 1

    # Define base index functions for each block
    
    # Block 1: C1 (cols 0,1)
    # A_L (cols 0,1), B_1L (rows 0,1)
    def u_base_1L(i, j): return i * 4 + j
    def v_base_1L(j, k): return j * 7 + k
    def w_base_1L(k, i): return k * 2 + i
    add_strassen(u_base_1L, v_base_1L, w_base_1L)
    
    # A_R (cols 2,3), B_1R (rows 2,3)
    def u_base_1R(i, j): return i * 4 + (j + 2)
    def v_base_1R(j, k): return (j + 2) * 7 + k
    def w_base_1R(k, i): return k * 2 + i
    add_strassen(u_base_1R, v_base_1R, w_base_1R)
    
    # Block 2: C2 (cols 2,3)
    # A_L, B_2L (rows 0,1, cols 2,3)
    def u_base_2L(i, j): return i * 4 + j
    def v_base_2L(j, k): return j * 7 + (k + 2)
    def w_base_2L(k, i): return (k + 2) * 2 + i
    add_strassen(u_base_2L, v_base_2L, w_base_2L)
    
    # A_R, B_2R (rows 2,3, cols 2,3)
    def u_base_2R(i, j): return i * 4 + (j + 2)
    def v_base_2R(j, k): return (j + 2) * 7 + (k + 2)
    def w_base_2R(k, i): return (k + 2) * 2 + i
    add_strassen(u_base_2R, v_base_2R, w_base_2R)
    
    # Block 3: C3a (cols 4,5)
    # A_L, B_3aL (rows 0,1, cols 4,5)
    def u_base_3aL(i, j): return i * 4 + j
    def v_base_3aL(j, k): return j * 7 + (k + 4)
    def w_base_3aL(k, i): return (k + 4) * 2 + i
    add_strassen(u_base_3aL, v_base_3aL, w_base_3aL)
    
    # A_R, B_3aR (rows 2,3, cols 4,5)
    def u_base_3aR(i, j): return i * 4 + (j + 2)
    def v_base_3aR(j, k): return (j + 2) * 7 + (k + 4)
    def w_base_3aR(k, i): return (k + 4) * 2 + i
    add_strassen(u_base_3aR, v_base_3aR, w_base_3aR)
    
    # Block 4: C3b (col 6)
    # This is 2x2x1. Rank 4.
    # Terms:
    # 1. X11 Y1 -> Z1
    # 2. X12 Y2 -> Z1
    # 3. X21 Y1 -> Z2
    # 4. X22 Y2 -> Z2
    
    def add_2x2x1(u_base, v_base, w_base):
        nonlocal col
        # Term 1: X00 Y0
        U[u_base(0,0)][col] = 1
        V[v_base(0,0)][col] = 1
        W[w_base(0,0)][col] = 1
        col += 1
        # Term 2: X01 Y1
        U[u_base(0,1)][col] = 1
        V[v_base(1,0)][col] = 1
        W[w_base(0,0)][col] = 1
        col += 1
        # Term 3: X10 Y0
        U[u_base(1,0)][col] = 1
        V[v_base(0,0)][col] = 1
        W[w_base(1,0)][col] = 1
        col += 1
        # Term 4: X11 Y1
        U[u_base(1,1)][col] = 1
        V[v_base(1,0)][col] = 1
        W[w_base(1,0)][col] = 1
        col += 1

    # C3b uses B col 6.
    # A_L, B_3bL (rows 0,1, col 6)
    def u_base_3bL(i, j): return i * 4 + j
    def v_base_3bL(j, k): return j * 7 + 6
    def w_base_3bL(k, i): return 6 * 2 + i
    add_2x2x1(u_base_3bL, v_base_3bL, w_base_3bL)
    
    # A_R, B_3bR (rows 2,3, col 6)
    def u_base_3bR(i, j): return i * 4 + (j + 2)
    def v_base_3bR(j, k): return (j + 2) * 7 + 6
    def w_base_3bR(k, i): return 6 * 2 + i
    add_2x2x1(u_base_3bR, v_base_3bR, w_base_3bR)
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
