import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We will use a block decomposition strategy.
    # Split m=5 into m1=4, m2=1.
    # Split p=5 into p1=4, p2=1.
    # 
    # C (4x5) = C1 (4x4) + C2 (4x1)
    # C1 = A1 B11 + A2 B21
    # C2 = A1 B12 + A2 B22
    #
    # A1 is 4x4, A2 is 4x1
    # B11 is 4x4, B12 is 4x1, B21 is 1x4, B22 is 1x1
    #
    # Block 1: A1 B11 (4x4x4). We use Strassen (Rank 56).
    # Block 2: A2 B21 (4x1x4). Schoolbook (Rank 16).
    # Block 3: A1 B12 (4x4x1). Schoolbook (Rank 16).
    # Block 4: A2 B22 (4x1x1). Schoolbook (Rank 4).
    #
    # Total Rank = 56 + 16 + 16 + 4 = 92.
    
    rank = 92
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    
    # Helper to add a schoolbook term
    def add_term(i, j, k):
        nonlocal col
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
        
    # Helper to add a Strassen term for a 2x2x2 block
    # The block is defined by offsets (i_off, j_off, k_off) and size 2
    # Strassen requires 7 multiplications.
    # We need to map the Strassen multipliers to the global U, V, W indices.
    # A Strassen multiplier M is a product of a linear combination of A elements and B elements.
    # This results in multiple U, V, W entries for one "logical" multiplication?
    # No, the tensor decomposition is a sum of rank-1 tensors.
    # Strassen replaces 8 rank-1 tensors with 7 rank-1 tensors?
    # No, Strassen uses 7 *multiplications*, but each multiplication is of sums.
    # In the tensor decomposition context, a "multiplication" of sums corresponds to multiple rank-1 terms?
    # Actually, the rank of the tensor is the number of terms in the decomposition.
    # Strassen's algorithm computes the product using 7 multiplications.
    # This implies the rank is 7 for 2x2x2.
    # The 7 terms in the decomposition are not simply $A_{ij} B_{jk}$.
    # They are terms like $(A_{00}+A_{11}) (B_{00}+B_{11})$.
    # This expands to $A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
    # This is a sum of 4 rank-1 tensors.
    # So one Strassen "multiplication" step corresponds to 4 rank-1 terms in the expansion?
    # No, the rank of the tensor is the minimum number of rank-1 tensors needed to sum to it.
    # Strassen shows that the tensor can be written as a sum of 7 rank-1 tensors?
    # No, Strassen shows that the *algorithm* uses 7 multiplications.
    # The tensor decomposition rank is 7.
    # The 7 terms are:
    # 1. (A00+A11)(B00+B11) -> This is NOT a rank-1 tensor. It is a sum of 4.
    # Wait. The rank of the matrix multiplication tensor $T_{ijk}$ is the number of terms in $\sum_r U_{ir} V_{jr} W_{kr}$.
    # If we use Strassen, we define 7 intermediate variables $M_r$.
    # Each $M_r$ is a product of two linear forms.
    # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
    # This $M_1$ contributes to $C_{00}, C_{01}, C_{10}, C_{11}$.
    # The contribution of $M_1$ to $C_{00}$ is 1.
    # The contribution of $M_1$ to $C_{01}$ is 0?
    # Actually, $C_{00} = M_1 + M_4 - M_5 + M_7$.
    # This means the tensor entry $T_{000}$ (for $C_{00}$ from $A_{00}B_{00}$) is formed by the sum of these $M$'s.
    # The rank-1 tensors are not the $M$'s directly.
    # The decomposition is:
    # $C_{ik} = \sum_j A_{ij} B_{jk}$.
    # Strassen rewrites this as $\sum_{r=1}^7 L_{ik}^{(r)} M_r$?
    # No.
    # The standard definition of rank is the number of terms in $\sum_r U_{ir} V_{jr} W_{kr}$.
    # For Strassen, the 7 "multiplications" are not rank-1 tensors in the original $A,B$ space?
    # Actually, they are.
    # $M_1 = (A_{00}+A_{11})(B_{00}+B_{11}) = A_{00}B_{00} + A_{00}B_{11} + A_{11}B_{00} + A_{11}B_{11}$.
    # This sum of 4 products is represented by 4 rank-1 tensors in the expansion of $C$.
    # But Strassen counts this as 1 multiplication.
    # The *rank* of the tensor is the number of rank-1 tensors.
    # Strassen's algorithm implies the rank is 7?
    # Yes, the rank of the $2 \times 2 \times 2$ matrix multiplication tensor is 7.
    # The 7 rank-1 tensors are:
    # 1. $U = (1,0,0,1)^T, V = (1,0,0,1)^T, W = (1,0,0,1)^T$? No.
    # The vectors $U, V, W$ for the 7 terms are:
    # Term 1: $U=(1,0,0,1), V=(1,0,0,1), W=(1,1,1,1)$?
    # Let's check $W_{ik}$.
    # $C_{00} = M_1 + M_4 - M_5 + M_7$.
    # $C_{01} = M_3 + M_5$.
    # $C_{10} = M_2 + M_4$.
    # $C_{11} = M_1 - M_2 + M_3 + M_6$.
    #
    # So for Term 1 ($M_1$):
    # $U_1$ corresponds to $A_{00} + A_{11}$. So $U_1 = (1, 0, 0, 1)$ (indices 00, 01, 10, 11).
    # $V_1$ corresponds to $B_{00} + B_{11}$. So $V_1 = (1, 0, 0, 1)$.
    # $W_1$ corresponds to the coefficients of $M_1$ in $C$.
    # $C_{00}$ gets 1. $C_{01}$ gets 0. $C_{10}$ gets 0. $C_{11}$ gets 1.
    # So $W_1 = (1, 0, 0, 1)$.
    #
    # This is a valid rank-1 tensor.
    #
    # So I can hardcode the 7 Strassen vectors for $2 \times 2 \times 2$.
    # And use them for the 8 blocks in $4 \times 4 \times 4$.
    
    # Strassen vectors for 2x2x2
    # Indices: A: 00, 01, 10, 11. B: 00, 01, 10, 11. C: 00, 01, 10, 11.
    # U, V, W are length 4.
    S_U = [
        [1, 0, 0, 1], # M1: A00+A11
        [0, 0, 1, 1], # M2: A10+A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 0, 1, 0], # M5: A00+A10
        [-1, 0, 1, 0],# M6: A10-A00
        [0, 1, 0, -1] # M7: A01-A11
    ]
    S_V = [
        [1, 0, 0, 1], # M1: B00+B11
        [1, 0, 0, 0], # M2: B00
        [0, 1, 0, -1],# M3: B10-B11
        [1, -1, 0, 0],# M4: B01-B00
        [0, 0, 0, 1], # M5: B11
        [1, 1, 0, 0], # M6: B00+B01
        [0, 0, 1, 1]  # M7: B10+B11
    ]
    S_W = [
        [1, 0, 0, 1], # M1: C00, C11
        [0, 0, 1, 0], # M2: C10
        [0, 1, 0, 0], # M3: C01
        [1, 0, 1, 0], # M4: C00, C10
        [0, 1, 0, 0], # M5: C01 (Wait, C01 = M3+M5. C11 = M1-M2+M3+M6. C00 = M1+M4-M5+M7. C10 = M2+M4)
        # Let's re-verify W vectors.
        # C00: M1 + M4 - M5 + M7
        # C01: M3 + M5
        # C10: M2 + M4
        # C11: M1 - M2 + M3 + M6
        
        # W1: C00=1, C01=0, C10=0, C11=1 -> [1, 0, 0, 1]
        # W2: C00=0, C01=0, C10=1, C11=-1 -> [0, 0, 1, -1]
        # W3: C00=0, C01=1, C10=0, C11=1 -> [0, 1, 0, 1]
        # W4: C00=1, C01=0, C10=1, C11=0 -> [1, 0, 1, 0]
        # W5: C00=-1, C01=1, C10=0, C11=0 -> [-1, 1, 0, 0]
        # W6: C00=0, C01=0, C10=0, C11=1 -> [0, 0, 0, 1]
        # W7: C00=1, C01=0, C10=0, C11=0 -> [1, 0, 0, 0]
    ]
    
    # Corrected S_W
    S_W = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    # Block 1: A1 B11 (4x4x4)
    # 8 blocks of 2x2x2
    for bi in range(2): # 0, 1
        for bj in range(2): # 0, 1
            for bk in range(2): # 0, 1
                for r in range(7):
                    # U indices
                    for ui in range(2):
                        for uj in range(2):
                            val = S_U[r][ui*2+uj]
                            if val != 0:
                                i = bi*2 + ui
                                j = bj*2 + uj
                                U[i*m + j, col] = val
                    # V indices
                    for vi in range(2):
                        for vj in range(2):
                            val = S_V[r][vi*2+vj]
                            if val != 0:
                                j = bj*2 + vi
                                k = bk*2 + vj
                                V[j*p + k, col] = val
                    # W indices
                    for wi in range(2):
                        for wj in range(2):
                            val = S_W[r][wi*2+wj]
                            if val != 0:
                                k = bk*2 + wi
                                i = bi*2 + wj
                                W[k*n + i, col] = val
                    col += 1
                    
    # Block 2: A2 B21 (4x1x4)
    # A2 is col 4 of A. B21 is row 4 of B.
    # i in 0..3, k in 0..3.
    for i in range(4):
        for k in range(4):
            add_term(i, 4, k)
            
    # Block 3: A1 B12 (4x4x1)
    # A1 is cols 0..3. B12 is col 4 of B (rows 0..3).
    # i in 0..3, j in 0..3.
    for i in range(4):
        for j in range(4):
            add_term(i, j, 4)
            
    # Block 4: A2 B22 (4x1x1)
    # A2 is col 4. B22 is (4,4).
    for i in range(4):
        add_term(i, 4, 4)
        
    # Verify col is 92
    assert col == 92, f"Col is {col}"
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
