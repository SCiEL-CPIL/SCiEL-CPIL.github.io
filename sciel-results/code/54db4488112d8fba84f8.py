import time
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Base 3x2x2 decomposition (Rank 11)
    # Local dims: n=3, m=2, p=2
    # U: 6x11, V: 4x11, W: 6x11
    
    # We store base terms as (u_idx, u_val, v_idx, v_val, w_idx, w_val)
    # But it's easier to store full vectors for the base case and then shift.
    
    base_U = np.zeros((6, 11))
    base_V = np.zeros((4, 11))
    base_W = np.zeros((6, 11))
    
    # Helper to add a term
    def add_base_term(u_idx, u_val, v_idx, v_val, w_idx, w_val, col):
        base_U[u_idx, col] += u_val
        base_V[v_idx, col] += v_val
        base_W[w_idx, col] += w_val

    # Strassen Terms for i in {0,1}
    # M1: (A00+A11)(B00+B11) -> C00+C01+C10+C11
    # A00: u[0], A11: u[3]
    # B00: v[0], B11: v[3]
    # C00: w[0], C01: w[3], C10: w[1], C11: w[4]
    add_base_term(0, 1, 0, 1, 0, 1, 0)
    add_base_term(3, 1, 0, 1, 0, 1, 0)
    add_base_term(0, 1, 3, 1, 0, 1, 0)
    add_base_term(3, 1, 3, 1, 0, 1, 0)
    # Wait, Strassen term is a product of sums.
    # (A00+A11) is one factor. (B00+B11) is another.
    # In tensor decomposition, a term is a rank-1 tensor.
    # M1 corresponds to 4 rank-1 components? No.
    # The term is $M_1 = (A_{00}+A_{11})(B_{00}+B_{11})$.
    # This contributes to $C_{00}, C_{01}, C_{10}, C_{11}$.
    # In the tensor decomposition $T_{ijk} = \sum_r U_{ij}^{(r)} V_{jk}^{(r)} W_{ki}^{(r)}$,
    # each $r$ is a rank-1 term.
    # Strassen's algorithm replaces 8 multiplications with 7.
    # The 7 "multiplications" are the rank-1 terms in the decomposition of the $2 \times 2 \times 2$ tensor.
    # So M1 is ONE rank-1 term?
    # No. $M_1$ is a scalar computed from A and B.
    # Then $C_{00} = M_1 + \dots$
    # This means the tensor is decomposed into 7 terms.
    # Term 1: $U$ has 1s at A00, A11. $V$ has 1s at B00, B11. $W$ has 1s at C00, C01, C10, C11.
    # Yes, this is a valid rank-1 term (outer product of vectors).
    # $U^{(1)} = [1, 0, 0, 1, 0, 0]^T$ (for 3x2x2, A is 2x2 part)
    # $V^{(1)} = [1, 0, 0, 1]^T$
    # $W^{(1)} = [1, 1, 0, 1, 1, 0]^T$ (C00, C10, C20, C01, C11, C21)
    # Indices for W in 3x2x2:
    # C[i,k] -> index k*3 + i.
    # C00 -> 0*3+0=0
    # C10 -> 0*3+1=1
    # C20 -> 0*3+2=2
    # C01 -> 1*3+0=3
    # C11 -> 1*3+1=4
    # C21 -> 1*3+2=5
    
    # Let's re-verify the W indices for Strassen terms.
    # M1 -> C00, C01, C10, C11.
    # W[0]=1, W[1]=1, W[3]=1, W[4]=1.
    
    # Correcting the add_base_term calls:
    # I need to clear base arrays first or just build them.
    
    base_U = np.zeros((6, 11))
    base_V = np.zeros((4, 11))
    base_W = np.zeros((6, 11))
    
    # Term 0: M1
    base_U[0, 0] = 1; base_U[3, 0] = 1
    base_V[0, 0] = 1; base_V[3, 0] = 1
    base_W[0, 0] = 1; base_W[1, 0] = 1; base_W[3, 0] = 1; base_W[4, 0] = 1
    
    # Term 1: M2 = (A10+A11)B00 -> C10, C11
    # A10: u[2], A11: u[3]
    # B00: v[0]
    # C10: w[1], C11: w[4]
    base_U[2, 1] = 1; base_U[3, 1] = 1
    base_V[0, 1] = 1
    base_W[1, 1] = 1; base_W[4, 1] = 1
    
    # Term 2: M3 = A00(B10-B11) -> C01, C11
    # A00: u[0]
    # B10: v[2], B11: v[3]
    # C01: w[3], C11: w[4]
    base_U[0, 2] = 1
    base_V[2, 2] = 1; base_V[3, 2] = -1
    base_W[3, 2] = 1; base_W[4, 2] = 1
    
    # Term 3: M4 = A11(B01-B00) -> C00, C10
    # A11: u[3]
    # B01: v[1], B00: v[0]
    # C00: w[0], C10: w[1]
    base_U[3, 3] = 1
    base_V[1, 3] = 1; base_V[0, 3] = -1
    base_W[0, 3] = 1; base_W[1, 3] = 1
    
    # Term 4: M5 = (A00+A10)B11 -> C01, C11
    # A00: u[0], A10: u[2]
    # B11: v[3]
    # C01: w[3], C11: w[4]
    base_U[0, 4] = 1; base_U[2, 4] = 1
    base_V[3, 4] = 1
    base_W[3, 4] = 1; base_W[4, 4] = 1
    
    # Term 5: M6 = (A10-A00)(B00+B01) -> C11
    # A10: u[2], A00: u[0]
    # B00: v[0], B01: v[1]
    # C11: w[4]
    base_U[2, 5] = 1; base_U[0, 5] = -1
    base_V[0, 5] = 1; base_V[1, 5] = 1
    base_W[4, 5] = 1
    
    # Term 6: M7 = (A01-A11)(B10+B11) -> C00, C10
    # A01: u[1], A11: u[3]
    # B10: v[2], B11: v[3]
    # C00: w[0], C10: w[1]
    base_U[1, 6] = 1; base_U[3, 6] = -1
    base_V[2, 6] = 1; base_V[3, 6] = 1
    base_W[0, 6] = 1; base_W[1, 6] = 1
    
    # Term 7: Schoolbook A20 B00 -> C20
    # A20: u[4]
    # B00: v[0]
    # C20: w[2]
    base_U[4, 7] = 1
    base_V[0, 7] = 1
    base_W[2, 7] = 1
    
    # Term 8: Schoolbook A21 B10 -> C20
    # A21: u[5]
    # B10: v[2]
    # C20: w[2]
    base_U[5, 8] = 1
    base_V[2, 8] = 1
    base_W[2, 8] = 1
    
    # Term 9: Schoolbook A20 B01 -> C21
    # A20: u[4]
    # B01: v[1]
    # C21: w[5]
    base_U[4, 9] = 1
    base_V[1, 9] = 1
    base_W[5, 9] = 1
    
    # Term 10: Schoolbook A21 B11 -> C21
    # A21: u[5]
    # B11: v[3]
    # C21: w[5]
    base_U[5, 10] = 1
    base_V[3, 10] = 1
    base_W[5, 10] = 1
    
    # Now construct full 3x4x4 matrices
    # Rank = 44
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for j_off in [0, 2]:
        for k_off in [0, 2]:
            for r in range(11):
                # Map base indices to global
                # Base U indices 0..5 correspond to i in 0..2, j' in 0..1
                # Base U index = i*2 + j'
                # Global U index = i*4 + (j_off + j')
                for i in range(3):
                    for jp in range(2):
                        base_u_idx = i*2 + jp
                        global_u_idx = i*4 + (j_off + jp)
                        U[global_u_idx, col] = base_U[base_u_idx, r]
                        
                # Base V indices 0..3 correspond to j' in 0..1, k' in 0..1
                # Base V index = j'*2 + k'
                # Global V index = (j_off + j')*4 + (k_off + k')
                for jp in range(2):
                    for kp in range(2):
                        base_v_idx = jp*2 + kp
                        global_v_idx = (j_off + jp)*4 + (k_off + kp)
                        V[global_v_idx, col] = base_V[base_v_idx, r]
                        
                # Base W indices 0..5 correspond to k' in 0..1, i in 0..2
                # Base W index = k'*3 + i
                # Global W index = (k_off + k')*3 + i
                for kp in range(2):
                    for i in range(3):
                        base_w_idx = kp*3 + i
                        global_w_idx = (k_off + kp)*3 + i
                        W[global_w_idx, col] = base_W[base_w_idx, r]
                        
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
