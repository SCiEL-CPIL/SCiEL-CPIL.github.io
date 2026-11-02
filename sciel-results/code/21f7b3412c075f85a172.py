import numpy as np
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-49 decomposition for 4x4 matrix multiplication via recursive Strassen.
    """
    n = 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * n)]
    V = [[0] * rank for _ in range(n * n)]
    W = [[0] * rank for _ in range(n * n)]
    
    def u_idx(i, k):
        return i * n + k
    
    def v_idx(k, j):
        return k * n + j
    
    def w_idx(i, j):
        return j * n + i
    
    # Outer Strassen terms
    outer_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    # Inner Strassen terms
    inner_terms = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1, (1,1): -1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}, {(0,1): 1, (1,1): 1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): -1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1}),
    ]
    
    for t in range(7):
        a_blocks, b_blocks, c_blocks = outer_terms[t]
        for s in range(7):
            a_elems, b_elems, c_elems = inner_terms[s]
            rank_idx = t * 7 + s
            
            for (br, bc), block_coeff in a_blocks.items():
                for (er, ec), elem_coeff in a_elems.items():
                    row = 2 * br + er
                    col = 2 * bc + ec
                    u = u_idx(row, col)
                    U[u][rank_idx] += block_coeff * elem_coeff
            
            for (br, bc), block_coeff in b_blocks.items():
                for (er, ec), elem_coeff in b_elems.items():
                    row = 2 * br + er
                    col = 2 * bc + ec
                    v = v_idx(row, col)
                    V[v][rank_idx] += block_coeff * elem_coeff
            
            for (br, bc), block_coeff in c_blocks.items():
                for (er, ec), elem_coeff in c_elems.items():
                    row = 2 * br + er
                    col = 2 * bc + ec
                    w = w_idx(row, col)
                    W[w][rank_idx] += block_coeff * elem_coeff
    
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    np.random.seed(42)
    A_test = np.random.randn(4, 4)
    B_test = np.random.randn(4, 4)
    C_expected = A_test @ B_test
    
    U4 = U_np.reshape(4, 4, 49)
    V4 = V_np.reshape(4, 4, 49)
    W4 = W_np.reshape(4, 4, 49)
    
    u_vals = np.einsum('ikr,ik->ir', U4, A_test)
    v_vals = np.einsum('kjr,kj->jr', V4, B_test)
    C_computed = np.einsum('jir,ir,jr->ij', W4, u_vals, v_vals)
    
    if not np.allclose(C_computed, C_expected, atol=1e-9):
        rank_fallback = 64
        U_f = [[0] * rank_fallback for _ in range(16)]
        V_f = [[0] * rank_fallback for _ in range(16)]
        W_f = [[0] * rank_fallback for _ in range(16)]
        
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U_f[i*4+k][r] = 1
                    V_f[k*4+j][r] = 1
                    W_f[j*4+i][r] = 1
        
        return {"U": U_f, "V": V_f, "W": W_f}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
