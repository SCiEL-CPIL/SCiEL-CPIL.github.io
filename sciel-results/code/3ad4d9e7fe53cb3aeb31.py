import numpy as np

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 102
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen_2x2x2(u_idx, v_idx, w_idx, scale=1):
        nonlocal col
        # u_idx: [u00, u01, u10, u11]
        # v_idx: [v00, v01, v10, v11]
        # w_idx: [w00, w01, w10, w11]
        
        # Term 1: (A00+A11)(B00+B11)
        U[u_idx[0]][col] += scale
        U[u_idx[3]][col] += scale
        V[v_idx[0]][col] += scale
        V[v_idx[3]][col] += scale
        W[w_idx[0]][col] += scale
        W[w_idx[3]][col] += scale
        col += 1
        
        # Term 2: (A10+A11)B00
        U[u_idx[2]][col] += scale
        U[u_idx[3]][col] += scale
        V[v_idx[0]][col] += scale
        W[w_idx[2]][col] += scale
        col += 1
        
        # Term 3: A00(B01-B11)
        U[u_idx[0]][col] += scale
        V[v_idx[1]][col] += scale
        V[v_idx[3]][col] += -scale
        W[w_idx[1]][col] += scale
        col += 1
        
        # Term 4: A11(B10-B00)
        U[u_idx[3]][col] += scale
        V[v_idx[2]][col] += scale
        V[v_idx[0]][col] += -scale
        W[w_idx[0]][col] += scale
        W[w_idx[2]][col] += scale
        col += 1
        
        # Term 5: (A00+A01)B11
        U[u_idx[0]][col] += scale
        U[u_idx[1]][col] += scale
        V[v_idx[3]][col] += scale
        W[w_idx[1]][col] += scale
        W[w_idx[3]][col] += scale
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[u_idx[2]][col] += scale
        U[u_idx[0]][col] += -scale
        V[v_idx[0]][col] += scale
        V[v_idx[1]][col] += scale
        W[w_idx[3]][col] += scale
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[u_idx[1]][col] += scale
        U[u_idx[3]][col] += -scale
        V[v_idx[2]][col] += scale
        V[v_idx[3]][col] += scale
        W[w_idx[0]][col] += scale
        W[w_idx[3]][col] += scale
        col += 1

    def get_u_idx(i, j):
        # i, j are 0..1 for 2x2 block
        return [i*m + j, i*m + (j+1), (i+1)*m + j, (i+1)*m + (j+1)]
        # Wait, U index is i*m+j.
        # For a 2x2 block starting at i0, j0:
        # u00: i0*m + j0
        # u01: i0*m + (j0+1)
        # u10: (i0+1)*m + j0
        # u11: (i0+1)*m + (j0+1)
        
    def get_v_idx(j, k):
        # j, k are 0..1 for 2x2 block
        # V index is j*p+k
        # For a 2x2 block starting at j0, k0:
        # v00: j0*p + k0
        # v01: j0*p + (k0+1)
        # v10: (j0+1)*p + k0
        # v11: (j0+1)*p + (k0+1)
        return [j*p + k, j*p + (k+1), (j+1)*p + k, (j+1)*p + (k+1)]

    def get_w_idx(k, i):
        # k, i are 0..1 for 2x2 block
        # W index is k*n+i
        # For a 2x2 block starting at k0, i0:
        # w00: k0*n + i0
        # w01: k0*n + (i0+1)
        # w10: (k0+1)*n + i0
        # w11: (k0+1)*n + (i0+1)
        return [k*n + i, k*n + (i+1), (k+1)*n + i, (k+1)*n + (i+1)]

    # 4x4x4 Strassen
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (1,0), (1,1)
    # C blocks: (0,0), (0,1), (1,0), (1,1)
    
    # M1 = (A00+A11)(B00+B11)
    u_m1 = get_u_idx(0, 0) + get_u_idx(1, 1) # No, this is not right.
    # We need to pass the indices for A00 and A11 as the "inputs" to the 2x2x2 Strassen.
    # But the 2x2x2 Strassen expects 4 indices for U.
    # Here we have 8 indices (4 for A00, 4 for A11).
    # So we cannot use the helper directly.
    
    # We must expand M1 = (A00+A11)(B00+B11) = A00B00 + A00B11 + A11B00 + A11B11.
    # Each term is a 2x2x2 multiplication.
    # So we call add_strassen_2x2x2 4 times for M1.
    # This will use 4*7 = 28 terms for M1.
    # Total 7*28 = 196 terms.
    # This is worse than 49.
    
    # So we must implement the 4x4x4 Strassen directly.
    # I will hardcode the 49 terms.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
