import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _seed():
    """Schoolbook rank-42 decomposition (fallback)."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W exactly decompose the tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = i * m + j
                v = j * p + k
                w = k * n + i
                s = 0
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 1:
                    return False
    for u in range(n * m):
        i, j = divmod(u, m)
        for v in range(m * p):
            j2, k = divmod(v, p)
            for w in range(p * n):
                k2, i2 = divmod(w, n)
                if j == j2 and k == k2 and i == i2:
                    continue
                s = 0
                for r in range(rank):
                    s += U[u][r] * V[v][r] * W[w][r]
                if s != 0:
                    return False
    return True

def _build_strassen_222():
    """
    Returns U, V, W for M_{2,2,2} with rank 7.
    U: 4x7, V: 4x7, W: 4x7.
    Indices: u=2i+j, v=2j+k, w=2k+i.
    """
    U = [[0]*7 for _ in range(4)]
    V = [[0]*7 for _ in range(4)]
    W = [[0]*7 for _ in range(4)]
    
    # m1: (a00+a11)(b00+b11) -> C00+=1, C11+=1
    U[0][0] = 1; U[3][0] = 1
    V[0][0] = 1; V[3][0] = 1
    W[0][0] = 1; W[3][0] = 1
    
    # m2: (a10+a11)b00 -> C10+=1, C11-=1
    U[2][1] = 1; U[3][1] = 1
    V[0][1] = 1
    W[2][1] = 1; W[3][1] = -1
    
    # m3: a00(b01-b11) -> C01+=1, C11+=1
    U[0][2] = 1
    V[1][2] = 1; V[3][2] = -1
    W[1][2] = 1; W[3][2] = 1
    
    # m4: a11(b10-b00) -> C00+=1, C10+=1
    U[3][3] = 1
    V[2][3] = 1; V[0][3] = -1
    W[0][3] = 1; W[2][3] = 1
    
    # m5: (a00+a01)b11 -> C00-=1, C01+=1
    U[0][4] = 1; U[1][4] = 1
    V[3][4] = 1
    W[0][4] = -1; W[1][4] = 1
    
    # m6: (a10-a00)(b00+b01) -> C11+=1
    U[2][5] = 1; U[0][5] = -1
    V[0][6] = 1; V[1][6] = 1
    W[3][6] = 1
    
    # m7: (a01-a11)(b10+b11) -> C00+=1
    U[1][6] = 1; U[3][6] = -1
    V[2][6] = 1; V[3][6] = 1
    W[0][6] = 1
    
    return U, V, W

def _build_linear_21p(p, u_offset, v_offset, w_offset):
    """
    Build M_{2,1,p} rank 2p.
    u = u_offset + i (i in 0,1)
    v = v_offset + k (k in 0..p-1)
    w = w_offset + 2k + i
    """
    rank = 2 * p
    U = [[0]*rank for _ in range(2)] # Only 2 rows used locally, but we return full 6x? No, return local
    # Actually, let's just return the non-zero entries as a list of tuples
    entries = []
    for k in range(p):
        for i in range(2):
            r = k * 2 + i
            u_idx = u_offset + i
            v_idx = v_offset + k
            w_idx = w_offset + 2 * k + i
            entries.append((u_idx, v_idx, w_idx, r, 1, 1, 1))
    return entries

def _build_linear_221(u_offset, v_offset, w_offset):
    """
    Build M_{2,2,1} rank 4.
    u = u_offset + 2i + j
    v = v_offset + 2j
    w = w_offset + i
    """
    rank = 4
    entries = []
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            u_idx = u_offset + 2 * i + j
            v_idx = v_offset + 2 * j
            w_idx = w_offset + i
            entries.append((u_idx, v_idx, w_idx, r, 1, 1, 1))
    return entries

def construct():
    n, m, p = DIMENSIONS
    total_rank = 39
    U = [[0] * total_rank for _ in range(6)]
    V = [[0] * total_rank for _ in range(21)]
    W = [[0] * total_rank for _ in range(14)]
    
    col_offset = 0
    
    # 1. M_{2,2,2} (part of M_{2,3,3})
    # k in {0,1}
    U_sub, V_sub, W_sub = _build_strassen_222()
    for i in range(4):
        for j in range(4):
            for k in range(7):
                # Map local indices to global
                # u_local = 2i_l + j_l -> i_l, j_l
                i_l, j_l = divmod(i, 2)
                j_l2, k_l = divmod(j, 2)
                k_l2, i_l2 = divmod(k, 2)
                
                # Global indices for M_{2,3,7}
                # u = i*m + j = 2*i_l + j_l
                # v = j*p + k = j_l*7 + k_l (since k_l in 0,1)
                # w = k*n + i = k_l*2 + i_l
                u_g = 2 * i_l + j_l
                v_g = j_l * 7 + k_l
                w_g = k_l * 2 + i_l
                
                val = U_sub[i][k]
                if val != 0:
                    U[u_g][col_offset + k] += val
                
                val = V_sub[j][k]
                if val != 0:
                    V[v_g][col_offset + k] += val
                
                val = W_sub[k][k] # Wait, W index is k_local
                # W_sub row index is w_local = 2*k_l + i_l
                # In _build_strassen, W is 4x7. Row index is w_local.
                # So W_sub[k][col] is correct if k is w_local.
                # But in the loop above, k is the column index (rank dim).
                # We need to map w_local to w_g.
                # w_local = 2*k_l + i_l.
                # We already have i_l, k_l from divmod of local indices?
                # No, the loop variable k is the rank index (0..6).
                # The row index of W_sub is w_local.
                # So we iterate over w_local.
                pass # Logic fix below

    # Let's re-do the assembly more carefully using explicit mapping functions.
    
    U = [[0] * total_rank for _ in range(6)]
    V = [[0] * total_rank for _ in range(21)]
    W = [[0] * total_rank for _ in range(14)]
    
    col = 0
    
    # Part 1: M_{2,3,3}
    
    # 1a. M_{2,2,2} for k in {0,1}
    U_s, V_s, W_s = _build_strassen_222()
    for u_l in range(4):
        i_l, j_l = divmod(u_l, 2)
        u_g = 2 * i_l + j_l
        for r in range(7):
            if U_s[u_l][r] != 0:
                U[u_g][col + r] = U_s[u_l][r]
                
    for v_l in range(4):
        j_l, k_l = divmod(v_l, 2)
        v_g = j_l * 7 + k_l
        for r in range(7):
            if V_s[v_l][r] != 0:
                V[v_g][col + r] = V_s[v_l][r]
                
    for w_l in range(4):
        k_l, i_l = divmod(w_l, 2)
        w_g = k_l * 2 + i_l
        for r in range(7):
            if W_s[w_l][r] != 0:
                W[w_g][col + r] = W_s[w_l][r]
    col += 7
    
    # 1b. M_{2,2,1} for k=2
    # u = 2i+j, v = 2j+2, w = 4+i
    for i in range(2):
        for j in range(2):
            r = i * 2 + j
            u_g = 2 * i + j
            v_g = 2 * j + 2
            w_g = 4 + i
            U[u_g][col + r] = 1
            V[v_g][col + r] = 1
            W[w_g][col + r] = 1
    col += 4
    
    # 1c. M_{2,1,3} for j=2, k in {0,1,2}
    # u = 2i+2, v = 4+k, w = 2k+i
    for k in range(3):
        for i in range(2):
            r = k * 2 + i
            u_g = 2 * i + 2
            v_g = 4 + k
            w_g = 2 * k + i
            U[u_g][col + r] = 1
            V[v_g][col + r] = 1
            W[w_g][col + r] = 1
    col += 6
    
    # Part 2: M_{2,3,4}
    
    # 2a. M_{2,2,2} for k in {3,4} (local k' in 0,1)
    U_s, V_s, W_s = _build_strassen_222()
    for u_l in range(4):
        i_l, j_l = divmod(u_l, 2)
        u_g = 2 * i_l + j_l
        for r in range(7):
            if U_s[u_l][r] != 0:
                U[u_g][col + r] = U_s[u_l][r]
                
    for v_l in range(4):
        j_l, k_l = divmod(v_l, 2)
        # Global k = 3 + k_l
        v_g = j_l * 7 + (3 + k_l)
        for r in range(7):
            if V_s[v_l][r] != 0:
                V[v_g][col + r] = V_s[v_l][r]
                
    for w_l in range(4):
        k_l, i_l = divmod(w_l, 2)
        # Global k = 3 + k_l
        w_g = (3 + k_l) * 2 + i_l
        for r in range(7):
            if W_s[w_l][r] != 0:
                W[w_g][col + r] = W_s[w_l][r]
    col += 7
    
    # 2b. M_{2,2,2} for k in {5,6} (local k' in 2,3)
    U_s, V_s, W_s = _build_strassen_222()
    for u_l in range(4):
        i_l, j_l = divmod(u_l, 2)
        u_g = 2 * i_l + j_l
        for r in range(7):
            if U_s[u_l][r] != 0:
                U[u_g][col + r] = U_s[u_l][r]
                
    for v_l in range(4):
        j_l, k_l = divmod(v_l, 2)
        # Global k = 5 + k_l (since local 0,1 map to 5,6)
        v_g = j_l * 7 + (5 + k_l)
        for r in range(7):
            if V_s[v_l][r] != 0:
                V[v_g][col + r] = V_s[v_l][r]
                
    for w_l in range(4):
        k_l, i_l = divmod(w_l, 2)
        # Global k = 5 + k_l
        w_g = (5 + k_l) * 2 + i_l
        for r in range(7):
            if W_s[w_l][r] != 0:
                W[w_g][col + r] = W_s[w_l][r]
    col += 7
    
    # 2c. M_{2,1,4} for j=2, k in {3,4,5,6}
    # u = 2i+2, v = 14+k, w = 2k+i
    for k in range(3, 7):
        for i in range(2):
            r = (k - 3) * 2 + i
            u_g = 2 * i + 2
            v_g = 14 + k
            w_g = 2 * k + i
            U[u_g][col + r] = 1
            V[v_g][col + r] = 1
            W[w_g][col + r] = 1
    col += 8
    
    # Convert to Fractions for exactness check
    Uf = [[Fraction(x) for x in row] for row in U]
    Vf = [[Fraction(x) for x in row] for row in V]
    Wf = [[Fraction(x) for x in row] for row in W]
    
    # Verify
    if _check_decomposition(Uf, Vf, Wf, n, m, p):
        return {"U": Uf, "V": Vf, "W": Wf}
    else:
        return _seed()

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
