import time
import sys

DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_entries, v_entries, w_entries):
        """Add a rank-1 term."""
        U_cols.append([0] * (n * m))
        V_cols.append([0] * (m * p))
        W_cols.append([0] * (p * n))
        for idx, val in u_entries.items():
            U_cols[-1][idx] = val
        for idx, val in v_entries.items():
            V_cols[-1][idx] = val
        for idx, val in w_entries.items():
            W_cols[-1][idx] = val
    
    def add_laderman_block(i1, i2, i3, j1, j2, j3, k1, k2, k3):
        """Add Laderman's rank-23 decomposition for a 3x3x3 block."""
        # A positions
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a13 = i1 * m + j3
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        a23 = i2 * m + j3
        a31 = i3 * m + j1
        a32 = i3 * m + j2
        a33 = i3 * m + j3
        
        # B positions
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b13 = j1 * p + k3
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        b23 = j2 * p + k3
        b31 = j3 * p + k1
        b32 = j3 * p + k2
        b33 = j3 * p + k3
        
        # C positions
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c13 = k3 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        c23 = k3 * n + i2
        c31 = k1 * n + i3
        c32 = k2 * n + i3
        c33 = k3 * n + i3
        
        # Laderman's 23 multiplications for 3x3x3
        # M1 = (a11+a22+a33)(b11+b22+b33)
        add_term({a11: 1, a22: 1, a33: 1}, {b11: 1, b22: 1, b33: 1},
                 {c11: 1, c12: 1, c13: 1, c21: 1, c22: 1, c23: 1, c31: 1, c32: 1, c33: 1})
        
        # M2 = (a11+a22-a33)(b11+b22-b33)
        add_term({a11: 1, a22: 1, a33: -1}, {b11: 1, b22: 1, b33: -1},
                 {c11: 1, c12: 1, c13: -1, c21: 1, c22: 1, c23: -1, c31: -1, c32: -1, c33: 1})
        
        # M3 = (a11-a22+a33)(b11-b22+b33)
        add_term({a11: 1, a22: -1, a33: 1}, {b11: 1, b22: -1, b33: 1},
                 {c11: 1, c12: -1, c13: 1, c21: -1, c22: 1, c23: -1, c31: 1, c32: -1, c33: 1})
        
        # M4 = (a11-a22-a33)(b11-b22-b33)
        add_term({a11: 1, a22: -1, a33: -1}, {b11: 1, b22: -1, b33: -1},
                 {c11: 1, c12: -1, c13: -1, c21: -1, c22: 1, c23: 1, c31: -1, c32: 1, c33: 1})
        
        # M5 = a11*(b12+b13-b22-b23)
        add_term({a11: 1}, {b12: 1, b13: 1, b22: -1, b23: -1},
                 {c12: 1, c13: 1, c22: -1, c23: -1})
        
        # M6 = a12*(b11+b13-b21-b23)
        add_term({a12: 1}, {b11: 1, b13: 1, b21: -1, b23: -1},
                 {c11: 1, c13: 1, c21: -1, c23: -1})
        
        # M7 = a13*(b11+b12-b21-b22)
        add_term({a13: 1}, {b11: 1, b12: 1, b21: -1, b22: -1},
                 {c11: 1, c12: 1, c21: -1, c22: -1})
        
        # M8 = a21*(b12+b23-b13-b22)
        add_term({a21: 1}, {b12: 1, b23: 1, b13: -1, b22: -1},
                 {c12: 1, c23: 1, c13: -1, c22: -1})
        
        # M9 = a22*(b13+b21-b11-b23)
        add_term({a22: 1}, {b13: 1, b21: 1, b11: -1, b23: -1},
                 {c13: 1, c21: 1, c11: -1, c23: -1})
        
        # M10 = a23*(b11+b22-b12-b21)
        add_term({a23: 1}, {b11: 1, b22: 1, b12: -1, b21: -1},
                 {c11: 1, c22: 1, c12: -1, c21: -1})
        
        # M11 = a31*(b12+b33-b13-b32)
        add_term({a31: 1}, {b12: 1, b33: 1, b13: -1, b32: -1},
                 {c12: 1, c33: 1, c13: -1, c32: -1})
        
        # M12 = a32*(b13+b31-b11-b33)
        add_term({a32: 1}, {b13: 1, b31: 1, b11: -1, b33: -1},
                 {c13: 1, c31: 1, c11: -1, c33: -1})
        
        # M13 = a33*(b11+b32-b12-b31)
        add_term({a33: 1}, {b11: 1, b32: 1, b12: -1, b31: -1},
                 {c11: 1, c32: 1, c12: -1, c31: -1})
        
        # M14 = (a11+a33)(b21+b23-b11-b13)
        add_term({a11: 1, a33: 1}, {b21: 1, b23: 1, b11: -1, b13: -1},
                 {c21: 1, c23: 1, c11: -1, c13: -1, c31: -1, c33: -1})
        
        # M15 = (a11+a22)(b31+b33-b11-b13)
        add_term({a11: 1, a22: 1}, {b31: 1, b33: 1, b11: -1, b13: -1},
                 {c31: 1, c33: 1, c11: -1, c13: -1, c21: -1, c23: -1})
        
        # M16 = (a12+a32)(b21+b23-b11-b13)
        add_term({a12: 1, a32: 1}, {b21: 1, b23: 1, b11: -1, b13: -1},
                 {c21: 1, c23: 1, c11: -1, c13: -1, c31: -1, c33: -1})
        
        # M17 = (a12+a23)(b31+b33-b11-b13)
        add_term({a12: 1, a23: 1}, {b31: 1, b33: 1, b11: -1, b13: -1},
                 {c31: 1, c33: 1, c11: -1, c13: -1, c21: -1, c23: -1})
        
        # M18 = (a13+a22)(b21+b23-b11-b13)
        add_term({a13: 1, a22: 1}, {b21: 1, b23: 1, b11: -1, b13: -1},
                 {c21: 1, c23: 1, c11: -1, c13: -1, c31: -1, c33: -1})
        
        # M19 = (a13+a31)(b31+b33-b11-b13)
        add_term({a13: 1, a31: 1}, {b31: 1, b33: 1, b11: -1, b13: -1},
                 {c31: 1, c33: 1, c11: -1, c13: -1, c21: -1, c23: -1})
        
        # M20 = (a21+a33)(b12+b13-b22-b23)
        add_term({a21: 1, a33: 1}, {b12: 1, b13: 1, b22: -1, b23: -1},
                 {c12: 1, c13: 1, c22: -1, c23: -1, c32: -1, c33: -1})
        
        # M21 = (a21+a32)(b11+b13-b21-b23)
        add_term({a21: 1, a32: 1}, {b11: 1, b13: 1, b21: -1, b23: -1},
                 {c11: 1, c13: 1, c21: -1, c23: -1, c31: -1, c33: -1})
        
        # M22 = (a22+a31)(b11+b12-b21-b22)
        add_term({a22: 1, a31: 1}, {b11: 1, b12: 1, b21: -1, b22: -1},
                 {c11: 1, c12: 1, c21: -1, c22: -1, c31: -1, c32: -1})
        
        # M23 = (a23+a32)(b11+b12-b21-b22)
        add_term({a23: 1, a32: 1}, {b11: 1, b12: 1, b21: -1, b22: -1},
                 {c11: 1, c12: 1, c21: -1, c22: -1, c31: -1, c32: -1})
    
    def add_strassen_block(i1, i2, j1, j2, k1, k2):
        """Add Strassen's rank-7 decomposition for a 2x2x2 block."""
        a11 = i1 * m + j1
        a12 = i1 * m + j2
        a21 = i2 * m + j1
        a22 = i2 * m + j2
        
        b11 = j1 * p + k1
        b12 = j1 * p + k2
        b21 = j2 * p + k1
        b22 = j2 * p + k2
        
        c11 = k1 * n + i1
        c12 = k2 * n + i1
        c21 = k1 * n + i2
        c22 = k2 * n + i2
        
        # M1 = (a11+a22)(b11+b22)
        add_term({a11: 1, a22: 1}, {b11: 1, b22: 1}, {c11: 1, c22: 1})
        
        # M2 = (a21+a22)(b11)
        add_term({a21: 1, a22: 1}, {b11: 1}, {c21: 1, c22: -1})
        
        # M3 = a11*(b12-b22)
        add_term({a11: 1}, {b12: 1, b22: -1}, {c12: 1, c22: 1})
        
        # M4 = a22*(b21-b11)
        add_term({a22: 1}, {b21: 1, b11: -1}, {c11: 1, c21: 1})
        
        # M5 = (a11+a12)(b22)
        add_term({a11: 1, a12: 1}, {b22: 1}, {c11: -1, c12: 1})
        
        # M6 = (a21-a11)(b11+b12)
        add_term({a21: 1, a11: -1}, {b11: 1, b12: 1}, {c22: 1})
        
        # M7 = (a12-a22)(b21+b22)
        add_term({a12: 1, a22: -1}, {b21: 1, b22: 1}, {c11: 1})
    
    def add_schoolbook(i1, i2, j1, j2, k1, k2):
        """Add schoolbook terms for a block."""
        for i in range(i1, i2):
            for j in range(j1, j2):
                for k in range(k1, k2):
                    if time.time() > deadline:
                        return
                    add_term({i * m + j: 1}, {j * p + k: 1}, {k * n + i: 1})
    
    # Strategy: Use Laderman on 3x3x3 blocks where possible
    # Block 1: i={0,1,2}, j={0,1,2}, k={0,1,2} -> Laderman rank 23
    add_laderman_block(0, 1, 2, 0, 1, 2, 0, 1, 2)
    
    # Block 2: i={0,1,2}, j={0,1,2}, k={3,4,5} -> Laderman rank 23
    add_laderman_block(0, 1, 2, 0, 1, 2, 3, 4, 5)
    
    # Remaining blocks with j={3,4} (2 cols)
    # i={0,1,2}, j={3,4}, k={0,1,2} -> schoolbook 3*2*3=18
    add_schoolbook(0, 3, 3, 5, 0, 3)
    
    # i={0,1,2}, j={3,4}, k={3,4,5} -> schoolbook 3*2*3=18
    add_schoolbook(0, 3, 3, 5, 3, 6)
    
    # i={0,1,2}, j={3,4}, k={6} -> schoolbook 3*2*1=6
    add_schoolbook(0, 3, 3, 5, 6, 7)
    
    rank = len(U_cols)
    
    # Convert to matrix format
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for c in range(rank):
        for r in range(n * m):
            U[r][c] = U_cols[c][r]
        for r in range(m * p):
            V[r][c] = V_cols[c][r]
        for r in range(p * n):
            W[r][c] = W_cols[c][r]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END

if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    rank = len(result['U'][0])
    
    assert len(result['U']) == n * m
    assert len(result['V']) == m * p
    assert len(result['W']) == p * n
    
    import random
    random.seed(42)
    A = [[random.randint(-5, 5) for _ in range(m)] for _ in range(n)]
    B = [[random.randint(-5, 5) for _ in range(p)] for _ in range(m)]
    
    C = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
    
    C_decomp = [[0 for _ in range(p)] for _ in range(n)]
    for i in range(n):
        for k in range(p):
            val = 0
            for r in range(rank):
                ua = 0
                for jj in range(m):
                    ua += result['U'][i * m + jj][r] * A[i][jj]
                vb = 0
                for jj in range(m):
                    vb += result['V'][jj * p + k][r] * B[jj][k]
                val += result['W'][k * n + i][r] * ua * vb
            C_decomp[i][k] = val
            
    if C_decomp != C:
        print("Verification failed!", file=sys.stderr)
    else:
        print(f"Verification passed. Rank: {rank}", file=sys.stderr)
