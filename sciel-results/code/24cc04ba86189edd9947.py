import numpy as np
import time

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140

    # --- Part 1: Strassen 4x4x4 (Rank 49) ---
    
    # Define 2x2x2 Strassen terms (rank 7)
    # U, V, W are length 4 vectors.
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    def get_strassen_2x2_terms():
        terms = []
        # Term 0: A00*B00 -> C00
        U = np.zeros(4); U[0] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1
        terms.append((U, V, W))
        
        # Term 1: A01*B11 -> C01
        U = np.zeros(4); U[1] = 1
        V = np.zeros(4); V[3] = 1
        W = np.zeros(4); W[2] = 1
        terms.append((U, V, W))
        
        # Term 2: A10*B00 -> C10
        U = np.zeros(4); U[2] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[1] = 1
        terms.append((U, V, W))
        
        # Term 3: A11*B10 -> C11
        U = np.zeros(4); U[3] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[3] = 1
        terms.append((U, V, W))
        
        # Term 4: (A00+A10)*B10 -> C01+C11
        # U: (0,0) and (1,0)
        U = np.zeros(4); U[0] = 1; U[2] = 1
        V = np.zeros(4); V[2] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        # Term 5: (A01+A11)*B00 -> C00+C10
        U = np.zeros(4); U[1] = 1; U[3] = 1
        V = np.zeros(4); V[0] = 1
        W = np.zeros(4); W[0] = 1; W[1] = 1
        terms.append((U, V, W))
        
        # Term 6: (A00-A11)*(B01+B11) -> C01+C11
        U = np.zeros(4); U[0] = 1; U[3] = -1
        V = np.zeros(4); V[1] = 1; V[3] = 1
        W = np.zeros(4); W[2] = 1; W[3] = 1
        terms.append((U, V, W))
        
        return terms

    base_terms = get_strassen_2x2_terms()
    
    # Define Strassen 4x4x4 block mappings
    # Each entry is a dict with 'A', 'B', 'C' keys.
    # Values are lists of (block_row, block_col, sign).
    strassen_defs = [
        { # M1
            'A': [(0,0,1), (1,1,1)],
            'B': [(0,0,1), (1,1,1)],
            'C': [(0,0,1), (1,1,1)]
        },
        { # M2
            'A': [(1,0,1), (1,1,1)],
            'B': [(0,0,1)],
            'C': [(1,0,1), (1,1,-1)]
        },
        { # M3
            'A': [(0,0,1)],
            'B': [(1,0,1), (1,1,-1)],
            'C': [(0,1,1), (1,1,1)]
        },
        { # M4
            'A': [(1,1,1)],
            'B': [(0,1,1), (0,0,-1)],
            'C': [(0,0,1), (1,0,1)]
        },
        { # M5
            'A': [(0,0,1), (0,1,1)],
            'B': [(1,1,1)],
            'C': [(0,1,1)]
        },
        { # M6
            'A': [(1,0,1), (0,0,-1)],
            'B': [(0,0,1), (0,1,1)],
            'C': [(1,1,1)]
        },
        { # M7
            'A': [(0,1,1), (1,1,-1)],
            'B': [(1,0,1), (1,1,1)],
            'C': [(0,0,1)]
        }
    ]

    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))

    for k, prod_def in enumerate(strassen_defs):
        for i, (u2, v2, w2) in enumerate(base_terms):
            col_idx = k * 7 + i
            
            u4 = np.zeros(16)
            v4 = np.zeros(16)
            w4 = np.zeros(16)
            
            # Expand U
            for (br, bc, sign) in prod_def['A']:
                for r in range(2):
                    for c in range(2):
                        u4[(2*br+r)*4 + (2*bc+c)] += sign * u2[r*2+c]
                        
            # Expand V
            for (br, bc, sign) in prod_def['B']:
                for r in range(2):
                    for c in range(2):
                        v4[(2*br+r)*4 + (2*bc+c)] += sign * v2[r*2+c]
                        
            # Expand W
            # W is indexed by (k, i) in 2x2, mapping to (col, row) in 4x4?
            # In 2x2x2, W[k*2+i] corresponds to C[i][k].
            # In 4x4x4, W[(k*4+i)] corresponds to C[i][k].
            # Block (br, bc) of C corresponds to rows 2*br..2*br+1, cols 2*bc..2*bc+1.
            # So global row i_g = 2*br + r, global col k_g = 2*bc + c.
            # W index = k_g * 4 + i_g.
            # 2x2 W index = c * 2 + r.
            for (br, bc, sign) in prod_def['C']:
                for r in range(2):
                    for c in range(2):
                        # r is row offset, c is col offset in 2x2 block
                        # 2x2 W index is c*2 + r
                        # 4x4 W index is (2*bc+c)*4 + (2*br+r)
                        w4[(2*bc+c)*4 + (2*br+r)] += sign * w2[c*2+r]
                        
            U_4x4[:, col_idx] = u4
            V_4x4[:, col_idx] = v4
            W_4x4[:, col_idx] = w4

    # --- Part 2: Schoolbook 4x4x3 (Rank 48) ---
    
    U_sb = np.zeros((16, 48))
    V_sb = np.zeros((28, 48))
    W_sb = np.zeros((28, 48))
    
    col = 0
    for i in range(4):
        for j in range(4):
            for k in range(3):
                # Global k is 4+k
                gk = 4 + k
                U_sb[i*4+j, col] = 1
                V_sb[j*7+gk, col] = 1
                W_sb[gk*4+i, col] = 1
                col += 1

    # --- Combine ---
    
    U = np.hstack([U_4x4, U_sb])
    V = np.vstack([V_4x4, np.zeros((12, 49))]) # V_4x4 is 16x49, V_sb is 28x48
    # Wait, V_4x4 is for the first 4 columns of B (m=4, p=4 part).
    # V_sb is for the last 3 columns of B (m=4, p=3 part).
    # Total V shape: (m*p, rank) = (28, 97).
    # V_4x4 is 16x49. We need to pad it to 28x49.
    # The first 16 rows of V correspond to B columns 0-3.
    # The next 12 rows correspond to B columns 4-6.
    # In the Strassen part, B is 4x4, so only first 16 rows are used.
    # In the Schoolbook part, B is 4x3, so rows 16-27 are used.
    
    V_full = np.zeros((28, 97))
    V_full[:16, :49] = V_4x4
    V_full[:, 49:] = V_sb
    
    W_full = np.zeros((28, 97))
    W_full[:16, :49] = W_4x4
    W_full[:, 49:] = W_sb
    
    # Verify
    def verify(U, V, W, n, m, p, rank):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    if abs(total - 1.0) > 1e-9:
                        return False
        return True

    if verify(U, V, W, n, m, p, 97):
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook 112
        U_fb = np.zeros((16, 112))
        V_fb = np.zeros((28, 112))
        W_fb = np.zeros((28, 112))
        col = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U_fb[i*4+j, col] = 1
                    V_fb[j*7+k, col] = 1
                    W_fb[k*4+i, col] = 1
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
