import numpy as np
import sys
import time

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Target tensor for verification
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def _verify(U, V, W):
        """Verify that U, V, W decompose T exactly (within tolerance)."""
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(R, T, atol=1e-10, rtol=0)

    def _build_rank_18():
        """
        Constructs the rank-18 decomposition for 2x2 * 2x5.
        """
        rank = 18
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        # Helper to add a term
        def add_term(u_vals, v_vals, w_vals, idx):
            for idx_u, val in u_vals.items():
                U[idx_u, idx] += val
            for idx_v, val in v_vals.items():
                V[idx_v, idx] += val
            for idx_w, val in w_vals.items():
                W[idx_w, idx] += val

        # --- Part 1: Strassen for Cols 0,1 ---
        # M1 = (a00+a11)(b00+b11)
        add_term({0:1, 3:1}, {0:1, 6:1}, {0:1, 3:1}, 0)
        
        # M2 = (a10+a11)b00
        add_term({2:1, 3:1}, {0:1}, {1:1}, 1)
        
        # M3 = a00(b11-b10)
        add_term({0:1}, {6:1, 5:-1}, {1:1, 3:-1}, 2)
        
        # M4 = a11(b00-b10)
        add_term({3:1}, {0:1, 5:-1}, {0:-1, 2:1}, 3)
        
        # M5 = (a00+a01)b10
        add_term({0:1, 1:1}, {5:1}, {0:1}, 4)
        
        # M6 = (a10+a11)(b01-b00)
        add_term({2:1, 3:1}, {1:1, 0:-1}, {2:1}, 5)
        
        # M7 = (a01-a10)(b00+b10)
        add_term({1:1, 2:-1}, {0:1, 5:1}, {3:1}, 6)

        # --- Part 2: Strassen for Cols 2,3 ---
        # M8 = (a00+a11)(b02+b12)
        add_term({0:1, 3:1}, {2:1, 7:1}, {4:1, 7:1}, 7)
        
        # M9 = (a10+a11)b02
        add_term({2:1, 3:1}, {2:1}, {5:1}, 8)
        
        # M10 = a00(b13-b12)
        add_term({0:1}, {8:1, 7:-1}, {5:1, 7:-1}, 9)
        
        # M11 = a11(b02-b12)
        add_term({3:1}, {2:1, 7:-1}, {4:-1, 6:1}, 10)
        
        # M12 = (a00+a01)b12
        add_term({0:1, 1:1}, {7:1}, {4:1}, 11)
        
        # M13 = (a10+a11)(b03-b02)
        add_term({2:1, 3:1}, {3:1, 2:-1}, {6:1}, 12)
        
        # M14 = (a01-a10)(b02+b12)
        add_term({1:1, 2:-1}, {2:1, 7:1}, {7:1}, 13)

        # --- Part 3: Schoolbook for Col 4 ---
        # c04 = a00*b04 + a01*b14
        add_term({0:1}, {4:1}, {8:1}, 14)
        add_term({1:1}, {9:1}, {8:1}, 15)
        
        # c14 = a10*b04 + a11*b14
        add_term({2:1}, {4:1}, {9:1}, 16)
        add_term({3:1}, {9:1}, {9:1}, 17)

        if _verify(U, V, W):
            return U, V, W
        return None

    U, V, W = _build_rank_18()
    
    if U is None:
        # Fallback to schoolbook if verification fails (should not happen)
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        for c in range(rank):
            i = c // (m*p)
            rem = c % (m*p)
            j = rem // p
            k = rem % p
            U[i*m+j, c] = 1.0
            V[j*p+k, c] = 1.0
            W[k*n+i, c] = 1.0

    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
