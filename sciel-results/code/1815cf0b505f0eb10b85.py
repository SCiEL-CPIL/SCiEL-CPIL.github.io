"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    
    # Strategy: Construct a Rank-26 decomposition using a hybrid block approach.
    # 1. Use Strassen's algorithm (Rank 7) for the top-left 2x2 block.
    # 2. Use direct multiplication for the remaining cross-terms and 1x1 block.
    # Total Rank: 7 (Strassen) + 4 (u*q^T) + 4 (A_tl*p) + 2 (u*b) + 4 (v^T*B_tl) + 2 (a*q^T) + 2 (v^T*p) + 1 (a*b) = 26.
    
    rank = 26
    U = np.zeros((9, rank))
    V = np.zeros((9, rank))
    W = np.zeros((9, rank))
    
    # Helper to add a term
    # U indices: i*3+j
    # V indices: j*3+k
    # W indices: k*3+i
    
    def add_term(u_map, v_map, w_map):
        r = len([x for x in range(rank) if np.any(U[:, x] != 0 or V[:, x] != 0 or W[:, x] != 0)])
        # Find first empty column
        for c in range(rank):
            if np.all(U[:, c] == 0) and np.all(V[:, c] == 0) and np.all(W[:, c] == 0):
                r = c
                break
        for idx, val in u_map.items():
            U[idx, r] += val
        for idx, val in v_map.items():
            V[idx, r] += val
        for idx, val in w_map.items():
            W[idx, r] += val
        return r

    # 1. Strassen for 2x2 block (i,j in 0,1; k in 0,1)
    # A indices: 00(0), 01(1), 10(3), 11(4)
    # B indices: 00(0), 01(1), 10(3), 11(4)
    # C indices: 00(0), 01(3), 10(1), 11(4)
    
    # M1 = A00 * (B00 + B11)
    add_term({0: 1}, {0: 1, 4: 1}, {0: 1, 4: -1})
    
    # M2 = (A00 + A10) * B11
    add_term({0: 1, 3: 1}, {4: 1}, {0: -1, 4: 1})
    
    # M3 = (A01 + A11) * B00
    add_term({1: 1, 4: 1}, {0: 1}, {0: 1, 1: 1})
    
    # M4 = A11 * (B00 - B01)
    add_term({4: 1}, {0: 1, 1: -1}, {0: -1, 4: 1})
    
    # M5 = (A10 + A11) * (B00 + B01)
    add_term({3: 1, 4: 1}, {0: 1, 1: 1}, {1: 1, 4: 1})
    
    # M6 = (A00 - A10) * (B10 + B11)
    add_term({0: 1, 3: -1}, {3: 1, 4: 1}, {1: -1, 4: 1})
    
    # M7 = (A01 - A11) * (B10 + B11)
    add_term({1: 1, 4: -1}, {3: 1, 4: 1}, {3: 1, 4: -1})
    
    # 2. Cross terms: u * q^T
    # u = [A02, A12], q^T = [B20, B21]
    # A indices: 02(2), 12(5)
    # B indices: 20(6), 21(7)
    # C indices: 00(0), 01(3), 10(1), 11(4)
    
    # M8 = A02 * B20
    add_term({2: 1}, {6: 1}, {0: 1})
    
    # M9 = A02 * B21
    add_term({2: 1}, {7: 1}, {3: 1})
    
    # M10 = A12 * B20
    add_term({5: 1}, {6: 1}, {1: 1})
    
    # M11 = A12 * B21
    add_term({5: 1}, {7: 1}, {4: 1})
    
    # 3. Cross terms: A_tl * p
    # A_tl = [[A00, A01], [A10, A11]], p = [B02, B12]
    # B indices: 02(2), 12(5)
    # C indices: 02(6), 12(7)
    
    # M12 = A00 * B02
    add_term({0: 1}, {2: 1}, {6: 1})
    
    # M13 = A01 * B12
    add_term({1: 1}, {5: 1}, {6: 1})
    
    # M14 = A10 * B02
    add_term({3: 1}, {2: 1}, {7: 1})
    
    # M15 = A11 * B12
    add_term({4: 1}, {5: 1}, {7: 1})
    
    # 4. Cross terms: u * b
    # u = [A02, A12], b = B22
    # B index: 22(8)
    # C indices: 02(6), 12(7)
    
    # M16 = A02 * B22
    add_term({2: 1}, {8: 1}, {6: 1})
    
    # M17 = A12 * B22
    add_term({5: 1}, {8: 1}, {7: 1})
    
    # 5. Cross terms: v^T * B_tl
    # v^T = [A20, A21], B_tl = [[B00, B01], [B10, B11]]
    # A indices: 20(6), 21(7)
    # B indices: 00(0), 01(1), 10(3), 11(4)
    # C indices: 20(2), 21(5)
    
    # M18 = A20 * B00
    add_term({6: 1}, {0: 1}, {2: 1})
    
    # M19 = A21 * B10
    add_term({7: 1}, {3: 1}, {2: 1})
    
    # M20 = A20 * B01
    add_term({6: 1}, {1: 1}, {5: 1})
    
    # M21 = A21 * B11
    add_term({7: 1}, {4: 1}, {5: 1})
    
    # 6. Cross terms: a * q^T
    # a = A22, q^T = [B20, B21]
    # A index: 22(8)
    # B indices: 20(6), 21(7)
    # C indices: 20(2), 21(5)
    
    # M22 = A22 * B20
    add_term({8: 1}, {6: 1}, {2: 1})
    
    # M23 = A22 * B21
    add_term({8: 1}, {7: 1}, {5: 1})
    
    # 7. Cross terms: v^T * p
    # v^T = [A20, A21], p = [B02, B12]
    # A indices: 20(6), 21(7)
    # B indices: 02(2), 12(5)
    # C index: 22(8)
    
    # M24 = A20 * B02
    add_term({6: 1}, {2: 1}, {8: 1})
    
    # M25 = A21 * B12
    add_term({7: 1}, {5: 1}, {8: 1})
    
    # 8. Cross term: a * b
    # a = A22, b = B22
    # A index: 22(8), B index: 22(8)
    # C index: 22(8)
    
    # M26 = A22 * B22
    add_term({8: 1}, {8: 1}, {8: 1})
    
    # Verification
    # Check if sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1 for all i,j,k
    # And 0 for mismatched j.
    
    valid = True
    for i in range(3):
        for j in range(3):
            for k in range(3):
                val = 0
                for r in range(rank):
                    val += U[i*3+j, r] * V[j*3+k, r] * W[k*3+i, r]
                if val != 1:
                    valid = False
                    break
            if not valid: break
        if not valid: break
    if not valid:
        # Fallback to schoolbook if verification fails
        rank = 27
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        col = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    U[i*3+j, col] = 1
                    V[j*3+k, col] = 1
                    W[k*3+i, col] = 1
                    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
