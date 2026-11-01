"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np
import sys

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 5
    nm, mp, pn = n*m, m*p, p*n  # 4, 10, 10
    
    # Build target tensor for verification
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    def _verify(U, V, W):
        """Verify that U, V, W decompose T exactly (within tolerance)."""
        R = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(R, T, atol=1e-10, rtol=0)

    def _try_rank_decomposition(rank):
        """
        Try to find a rank-`rank` decomposition using a structured approach.
        For 2x2 * 2x5, we exploit the rectangular structure.
        
        Strategy: 
        - The 2x2 * 2x5 multiplication can be seen as 2x2 * 2x4 (Strassen, rank 7) 
          plus 2x2 * 2x1 (schoolbook, rank 4) = rank 11? No, that's not quite right.
        
        Actually, let's think more carefully.
        C = A * B where A is 2x2, B is 2x5.
        
        We can partition B = [B1 | B2] where B1 is 2x4 and B2 is 2x1.
        C = [A*B1 | A*B2]
        
        A*B1: 2x2 * 2x4. We can split B1 into two 2x2 blocks: B1 = [B1a | B1b].
        A*B1a: 2x2 * 2x2, rank 7 (Strassen)
        A*B1b: 2x2 * 2x2, rank 7 (Strassen)
        A*B2: 2x2 * 2x1, rank 4 (schoolbook)
        Total: 7 + 7 + 4 = 18.
        
        Can we do better? Let's try to merge some terms.
        
        Alternative: Use the fact that for 2x2 * 2xp, the rank is at most 7*ceil(p/2) + 4*(p mod 2).
        For p=5: 7*2 + 4*1 = 18.
        
        Let's try to find a rank 17 or lower decomposition by careful merging.
        
        Actually, let's first verify the rank 18 decomposition works, then try to optimize.
        """
        # Rank 18 decomposition using Strassen for 2x2 blocks
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        term_idx = 0
        
        def add_term(u_vals, v_vals, w_vals):
            nonlocal term_idx
            if term_idx >= rank:
                return False
            for idx, val in u_vals.items():
                U[idx, term_idx] += val
            for idx, val in v_vals.items():
                V[idx, term_idx] += val
            for idx, val in w_vals.items():
                W[idx, term_idx] += val
            term_idx += 1
            return True
        
        # Indices:
        # U: i*m + j -> i in {0,1}, j in {0,1} -> 0,1,2,3
        # V: j*p + k -> j in {0,1}, k in {0..4} -> 0-4 (j=0), 5-9 (j=1)
        # W: k*n + i -> k in {0..4}, i in {0,1} -> 0-1 (k=0), 2-3 (k=1), 4-5 (k=2), 6-7 (k=3), 8-9 (k=4)
        
        # Strassen for A*B1a (columns 0,1)
        # M1 = (a00+a11)(b00+b11)
        add_term({0:1, 3:1}, {0:1, 5:1}, {0:1, 1:1, 2:1, 3:1})
        
        # M2 = (a10+a11)b00
        add_term({2:1, 3:1}, {0:1}, {0:-1, 1:1})
        
        # M3 = a00(b11-b10)
        add_term({0:1}, {5:1, 4:-1}, {0:1, 1:1})
        
        # M4 = a11(b00-b10)
        add_term({3:1}, {0:1, 4:-1}, {0:-1, 1:1})
        
        # M5 = (a00+a01)b10
        add_term({0:1, 1:1}, {4:1}, {0:1, 1:-1})
        
        # M6 = (a10+a11)(b01-b00)
        add_term({2:1, 3:1}, {1:1, 0:-1}, {0:1, 1:1})
        
        # M7 = (a01-a10)(b00+b10)
        add_term({1:1, 2:-1}, {0:1, 4:1}, {0:1, 1:-1})
        
        # Strassen for A*B1b (columns 2,3)
        # M8 = (a00+a11)(b02+b12)
        add_term({0:1, 3:1}, {2:1, 6:1}, {4:1, 5:1, 6:1, 7:1})
        
        # M9 = (a10+a11)b02
        add_term({2:1, 3:1}, {2:1}, {4:-1, 5:1})
        
        # M10 = a00(b12-b11)
        add_term({0:1}, {6:1, 5:-1}, {4:1, 5:1})
        
        # M11 = a11(b02-b11)
        add_term({3:1}, {2:1, 5:-1}, {4:-1, 5:1})
        
        # M12 = (a00+a01)b11
        add_term({0:1, 1:1}, {5:1}, {4:1, 5:-1})
        
        # M13 = (a10+a11)(b03-b02)
        add_term({2:1, 3:1}, {3:1, 2:-1}, {4:1, 5:1})
        
        # M14 = (a01-a10)(b02+b11)
        add_term({1:1, 2:-1}, {2:1, 5:1}, {4:1, 5:-1})
        
        # Schoolbook for A*B2 (column 4)
        # c04 = a00*b04 + a01*b14
        add_term({0:1}, {4:1}, {8:1})
        add_term({1:1}, {9:1}, {8:1})
        
        # c14 = a10*b04 + a11*b14
        add_term({2:1}, {4:1}, {9:1})
        add_term({3:1}, {9:1}, {9:1})
        
        if term_idx != rank:
            return None
        
        if _verify(U, V, W):
            return U, V, W
        return None

    # Try to find the minimal rank
    # Start from a reasonable lower bound
    # Lower bound: max(nm, mp, pn) = 10, but tighter bounds exist
    # For 2x2 * 2x5, known rank is 18 (from Strassen + schoolbook)
    # Let's try to find if we can do better
    
    best_rank = n * m * p  # Start with schoolbook rank 20
    best_decomp = None
    
    # Try ranks from 15 to 19
    for r in range(15, best_rank):
        result = _try_rank_decomposition(r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_decomp = (U, V, W)
            break
    
    # If no better decomposition found, use the rank 18 one
    if best_decomp is None:
        result = _try_rank_decomposition(18)
        if result is not None:
            U, V, W = result
            best_rank = 18
            best_decomp = (U, V, W)
    
    # Fallback to schoolbook if all else fails
    if best_decomp is None:
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
        best_decomp = (U, V, W)
        best_rank = rank
    
    U, V, W = best_decomp
    
    # Convert to lists
    U_list = U.astype(int).tolist()
    V_list = V.astype(int).tolist()
    W_list = W.astype(int).tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-END
