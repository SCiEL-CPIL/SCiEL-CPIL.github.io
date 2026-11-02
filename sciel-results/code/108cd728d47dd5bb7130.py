import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Initialize U, V, W
    # U: (n*m, rank) -> (9, rank)
    # V: (m*p, rank) -> (12, rank)
    # W: (p*n, rank) -> (12, rank)
    
    # We don't know the rank yet, so we'll collect terms in a list
    terms = []
    
    def add_term(u_coeffs, v_coeffs, w_coeffs):
        """
        u_coeffs: dict {index: coeff}
        v_coeffs: dict {index: coeff}
        w_coeffs: dict {index: coeff}
        """
        terms.append((u_coeffs, v_coeffs, w_coeffs))
        
    # Helper to get index in U (n*m)
    def u_idx(i, j): return i * m + j
    # Helper to get index in V (m*p)
    def v_idx(j, k): return j * p + k
    # Helper to get index in W (p*n) -> W[k*n + i]
    def w_idx(k, i): return k * n + i
    
    # --- Block Definitions ---
    # A: 3x3
    # A11: A[0,0]
    # A12: A[0,1], A[0,2]
    # A21: A[1,0], A[2,0]
    # A22: A[1,1], A[1,2], A[2,1], A[2,2]
    
    # B: 3x4
    # B11: B[0,0], B[0,1]
    # B12: B[0,2], B[0,3]
    # B21: B[1,0], B[1,1], B[2,0], B[2,1]
    # B22: B[1,2], B[1,3], B[2,2], B[2,3]
    
    # --- Linear Terms ---
    
    # A11 * B11
    # C11 += A11 * B11
    # C[0,0] += A[0,0] * B[0,0]
    add_term({u_idx(0,0): 1}, {v_idx(0,0): 1}, {w_idx(0,0): 1})
    # C[0,1] += A[0,0] * B[0,1]
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1}, {w_idx(1,0): 1})
    
    # A11 * B12
    # C12 += A11 * B12
    # C[0,2] += A[0,0] * B[0,2]
    add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
    # C[0,3] += A[0,0] * B[0,3]
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1}, {w_idx(3,0): 1})
    
    # A12 * B21
    # C11 += A12 * B21
    # A12 = [A[0,1], A[0,2]]
    # B21 = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    # C[0,0] += A[0,1]*B[1,0] + A[0,2]*B[2,0]
    add_term({u_idx(0,1): 1}, {v_idx(1,0): 1}, {w_idx(0,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,0): 1}, {w_idx(0,0): 1})
    # C[0,1] += A[0,1]*B[1,1] + A[0,2]*B[2,1]
    add_term({u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(1,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,1): 1}, {w_idx(1,0): 1})
    
    # A12 * B22
    # C12 += A12 * B22
    # B22 = [[B[1,2], B[1,3]], [B[2,2], B[2,3]]]
    # C[0,2] += A[0,1]*B[1,2] + A[0,2]*B[2,2]
    add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,2): 1}, {w_idx(2,0): 1})
    # C[0,3] += A[0,1]*B[1,3] + A[0,2]*B[2,3]
    add_term({u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(3,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,3): 1}, {w_idx(3,0): 1})
    
    # A21 * B11
    # C21 += A21 * B11
    # A21 = [A[1,0], A[2,0]]
    # B11 = [B[0,0], B[0,1]]
    # C[1,0] += A[1,0]*B[0,0]
    add_term({u_idx(1,0): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1})
    # C[1,1] += A[1,0]*B[0,1]
    add_term({u_idx(1,0): 1}, {v_idx(0,1): 1}, {w_idx(1,1): 1})
    # C[2,0] += A[2,0]*B[0,0]
    add_term({u_idx(2,0): 1}, {v_idx(0,0): 1}, {w_idx(0,2): 1})
    # C[2,1] += A[2,0]*B[0,1]
    add_term({u_idx(2,0): 1}, {v_idx(0,1): 1}, {w_idx(1,2): 1})
    
    # A21 * B12
    # C22 += A21 * B12
    # B12 = [B[0,2], B[0,3]]
    # C[1,2] += A[1,0]*B[0,2]
    add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
    # C[1,3] += A[1,0]*B[0,3]
    add_term({u_idx(1,0): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1})
    # C[2,2] += A[2,0]*B[0,2]
    add_term({u_idx(2,0): 1}, {v_idx(0,2): 1}, {w_idx(2,2): 1})
    # C[2,3] += A[2,0]*B[0,3]
    add_term({u_idx(2,0): 1}, {v_idx(0,3): 1}, {w_idx(3,2): 1})
    
    # --- Strassen Terms ---
    
    # Strassen for A22 * B21
    # A22 = [[A[1,1], A[1,2]], [A[2,1], A[2,2]]]
    # B21 = [[B[1,0], B[1,1]], [B[2,0], B[2,1]]]
    
    # M1 = (A22[0,0] + A22[1,1]) * (B21[0,0] + B21[1,1])
    add_term({u_idx(1,1): 1, u_idx(2,2): 1}, {v_idx(1,0): 1, v_idx(2,1): 1}, {})
    # Wait, Strassen products contribute to the result C21.
    # C21[0,0] = M1 + M4 - M5 + M7
    # C21[0,1] = M3 + M5
    # C21[1,0] = M2 + M4
    # C21[1,1] = M1 - M2 + M3 + M6
    
    # We need to add the contributions of each Mk to the W matrix.
    # But the Mk terms are defined by their U and V parts.
    # We must add the U, V parts for each Mk, and then add the W parts for each C element.
    # This means we need to keep track of which "column" in U/V corresponds to which Mk.
    # Since we are building a list of terms, we can just add the full term for each contribution.
    # But that would increase the rank!
    # Strassen's algorithm works because the SAME M_k is used in multiple C elements.
    # So we must add the U and V parts ONCE, and then add the W parts for each C element using the SAME rank index.
    
    # My `add_term` function creates a NEW rank index for each call.
    # This is incorrect for Strassen.
    # I need a way to share the U and V parts.
    
    # Let's restructure.
    # I will create the U and V columns for the Strassen products first.
    # Then I will add the W columns corresponding to the linear combinations.
    
    # However, the output format is just U, V, W matrices.
    # If I add a term for M1 to C21[0,0], I add a column to U, V, W.
    # If I add M1 to C21[1,1], I MUST use the SAME column in U and V, but add to W.
    # This implies that for a single rank-1 tensor, U, V, W are fixed.
    # But Strassen's M1 is a scalar product.
    # The tensor term is $u \otimes v \otimes w$.
    # If M1 contributes to C21[0,0] with coeff 1 and C21[1,1] with coeff 1,
    # then we have two terms?
    # No. The product is $M1 = u_{M1} \cdot v_{M1}$.
    # $C_{21,00} += 1 \cdot M1$.
    # $C_{21,11} += 1 \cdot M1$.
    # This can be represented as:
    # Term 1: $u_{M1} \otimes v_{M1} \otimes w_{21,00}$ (where $w_{21,00}$ has 1 at C21[0,0])
    # Term 2: $u_{M1} \otimes v_{M1} \otimes w_{21,11}$ (where $w_{21,11}$ has 1 at C21[1,1])
    # This would double the rank for M1!
    
    # Wait. The tensor decomposition is $\sum_r u_r \otimes v_r \otimes w_r$.
    # If $M1$ appears in two places, we can combine them if the W vectors are compatible?
    # No, W is a vector in $\mathbb{R}^{p \times n}$.
    # If $M1$ contributes to $C_{21,00}$ and $C_{21,11}$, then the W vector for that term would have non-zero entries at both positions.
    # So, $w_{M1}$ would have $w_{21,00}=1$ and $w_{21,11}=1$.
    # Then the term is $u_{M1} \otimes v_{M1} \otimes (w_{21,00} + w_{21,11})$.
    # This is a SINGLE rank-1 term!
    
    # So, for each Strassen product $M_k$, we define:
    # $u_k$: linear combo of A elements.
    # $v_k$: linear combo of B elements.
    # $w_k$: vector in $\mathbb{R}^{12}$ with coefficients at the C positions where $M_k$ appears.
    
    # This is correct.
    
    # Let's restart the term collection with this understanding.
    
    terms.clear()
    
    # Re-add linear terms (they are simple, 1-to-1)
    # (Already added above, but I cleared terms. I'll re-add them in the final code block properly)
    
    # Strassen M1 for A22*B21
    u_M1 = {u_idx(1,1): 1, u_idx(2,2): 1}
    v_M1 = {v_idx(1,0): 1, v_idx(2,1): 1}
    w_M1 = {w_idx(0,1): 1, w_idx(1,2): 1} # C[1,0] and C[2,1]
    # Check Strassen formula:
    # C[0,0] (local) = M1 + M4 - M5 + M7 -> Global C[1,0]
    # C[1,1] (local) = M1 - M2 + M3 + M6 -> Global C[2,1]
    # So w_M1 has 1 at C[1,0] and C[2,1].
    add_term(u_M1, v_M1, w_M1)
    
    # Strassen M2 for A22*B21
    u_M2 = {u_idx(2,1): 1, u_idx(2,2): 1} # A22[1,0] + A22[1,1]
    v_M2 = {v_idx(1,0): 1} # B21[0,0]
    w_M2 = {w_idx(0,1): 1, w_idx(1,2): -1} # C[1,0] += M2, C[2,1] -= M2
    add_term(u_M2, v_M2, w_M2)
    
    # Strassen M3 for A22*B21
    u_M3 = {u_idx(1,1): 1} # A22[0,0]
    v_M3 = {v_idx(1,1): 1, v_idx(2,1): -1} # B21[0,1] - B21[1,1]
    w_M3 = {w_idx(1,1): 1, w_idx(1,2): 1} # C[1,1] += M3, C[2,1] += M3
    add_term(u_M3, v_M3, w_M3)
    
    # Strassen M4 for A22*B21
    u_M4 = {u_idx(2,2): 1} # A22[1,1]
    v_M4 = {v_idx(2,1): 1} # B21[1,1]
    w_M4 = {w_idx(0,1): 1, w_idx(1,1): 1} # C[1,0] += M4, C[1,1] += M4
    add_term(u_M4, v_M4, w_M4)
    
    # Strassen M5 for A22*B21
    u_M5 = {u_idx(1,1): 1, u_idx(2,1): 1} # A22[0,0] + A22[1,0]
    v_M5 = {v_idx(2,1): 1} # B21[1,1]
    w_M5 = {w_idx(0,1): -1, w_idx(1,1): 1} # C[1,0] -= M5, C[1,1] += M5
    add_term(u_M5, v_M5, w_M5)
    
    # Strassen M6 for A22*B21
    u_M6 = {u_idx(1,2): 1, u_idx(1,1): -1} # A22[0,1] - A22[0,0]
    v_M6 = {v_idx(1,0): 1, v_idx(2,0): 1} # B21[0,0] + B21[1,0]
    w_M6 = {w_idx(1,2): 1} # C[2,1] += M6
    add_term(u_M6, v_M6, w_M6)
    
    # Strassen M7 for A22*B21
    u_M7 = {u_idx(2,2): 1, u_idx(2,1): -1} # A22[1,1] - A22[1,0]
    v_M7 = {v_idx(1,1): 1, v_idx(2,1): 1} # B21[0,1] + B21[1,1]
    w_M7 = {w_idx(0,1): 1} # C[1,0] += M7
    add_term(u_M7, v_M7, w_M7)
    
    # Strassen for A22 * B22
    # A22 = [[A[1,1], A[1,2]], [A[2,1], A[2,2]]]
    # B22 = [[B[1,2], B[1,3]], [B[2,2], B[2,3]]]
    # Result C22 (Global C[1:3, 2:4])
    # C[1,2] = M1' + M4' - M5' + M7'
    # C[1,3] = M3' + M5'
    # C[2,2] = M2' + M4'
    # C[2,3] = M1' - M2' + M3' + M6'
    
    # M1'
    u_M1p = {u_idx(1,1): 1, u_idx(2,2): 1}
    v_M1p = {v_idx(1,2): 1, v_idx(2,3): 1}
    w_M1p = {w_idx(2,1): 1, w_idx(3,2): 1}
    add_term(u_M1p, v_M1p, w_M1p)
    
    # M2'
    u_M2p = {u_idx(2,1): 1, u_idx(2,2): 1}
    v_M2p = {v_idx(1,2): 1}
    w_M2p = {w_idx(2,2): 1, w_idx(3,2): -1}
    add_term(u_M2p, v_M2p, w_M2p)
    
    # M3'
    u_M3p = {u_idx(1,1): 1}
    v_M3p = {v_idx(1,3): 1, v_idx(2,3): -1}
    w_M3p = {w_idx(3,1): 1, w_idx(3,2): 1}
    add_term(u_M3p, v_M3p, w_M3p)
    
    # M4'
    u_M4p = {u_idx(2,2): 1}
    v_M4p = {v_idx(2,3): 1}
    w_M4p = {w_idx(2,1): 1, w_idx(2,2): 1}
    add_term(u_M4p, v_M4p, w_M4p)
    
    # M5'
    u_M5p = {u_idx(1,1): 1, u_idx(2,1): 1}
    v_M5p = {v_idx(2,3): 1}
    w_M5p = {w_idx(2,1): -1, w_idx(3,1): 1}
    add_term(u_M5p, v_M5p, w_M5p)
    
    # M6'
    u_M6p = {u_idx(1,2): 1, u_idx(1,1): -1}
    v_M6p = {v_idx(1,2): 1, v_idx(2,2): 1}
    w_M6p = {w_idx(3,2): 1}
    add_term(u_M6p, v_M6p, w_M6p)
    
    # M7'
    u_M7p = {u_idx(2,2): 1, u_idx(2,1): -1}
    v_M7p = {v_idx(1,3): 1, v_idx(2,3): 1}
    w_M7p = {w_idx(2,1): 1}
    add_term(u_M7p, v_M7p, w_M7p)
    
    # Re-add linear terms
    # A11 * B11
    add_term({u_idx(0,0): 1}, {v_idx(0,0): 1}, {w_idx(0,0): 1})
    add_term({u_idx(0,0): 1}, {v_idx(0,1): 1}, {w_idx(1,0): 1})
    # A11 * B12
    add_term({u_idx(0,0): 1}, {v_idx(0,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,0): 1}, {v_idx(0,3): 1}, {w_idx(3,0): 1})
    # A12 * B21
    add_term({u_idx(0,1): 1}, {v_idx(1,0): 1}, {w_idx(0,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,0): 1}, {w_idx(0,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,1): 1}, {w_idx(1,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,1): 1}, {w_idx(1,0): 1})
    # A12 * B22
    add_term({u_idx(0,1): 1}, {v_idx(1,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,2): 1}, {w_idx(2,0): 1})
    add_term({u_idx(0,1): 1}, {v_idx(1,3): 1}, {w_idx(3,0): 1})
    add_term({u_idx(0,2): 1}, {v_idx(2,3): 1}, {w_idx(3,0): 1})
    # A21 * B11
    add_term({u_idx(1,0): 1}, {v_idx(0,0): 1}, {w_idx(0,1): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,1): 1}, {w_idx(1,1): 1})
    add_term({u_idx(2,0): 1}, {v_idx(0,0): 1}, {w_idx(0,2): 1})
    add_term({u_idx(2,0): 1}, {v_idx(0,1): 1}, {w_idx(1,2): 1})
    # A21 * B12
    add_term({u_idx(1,0): 1}, {v_idx(0,2): 1}, {w_idx(2,1): 1})
    add_term({u_idx(1,0): 1}, {v_idx(0,3): 1}, {w_idx(3,1): 1})
    add_term({u_idx(2,0): 1}, {v_idx(0,2): 1}, {w_idx(2,2): 1})
    add_term({u_idx(2,0): 1}, {v_idx(0,3): 1}, {w_idx(3,2): 1})
    
    # Construct U, V, W
    rank = len(terms)
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    for r, (u_c, v_c, w_c) in enumerate(terms):
        for idx, coeff in u_c.items():
            U[idx, r] = coeff
        for idx, coeff in v_c.items():
            V[idx, r] = coeff
        for idx, coeff in w_c.items():
            W[idx, r] = coeff
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

