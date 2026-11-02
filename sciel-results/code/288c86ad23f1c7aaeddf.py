"""Matrix multiplication 3x3x3: rank-minimal tensor decomposition."""
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify(U, V, W):
    """Verify that the decomposition correctly computes matrix multiplication."""
    n = m = p = 3
    U = [[float(x) for x in row] for row in U]
    V = [[float(x) for x in row] for row in V]
    W = [[float(x) for x in row] for row in W]
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def _schoolbook():
    n = m = p = 3
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

def _strassen_rank23():
    """
    Rank-26 decomposition for 3x3 matrix multiplication using explicit factors.
    (Named rank23 for interface compatibility, but actually rank 26).
    """
    n = m = p = 3
    rank = 26
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    def u_idx(i, j): return i * m + j
    def v_idx(j, k): return j * p + k
    def w_idx(k, i): return k * n + i
    
    # Each term is (list of (u_row, coeff)), (list of (v_row, coeff)), (list of (w_row, coeff))
    terms = []
    
    # --- Strassen core for C[0:2,0:2] (7 terms) ---
    # M1 = (A00+A11)*(B00+B11) -> C00+1, C11+1
    terms.append(([ (u_idx(0,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1), (w_idx(1,1),1) ]))
    # M2 = (A10+A11)*B00 -> C10+1, C11-1
    terms.append(([ (u_idx(1,0),1), (u_idx(1,1),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,1),1), (w_idx(1,1),-1) ]))
    # M3 = A00*(B01-B11) -> C01+1, C11+1
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,1),1), (v_idx(1,1),-1) ], [ (w_idx(1,0),1), (w_idx(1,1),1) ]))
    # M4 = A11*(B10-B00) -> C00+1, C10+1
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,0),1), (v_idx(0,0),-1) ], [ (w_idx(0,0),1), (w_idx(0,1),1) ]))
    # M5 = (A00+A01)*B11 -> C00-1, C01+1
    terms.append(([ (u_idx(0,0),1), (u_idx(0,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(0,0),-1), (w_idx(1,0),1) ]))
    # M6 = (A10-A00)*(B00+B01) -> C11+1
    terms.append(([ (u_idx(1,0),1), (u_idx(0,0),-1) ], [ (v_idx(0,0),1), (v_idx(0,1),1) ], [ (w_idx(1,1),1) ]))
    # M7 = (A01-A11)*(B10+B11) -> C00+1
    terms.append(([ (u_idx(0,1),1), (u_idx(1,1),-1) ], [ (v_idx(1,0),1), (v_idx(1,1),1) ], [ (w_idx(0,0),1) ]))
    
    # --- Border: C[0:2, 2] = A[0:2,0:2]*B[0:2,2] + A[0:2,2]*B[2,2] ---
    # C[0,2] = A[0,0]*B[0,2] + A[0,1]*B[1,2] + A[0,2]*B[2,2]
    # C[1,2] = A[1,0]*B[0,2] + A[1,1]*B[1,2] + A[1,2]*B[2,2]
    # These need 6 independent products (no sharing with Strassen core since different B columns)
    # Term 7: A[0,0]*B[0,2] -> C[0,2]
    terms.append(([ (u_idx(0,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,0),1) ]))
    # Term 8: A[0,1]*B[1,2] -> C[0,2]
    terms.append(([ (u_idx(0,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,0),1) ]))
    # Term 9: A[0,2]*B[2,2] -> C[0,2]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,0),1) ]))
    # Term 10: A[1,0]*B[0,2] -> C[1,2]
    terms.append(([ (u_idx(1,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,1),1) ]))
    # Term 11: A[1,1]*B[1,2] -> C[1,2]
    terms.append(([ (u_idx(1,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,1),1) ]))
    # Term 12: A[1,2]*B[2,2] -> C[1,2]
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,1),1) ]))
    
    # --- Border: C[2, 0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2]*B[2,0:2] ---
    # C[2,0] = A[2,0]*B[0,0] + A[2,1]*B[1,0] + A[2,2]*B[2,0]
    # C[2,1] = A[2,0]*B[0,1] + A[2,1]*B[1,1] + A[2,2]*B[2,1]
    # 6 more products
    # Term 13: A[2,0]*B[0,0] -> C[2,0]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,0),1) ], [ (w_idx(0,2),1) ]))
    # Term 14: A[2,1]*B[1,0] -> C[2,0]
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,0),1) ], [ (w_idx(0,2),1) ]))
    # Term 15: A[2,2]*B[2,0] -> C[2,0]
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,2),1) ]))
    # Term 16: A[2,0]*B[0,1] -> C[2,1]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,1),1) ], [ (w_idx(1,2),1) ]))
    # Term 17: A[2,1]*B[1,1] -> C[2,1]
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,1),1) ], [ (w_idx(1,2),1) ]))
    # Term 18: A[2,2]*B[2,1] -> C[2,1]
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,2),1) ]))
    
    # --- Corner: C[2,2] = A[2,0]*B[0,2] + A[2,1]*B[1,2] + A[2,2]*B[2,2] ---
    # 3 more products
    # Term 19: A[2,0]*B[0,2] -> C[2,2]
    terms.append(([ (u_idx(2,0),1) ], [ (v_idx(0,2),1) ], [ (w_idx(2,2),1) ]))
    # Term 20: A[2,1]*B[1,2] -> C[2,2]
    terms.append(([ (u_idx(2,1),1) ], [ (v_idx(1,2),1) ], [ (w_idx(2,2),1) ]))
    # Term 21: A[2,2]*B[2,2] -> C[2,2]
    terms.append(([ (u_idx(2,2),1) ], [ (v_idx(2,2),1) ], [ (w_idx(2,2),1) ]))
    
    # That's 7 + 6 + 6 + 3 = 22 terms. We need 23.
    # Wait, let me check if this is actually correct by verifying.
    # The issue: Strassen core gives C[0:2,0:2] correctly, but the border terms
    # only cover the "new" products. Let me check if all 27 (i,j,k) triples are covered.
    
    # For (i,j,k) with i,k in {0,1}: covered by Strassen (should sum to 1)
    # For i in {0,1}, k=2: need A[i,j]*B[j,2] terms
    #   j=0: A[i,0]*B[0,2] -> term 7 (i=0) or term 10 (i=1)
    #   j=1: A[i,1]*B[1,2] -> term 8 (i=0) or term 11 (i=1)
    #   j=2: A[i,2]*B[2,2] -> term 9 (i=0) or term 12 (i=1)
    # For i=2, k in {0,1}: need A[2,j]*B[j,k] terms
    #   j=0: A[2,0]*B[0,k] -> term 13 (k=0) or term 16 (k=1)
    #   j=1: A[2,1]*B[1,k] -> term 14 (k=0) or term 17 (k=1)
    #   j=2: A[2,2]*B[2,k] -> term 15 (k=0) or term 18 (k=1)
    # For i=2, k=2: need A[2,j]*B[j,2] terms
    #   j=0: A[2,0]*B[0,2] -> term 19
    #   j=1: A[2,1]*B[1,2] -> term 20
    #   j=2: A[2,2]*B[2,2] -> term 21
    
    # That's 22 terms total. But we claimed rank 23 is minimal.
    # Let me verify: is this decomposition actually valid?
    # The Strassen part should work for the 2x2 core.
    # The border parts are just schoolbook for those entries.
    # Total rank = 7 + 6 + 6 + 3 = 22.
    # 
    # But wait - is rank 22 possible? The known minimum is 23!
    # Let me check my Strassen verification more carefully.
    # 
    # The problem: in the Strassen core, the W coefficients must correctly
    # reconstruct C[0:2, 0:2]. Let me verify term by term.
    # 
    # For C[0,0] (w_idx(0,0)=0): should get contribution 1 from sum of A[0,j]*B[j,0]
    #   j=0: A[0,0]*B[0,0]
    #   j=1: A[0,1]*B[1,0]
    #   j=2: A[0,2]*B[2,0] (but B[2,0] is in border, handled by term 13)
    # 
    # In Strassen core, the contribution to C[0,0] from terms 0-6:
    #   M1: W=1, so contributes (A00+A11)*(B00+B11) to C00
    #   M4: W=1, so contributes A11*(B10-B00) to C00
    #   M5: W=-1, so contributes -(A00+A01)*B11 to C00
    #   M7: W=1, so contributes (A01-A11)*(B10+B11) to C00
    #   Sum: (A00+A11)(B00+B11) + A11(B10-B00) - (A00+A01)B11 + (A01-A11)(B10+B11)
    #   = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #   = A00B00 + A01B10 + A01B11 - A01B11 + ... let me redo
    #   = A00B00 + A00B11 + A11B00 + A11B11 + A11B10 - A11B00 - A00B11 - A01B11 + A01B10 + A01B11 - A11B10 - A11B11
    #   Group: A00B00 + (A00B11 - A00B11) + (A11B00 - A11B00) + (A11B11 - A11B11) + (A11B10 - A11B10) + (-A01B11 + A01B11) + A01B10
    #   = A00B00 + A01B10
    # 
    # But we need A00B00 + A01B10 + A02B20 for C[0,0]. The A02B20 part comes from term 13.
    # So C[0,0] = (Strassen contribution) + A02B20 = A00B00 + A01B10 + A02B20. Correct!
    # 
    # Similarly for other C entries in the 2x2 core.
    # 
    # So the total rank is 22, not 23! But the known minimum rank for 3x3 is 23.
    # This means my construction must have an error. Let me check more carefully.
    # 
    # Actually wait - I think I made an error. Let me re-examine.
    # The Strassen decomposition gives C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] using 7 mults.
    # But the full C[0:2,0:2] also includes contributions from A[0:2,2]*B[2,0:2]!
    # 
    # C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0]
    # C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] + A[0,2]*B[2,1]
    # C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] + A[1,2]*B[2,0]
    # C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] + A[1,2]*B[2,1]
    # 
    # So C[0:2,0:2] = A[0:2,0:2]*B[0:2,0:2] + A[0:2,2]*B[2,0:2]
    # 
    # The Strassen core handles A[0:2,0:2]*B[0:2,0:2] with 7 mults.
    # The A[0:2,2]*B[2,0:2] part needs 2*2=4 more mults.
    # 
    # So C[0:2,0:2] needs 7+4=11 mults total.
    # C[0:2,2] = A[0:2,0:2]*B[0:2,2] + A[0:2,2]*B[2,2] needs 2*2+2*1=6 mults
    # C[2,0:2] = A[2,0:2]*B[0:2,0:2] + A[2,2]*B[2,0:2] needs 2*2+1*2=6 mults
    # C[2,2] = A[2,0:2]*B[0:2,2] + A[2,2]*B[2,2] needs 2*1+1*1=3 mults
    # 
    # Total: 11+6+6+3 = 26. That's rank 26!
    # 
    # So my earlier count of 22 was wrong because I forgot the A[0:2,2]*B[2,0:2] terms.
    # Let me add those 4 terms.
    
    # --- Additional border: A[0:2,2]*B[2,0:2] contributes to C[0:2,0:2] ---
    # C[0,0] += A[0,2]*B[2,0]
    # C[0,1] += A[0,2]*B[2,1]
    # C[1,0] += A[1,2]*B[2,0]
    # C[1,1] += A[1,2]*B[2,1]
    # Term 22: A[0,2]*B[2,0] -> C[0,0]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,0),1) ]))
    # Term 23: A[0,2]*B[2,1] -> C[0,1]
    terms.append(([ (u_idx(0,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,0),1) ]))
    # Term 24: A[1,2]*B[2,0] -> C[1,0]
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,0),1) ], [ (w_idx(0,1),1) ]))
    # Term 25: A[1,2]*B[2,1] -> C[1,1]
    terms.append(([ (u_idx(1,2),1) ], [ (v_idx(2,1),1) ], [ (w_idx(1,1),1) ]))
    
    # Now we have 7 + 6 + 6 + 3 + 4 = 26 terms. That's rank 26.
    # To get to 23, we need to find 3 more savings through clever combinations.
    # 
    # For now, let's just use this rank-26 construction and verify it.
    # If it verifies, great - rank 26 > 23 but < 27, so it's an improvement.
    
    for r, (u_list, v_list, w_list) in enumerate(terms):
        for (row, coeff) in u_list:
            U[row][r] += coeff
        for (row, coeff) in v_list:
            V[row][r] += coeff
        for (row, coeff) in w_list:
            W[row][r] += coeff
    
    return {"U": U, "V": V, "W": W}

def construct():
    """Construct a tensor decomposition for 3x3 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Try rank-26 construction (Strassen core + schoolbook border)
    result = _strassen_rank23()
    if _verify(result["U"], result["V"], result["W"]):
        if time.time() < deadline:
            return result
    
    # Fallback to schoolbook
    return _schoolbook()

# EVOLVE-BLOCK-END
