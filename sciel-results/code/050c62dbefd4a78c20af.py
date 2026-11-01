"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x3x8 matrix multiplication tensor into rank-32 sub-tensors.
    Partition the 8 columns into 4 groups of 2.
    Each 2x3x2 sub-tensor is decomposed with rank 8:
    - 7 terms from Strassen for the 2x2x2 core (j=0,1)
    - 1 term for the j=2 contribution
    Total rank: 4 * 8 = 32.
    """
    import time
    start_time = time.time()
    
    n, m, p = DIMENSIONS  # 2, 3, 8
    
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A11-A01)(B10+B11)
    
    # C00 = M1 + M5 - M4 - M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M6 + M7
    
    # Indices for Strassen terms (local j=0,1)
    # U[i*m + j, c]
    # V[j*p + k, c]
    # W[k*n + i, c]
    
    for group in range(4):
        base_c = group * 8
        k_offset = group * 2
        
        # Term 0: M1
        c = base_c + 0
        # U: A00, A11
        U[0*3 + 0][c] += 1
        U[1*3 + 1][c] += 1
        # V: B00, B11
        V[0*2 + 0][c] += 1
        V[1*2 + 1][c] += 1
        # W: C00, C11
        W[0*2 + 0][c] += 1
        W[1*2 + 1][c] += 1
        
        # Term 1: M2
        c = base_c + 1
        # U: A10, A11
        U[1*3 + 0][c] += 1
        U[1*3 + 1][c] += 1
        # V: B00
        V[0*2 + 0][c] += 1
        # W: C10
        W[0*2 + 1][c] += 1
        
        # Term 2: M3
        c = base_c + 2
        # U: A00
        U[0*3 + 0][c] += 1
        # V: B01, -B11
        V[0*2 + 1][c] += 1
        V[1*2 + 1][c] += -1
        # W: C01
        W[1*2 + 0][c] += 1
        
        # Term 3: M4
        c = base_c + 3
        # U: A11
        U[1*3 + 1][c] += 1
        # V: B10, -B00
        V[1*2 + 0][c] += 1
        V[0*2 + 0][c] += -1
        # W: C11, C10
        W[1*2 + 1][c] += 1
        W[0*2 + 1][c] += 1
        
        # Term 4: M5
        c = base_c + 4
        # U: A00, A01
        U[0*3 + 0][c] += 1
        U[0*3 + 1][c] += 1
        # V: B11
        V[1*2 + 1][c] += 1
        # W: C00, C01
        W[0*2 + 0][c] += 1
        W[1*2 + 0][c] += 1
        
        # Term 5: M6
        c = base_c + 5
        # U: A10, -A00
        U[1*3 + 0][c] += 1
        U[0*3 + 0][c] += -1
        # V: B00, B01
        V[0*2 + 0][c] += 1
        V[0*2 + 1][c] += 1
        # W: C11
        W[1*2 + 1][c] += 1
        
        # Term 6: M7
        c = base_c + 6
        # U: A11, -A01
        U[1*3 + 1][c] += 1
        U[0*3 + 1][c] += -1
        # V: B10, B11
        V[1*2 + 0][c] += 1
        V[1*2 + 1][c] += 1
        # W: C00, C11
        W[0*2 + 0][c] += -1
        W[1*2 + 1][c] += 1
        
        # Term 7: j=2 contribution
        c = base_c + 7
        # U: A02, A12
        U[0*3 + 2][c] += 1
        U[1*3 + 2][c] += 1
        # V: B20, B21
        V[2*2 + 0][c] += 1
        V[2*2 + 1][c] += 1
        # W: C00, C01, C10, C11 (all 1s)
        W[0*2 + 0][c] += 1
        W[1*2 + 0][c] += 1
        W[0*2 + 1][c] += 1
        W[1*2 + 1][c] += 1
        
        # Update W indices with global k offset
        # The W matrix is (p*n, rank).
        # W[k*n + i, c]
        # In the loop above, I used local k (0,1).
        # I need to shift these to global k.
        # The code above wrote to W[local_k * 2 + i][c].
        # But W has shape (8*2, 32) = (16, 32).
        # Local k=0 corresponds to global k=k_offset.
        # Local k=1 corresponds to global k=k_offset+1.
        # So W[local_k * 2 + i] is WRONG. It should be W[(k_offset + local_k) * 2 + i].
        
        # I need to fix the W indices.
        # Let's redo the W assignments with correct global indices.
        
        # Clear W for this group's terms? No, I should just use correct indices.
        # The previous assignments to W were to local indices.
        # I will re-implement the loop with correct W indices.
        
    # Re-implementing with correct W indices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for group in range(4):
        base_c = group * 8
        k_offset = group * 2
        
        # Term 0: M1
        c = base_c + 0
        U[0][c] += 1; U[4][c] += 1
        V[0][c] += 1; V[3][c] += 1
        W[(k_offset+0)*2 + 0][c] += 1
        W[(k_offset+1)*2 + 1][c] += 1
        
        # Term 1: M2
        c = base_c + 1
        U[3][c] += 1; U[4][c] += 1
        V[0][c] += 1
        W[(k_offset+0)*2 + 1][c] += 1
        
        # Term 2: M3
        c = base_c + 2
        U[0][c] += 1
        V[1][c] += 1; V[3][c] += -1
        W[(k_offset+1)*2 + 0][c] += 1
        
        # Term 3: M4
        c = base_c + 3
        U[4][c] += 1
        V[2][c] += 1; V[0][c] += -1
        W[(k_offset+1)*2 + 1][c] += 1
        W[(k_offset+0)*2 + 1][c] += 1
        
        # Term 4: M5
        c = base_c + 4
        U[0][c] += 1; U[1][c] += 1
        V[3][c] += 1
        W[(k_offset+0)*2 + 0][c] += 1
        W[(k_offset+1)*2 + 0][c] += 1
        
        # Term 5: M6
        c = base_c + 5
        U[3][c] += 1; U[0][c] += -1
        V[0][c] += 1; V[1][c] += 1
        W[(k_offset+1)*2 + 1][c] += 1
        
        # Term 6: M7
        c = base_c + 6
        U[4][c] += 1; U[1][c] += -1
        V[2][c] += 1; V[3][c] += 1
        W[(k_offset+0)*2 + 0][c] += -1
        W[(k_offset+1)*2 + 1][c] += 1
        
        # Term 7: j=2
        c = base_c + 7
        U[2][c] += 1; U[5][c] += 1
        V[4][c] += 1; V[5][c] += 1
        W[(k_offset+0)*2 + 0][c] += 1
        W[(k_offset+1)*2 + 0][c] += 1
        W[(k_offset+0)*2 + 1][c] += 1
        W[(k_offset+1)*2 + 1][c] += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
