"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # Rank is 5 * 7 = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U indices: i*m + j
    # V indices: j*p + k
    # W indices: k*n + i
    
    # Terms P1..P7 defined by (U_coeffs, V_coeffs, W_coeffs)
    # U_coeffs: list of (index, value)
    # V_coeffs: list of (index_offset_in_block, value) -> actual index = j*p + (2*b + offset)
    # W_coeffs: list of (index_offset_in_block, value) -> actual index = (2*b + offset)*n + i? No.
    # W index is k*n + i. k = 2*b + r.
    # So W index = (2*b + r)*n + i = 2*b*n + r*n + i.
    # Since n=2, W index = 4*b + 2*r + i.
    
    # Let's predefine the static parts and the dynamic block offsets.
    
    # P1: (A00+A11)(B0e+B1h)
    # U: (0,1), (3,1)
    # V: B0e (j=0, k=2b+0) -> idx 2b. B1h (j=1, k=2b+1) -> idx 10+2b+1.
    # W: C0e (k=2b, i=0) -> 4b. C1h (k=2b+1, i=1) -> 4b+3.
    
    # P2: (C00+C11)B0e  -- Wait, standard Strassen P2 is (c+d)e.
    # c=A10, d=A11. e=B0e.
    # U: (2,1), (3,1)
    # V: B0e -> idx 2b.
    # W: C10 (k=2b, i=1) -> 4b+1. C11 (k=2b+1, i=1) -> 4b+3 (with -1).
    
    # P3: A00(B0f-B1h)
    # U: (0,1)
    # V: B0f (j=0, k=2b+1) -> idx 2b+1. B1h (j=1, k=2b+1) -> idx 10+2b+1 (with -1).
    # W: C0f (k=2b+1, i=0) -> 4b+2. C1h (k=2b+1, i=1) -> 4b+3.
    
    # P4: A11(B1g-B0f)
    # U: (3,1)
    # V: B1g (j=1, k=2b) -> idx 10+2b. B0f (j=0, k=2b+1) -> idx 2b+1 (with -1).
    # W: C0g (k=2b, i=0) -> 4b. C1g (k=2b, i=1) -> 4b+1.
    
    # P5: (A00+A01)B1h
    # U: (0,1), (1,1)
    # V: B1h -> idx 10+2b+1.
    # W: C0h (k=2b+1, i=0) -> 4b+2 (with +1). C0e (k=2b, i=0) -> 4b (with -1).
    
    # P6: (A10-A00)(B0e+B0f)
    # U: (2,1), (0,-1)
    # V: B0e -> idx 2b. B0f -> idx 2b+1.
    # W: C1h (k=2b+1, i=1) -> 4b+3.
    
    # P7: (A01-A11)(B1g+B1h)
    # U: (1,1), (3,-1)
    # V: B1g -> idx 10+2b. B1h -> idx 10+2b+1.
    # W: C0e (k=2b, i=0) -> 4b.

    terms = [
        # (U_indices, U_vals, V_offsets, V_vals, W_offsets, W_vals)
        # V_offset is r in k=2b+r. Actual V index = j*p + 2b + r.
        # W_offset is r*n+i in k=2b+r. Actual W index = (2b+r)*n + i = 4b + 2r + i.
        
        # P1
        ([(0, 1), (3, 1)], [(0, 1), (1, 1)], [(0, 1), (1, 1)]),
        # P2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # P3
        ([(0, 1)], [(0, 1), (1, -1)], [(2, 1), (3, 1)]),
        # P4
        ([(3, 1)], [(1, 1), (0, -1)], [(0, 1), (1, 1)]),
        # P5
        ([(0, 1), (1, 1)], [(1, 1)], [(0, -1), (2, 1)]),
        # P6
        ([(2, 1), (0, -1)], [(0, 1), (0, 1)], [(3, 1)]),
        # P7
        ([(1, 1), (3, -1)], [(1, 1), (1, 1)], [(0, 1)])
    ]
    
    # Note on V indices:
    # V index = j*p + k.
    # For P1 V: j=0, k=2b+0 => 2b. j=1, k=2b+1 => 10+2b+1.
    # My V_offsets above:
    # P1: [(0, 1), (1, 1)] means (j=0, r=0) and (j=1, r=1).
    # Index 1: 0*10 + 2b + 0 = 2b.
    # Index 2: 1*10 + 2b + 1 = 10+2b+1.
    # This matches.
    
    # For P2 V: [(0, 1)] means (j=0, r=0). Index 2b. Matches.
    # For P3 V: [(0, 1), (1, -1)] means (j=0, r=1) and (j=1, r=1).
    # Index 1: 0*10 + 2b + 1 = 2b+1.
    # Index 2: 1*10 + 2b + 1 = 10+2b+1.
    # Matches.
    
    # For P4 V: [(1, 1), (0, -1)] means (j=1, r=0) and (j=0, r=1).
    # Index 1: 1*10 + 2b + 0 = 10+2b.
    # Index 2: 0*10 + 2b + 1 = 2b+1.
    # Matches.
    
    # For P5 V: [(1, 1)] means (j=1, r=1). Index 10+2b+1. Matches.
    
    # For P6 V: [(0, 1), (0, 1)] means (j=0, r=0) and (j=0, r=1).
    # Index 1: 2b. Index 2: 2b+1. Matches.
    
    # For P7 V: [(1, 1), (1, 1)] means (j=1, r=0) and (j=1, r=1).
    # Index 1: 10+2b. Index 2: 10+2b+1. Matches.

    # Note on W indices:
    # W index = k*n + i = (2b+r)*2 + i = 4b + 2r + i.
    # My W_offsets are (r, i) pairs? No, I used offsets directly in the list above?
    # Let's re-check the W offsets in the terms list.
    # P1 W: [(0, 1), (1, 1)].
    # If offset is (r, i):
    # (0, 0) -> 4b. (1, 1) -> 4b+3.
    # My list had [(0, 1), (1, 1)] which I interpreted as (index_offset, value) in the comment?
    # No, I wrote `[(0, 1), (1, 1)]` for W in P1.
    # If the tuple is (offset, value), then offset 0 and 1.
    # But W index depends on r and i.
    # Let's refine the structure to be explicit (r, i, value) for W and (j, r, value) for V.
    
    # Revised terms structure:
    # U: list of (u_idx, val)
    # V: list of (j, r, val) -> idx = j*10 + 2b + r
    # W: list of (r, i, val) -> idx = 4b + 2r + i
    
    terms = [
        # P1
        (
            [(0, 1), (3, 1)],
            [(0, 0, 1), (1, 1, 1)],
            [(0, 0, 1), (1, 1, 1)]
        ),
        # P2
        (
            [(2, 1), (3, 1)],
            [(0, 0, 1)],
            [(0, 1, 1), (1, 1, -1)]
        ),
        # P3
        (
            [(0, 1)],
            [(0, 1, 1), (1, 1, -1)],
            [(1, 0, 1), (1, 1, 1)]
        ),
        # P4
        (
            [(3, 1)],
            [(1, 0, 1), (0, 1, -1)],
            [(0, 0, 1), (0, 1, 1)]
        ),
        # P5
        (
            [(0, 1), (1, 1)],
            [(1, 1, 1)],
            [(0, 0, -1), (1, 0, 1)]
        ),
        # P6
        (
            [(2, 1), (0, -1)],
            [(0, 0, 1), (0, 1, 1)],
            [(1, 1, 1)]
        ),
        # P7
        (
            [(1, 1), (3, -1)],
            [(1, 0, 1), (1, 1, 1)],
            [(0, 0, 1)]
        )
    ]
    
    for b in range(5):
        for t_idx in range(7):
            col = b * 7 + t_idx
            u_coeffs, v_coeffs, w_coeffs = terms[t_idx]
            
            for u_idx, val in u_coeffs:
                U[u_idx][col] = val
                
            for j, r, val in v_coeffs:
                v_idx = j * p + 2 * b + r
                V[v_idx][col] = val
                
            for r, i, val in w_coeffs:
                w_idx = (2 * b + r) * n + i
                W[w_idx][col] = val

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
