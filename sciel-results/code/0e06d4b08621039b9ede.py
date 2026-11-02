"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank is 5 * 7 = 35
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms for a single 2x2x2 block
    # Each term is (U_indices, U_vals, V_indices, V_vals, W_indices, W_vals)
    # Indices are local to the 2x2 block:
    # U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # V: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # W: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1: (a+d)(x+w)
    # U: 0(+1), 3(+1)
    # V: 0(+1), 2(+1)  -- Note: V index j*p+k. For 2x2, p=2.
    # Local V indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # M1 uses V[0,0] and V[1,1]? No.
    # (a+d) uses A[0,0], A[1,1].
    # (x+w) uses B[0,0], B[0,1]? No.
    # Let's stick to the indices in the previous code to be safe, but generate them.
    
    # Previous code mapping for block l:
    # k1 = 2l, k2 = 2l+1
    # U indices: 0,1,2,3
    # V indices: 0*p+k1, 0*p+k2, 1*p+k1, 1*p+k2
    # W indices: k1*n+0, k1*n+1, k2*n+0, k2*n+1
    
    # Local V indices for block l:
    # v00 = 0*p + k1
    # v01 = 0*p + k2
    # v10 = 1*p + k1
    # v11 = 1*p + k2
    
    # Local W indices for block l:
    # w00 = k1*n + 0
    # w01 = k1*n + 1
    # w10 = k2*n + 0
    # w11 = k2*n + 1
    
    # Strassen Terms (Local):
    # 1: (a+d)(x+w) -> U[0],U[3] ; V[0,0],V[1,1] ; W[0,0],W[1,1]
    #    Wait, (x+w) is B[0,0] + B[0,1]? No.
    #    Standard Strassen:
    #    M1 = (A11+A22)(B11+B22)
    #    M2 = (A21+A22)B11
    #    M3 = A11(B21-B11)
    #    M4 = A22(B12-B22)
    #    M5 = (A11+A12)B22
    #    M6 = (A21-A11)(B11+B12)
    #    M7 = (A12-A22)(B21+B22)
    
    # Let's re-verify the previous code's mapping to ensure I replicate the correct math.
    # Previous code:
    # M1: (a+d)(x+w) -> U[0],U[3]; V[0,k1],V[1,k2]; W[k1,0],W[k2,1]
    # Here x=B[0,k1], w=B[1,k2]? No.
    # B is 2x10. Block l is columns 2l, 2l+1.
    # B[0, 2l] = x, B[0, 2l+1] = y?
    # B[1, 2l] = z, B[1, 2l+1] = w?
    # Let's assume B = [[x, y], [z, w]] for the block.
    # Then V indices:
    # V[0, k1] is B[0, 2l] = x
    # V[0, k2] is B[0, 2l+1] = y
    # V[1, k1] is B[1, 2l] = z
    # V[1, k2] is B[1, 2l+1] = w
    
    # Previous M1: (a+d)(x+w).
    # U: A[0,0]=a, A[1,1]=d.
    # V: x=B[0,k1], w=B[1,k2].
    # So V indices are 0*p+k1 and 1*p+k2.
    # This corresponds to B[0,0] and B[1,1] of the block.
    # So M1 = (A11+A22)(B11+B22).
    
    # Previous M2: (c+d)x.
    # U: A[1,0]=c, A[1,1]=d.
    # V: x=B[0,k1] = B[0,0].
    # So M2 = (A21+A22)B11.
    
    # Previous M3: a(y-w).
    # U: A[0,0]=a.
    # V: y=B[0,k2]=B[0,1], w=B[1,k2]=B[1,1].
    # So M3 = A11(B12-B22).
    # Note: Standard Strassen M3 is A11(B21-B11).
    # The previous code uses a variant. I will replicate the previous code's math exactly to ensure correctness, just changing the generation method.
    
    # I will hardcode the 7 term definitions in a list of tuples and then loop over blocks.
    
    terms = [
        # (U_idxs, U_vals, V_local_idxs, V_vals, W_local_idxs, W_vals)
        # V_local: 0:(0,k1), 1:(0,k2), 2:(1,k1), 3:(1,k2)
        # W_local: 0:(k1,0), 1:(k1,1), 2:(k2,0), 3:(k2,1)
        
        # M1: (a+d)(x+w)
        ([0, 3], [1, 1], [0, 3], [1, 1], [0, 3], [1, 1]),
        # M2: (c+d)x
        ([2, 3], [1, 1], [0], [1], [1, 3], [1, -1]),
        # M3: a(y-w)
        ([0], [1], [1, 3], [1, -1], [2, 3], [1, 1]),
        # M4: d(z-x)
        ([3], [1], [2, 0], [1, -1], [0, 1], [1, 1]),
        # M5: (a+b)w
        ([0, 1], [1, 1], [3], [1], [0, 2], [-1, 1]),
        # M6: (c-a)(x+y)
        ([2, 0], [1, -1], [0, 1], [1, 1], [3], [1]),
        # M7: (b-d)(z+w)
        ([1, 3], [1, -1], [2, 3], [1, 1], [0], [1]),
    ]
    
    for l in range(5):
        k1 = 2 * l
        k2 = 2 * l + 1
        base_col = 7 * l
        
        # Map local V indices to global V indices
        # Local 0 -> 0*p + k1
        # Local 1 -> 0*p + k2
        # Local 2 -> 1*p + k1
        # Local 3 -> 1*p + k2
        v_map = [0*p + k1, 0*p + k2, 1*p + k1, 1*p + k2]
        
        # Map local W indices to global W indices
        # Local 0 -> k1*n + 0
        # Local 1 -> k1*n + 1
        # Local 2 -> k2*n + 0
        # Local 3 -> k2*n + 1
        w_map = [k1*n + 0, k1*n + 1, k2*n + 0, k2*n + 1]
        
        for t_idx, (u_idxs, u_vals, v_l_idxs, v_vals, w_l_idxs, w_vals) in enumerate(terms):
            c = base_col + t_idx
            for u_idx, val in zip(u_idxs, u_vals):
                U[u_idx][c] += val
            for v_l_idx, val in zip(v_l_idxs, v_vals):
                V[v_map[v_l_idx]][c] += val
            for w_l_idx, val in zip(w_l_idxs, w_vals):
                W[w_map[w_l_idx]][c] += val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
