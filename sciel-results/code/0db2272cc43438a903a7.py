<<<<<<< SEARCH
def _build_decomposition_with_3x2x2(tiling, u322, v322, w322):
    """Build a decomposition using Strassen for i=0,1 and 3x2x2 blocks for i=2."""
    n, m, p = 3, 4, 8
    # 8 Strassen blocks (rank 7 each) + 8 3x2x2 blocks (rank 3 each) = 56 + 24 = 80
    total_rank = 8 * 7 + 8 * 3
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    U2, V2, W2 = _strassen_2x2x2()
    
    col_offset = 0
    # Strassen blocks for i=0,1
    for jp in range(2):
        for kp in range(4):
            j0, j1 = 2 * jp, 2 * jp + 1
            k0, k1 = 2 * kp, 2 * kp + 1
            for i in range(2):  # i = 0, 1
                u_rows = [i * m + j0, i * m + j1]
                v_rows = [j0 * p + k0, j0 * p + k1, j1 * p + k0, j1 * p + k1]
                w_rows = [k0 * n + i, k1 * n + i]
                # Strassen U is 4x7, V is 4x7, W is 4x7
                # U rows correspond to a_{i,j} for j in {j0,j1} -> 2 rows
                # But Strassen expects 2x2, so we map:
                # U2[0] -> a_{i,j0}, U2[1] -> a_{i,j1}
                # V2[0] -> b_{j0,k0}, V2[1] -> b_{j0,k1}, V2[2] -> b_{j1,k0}, V2[3] -> b_{j1,k1}
                # W2[0] -> c_{k0,i}, W2[1] -> c_{k1,i}, W2[2] -> c_{k0,i+1}... wait
                # W2 is for 2x2x2: W2[0]=c00, W2[1]=c01, W2[2]=c10, W2[3]=c11
                # For our block: c_{k0,i}, c_{k1,i} are the outputs
                # W2[0] -> c_{k0, i}, W2[1] -> c_{k1, i}? No...
                # In 2x2x2: c_{0,0}=W2[0], c_{0,1}=W2[1], c_{1,0}=W2[2], c_{1,1}=W2[3]
                # Here: row index in W is k*n+i, so c_{k0,i} = W[k0*n+i], c_{k1,i} = W[k1*n+i]
                # But Strassen W2 has 4 rows for 2x2 output. We only need 2 outputs (k0 and k1 for fixed i).
                # This doesn't work directly. Let me reconsider.
                
                # Actually, for a 2x2x2 Strassen block covering A rows i,i+1 and B cols k0,k1:
                # It computes c_{i,k0}, c_{i,k1}, c_{i+1,k0}, c_{i+1,k1}
                # So for i=0: c_{0,k0}, c_{0,k1}, c_{1,k0}, c_{1,k1}
                # For i=1: c_{1,k0}, c_{1,k1}, c_{2,k0}, c_{2,k1} -- but we only want i=0,1 for Strassen
                
                # Let me restructure: each Strassen block handles 2 consecutive i values.
                pass
            col_offset += 7
    
    # This approach is getting complicated. Let me just use the original tiling.
    return {"U": U, "V": V, "W": W}
=======
def _build_decomposition_with_3x2x2(tiling, u322, v322, w322):
    """Build a decomposition using Strassen for i=0,1 and 3x2x2 blocks for i=2."""
    # Placeholder - not used in current best path
    return None
>>>>>>> REPLACE

