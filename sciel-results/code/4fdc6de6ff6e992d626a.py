"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 35 decomposition: 5 blocks of rank 7
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Rank-7 decomposition vectors for 2x2x2 multiplication
    # Local indices: u[i*m+j], v[j*p+k], w[k*n+i]
    # m=2, p=2 for the block
    # u indices: 0,1,2,3
    # v indices: 0,1,2,3 (mapped to global 10j+k)
    # w indices: 0,1,2,3 (mapped to global 2k+i)
    
    # The 7 rank-1 tensors
    # Format: (u_vals, v_vals, w_vals)
    # u_vals: [u00, u01, u10, u11]
    # v_vals: [v00, v01, v10, v11]
    # w_vals: [w00, w01, w10, w11]
    
    # Note: v index in block is j*2+k.
    # j=0, k=0 -> 0
    # j=0, k=1 -> 1
    # j=1, k=0 -> 2
    # j=1, k=1 -> 3
    
    # w index in block is k*2+i.
    # k=0, i=0 -> 0
    # k=0, i=1 -> 1
    # k=1, i=0 -> 2
    # k=1, i=1 -> 3
    
    vectors = [
        ((1, 0, 1, 1), (1, 1, 0, 1), (1, 0, 1, 1)),
        ((0, 0, 1, 1), (1, 1, 0, 1), (0, 0, 1, 1)),
        ((1, 0, 0, 0), (0, 1, 0, -1), (0, 1, 0, 0)),
        ((0, 0, 0, 1), (0, 0, 1, -1), (0, 0, 0, 1)),
        ((1, 1, 0, 0), (0, 0, 1, 0), (0, 0, 1, 0)),
        ((0, 0, 1, 1), (0, 1, 0, 0), (0, 1, 0, 0)),
        ((0, 1, 0, -1), (0, 0, 0, 1), (0, 1, 0, 0))
    ]
    
    for b in range(5):
        for r_idx in range(7):
            col = b * 7 + r_idx
            u_vals, v_vals, w_vals = vectors[r_idx]
            
            # Map U (global index 2i+j)
            # u_vals indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            for i in range(2):
                for j in range(2):
                    local_idx = i * 2 + j
                    U[i * m + j][col] = u_vals[local_idx]
                    
            # Map V (global index 10j+k)
            # v_vals indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # Block b corresponds to k in {2b, 2b+1}
            # v local j=0, k=0 -> global j=0, k=2b
            # v local j=0, k=1 -> global j=0, k=2b+1
            # v local j=1, k=0 -> global j=1, k=2b
            # v local j=1, k=1 -> global j=1, k=2b+1
            for j in range(2):
                for k_local in range(2):
                    local_idx = j * 2 + k_local
                    k_global = 2 * b + k_local
                    V[j * p + k_global][col] = v_vals[local_idx]
                    
            # Map W (global index 2k+i)
            # w_vals indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
            # w local k=0, i=0 -> global k=2b, i=0
            # w local k=0, i=1 -> global k=2b, i=1
            # w local k=1, i=0 -> global k=2b+1, i=0
            # w local k=1, i=1 -> global k=2b+1, i=1
            for k_local in range(2):
                for i in range(2):
                    local_idx = k_local * 2 + i
                    k_global = 2 * b + k_local
                    W[k_global * n + i][col] = w_vals[local_idx]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
