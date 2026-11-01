"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """Return a rank-minimal exact tensor decomposition for 2x2 * 2x10 matmul.
    
    Uses Strassen's algorithm for each of the 5 independent 2x2x2 block 
    multiplications. Rank is 5 * 7 = 35.
    """
    start_time = time.time()
    deadline = start_time + 140.0
    
    n, m, p = DIMENSIONS
    num_blocks = p // 2 # 5 blocks
    rank_per_block = 7
    rank = num_blocks * rank_per_block
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms for 2x2x2
    # Term 1: (a+d)(e+h) -> p, s
    # Term 2: (c+d)e -> r, -s
    # Term 3: a(f-h) -> q, s
    # Term 4: d(g-e) -> -p, q
    # Term 5: (a+b)h -> p, r
    # Term 6: (c-a)(e+f) -> s
    # Term 7: (b-d)(g+h) -> -p
    
    # U indices: a=0, b=1, c=2, d=3
    # Local V indices: e=0, f=1, g=2, h=3 (relative to block start)
    # Local W indices: p=0, q=2, r=1, s=3 (relative to block start, k*n+i)
    
    # Coefficients for U, V, W for each of the 7 terms
    # Format: list of (index, coeff)
    strassen_terms = [
        # T1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # T2
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # T3
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # T4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, -1), (2, 1)]),
        # T5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, 1), (1, 1)]),
        # T6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # T7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, -1)]),
    ]
    
    for l in range(num_blocks):
        if time.time() > deadline:
            break
        for t in range(rank_per_block):
            col = l * rank_per_block + t
            u_coeffs, v_coeffs, w_coeffs = strassen_terms[t]
            
            for idx, coeff in u_coeffs:
                U[idx][col] += coeff
                
            for idx, coeff in v_coeffs:
                # Map local V index to global
                # Local idx 0,1,2,3 correspond to j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
                # Global V index = j * p + k
                # Local idx -> j, r
                j = idx // 2
                r = idx % 2
                k = 2 * l + r
                global_v_idx = j * p + k
                V[global_v_idx][col] += coeff
                
            for idx, coeff in w_coeffs:
                # Map local W index to global
                # Local 0: k=0, i=0
                # Local 1: k=0, i=1
                # Local 2: k=1, i=0
                # Local 3: k=1, i=1
                k_local = idx // 2
                i_local = idx % 2
                k_global = 2 * l + k_local
                i_global = i_local
                global_w_idx = k_global * n + i_global
                W[global_w_idx][col] += coeff
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
