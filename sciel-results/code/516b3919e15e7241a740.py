"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct rank-35 decomposition for 2x2 * 2x10 matrix multiplication.
    Uses block-diagonal Strassen (rank 7 per 2x2 block, 5 blocks).
    """
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    # Initialize matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen terms defined as lists of (index, value) pairs for U, V, W
    # U index: i*m + j
    # V index: j*p + k (k is local block index 0 or 1)
    # W index: k*n + i (k is local block index 0 or 1)
    
    # Term 0: (a+d)(e+h) -> C[0,0]+, C[1,1]+
    # U: (0,0)=1, (1,1)=1
    # V: (0,0)=1, (1,1)=1
    # W: (0,0)=1, (1,1)=1
    t0_u = [(0, 1), (3, 1)]
    t0_v = [(0, 1), (3, 1)] # j*p+k: 0*p+0=0, 1*p+1=p+1? No, local k.
    # Wait, V index is j*p + k_global.
    # In the loop, we will add 2*r to the local k.
    # So let's define local indices and map them.
    
    # Let's define the terms with local indices (0,1) for j,k,i
    # U: (i,j)
    # V: (j,k)
    # W: (k,i)
    
    terms = [
        # (u_entries, v_entries, w_entries)
        # Each entry is (local_index, value)
        # local_index for U: i*m+j
        # local_index for V: j*p+k (k is 0 or 1) -> We will add 2*r to k part?
        # No, V[j, k_global]. k_global = 2*r + k_local.
        # So V_index = j*p + 2*r + k_local.
        # W[k_global, i]. W_index = (2*r + k_local)*n + i.
        
        # We store local indices and adjust in loop.
        
        # T0
        ([(0, 1), (3, 1)], [(0, 1), (p+1, 1)], [(0, 1), (n+1, 1)]), # V: j=0,k=0 -> 0; j=1,k=1 -> p+1. W: k=0,i=0 -> 0; k=1,i=1 -> n+1.
        # T1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (n+1, -1)]), # U: c(2),d(3). V: e(0). W: C[1,0] -> k=0,i=1 -> 1; C[1,1] -> k=1,i=1 -> n+1.
        # T2
        ([(0, 1)], [(1, 1), (p+1, -1)], [(n, 1), (n+1, 1)]), # U: a(0). V: f(1), h(p+1). W: C[0,1] -> k=1,i=0 -> n; C[1,1] -> k=1,i=1 -> n+1.
        # T3
        ([(3, 1)], [(p, 1), (0, -1)], [(0, 1), (1, 1)]), # U: d(3). V: g(p), e(0). W: C[0,0] -> 0; C[1,0] -> 1.
        # T4
        ([(0, 1), (1, 1)], [(p+1, 1)], [(0, -1), (n, 1)]), # U: a(0),b(1). V: h(p+1). W: C[0,0] -> 0; C[0,1] -> n.
        # T5
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(n+1, 1)]), # U: c(2),a(0). V: e(0),f(1). W: C[1,1] -> n+1.
        # T6
        ([(1, 1), (3, -1)], [(p, 1), (p+1, 1)], [(0, 1)]), # U: b(1),d(3). V: g(p),h(p+1). W: C[0,0] -> 0.
    ]
    
    # Note: The indices in terms above are for the FIRST block (r=0).
    # For block r, we need to shift the k indices in V and W by 2*r.
    # V index: j*p + k_global. k_global = k_local + 2*r.
    # So V_index = j*p + k_local + 2*r.
    # W index: k_global*n + i. k_global = k_local + 2*r.
    # So W_index = (k_local + 2*r)*n + i.
    
    # Let's re-verify the local indices in the terms list.
    # p=10, n=2, m=2.
    # V local indices used: 0, 1, p, p+1.
    # 0: j=0, k=0.
    # 1: j=0, k=1.
    # p: j=1, k=0.
    # p+1: j=1, k=1.
    # This matches j*p + k.
    
    # W local indices used: 0, 1, n, n+1.
    # 0: k=0, i=0.
    # 1: k=0, i=1.
    # n: k=1, i=0.
    # n+1: k=1, i=1.
    # This matches k*n + i.
    
    for r in range(num_blocks):
        shift = 2 * r
        for t in range(7):
            col = r * 7 + t
            u_entries, v_entries, w_entries = terms[t]
            
            for idx, val in u_entries:
                U[idx][col] = val
            
            for idx, val in v_entries:
                # idx is j*p + k_local
                # We need j*p + (k_local + shift)
                # This is not a simple addition to idx if idx = j*p + k_local.
                # idx + shift = j*p + k_local + shift. Correct.
                V[idx + shift][col] = val
            
            for idx, val in w_entries:
                # idx is k_local*n + i
                # We need (k_local + shift)*n + i = k_local*n + shift*n + i
                # So add shift * n to idx.
                W[idx + shift * n][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
