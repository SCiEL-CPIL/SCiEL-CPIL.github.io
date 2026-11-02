"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
_DIM_START = time.time()
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    _t0 = time.time()
    _deadline = 140.0  # safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Use Strassen's algorithm for each 2x2 block of B.
    # There are 4 such blocks (since p=8, m=2, so 8/2 = 4 blocks).
    # Each block uses rank 7. Total rank = 4 * 7 = 28.
    num_blocks = p // 2  # 4
    rank = num_blocks * 7  # 28
    
    U = [[0] * rank for _ in range(n * m)]   # 4 x 28
    V = [[0] * rank for _ in range(m * p)]   # 16 x 28
    W = [[0] * rank for _ in range(p * n)]   # 16 x 28
    
    # Correct Strassen terms for M_{2,2,2}
    # u, v, w are vectors of length 4 (flattened 2x2 matrices: [00, 01, 10, 11])
    # A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = A*B
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1: u=[a,0,c? no...
        # Let me be more careful. Flattened as [00,01,10,11] = [a,b,c,d]
        # M1 = (a+d)(e+h): u=[1,0,0,1], v=[1,0,0,1], w contribution to C[0,0] and C[1,1]
        # C[0,0] += M1, C[1,1] += M1
        # So w for M1: C[0,0]=1, C[0,1]=0, C[1,0]=0, C[1,1]=1 => [1,0,0,1]
        
        # M2 = (c+d)e: u=[0,0,1,1], v=[1,0,0,0], 
        # C[1,0] += M2, C[1,1] -= M2
        # w: C[0,0]=0, C[0,1]=0, C[1,0]=1, C[1,1]=-1 => [0,0,1,-1]
        
        # M3 = a(f-h): u=[1,0,0,0], v=[0,1,0,-1],
        # C[0,1] += M3, C[1,1] += M3
        # w: [0,1,0,1]
        
        # M4 = d(g-e): u=[0,0,0,1], v=[-1,0,1,0],
        # C[0,0] += M4, C[1,0] += M4
        # w: [1,0,1,0]
        
        # M5 = (a+b)h: u=[1,1,0,0], v=[0,0,0,1],
        # C[0,0] -= M5, C[0,1] += M5
        # w: [-1,1,0,0]
        
        # M6 = (c-a)(e+f): u=[-1,0,1,0], v=[1,1,0,0],
        # C[1,1] += M6
        # w: [0,0,0,1]
        
        # M7 = (b-d)(g+h): u=[0,1,0,-1], v=[0,0,1,1],
        # C[0,0] += M7
        # w: [1,0,0,0]
    ]
    
    # Correct terms list:
    terms = [
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),  # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),  # M2
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),   # M3
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),   # M4
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),   # M5
        ([-1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),   # M6
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]),   # M7
    ]
    
    for block_idx in range(num_blocks):
        base = block_idx * 7
        for term_idx in range(7):
            col = base + term_idx
            u, v, w = terms[term_idx]
            
            # Fill U: index is i*m+j, flattened as [00,01,10,11]
            for idx in range(4):
                U[idx][col] = u[idx]
                
            # Fill V: index is j*p+k
            # Local B indices: [00,01,10,11] -> j=0,k=0; j=0,k=1; j=1,k=0; j=1,k=1
            # Global k = 2*block_idx + k_local
            for idx in range(4):
                j = idx // 2
                k_local = idx % 2
                k_global = 2 * block_idx + k_local
                v_idx = j * p + k_global
                V[v_idx][col] = v[idx]
                
            # Fill W: index is k*n+i
            # Local C indices: [00,01,10,11] -> i=0,k=0; i=0,k=1; i=1,k=0; i=1,k=1
            # Global k = 2*block_idx + k_local
            for idx in range(4):
                i = idx % 2
                k_local = idx // 2
                k_global = 2 * block_idx + k_local
                w_idx = k_global * n + i
                W[w_idx][col] = w[idx]
    
    # Verify the decomposition
    def verify(U, V, W, n, m, p):
        nm, mp, pn = n*m, m*p, p*n
        rank = len(U[0])
        # Build valid index set
        valid = set()
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    valid.add((i*m+j, j*p+k, k*n+i))
        # Check all entries
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    expected = 1 if (a, b, c) in valid else 0
                    if s != expected:
                        return False
        return True
    
    if verify(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    
    # Fallback: schoolbook rank 32
    rank_school = n * m * p  # 32
    U2 = [[0] * rank_school for _ in range(n * m)]
    V2 = [[0] * rank_school for _ in range(m * p)]
    W2 = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U2[i * m + j][col] = 1
                V2[j * p + k][col] = 1
                W2[k * n + i][col] = 1
                col += 1
    
    return {"U": U2, "V": V2, "W": W2}
# EVOLVE-BLOCK-END
