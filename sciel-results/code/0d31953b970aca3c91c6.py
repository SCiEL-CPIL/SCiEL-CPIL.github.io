import numpy as np
import time

DIMENSIONS = [4, 4, 5]

def _strassen_2x2():
    """
    Return the 7 Strassen products for 2x2 matrix multiplication.
    Each product is (coeff_U, coeff_V, coeff_W) for the 4 entries.
    U entries: (r,c) in 2x2 -> index r*2+c
    V entries: (r,c) in 2x2 -> index r*2+c  
    W entries: (r,c) in 2x2 -> index r*2+c
    """
    # Strassen's 7 products for [[a,b],[c,d]] * [[e,f],[g,h]] = [[ae+bg, af+bh],[ce+dg, cf+dh]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-d)(f+g)
    # M7 = (b-d)(e+g)
    # Result: [[M1+M4-M5+M7, M3+M5], [M2+M4, M1-M2+M3+M6]]
    
    # For each of the 7 products, specify which U,V,W entries get which coefficient
    # U is 2x2 (4 entries), V is 2x2 (4 entries), W is 2x2 (4 entries)
    # U index: 0=(0,0)=a, 1=(0,1)=b, 2=(1,0)=c, 3=(1,1)=d
    # V index: 0=(0,0)=e, 1=(0,1)=f, 2=(1,0)=g, 3=(1,1)=h
    # W index: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    products = []
    
    # M1 = (a+d)(e+h): U: a=+1, d=+1; V: e=+1, h=+1
    products.append(([(0,1),(3,1)], [(0,1),(3,1)], []))
    
    # M2 = (c+d)e: U: c=+1, d=+1; V: e=+1
    products.append(([(2,1),(3,1)], [(0,1)], []))
    
    # M3 = a(f-h): U: a=+1; V: f=+1, h=-1
    products.append(([(0,1)], [(1,1),(3,-1)], []))
    
    # M4 = d(g-e): U: d=+1; V: g=+1, e=-1
    products.append(([(3,1)], [(2,1),(0,-1)], []))
    
    # M5 = (a+b)h: U: a=+1, b=+1; V: h=+1
    products.append(([(0,1),(1,1)], [(3,1)], []))
    
    # M6 = (c-d)(f+g): U: c=+1, d=-1; V: f=+1, g=+1
    products.append(([(2,1),(3,-1)], [(1,1),(2,1)], []))
    
    # M7 = (b-d)(e+g): U: b=+1, d=-1; V: e=+1, g=+1
    products.append(([(1,1),(3,-1)], [(0,1),(2,1)], []))
    
    # Now we need to map these to W contributions
    # M1+M4-M5+M7 -> W[0,0]
    # M3+M5 -> W[0,1]
    # M2+M4 -> W[1,0]
    # M1-M2+M3+M6 -> W[1,1]
    
    # For each product, specify W contributions
    w_contribs = [
        # M1: W[0,0]+=1, W[1,1]+=1
        [(0,1),(3,1)],
        # M2: W[1,0]+=1, W[1,1]-=1
        [(2,1),(3,-1)],
        # M3: W[0,1]+=1, W[1,1]+=1
        [(1,1),(3,1)],
        # M4: W[0,0]+=1, W[1,0]+=1
        [(0,1),(2,1)],
        # M5: W[0,0]-=1, W[0,1]+=1
        [(0,-1),(1,1)],
        # M6: W[1,1]+=1
        [(3,1)],
        # M7: W[0,0]+=1
        [(0,1)],
    ]
    
    return products, w_contribs

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid decomposition of the matmul tensor."""
    rank = len(U[0]) if U else 0
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _build_hybrid_decomposition(n, m, p):
    """
    Build hybrid decomposition:
    - Partition B (m x p) into B1 (m x m) and B2 (m x (p-m))
    - C1 = A @ B1 using Strassen per 2x2 block (rank 7 per block)
    - C2 = A @ B2 using schoolbook (rank m per column)
    """
    assert m == 4 and p == 5 and n == 4
    
    # Number of 2x2 blocks in 4x4
    n_blocks = (n // 2) * (m // 2)  # 4 blocks
    strassen_rank_per_block = 7
    strassen_total_rank = n_blocks * strassen_rank_per_block  # 28
    
    # C2 = A @ B2 where B2 is 4x1, rank = 4
    vec_rank = m  # 4
    
    total_rank = strassen_total_rank + vec_rank  # 32
    
    U = [[0] * total_rank for _ in range(n * m)]  # 16 x 32
    V = [[0] * total_rank for _ in range(m * p)]  # 20 x 32
    W = [[0] * total_rank for _ in range(p * n)]  # 20 x 32
    
    # Part 1: C1 = A @ B1 (4x4 * 4x4 -> 4x4)
    # W index for C1: k*n + i where k in [0,4), i in [0,4)
    # But C1 only contributes to k in [0,4) (the first 4 columns of C)
    
    products, w_contribs = _strassen_2x2()
    
    col = 0
    for bi in range(n // 2):  # block row in A
        for bj in range(m // 2):  # block col in A (= block col in B1)
            # Block A[2*bi:2*bi+2, 2*bj:2*bj+2]
            # Block B1[2*bj:2*bj+2, 2*bj:2*bj+2]
            # Block C1[2*bi:2*bi+2, 2*bj:2*bj+2]
            
            for prod_idx in range(7):
                u_entries = products[prod_idx][0]
                v_entries = products[prod_idx][1]
                w_entries = w_contribs[prod_idx]
                
                for (u_idx, u_coeff) in u_entries:
                    # U index: (2*bi + u_idx//2) * m + (2*bj + u_idx%2)
                    ui = 2 * bi + u_idx // 2
                    uj = 2 * bj + u_idx % 2
                    U[ui * m + uj][col] += u_coeff
                
                for (v_idx, v_coeff) in v_entries:
                    # V index: (2*bj + v_idx//2) * p + (2*bj + v_idx%2)
                    vi = 2 * bj + v_idx // 2
                    vk = 2 * bj + v_idx % 2
                    V[vi * p + vk][col] += v_coeff
                
                for (w_idx, w_coeff) in w_entries:
                    # W index: (2*bj + w_idx%2) * n + (2*bi + w_idx//2)
                    # w_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
                    wk = 2 * bj + w_idx % 2
                    wi = 2 * bi + w_idx // 2
                    W[wk * n + wi][col] += w_coeff
                
                col += 1
    
    # Part 2: C2 = A @ B2 (4x4 * 4x1 -> 4x1)
    # B2 is column 4 (0-indexed) of B
    # C2[k, i] = sum_j A[i,j] * B2[j]
    # W index for C2: k*n + i where k=4 (the 5th column), i in [0,4)
    
    for j in range(m):
        # U[i,j] for all i
        for i in range(n):
            U[i * m + j][col] += 1
        # V[j, 4] (B2[j])
        V[j * p + 4][col] += 1
        # W[4, i] for all i
        for i in range(n):
            W[4 * n + i][col] += 1
        col += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Build the hybrid Strassen + vector decomposition (rank 32)
    result = _build_hybrid_decomposition(n, m, p)
    
    # Verify
    if _verify_decomposition(result["U"], result["V"], result["W"], n, m, p):
        return result
    
    # Fallback: schoolbook (rank 80)
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

# EVOLVE-BLOCK-END
