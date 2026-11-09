import time

def _strassen_2x2x2():
    """
    Returns the 7 rank-1 terms for 2x2x2 Strassen decomposition.
    Each term is a tuple (U, V, W) of lists of 4 integers.
    U index: i*2+j
    V index: j*2+k
    W index: k*2+i
    """
    terms = []
    # p1 = (a00+a11)(b00+b11)
    terms.append(([1,0,0,1], [1,0,1,0], [1,0,0,1]))
    # p2 = (a10+a11)b00
    terms.append(([0,0,1,1], [1,0,0,0], [0,1,0,1]))
    # p3 = a00(b01-b11)
    terms.append(([1,0,0,0], [0,1,0,-1], [0,0,1,1]))
    # p4 = a11(b10-b00)
    terms.append(([0,0,0,1], [0,-1,1,0], [1,1,0,0]))
    # p5 = (a00+a01)b11
    terms.append(([1,1,0,0], [0,0,0,1], [-1,0,1,0]))
    # p6 = (a10-a00)(b00+b01)
    terms.append(([-1,0,1,0], [1,1,0,0], [0,0,0,1]))
    # p7 = (a01-a11)(b10+b11)
    terms.append(([0,1,0,-1], [0,0,1,1], [1,0,0,-1]))
    return terms

def construct():
    t_start = time.time()
    DEADLINE = 110.0 # Safety margin
    
    n, m, p = 4, 4, 6
    
    # Part 1: 4x4x4 (first 4 columns of B)
    # Rank 49 via tensor product of two Strassen decompositions
    strassen_terms = _strassen_2x2x2()
    
    # Part 2: 4x4x2 (last 2 columns of B)
    # Rank 28 via 4 independent Strassen decompositions for the block products
    
    R = 49 + 28
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 ---
    # Block indices I, J, K in {0,1}
    # Internal indices i, j, k in {0,1}
    for t1 in strassen_terms:
        U1, V1, W1 = t1
        for t2 in strassen_terms:
            U2, V2, W2 = t2
            
            # U index: (2*I + i)*4 + (2*J + j)
            # Value: U1[I*2+i] * U2[J*2+j]
            # Note: t1 corresponds to block level (I,J,K), t2 to internal (i,j,k)
            # Actually, the tensor product is symmetric.
            # Let's say t1 is for the "outer" block indices and t2 for "inner".
            # U_outer has indices I, J. U_inner has indices i, j.
            # U_total[ (2I+i)*4 + (2J+j) ] = U_outer[I*2+J] * U_inner[i*2+j]
            
            for I in range(2):
                for J in range(2):
                    for i in range(2):
                        for j in range(2):
                            u_val = U1[I*2 + J] * U2[i*2 + j]
                            if u_val != 0:
                                U[(2*I + i)*4 + (2*J + j)][col] += u_val
                                
            # V index: (2*J + j)*4 + (2*K + k)
            # Value: V1[J*2+K] * V2[j*2+k]
            for J in range(2):
                for K in range(2):
                    for j in range(2):
                        for k in range(2):
                            v_val = V1[J*2 + K] * V2[j*2 + k]
                            if v_val != 0:
                                V[(2*J + j)*4 + (2*K + k)][col] += v_val
                                
            # W index: (2*K + k)*4 + (2*I + i)
            # Value: W1[K*2+I] * W2[k*2+i]
            for K in range(2):
                for I in range(2):
                    for k in range(2):
                        for i in range(2):
                            w_val = W1[K*2 + I] * W2[k*2 + i]
                            if w_val != 0:
                                W[(2*K + k)*4 + (2*I + i)][col] += w_val
                                
            col += 1
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
    
    # --- Part 2: 4x4x2 ---
    # Columns 4,5 of B correspond to K=2 in block index (since 0,1 are cols 0-3)
    # Block products:
    # A00 * B02 -> I=0, J=0, K=2
    # A01 * B12 -> I=0, J=1, K=2
    # A10 * B02 -> I=1, J=0, K=2
    # A11 * B12 -> I=1, J=1, K=2
    
    block_configs = [
        (0, 0, 2), # A00, B02
        (0, 1, 2), # A01, B12
        (1, 0, 2), # A10, B02
        (1, 1, 2)  # A11, B12
    ]
    
    for I, J, K in block_configs:
        for t in strassen_terms:
            U_t, V_t, W_t = t
            
            # U index: (2*I + i)*4 + (2*J + j)
            # Value: U_t[i*2+j] (since block part is fixed/implicit 1)
            for i in range(2):
                for j in range(2):
                    u_val = U_t[i*2 + j]
                    if u_val != 0:
                        U[(2*I + i)*4 + (2*J + j)][col] += u_val
                        
            # V index: (2*J + j)*6 + (2*K + k)
            # Note: p=6, so V index is row*6 + col.
            # Row is 2*J+j, Col is 2*K+k.
            for j in range(2):
                for k in range(2):
                    v_val = V_t[j*2 + k]
                    if v_val != 0:
                        V[(2*J + j)*6 + (2*K + k)][col] += v_val
                        
            # W index: (2*K + k)*4 + (2*I + i)
            # Note: W index is col*n + row.
            # Col is 2*K+k, Row is 2*I+i.
            for k in range(2):
                for i in range(2):
                    w_val = W_t[k*2 + i]
                    if w_val != 0:
                        W[(2*K + k)*4 + (2*I + i)][col] += w_val
                        
            col += 1
            if time.time() > DEADLINE:
                break
        if time.time() > DEADLINE:
            break
            
    # If we hit deadline early, we might have incomplete columns.
    # But the loops are small (49 + 28 = 77 iterations), so it's very fast.
    # 77 * (16+24+24) ops is trivial.
    
    return {"U": U, "V": V, "W": W}
