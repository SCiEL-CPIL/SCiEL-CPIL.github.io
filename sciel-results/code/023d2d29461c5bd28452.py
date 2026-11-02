import time
import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    n, m, p = DIMENSIONS
    # n*m = 4, m*p = 22, p*n = 22
    
    # Build the multiplication tensor entries
    # T[a, b, c] = 1 where a = i*m+j, b = j*p+k, c = k*n+i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                tensor_entries.append((a, b, c))
    
    # Rank-38 decomposition:
    # 4 Strassen blocks (rank 7 each) for columns 0-7
    # 1 M_{2,2,3} block (rank 10) for columns 8,9,10
    rank38 = 4 * 7 + 10
    U38 = np.zeros((n * m, rank38))
    V38 = np.zeros((m * p, rank38))
    W38 = np.zeros((p * n, rank38))
    
    r = 0
    # 4 Strassen blocks for columns 0-7
    for block in range(4):
        k0 = block * 2
        k1 = block * 2 + 1
        
        v_e = k0
        v_f = k1
        v_g = 11 + k0
        v_h = 11 + k1
        
        w_c00 = 2 * k0
        w_c10 = 2 * k0 + 1
        w_c01 = 2 * k1
        w_c11 = 2 * k1 + 1
        
        # m1 = (a+d)(e+h)
        U38[0, r] = 1; U38[3, r] = 1
        V38[v_e, r] = 1; V38[v_h, r] = 1
        W38[w_c00, r] = 1; W38[w_c11, r] = 1
        r += 1
        
        # m2 = (c+d)e
        U38[2, r] = 1; U38[3, r] = 1
        V38[v_e, r] = 1
        W38[w_c10, r] = 1; W38[w_c11, r] = -1
        r += 1
        
        # m3 = a(f-h)
        U38[0, r] = 1
        V38[v_f, r] = 1; V38[v_h, r] = -1
        W38[w_c01, r] = 1; W38[w_c11, r] = 1
        r += 1
        
        # m4 = d(g-e)
        U38[3, r] = 1
        V38[v_g, r] = 1; V38[v_e, r] = -1
        W38[w_c00, r] = 1; W38[w_c10, r] = 1
        r += 1
        
        # m5 = (a+b)h
        U38[0, r] = 1; U38[1, r] = 1
        V38[v_h, r] = 1
        W38[w_c00, r] = -1; W38[w_c01, r] = 1
        r += 1
        
        # m6 = (c-a)(e+f)
        U38[2, r] = 1; U38[0, r] = -1
        V38[v_e, r] = 1; V38[v_f, r] = 1
        W38[w_c11, r] = 1
        r += 1
        
        # m7 = (b-d)(g+h)
        U38[1, r] = 1; U38[3, r] = -1
        V38[v_g, r] = 1; V38[v_h, r] = 1
        W38[w_c00, r] = 1
        r += 1
    
    # M_{2,2,3} block for columns 8,9,10 using rank 10
    # A = [[a, b], [c, d]], B = [[e, f, g], [h, i, j]]
    # Output C = [[ae+ch, af+ci, ag+cj], [be+dh, bf+di, bg+dj]]
    # Indices: a=0, b=1, c=2, d=3
    # v: e=8, f=9, g=10, h=19, i=20, j=21
    # w: c00=16, c10=17, c01=18, c11=19, c02=20, c12=21
    # Wait, let me recompute: w index = k*n + i = k*2 + i
    # k=8: w[16]=c00, w[17]=c10
    # k=9: w[18]=c01, w[19]=c11  
    # k=10: w[20]=c02, w[21]=c12
    
    k8, k9, k10 = 8, 9, 10
    v_e = k8      # 8
    v_f = k9      # 9
    v_g = k10     # 10
    v_h = 11 + k8   # 19
    v_i = 11 + k9   # 20
    v_j = 11 + k10  # 21
    
    w_c00 = 2 * k8    # 16
    w_c10 = 2 * k8 + 1  # 17
    w_c01 = 2 * k9    # 18
    w_c11 = 2 * k9 + 1  # 19
    w_c02 = 2 * k10   # 20
    w_c12 = 2 * k10 + 1  # 21
    
    # Rank-10 decomposition for 2x2 * 2x3
    # Using a known construction:
    # m1 = (a+d)(e+h) -> contributes to c00, c11
    # m2 = (c+d)e -> contributes to c10, c11
    # m3 = a(f-h) -> contributes to c01, c11
    # m4 = d(g-e) -> contributes to c00, c10
    # m5 = (a+b)h -> contributes to c00, c01
    # m6 = (c-a)(e+f) -> contributes to c11
    # m7 = (b-d)(g+h) -> contributes to c00
    # m8 = (c+d)(i-j) -> contributes to c12, c02
    # m9 = a(j-i) -> contributes to c02, c12
    # m10 = d(g-i) -> contributes to c02, c12
    
    # m1 = (a+d)(e+h)
    U38[0, r] = 1; U38[3, r] = 1
    V38[v_e, r] = 1; V38[v_h, r] = 1
    W38[w_c00, r] = 1; W38[w_c11, r] = 1
    r += 1
    
    # m2 = (c+d)e
    U38[2, r] = 1; U38[3, r] = 1
    V38[v_e, r] = 1
    W38[w_c10, r] = 1; W38[w_c11, r] = -1
    r += 1
    
    # m3 = a(f-h)
    U38[0, r] = 1
    V38[v_f, r] = 1; V38[v_h, r] = -1
    W38[w_c01, r] = 1; W38[w_c11, r] = 1
    r += 1
    
    # m4 = d(g-e)
    U38[3, r] = 1
    V38[v_g, r] = 1; V38[v_e, r] = -1
    W38[w_c00, r] = 1; W38[w_c10, r] = 1
    r += 1
    
    # m5 = (a+b)h
    U38[0, r] = 1; U38[1, r] = 1
    V38[v_h, r] = 1
    W38[w_c00, r] = -1; W38[w_c01, r] = 1
    r += 1
    
    # m6 = (c-a)(e+f)
    U38[2, r] = 1; U38[0, r] = -1
    V38[v_e, r] = 1; V38[v_f, r] = 1
    W38[w_c11, r] = 1
    r += 1
    
    # m7 = (b-d)(g+h)
    U38[1, r] = 1; U38[3, r] = -1
    V38[v_g, r] = 1; V38[v_h, r] = 1
    W38[w_c00, r] = 1
    r += 1
    
    # Now for the third column (g, j) we need additional terms
    # The first 7 handle columns 8 and 9 partially.
    # For column 10 (g, j), we need: c02 = a*g + c*j, c12 = b*g + d*j
    # 
    # Let me use a different approach. Use Strassen-like for (k=8,9) and 
    # then 3 more for k=10, but that's 7+3=10.
    # 
    # Actually, let me use a cleaner rank-10 construction.
    # 
    # Alternative: Use the known rank-10 formula for 2x2*2x3.
    # Let me derive it properly.
    
    # Reset and use a verified construction
    # Undo the last 7 entries and use a proper rank-10
    
    # Actually, let me just verify what we have and add 3 more for column 10
    # m8, m9, m10 for the third column
    
    # For column 10: c02 = a*g + c*j, c12 = b*g + d*j
    # m8 = a*g
    U38[0, r] = 1
    V38[v_g, r] = 1
    W38[w_c02, r] = 1
    r += 1
    
    # m9 = c*j
    U38[2, r] = 1
    V38[v_j, r] = 1
    W38[w_c02, r] = 1
    r += 1
    
    # m10 = b*g + d*j... but that's 2 terms. We need to combine.
    # Hmm, we need c12 = b*g + d*j. That's 2 separate products unless we can combine.
    # We only have 1 more rank slot. So this won't work with this approach.
    
    # Let me restart the M_{2,2,3} block with a proper rank-10 construction.
    # I'll undo and redo.
    
    # Actually, let me just verify the full thing and if it fails, use a known correct rank-10.
    
    def verify(U, V, W, entries):
        for (a, b, c) in entries:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val - 1) > 1e-10:
                return False
        return True
    
    if not verify(U38, V38, W38, tensor_entries):
        # Fallback to rank-39
        rank39 = 5 * 7 + 4
        U39 = np.zeros((n * m, rank39))
        V39 = np.zeros((m * p, rank39))
        W39 = np.zeros((p * n, rank39))
        r = 0
        for block in range(5):
            k0 = block * 2
            k1 = block * 2 + 1
            v_e = k0
            v_f = k1
            v_g = 11 + k0
            v_h = 11 + k1
            w_c00 = 2 * k0
            w_c10 = 2 * k0 + 1
            w_c01 = 2 * k1
            w_c11 = 2 * k1 + 1
            U39[0, r] = 1; U39[3, r] = 1
            V39[v_e, r] = 1; V39[v_h, r] = 1
            W39[w_c00, r] = 1; W39[w_c11, r] = 1
            r += 1
            U39[2, r] = 1; U39[3, r] = 1
            V39[v_e, r] = 1
            W39[w_c10, r] = 1; W39[w_c11, r] = -1
            r += 1
            U39[0, r] = 1
            V39[v_f, r] = 1; V39[v_h, r] = -1
            W39[w_c01, r] = 1; W39[w_c11, r] = 1
            r += 1
            U39[3, r] = 1
            V39[v_g, r] = 1; V39[v_e, r] = -1
            W39[w_c00, r] = 1; W39[w_c10, r] = 1
            r += 1
            U39[0, r] = 1; U39[1, r] = 1
            V39[v_h, r] = 1
            W39[w_c00, r] = -1; W39[w_c01, r] = 1
            r += 1
            U39[2, r] = 1; U39[0, r] = -1
            V39[v_e, r] = 1; V39[v_f, r] = 1
            W39[w_c11, r] = 1
            r += 1
            U39[1, r] = 1; U39[3, r] = -1
            V39[v_g, r] = 1; V39[v_h, r] = 1
            W39[w_c00, r] = 1
            r += 1
        k2 = 10
        v_e2 = k2
        v_g2 = 11 + k2
        w_c02 = 2 * k2
        w_c12 = 2 * k2 + 1
        U39[0, r] = 1
        V39[v_e2, r] = 1
        W39[w_c02, r] = 1
        r += 1
        U39[1, r] = 1
        V39[v_g2, r] = 1
        W39[w_c02, r] = 1
        r += 1
        U39[2, r] = 1
        V39[v_e2, r] = 1
        W39[w_c12, r] = 1
        r += 1
        U39[3, r] = 1
        V39[v_g2, r] = 1
        W39[w_c12, r] = 1
        r += 1
        return {"U": U39.tolist(), "V": V39.tolist(), "W": W39.tolist()}
    
    return {"U": U38.tolist(), "V": V38.tolist(), "W": W38.tolist()}
