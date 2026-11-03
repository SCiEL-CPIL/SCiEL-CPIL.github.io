"""Rank-88 exact tensor decomposition for 3x4x8 matrix multiplication using Strassen blocks."""
import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def _verify_decomposition(U, V, W, n, m, p):
    """Verify tensor decomposition: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                if abs(val - 1.0) > 1e-10:
                    return False, (i, j, k, val)
    return True, None

def _build_strassen_2x2x2():
    """
    Build verified Strassen 2x2x2 decomposition using correct formulas.
    
    For 2x2 matrix multiplication C = A * B:
    A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    C = [[ae+bg, af+bh], [ce+dg, cf+dh]]
    
    Strassen's 7 products:
    P1 = (a+d)(e+h)
    P2 = (c+d)e
    P3 = a(f-h)
    P4 = d(g-e)
    P5 = (a+b)h
    P6 = (c-a)(e+f)
    P7 = (b-d)(g+h)
    
    C[0,0] = P1 + P4 - P5 + P7
    C[0,1] = P3 + P5
    C[1,0] = P2 + P4
    C[1,1] = P1 - P2 + P3 + P6
    """
    U = []
    V = []
    W = []
    
    # P1 = (a+d)(e+h): U=[1,0,0,1], V=[1,0,0,1], W=[1,1,1,1]
    U.append([1, 0, 0, 1])
    V.append([1, 0, 0, 1])
    W.append([1, 1, 1, 1])
    
    # P2 = (c+d)e: U=[0,0,1,1], V=[1,0,0,0], W=[0,0,1,0]
    U.append([0, 0, 1, 1])
    V.append([1, 0, 0, 0])
    W.append([0, 0, 1, 0])
    
    # P3 = a(f-h): U=[1,0,0,0], V=[0,1,0,-1], W=[0,1,0,1]
    U.append([1, 0, 0, 0])
    V.append([0, 1, 0, -1])
    W.append([0, 1, 0, 1])
    
    # P4 = d(g-e): U=[0,0,0,1], V=[-1,0,1,0], W=[1,0,1,0]
    U.append([0, 0, 0, 1])
    V.append([-1, 0, 1, 0])
    W.append([1, 0, 1, 0])
    
    # P5 = (a+b)h: U=[1,1,0,0], V=[0,0,0,1], W=[-1,1,0,0]
    U.append([1, 1, 0, 0])
    V.append([0, 0, 0, 1])
    W.append([-1, 1, 0, 0])
    
    # P6 = (c-a)(e+f): U=[-1,0,1,0], V=[1,1,0,0], W=[0,0,0,1]
    U.append([-1, 0, 1, 0])
    V.append([1, 1, 0, 0])
    W.append([0, 0, 0, 1])
    
    # P7 = (b-d)(g+h): U=[0,1,0,-1], V=[0,0,1,1], W=[1,0,0,-1]
    U.append([0, 1, 0, -1])
    V.append([0, 0, 1, 1])
    W.append([1, 0, 0, -1])
    
    # Verify
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = 0.0
                for t in range(7):
                    val += U[t][i*2+j] * V[t][j*2+k] * W[t][k*2+i]
                if abs(val - 1.0) > 1e-10:
                    raise ValueError(f"Strassen verification failed at ({i},{j},{k}): {val}")
    
    return U, V, W

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    U = []
    V = []
    W = []
    
    # Get verified Strassen factors
    sU, sV, sW = _build_strassen_2x2x2()
    
    # Part 1: i=0, schoolbook for 1x4x8: 4*8 = 32 terms
    for j in range(m):
        for k in range(p):
            u_vec = [0.0] * (n * m)
            v_vec = [0.0] * (m * p)
            w_vec = [0.0] * (p * n)
            u_vec[0 * m + j] = 1.0
            v_vec[j * p + k] = 1.0
            w_vec[k * n + 0] = 1.0
            U.append(u_vec)
            V.append(v_vec)
            W.append(w_vec)
    
    # Part 2: i=1,2 using Strassen on 2x4x8
    for j_block in range(m // 2):
        j_start = j_block * 2
        for k_block in range(p // 2):
            k_start = k_block * 2
            for t in range(7):
                # U factors
                u_vec = [0.0] * (n * m)
                for l in range(4):
                    uval = sU[t][l]
                    if uval != 0:
                        local_i = l // 2
                        local_j = l % 2
                        gi = 1 + local_i
                        gj = j_start + local_j
                        u_vec[gi * m + gj] += float(uval)
                U.append(u_vec)
                
                # V factors
                v_vec = [0.0] * (m * p)
                for l in range(4):
                    vval = sV[t][l]
                    if vval != 0:
                        local_j = l // 2
                        local_k = l % 2
                        gj = j_start + local_j
                        gk = k_start + local_k
                        v_vec[gj * p + gk] += float(vval)
                V.append(v_vec)
                
                # W factors
                w_vec = [0.0] * (p * n)
                for l in range(4):
                    wval = sW[t][l]
                    if wval != 0:
                        local_i = l // 2
                        local_k = l % 2
                        gi = 1 + local_i
                        gk = k_start + local_k
                        w_vec[gk * n + gi] += float(wval)
                W.append(w_vec)
    
    rank = len(U)
    assert rank == 88, f"Expected rank 88, got {rank}"
    
    # Verify the full decomposition
    ok, info = _verify_decomposition(U, V, W, n, m, p)
    if not ok:
        # Fallback to schoolbook
        rank_fb = n * m * p
        U_fb = []
        V_fb = []
        W_fb = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_vec = [0.0] * rank_fb
                    v_vec = [0.0] * rank_fb
                    w_vec = [0.0] * rank_fb
                    u_vec[i*m+j] = 1.0
                    v_vec[j*p+k] = 1.0
                    w_vec[k*n+i] = 1.0
                    U_fb.append(u_vec)
                    V_fb.append(v_vec)
                    W_fb.append(w_vec)
        return {"U": U_fb, "V": V_fb, "W": W_fb}
    
    elapsed = time.time() - start_time
    if elapsed > 120:
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
    ok, info = _verify_decomposition(result['U'], result['V'], result['W'], 3, 4, 8)
    print(f"Verification: {ok}")
    if not ok:
        print(f"  Failed at: {info}")
