import numpy as np
import time

def _verify(U, V, W, tol=1e-6):
    """Verify that the tensor decomposition is valid:
    sum_t U[i*4+j, t] * V[j*4+k, t] * W[k*4+i, t] == 1 for all (i,j,k)
    """
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                s = 0.0
                u = U[i*4+j, :]
                v = V[j*4+k, :]
                w = W[k*4+i, :]
                for t in range(rank):
                    s += u[t] * v[t] * w[t]
                if abs(s - 1.0) > tol:
                    valid = False
    return valid

def _schoolbook():
    """Return the schoolbook 4x4x4 decomposition (rank 64)."""
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    t0 = time.time()
    
    # Strassen's 2x2x2 algorithm
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)
    
    # Kronecker products give 16x49
    U4k = np.kron(U2, U2)
    V4k = np.kron(V2, V2)
    W4k = np.kron(W2, W2)
    
    # Build permutation for U (input A, row-major i*4+j):
    # i = 2*i_out + i_in, j = 2*j_out + j_in
    # K = (i_out*2 + j_out)*4 + (i_in*2 + j_in)
    perm_U = np.zeros(16, dtype=int)
    for S in range(16):
        i = S // 4
        j = S % 4
        i_out, i_in = i // 2, i % 2
        j_out, j_in = j // 2, j % 2
        K = (i_out * 2 + j_out) * 4 + (i_in * 2 + j_in)
        perm_U[S] = K
    
    # For V (input B, row-major j*4+k):
    # j = 2*j_out + j_in, k = 2*k_out + k_in
    # K = (j_out*2 + k_out)*4 + (j_in*2 + k_in)
    perm_V = np.zeros(16, dtype=int)
    for S in range(16):
        j = S // 4
        k = S % 4
        j_out, j_in = j // 2, j % 2
        k_out, k_in = k // 2, k % 2
        K = (j_out * 2 + k_out) * 4 + (j_in * 2 + k_in)
        perm_V[S] = K
    
    # For W (output C, trace convention W[k*4+i]):
    # k = 2*k_out + k_in, i = 2*i_out + i_in
    # K = (k_out*2 + i_out)*4 + (k_in*2 + i_in)
    perm_W = np.zeros(16, dtype=int)
    for S in range(16):
        k = S // 4
        i = S % 4
        k_out, k_in = k // 2, k % 2
        i_out, i_in = i // 2, i % 2
        K = (k_out * 2 + i_out) * 4 + (k_in * 2 + i_in)
        perm_W[S] = K
    
    U4 = U4k[perm_U, :]
    V4 = V4k[perm_V, :]
    W4 = W4k[perm_W, :]
    
    # Verify the rank-49 Kronecker decomposition using vectorized approach
    rank49 = U4.shape[1]
    ii = np.repeat(np.arange(4), 16)
    jj = np.tile(np.repeat(np.arange(4), 4), 4)
    kk = np.tile(np.arange(4), 16)
    
    ur = U4[ii * 4 + jj, :]
    vr = V4[jj * 4 + kk, :]
    wr = W4[kk * 4 + ii, :]
    totals = (ur * vr * wr).sum(axis=1)
    
    if not np.allclose(totals, 1.0, atol=1e-10):
        return _schoolbook()
    
    # Try rank-48 by removing columns from Kronecker product
    # The Kronecker product has rank 49. Try removing one column at a time
    # and see if the decomposition still works.
    removed_ok = False
    U4_r48 = None
    V4_r48 = None
    W4_r48 = None
    
    # Try all 49 columns - check if removing any single column still gives valid decomposition
    for rem in range(49):
        keep = [x for x in range(49) if x != rem]
        U4_t = U4k[perm_U, :][:, keep]
        V4_t = V4k[perm_V, :][:, keep]
        W4_t = W4k[perm_W, :][:, keep]
        
        ur2 = U4_t[ii * 4 + jj, :]
        vr2 = V4_t[jj * 4 + kk, :]
        wr2 = W4_t[kk * 4 + ii, :]
        t2 = (ur2 * vr2 * wr2).sum(axis=1)
        
        if np.allclose(t2, 1.0, atol=1e-10):
            removed_ok = True
            U4_r48 = U4_t
            V4_r48 = V4_t
            W4_r48 = W4_t
            break
    
    if removed_ok:
        # Now try removing a second column from the rank-48 solution
        rank48 = U4_r48.shape[1]
        for rem2 in range(rank48):
            keep2 = [x for x in range(rank48) if x != rem2]
            U4_t2 = U4_r48[:, keep2]
            V4_t2 = V4_r48[:, keep2]
            W4_t2 = W4_r48[:, keep2]
            
            ur3 = U4_t2[ii * 4 + jj, :]
            vr3 = V4_t2[jj * 4 + kk, :]
            wr3 = W4_t2[kk * 4 + ii, :]
            t3 = (ur3 * vr3 * wr3).sum(axis=1)
            
            if np.allclose(t3, 1.0, atol=1e-10):
                # Found rank-47!
                # Convert to list of lists with int or half-int entries
                def to_outlist(A):
                    out = []
                    for row in A:
                        r = []
                        for v in row:
                            if abs(v - round(v)) < 1e-10:
                                r.append(int(round(v)))
                            else:
                                # Check if it's a half-integer
                                if abs(2*v - round(2*v)) < 1e-10:
                                    r.append(int(round(2*v)) / 2.0)
                                else:
                                    r.append(float(v))
                        out.append(r)
                    return out
                return {
                    "U": to_outlist(U4_t2),
                    "V": to_outlist(V4_t2),
                    "W": to_outlist(W4_t2)
                }
        
        # No rank-47 found, return rank-48
        def to_outlist2(A):
            out = []
            for row in A:
                r = []
                for v in row:
                    if abs(v - round(v)) < 1e-10:
                        r.append(int(round(v)))
                    else:
                        if abs(2*v - round(2*v)) < 1e-10:
                            r.append(int(round(2*v)) / 2.0)
                        else:
                            r.append(float(v))
                out.append(r)
            return out
        return {
            "U": to_outlist2(U4_r48),
            "V": to_outlist2(V4_r48),
            "W": to_outlist2(W4_r48)
        }
    
    # Safety margin check
    if time.time() - t0 > 110:
        return _schoolbook()
    
    # Return rank-49 Kronecker product decomposition
    def to_outlist3(A):
        out = []
        for row in A:
            r = []
            for v in row:
                if abs(v - round(v)) < 1e-10:
                    r.append(int(round(v)))
                else:
                    r.append(float(v))
            out.append(r)
        return out
    return {
        "U": to_outlist3(U4),
        "V": to_outlist3(V4),
        "W": to_outlist3(W4)
    }
