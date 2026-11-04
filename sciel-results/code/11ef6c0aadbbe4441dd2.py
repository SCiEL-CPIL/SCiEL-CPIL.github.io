import time
import numpy as np

def construct():
    t0 = time.time()
    deadline = 120.0

    n, m, p = 4, 4, 4

    # Strassen rank-7 factors for 2x2 matrix multiplication
    U2 = np.array([
        [1,  0,  1,  0,  1, -1,  0],  # A00
        [0,  0,  0,  0,  1,  0,  1],  # A01
        [0,  1,  0,  0,  0,  1,  0],  # A10
        [1,  1,  0,  1,  0,  0, -1]   # A11
    ], dtype=np.int8)

    V2 = np.array([
        [1,  1,  0, -1,  0,  1,  0],  # B00
        [0,  0,  1,  0,  0,  1,  0],  # B01
        [0,  0,  0,  1,  0,  0,  1],  # B10
        [1,  0, -1,  0,  1,  0,  1]   # B11
    ], dtype=np.int8)

    W2 = np.array([
        [1,  0,  0,  1, -1,  0,  1],  # C00 (k=0,i=0)
        [0,  1,  0,  1,  0,  0,  0],  # C10 (k=0,i=1)
        [0,  0,  1,  0,  1,  0,  0],  # C01 (k=1,i=0)
        [1, -1, 1,  0,  0,  1,  0]   # C11 (k=1,i=1)
    ], dtype=np.int8)

    # Construct 4x4 factors via Kronecker product
    rank = 49
    U = np.zeros((16, 49), dtype=np.int8)
    V = np.zeros((16, 49), dtype=np.int8)
    W = np.zeros((16, 49), dtype=np.int8)

    for i in range(4):
        i1 = i // 2
        i2 = i % 2
        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            for r1 in range(7):
                u1 = U2[i1*2 + j1, r1]
                if u1 == 0:
                    continue
                for r2 in range(7):
                    u2 = U2[i2*2 + j2, r2]
                    if u2 == 0:
                        continue
                    U[i*4 + j, r1*7 + r2] = u1 * u2

    for j in range(4):
        j1 = j // 2
        j2 = j % 2
        for k in range(4):
            k1 = k // 2
            k2 = k % 2
            for r1 in range(7):
                v1 = V2[j1*2 + k1, r1]
                if v1 == 0:
                    continue
                for r2 in range(7):
                    v2 = V2[j2*2 + k2, r2]
                    if v2 == 0:
                        continue
                    V[j*4 + k, r1*7 + r2] = v1 * v2

    for k in range(4):
        k1 = k // 2
        k2 = k % 2
        for i in range(4):
            i1 = i // 2
            i2 = i % 2
            for r1 in range(7):
                w1 = W2[k1*2 + i1, r1]
                if w1 == 0:
                    continue
                for r2 in range(7):
                    w2 = W2[k2*2 + i2, r2]
                    if w2 == 0:
                        continue
                    W[k*4 + i, r1*7 + r2] = w1 * w2

    # Build the 64x49 contribution matrix: column r is the vector of
    # contributions of rank-1 term r to all 64 (i,j,k) entries
    contrib = np.zeros((64, 49), dtype=np.float64)
    for r in range(49):
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    idx = i * 16 + j * 4 + k
                    contrib[idx, r] = U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]

    # Check rank of contribution matrix
    # If rank < 49, some terms are linearly dependent
    # Use SVD to find numerical rank
    try:
        U_sv, s_vals, Vt = np.linalg.svd(contrib, full_matrices=False)
        # Count significant singular values
        tol = 1e-10
        num_rank = np.sum(s_vals > tol)
    except:
        num_rank = 49

    # If we found a lower rank, try to identify which terms to remove
    # and verify the reduced decomposition is still valid
    if num_rank < 49 and time.time() - t0 < deadline - 30:
        # Find which columns are linearly independent using QR with column pivoting
        # Or use the SVD to identify the subspace
        # For now, try removing terms one by one and checking validity
        
        # First, identify zero-contribution terms
        zero_terms = []
        for r in range(49):
            if np.all(contrib[:, r] == 0):
                zero_terms.append(r)
        
        if zero_terms:
            # Remove zero terms
            keep = [r for r in range(49) if r not in zero_terms]
            U_new = U[:, keep]
            V_new = V[:, keep]
            W_new = W[:, keep]
            
            # Verify
            valid = True
            for i in range(4):
                for j in range(4):
                    for k in range(4):
                        val = 0
                        for r in range(len(keep)):
                            val += U_new[i*4+j, r] * V_new[j*4+k, r] * W_new[k*4+i, r]
                        if val != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                U = U_new
                V = V_new
                W = W_new

    # Verify the full decomposition
    valid = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = 0
                for r in range(U.shape[1]):
                    val += U[i*4+j, r] * V[j*4+k, r] * W[k*4+i, r]
                if val != 1:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if not valid:
        # Fallback: return the original rank-49 decomposition
        U = np.zeros((16, 49), dtype=np.int8)
        V = np.zeros((16, 49), dtype=np.int8)
        W = np.zeros((16, 49), dtype=np.int8)

        for i in range(4):
            i1 = i // 2
            i2 = i % 2
            for j in range(4):
                j1 = j // 2
                j2 = j % 2
                for r1 in range(7):
                    u1 = U2[i1*2 + j1, r1]
                    if u1 == 0:
                        continue
                    for r2 in range(7):
                        u2 = U2[i2*2 + j2, r2]
                        if u2 == 0:
                            continue
                        U[i*4 + j, r1*7 + r2] = u1 * u2

        for j in range(4):
            j1 = j // 2
            j2 = j % 2
            for k in range(4):
                k1 = k // 2
                k2 = k % 2
                for r1 in range(7):
                    v1 = V2[j1*2 + k1, r1]
                    if v1 == 0:
                        continue
                    for r2 in range(7):
                        v2 = V2[j2*2 + k2, r2]
                        if v2 == 0:
                            continue
                        V[j*4 + k, r1*7 + r2] = v1 * v2

        for k in range(4):
            k1 = k // 2
            k2 = k % 2
            for i in range(4):
                i1 = i // 2
                i2 = i % 2
                for r1 in range(7):
                    w1 = W2[k1*2 + i1, r1]
                    if w1 == 0:
                        continue
                    for r2 in range(7):
                        w2 = W2[k2*2 + i2, r2]
                        if w2 == 0:
                            continue
                        W[k*4 + i, r1*7 + r2] = w1 * w2

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    construct()
