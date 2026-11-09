import numpy as np
import time

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n

    # Build the multiplication tensor T
    # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for some i,j,k
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1.0

    # Schoolbook baseline
    rank_sb = n * m * p  # 54
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1
                V_sb[j*p + k, col] = 1
                W_sb[k*n + i, col] = 1
                col += 1

    start_time = time.time()
    time_limit = 140  # Leave 10s margin for safety

    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()

    # Strategy: Use Kronecker product structure to decompose 3x9 into blocks
    # The 3x9 matrix can be viewed as 3x(3*3), so we can use 3x3 multiplication structure
    # For 3x3 matrix multiplication, we know rank 5 (Strassen-like) is possible
    # This gives us a candidate rank of 2*3*5 = 30 for the full problem

    # First, try to find a rank-5 decomposition for 3x3 matrix multiplication
    # The 3x3 multiplication tensor has entries T[i,j,k] = 1 if output (i,k) gets input (i,j)*(j,k)
    # Let's build the 3x3 tensor
    T33 = np.zeros((9, 9, 9))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                a = i * 3 + j
                b = j * 3 + k
                c = k * 3 + i
                T33[a, b, c] = 1.0

    # Try to find rank-5 decomposition for 3x3
    found_33 = False
    U33 = None
    V33 = None
    W33 = None

    # Known rank-5 decomposition for 3x3 (Strassen-like)
    # This is a known result; let's construct it
    # The rank-5 decomposition uses the following structure:
    # We need to find U33, V33, W33 of shape (9,5) such that T33 = sum_r U33[:,r] outer V33[:,r] outer W33[:,r]

    # Let's try to find this using alternating least squares
    for rank33 in [5, 6, 7, 8]:
        if time.time() - start_time > time_limit - 30:
            break

        found_33 = False
        max_restarts_33 = 20
        for restart in range(max_restarts_33):
            if time.time() - start_time > time_limit - 30:
                break

            np.random.seed(9999 + restart * 777 + rank33)
            U33_try = np.random.randn(9, rank33) * 0.5
            V33_try = np.random.randn(9, rank33) * 0.5
            W33_try = np.random.randn(9, rank33) * 0.5

            max_iter_33 = 300
            prev_error_33 = float('inf')
            for iteration in range(max_iter_33):
                if time.time() - start_time > time_limit - 30:
                    break

                # Solve for U33
                VW33 = np.einsum('br,cr->bcr', V33_try, W33_try)
                VW33_flat = VW33.transpose(2, 0, 1).reshape(rank33, 81)
                T33_flat = T33.reshape(9, 81)
                try:
                    U33_new, _, _, _ = np.linalg.lstsq(VW33_flat.T, T33_flat.T, rcond=None)
                    U33_try = U33_new.T
                except:
                    break

                # Solve for V33
                UW33 = np.einsum('ar,cr->acr', U33_try, W33_try)
                UW33_flat = UW33.transpose(2, 0, 1).reshape(rank33, 81)
                T33_flat_b = T33.transpose(1, 0, 2).reshape(9, 81)
                try:
                    V33_new, _, _, _ = np.linalg.lstsq(UW33_flat.T, T33_flat_b.T, rcond=None)
                    V33_try = V33_new.T
                except:
                    break

                # Solve for W33
                UV33 = np.einsum('ar,br->abr', U33_try, V33_try)
                UV33_flat = UV33.transpose(2, 0, 1).reshape(rank33, 81)
                T33_flat_c = T33.transpose(2, 0, 1).reshape(9, 81)
                try:
                    W33_new, _, _, _ = np.linalg.lstsq(UV33_flat.T, T33_flat_c.T, rcond=None)
                    W33_try = W33_new.T
                except:
                    break

                if iteration % 10 == 0:
                    T33_recon = np.einsum('ar,br,cr->abc', U33_try, V33_try, W33_try)
                    error_33 = np.linalg.norm(T33_recon - T33)
                    if error_33 < 1e-8:
                        found_33 = True
                        U33 = U33_try.copy()
                        V33 = V33_try.copy()
                        W33 = W33_try.copy()
                        break
                    if error_33 > prev_error_33 * 1.1 and iteration > 50:
                        break
                    prev_error_33 = error_33

            if found_33:
                break

        if found_33:
            break

    # Now use the 3x3 decomposition to build the full 2x3x9 decomposition
    # The 3x9 multiplication can be seen as 3 blocks of 3x3
    # For each output row i (0,1), we have 3 blocks corresponding to k=0..2
    # Each block is a 3x3 multiplication

    # If we found a rank-r33 decomposition for 3x3, we can use it
    # The full rank would be n * r33 = 2 * r33

    if found_33 and U33 is not None:
        r33 = U33.shape[1]
        full_rank = n * r33

        if full_rank < best_rank:
            # Build the full decomposition using Kronecker-like structure
            U_full = np.zeros((nm, full_rank))
            V_full = np.zeros((mp, full_rank))
            W_full = np.zeros((pn, full_rank))

            # For each i in {0,1}, we have a 3x3 multiplication for the block
            # The input U indices: i*m+j where i in {0,1}, j in {0,1,2}
            # The V indices: j*p+k where j in {0,1,2}, k in {0,1,2,3,4,5,6,7,8}
            # The W indices: k*n+i where k in {0,1,2,3,4,5,6,7,8}, i in {0,1}

            # For the 3x3 block at position (i, k_block) where k_block in {0,1,2}
            # We multiply the 3x3 matrix A[i,:] (row i of A) with the 3x3 matrix B[:, 3*k_block:3*k_block+3]
            # to get output C[i, 3*k_block:3*k_block+3]

            # Actually, let me reconsider the structure
            # A is 2x3, B is 3x9, C is 2x9
            # C[i,k] = sum_j A[i,j] * B[j,k]
            # For fixed i, C[i,:] = A[i,:] * B (1x3 times 3x9 = 1x9)
            # This is a row vector times a matrix

            # We can decompose B into three 3x3 blocks: B[:, 0:3], B[:, 3:6], B[:, 6:9]
            # Then C[i, 0:3] = A[i,:] * B[:, 0:3] (this is 1x3 times 3x3)
            # C[i, 3:6] = A[i,:] * B[:, 3:6]
            # C[i, 6:9] = A[i,:] * B[:, 6:9]

            # Each of these is a 1x3 times 3x3 multiplication
            # We can use the 3x3 decomposition for this

            # The 3x3 tensor T33 has indices:
            # a = i3*3+j3 for the row of the 3x3 input (but here we have 1x3, so i3=0)
            # b = j3*3+k3 for the column of the 3x3 input
            # c = k3*3+i3 for the output

            # For 1x3 times 3x3:
            # Input U: A[i, j3] where j3 in {0,1,2}
            # Input V: B[j3, k3] where j3 in {0,1,2}, k3 in {0,1,2}
            # Output W: C[i, k3] where k3 in {0,1,2}

            # So we need to extract from T33 the part where i3=0
            # T33[a, b, c] where a = 0*3+j3 = j3, b = j3*3+k3, c = k3*3+0 = k3

            # Let's build the 1x3x3 tensor
            T133 = np.zeros((3, 9, 3))
            for j3 in range(3):
                for k3 in range(3):
                    a = j3  # input index (j3)
                    b = j3 * 3 + k3  # V index
                    c = k3  # output index
                    T133[a, b, c] = 1.0

            # Now find rank-r33 decomposition for T133
            # We can use the same alternating least squares
            found_133 = False
            U133 = None
            V133 = None
            W133 = None

            for restart in range(20):
                if time.time() - start_time > time_limit - 20:
                    break

                np.random.seed(1111 + restart * 555 + r33)
                U133_try = np.random.randn(3, r33) * 0.5
                V133_try = np.random.randn(9, r33) * 0.5
                W133_try = np.random.randn(3, r33) * 0.5

                max_iter_133 = 200
                prev_error_133 = float('inf')
                for iteration in range(max_iter_133):
                    if time.time() - start_time > time_limit - 20:
                        break

                    # Solve for U133
                    VW133 = np.einsum('br,cr->bcr', V133_try, W133_try)
                    VW133_flat = VW133.transpose(2, 0, 1).reshape(r33, 27)
                    T133_flat = T133.reshape(3, 27)
                    try:
                        U133_new, _, _, _ = np.linalg.lstsq(VW133_flat.T, T133_flat.T, rcond=None)
                        U133_try = U133_new.T
                    except:
                        break

                    # Solve for V133
                    UW133 = np.einsum('ar,cr->acr', U133_try, W133_try)
                    UW133_flat = UW133.transpose(2, 0, 1).reshape(r33, 9)
                    T133_flat_b = T133.transpose(1, 0, 2).reshape(9, 9)
                    try:
                        V133_new, _, _, _ = np.linalg.lstsq(UW133_flat.T, T133_flat_b.T, rcond=None)
                        V133_try = V133_new.T
                    except:
                        break

                    # Solve for W133
                    UV133 = np.einsum('ar,br->abr', U133_try, V133_try)
                    UV133_flat = UV133.transpose(2, 0, 1).reshape(r33, 27)
                    T133_flat_c = T133.transpose(2, 0, 1).reshape(3, 27)
                    try:
                        W133_new, _, _, _ = np.linalg.lstsq(UV133_flat.T, T133_flat_c.T, rcond=None)
                        W133_try = W133_new.T
                    except:
                        break

                    if iteration % 10 == 0:
                        T133_recon = np.einsum('ar,br,cr->abc', U133_try, V133_try, W133_try)
                        error_133 = np.linalg.norm(T133_recon - T133)
                        if error_133 < 1e-8:
                            found_133 = True
                            U133 = U133_try.copy()
                            V133 = V133_try.copy()
                            W133 = W133_try.copy()
                            break
                        if error_133 > prev_error_133 * 1.1 and iteration > 30:
                            break
                        prev_error_133 = error_133

                if found_133:
                    break

            if found_133 and U133 is not None:
                # Build the full decomposition
                # For each i in {0,1} and each block b in {0,1,2}
                # We use the 1x3x3 decomposition
                # U_full[i*3+j3, i*r33 + b*r33 + r] = U133[j3, r]
                # V_full[j3*9 + b*3 + k3, i*r33 + b*r33 + r] = V133[j3*3+k3, r]
                # W_full[(b*3+k3)*2 + i, i*r33 + b*r33 + r] = W133[k3, r]

                for i in range(n):
                    for b in range(3):
                        for r in range(r33):
                            for j3 in range(3):
                                U_full[i * m + j3, i * r33 + b * r33 + r] = U133[j3, r]

                            for j3 in range(3):
                                for k3 in range(3):
                                    V_full[j3 * p + b * 3 + k3, i * r33 + b * r33 + r] = V133[j3 * 3 + k3, r]

                            for k3 in range(3):
                                W_full[(b * 3 + k3) * n + i, i * r33 + b * r33 + r] = W133[k3, r]

                # Verify
                T_recon = np.einsum('ar,br,cr->abc', U_full, V_full, W_full)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_rank = full_rank
                    best_U = U_full.copy()
                    best_V = V_full.copy()
                    best_W = W_full.copy()
                    # We found a good solution, return it
                    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # If the Kronecker approach didn't work, fall back to the original alternating least squares
    # For [2,3,9], try ranks from 36 down to 30
    ranks_to_try = list(range(36, 29, -1))

    for rank in ranks_to_try:
        if time.time() - start_time > time_limit:
            break

        found = False
        max_restarts = 15
        for restart in range(max_restarts):
            if time.time() - start_time > time_limit:
                break

            np.random.seed(12345 + restart * 1000 + rank)

            if restart == 0:
                U = np.zeros((nm, rank))
                V = np.zeros((mp, rank))
                W = np.zeros((pn, rank))
                for r in range(rank):
                    i = r % n
                    j = (r // n) % m
                    k = (r // (n*m)) % p
                    if r < n*m*p:
                        U[i*m + j, r] = 1.0
                        V[j*p + k, r] = 1.0
                        W[k*n + i, r] = 1.0
            else:
                scale = 0.5 + restart * 0.1
                U = np.random.randn(nm, rank) * scale
                V = np.random.randn(mp, rank) * scale
                W = np.random.randn(pn, rank) * scale

            max_iter = 500
            prev_error = float('inf')
            for iteration in range(max_iter):
                if time.time() - start_time > time_limit:
                    break

                VW = np.einsum('br,cr->bcr', V, W)
                VW_flat = VW.transpose(2, 0, 1).reshape(rank, mp * pn)
                T_flat_a = T.reshape(nm, mp * pn)

                try:
                    U_new, _, _, _ = np.linalg.lstsq(VW_flat.T, T_flat_a.T, rcond=None)
                    U = U_new.T
                except:
                    break

                UW = np.einsum('ar,cr->acr', U, W)
                UW_flat = UW.transpose(2, 0, 1).reshape(rank, nm * pn)
                T_flat_b = T.transpose(1, 0, 2).reshape(mp, nm * pn)

                try:
                    V_new, _, _, _ = np.linalg.lstsq(UW_flat.T, T_flat_b.T, rcond=None)
                    V = V_new.T
                except:
                    break

                UV = np.einsum('ar,br->abr', U, V)
                UV_flat = UV.transpose(2, 0, 1).reshape(rank, nm * mp)
                T_flat_c = T.transpose(2, 0, 1).reshape(pn, nm * mp)

                try:
                    W_new, _, _, _ = np.linalg.lstsq(UV_flat.T, T_flat_c.T, rcond=None)
                    W = W_new.T
                except:
                    break

                if iteration % 10 == 0:
                    T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                    error = np.linalg.norm(T_recon - T)
                    if error < 1e-8:
                        found = True
                        break
                    if error > prev_error * 1.1 and iteration > 50:
                        break
                    prev_error = error

            if found:
                T_recon = np.einsum('ar,br,cr->abc', U, V, W)
                if np.allclose(T_recon, T, atol=1e-10):
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                    break

        if found:
            continue

    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    else:
        return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
