import numpy as np
import time

DIMENSIONS = [3, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = DIMENSIONS  # 3, 4, 8
    nm, mp, pn = n * m, m * p, p * n  # 12, 32, 24

    # Construct the matrix multiplication tensor T of shape (nm, mp, pn)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((nm, mp, pn), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1

    def validate(U, V, W):
        """Check if U, V, W correctly decompose T."""
        rank = U.shape[1]
        T_recon = np.zeros((nm, mp, pn), dtype=np.int64)
        for r in range(rank):
            T_recon += np.einsum('i,j,k->ijk', U[:, r], V[:, r], W[:, r])
        return np.array_equal(T_recon, T)

    # Schoolbook baseline
    rank_school = n * m * p
    U_school = np.zeros((nm, rank_school), dtype=np.int64)
    V_school = np.zeros((mp, rank_school), dtype=np.int64)
    W_school = np.zeros((pn, rank_school), dtype=np.int64)

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1

    # Correct Strassen 2x2x2 decomposition
    # U, V, W are dicts mapping (local_index) to coeff
    def strassen_2x2x2():
        terms = []
        # r=1: (a11-a22)(b21+b22) -> c12, c22
        # U: (0,0)=1, (1,1)=-1
        # V: (1,0)=1, (1,1)=1
        # W: (1,0)=1, (1,1)=1
        terms.append(({(0,0):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(1,0):1, (1,1):1}))
        
        # r=2: (a11+a12)b22 -> c21
        # U: (0,0)=1, (0,1)=1
        # V: (1,1)=1
        # W: (1,1)=1
        terms.append(({(0,0):1, (0,1):1}, {(1,1):1}, {(1,1):1}))
        
        # r=3: a11(b11-b21) -> c11, c21
        # U: (0,0)=1
        # V: (0,0)=1, (1,0)=-1
        # W: (0,0)=1, (1,0)=1
        terms.append(({(0,0):1}, {(0,0):1, (1,0):-1}, {(0,0):1, (1,0):1}))
        
        # r=4: a22(b12-b11) -> c11, c22
        # U: (1,1)=1
        # V: (0,1)=1, (0,0)=-1
        # W: (1,0)=1, (1,1)=1
        terms.append(({(1,1):1}, {(0,1):1, (0,0):-1}, {(1,0):1, (1,1):1}))
        
        # r=5: (a11+a22)(b11+b22) -> c11, c12, c21, c22
        # U: (0,0)=1, (1,1)=1
        # V: (0,0)=1, (1,1)=1
        # W: (0,0)=1, (1,1)=1
        terms.append(({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}))
        
        # r=6: (a12-a22)(b11+b21) -> c21, c22
        # U: (0,1)=1, (1,1)=-1
        # V: (0,0)=1, (1,0)=1
        # W: (1,0)=1, (1,1)=1
        terms.append(({(0,1):1, (1,1):-1}, {(0,0):1, (1,0):1}, {(1,0):1, (1,1):1}))
        
        # r=7: (a11-a12)(b12+b22) -> c11, c12
        # U: (0,0)=1, (0,1)=-1
        # V: (0,1)=1, (1,1)=1
        # W: (0,0)=1, (1,0)=1
        terms.append(({(0,0):1, (0,1):-1}, {(0,1):1, (1,1):1}, {(0,0):1, (1,0):1}))
        
        return terms

    # Verify Strassen decomposition locally
    T_check = np.zeros((4, 4, 4), dtype=np.int64)
    for u_dict, v_dict, w_dict in strassen_2x2x2():
        for (i, j), u_val in u_dict.items():
            for (j2, k), v_val in v_dict.items():
                if j == j2:
                    for (k2, i2), w_val in w_dict.items():
                        if k == k2:
                            T_check[i, j, k] += u_val * v_val * w_val
    # Expected: T_check[i,j,k] == 1 for all i,j,k in 0..1
    # Note: T_check is 4x4x4, but we only care about indices 0,1
    valid_strassen = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                if T_check[i, j, k] != 1:
                    valid_strassen = False
    if not valid_strassen:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    # Build the full decomposition
    # Strategy:
    # Split m=4 into m1=2, m2=2.
    # For each m_part, split n=3 into n1=2, n2=1.
    # For each n_part, split p=8 into p1..p4 (each 2).
    
    U_full = np.zeros((nm, 200), dtype=np.int64)
    V_full = np.zeros((mp, 200), dtype=np.int64)
    W_full = np.zeros((pn, 200), dtype=np.int64)
    col = 0

    for m0 in [0, 2]:
        # A is 3x2 (local m), B is 2x8 (local m)
        # Split n into 2 and 1
        for n0, n_size in [(0, 2), (2, 1)]:
            # A_part is n_size x 2
            # B_part is 2 x 8
            # Split p into 4 blocks of 2
            for p0 in range(0, 8, 2):
                # B_block is 2 x 2
                if n_size == 2:
                    # 2x2x2 Strassen
                    terms = strassen_2x2x2()
                    for u_dict, v_dict, w_dict in terms:
                        for (il, jl), u_val in u_dict.items():
                            i_global = n0 + il
                            j_global = m0 + jl
                            U_full[i_global*m + j_global, col] += u_val
                        for (jl, kl), v_val in v_dict.items():
                            j_global = m0 + jl
                            k_global = p0 + kl
                            V_full[j_global*p + k_global, col] += v_val
                        for (kl, il), w_val in w_dict.items():
                            k_global = p0 + kl
                            i_global = n0 + il
                            W_full[k_global*n + i_global, col] += w_val
                        col += 1
                else:
                    # 1x2x2 Schoolbook (rank 4)
                    # A is 1x2, B is 2x2
                    # i=0, j in {0,1}, k in {0,1}
                    for jl in range(2):
                        for kl in range(2):
                            i_global = n0
                            j_global = m0 + jl
                            k_global = p0 + kl
                            U_full[i_global*m + j_global, col] = 1
                            V_full[j_global*p + k_global, col] = 1
                            W_full[k_global*n + i_global, col] = 1
                            col += 1

    U_trim = U_full[:, :col]
    V_trim = V_full[:, :col]
    W_trim = W_full[:, :col]

    if validate(U_trim, V_trim, W_trim):
        return {
            "U": U_trim.tolist(),
            "V": V_trim.tolist(),
            "W": W_trim.tolist()
        }
    else:
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }
