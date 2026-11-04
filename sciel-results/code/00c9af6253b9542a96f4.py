import time
import random
import numpy as np

DIMENSIONS = [4, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 115

    def verify(rank, U, V, W):
        """Check T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c] == 1 for all i,j,k."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i*m + j][c] * V[j*p + k][c] * W[k*n + i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True

    def make_strassen_4x4x5():
        """Construct rank-65 decomposition for 4x4x5 using Strassen at block level."""
        rank = 65
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]

        # Strassen 2x2x2 scalar decomposition
        a_scalar_defs = [
            [(0, 0, 1), (1, 1, 1)],    # s=0
            [(1, 0, 1), (1, 1, 1)],    # s=1
            [(0, 0, 1)],                # s=2
            [(1, 1, 1)],                # s=3
            [(0, 0, 1), (0, 1, 1)],    # s=4
            [(1, 0, 1), (0, 0, -1)],   # s=5
            [(0, 1, 1), (1, 1, -1)]    # s=6
        ]
        
        b_scalar_defs = [
            [(0, 0, 1), (1, 1, 1)],    # s=0
            [(0, 0, 1)],                # s=1
            [(0, 1, 1), (1, 1, -1)],   # s=2
            [(1, 0, 1), (0, 0, -1)],   # s=3
            [(1, 1, 1)],                # s=4
            [(0, 0, 1), (0, 1, 1)],    # s=5
            [(1, 0, 1), (1, 1, 1)]     # s=6
        ]
        
        c_contrib_defs = [
            [(0, 0, 1), (1, 1, 1)],    # s=0
            [(1, 0, 1), (1, 1, -1)],   # s=1
            [(0, 1, 1), (1, 1, 1)],    # s=2
            [(0, 0, 1), (1, 0, 1)],    # s=3
            [(0, 0, -1), (0, 1, 1)],   # s=4
            [(1, 1, 1)],                # s=5
            [(0, 0, 1)]                 # s=6
        ]
        
        a_block_defs = [
            [(0, 0, 1), (1, 1, 1)],    # t=0
            [(1, 0, 1), (1, 1, 1)],    # t=1
            [(0, 0, 1)],                # t=2
            [(1, 1, 1)],                # t=3
            [(0, 0, 1), (0, 1, 1)],    # t=4
            [(1, 0, 1), (0, 0, -1)],   # t=5
            [(0, 1, 1), (1, 1, -1)]    # t=6
        ]
        
        b_block_defs = [
            [(0, 0, 1), (1, 1, 1)],    # t=0
            [(0, 0, 1)],                # t=1
            [(0, 1, 1), (1, 1, -1)],   # t=2
            [(1, 0, 1), (0, 0, -1)],   # t=3
            [(1, 1, 1)],                # t=4
            [(0, 0, 1), (0, 1, 1)],    # t=5
            [(1, 0, 1), (1, 1, 1)]     # t=6
        ]
        
        c_block_defs = [
            [(0, 0, 1), (1, 1, 1)],    # t=0
            [(1, 0, 1), (1, 1, -1)],   # t=1
            [(0, 1, 1), (1, 1, 1)],    # t=2
            [(0, 0, 1), (1, 0, 1)],    # t=3
            [(0, 0, -1), (0, 1, 1)],   # t=4
            [(1, 1, 1)],                # t=5
            [(0, 0, 1)]                 # t=6
        ]

        for t in range(7):
            for s in range(7):
                c_idx = t * 7 + s
                for (abr, abc, ca) in a_block_defs[t]:
                    for (ar, ac, cs) in a_scalar_defs[s]:
                        i = 2 * abr + ar
                        j = 2 * abc + ac
                        U[i * m + j][c_idx] += ca * cs
                for (bbr, bbc, cb) in b_block_defs[t]:
                    for (br, bc, cs) in b_scalar_defs[s]:
                        j = 2 * bbr + br
                        k = 2 * bbc + bc
                        V[j * p + k][c_idx] += cb * cs
                for (cbr, cbc, cc) in c_block_defs[t]:
                    for (cr, cc2, cs2) in c_contrib_defs[s]:
                        k = 2 * cbr + cr
                        i = 2 * cbc + cc2
                        W[k * n + i][c_idx] += cc * cs2

        for i in range(n):
            for j in range(m):
                c_idx = 49 + i * m + j
                U[i * m + j][c_idx] = 1
                V[j * p + 4][c_idx] = 1
                W[4 * n + i][c_idx] = 1
        
        return {"U": U, "V": V, "W": W}

    def try_rank_64_search():
        """Randomized local search for rank 64."""
        rank = 64
        U = np.random.randint(-2, 3, (n*m, rank))
        V = np.random.randint(-2, 3, (m*p, rank))
        W = np.random.randint(-2, 3, (p*n, rank))
        
        # Compute initial error
        def compute_error(U, V, W):
            err = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for c in range(rank):
                            s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        err += (s - 1.0) ** 2
            return err

        current_err = compute_error(U, V, W)
        
        # Local search
        steps = 0
        while time.time() < deadline - 5:
            # Pick random variable
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                idx = random.randint(0, n*m-1)
                c = random.randint(0, rank-1)
                old_val = U[idx, c]
                new_val = old_val + random.choice([-1, 1])
                U[idx, c] = new_val
            elif mat == 'V':
                idx = random.randint(0, m*p-1)
                c = random.randint(0, rank-1)
                old_val = V[idx, c]
                new_val = old_val + random.choice([-1, 1])
                V[idx, c] = new_val
            else:
                idx = random.randint(0, p*n-1)
                c = random.randint(0, rank-1)
                old_val = W[idx, c]
                new_val = old_val + random.choice([-1, 1])
                W[idx, c] = new_val
            
            new_err = compute_error(U, V, W)
            
            if new_err < current_err:
                current_err = new_err
                if current_err < 1e-6:
                    # Convert to lists
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            else:
                # Revert
                if mat == 'U':
                    U[idx, c] = old_val
                elif mat == 'V':
                    V[idx, c] = old_val
                else:
                    W[idx, c] = old_val
            
            steps += 1
            if steps % 1000 == 0:
                # Restart if stuck?
                if steps > 5000 and current_err > 100:
                    U = np.random.randint(-2, 3, (n*m, rank))
                    V = np.random.randint(-2, 3, (m*p, rank))
                    W = np.random.randint(-2, 3, (p*n, rank))
                    current_err = compute_error(U, V, W)
                    steps = 0

        return None

    best = make_strassen_4x4x5()
    best_rank = 65

    # Try to find rank 64
    if time.time() < deadline - 10:
        res = try_rank_64_search()
        if res:
            if verify(64, res["U"], res["V"], res["W"]):
                best = res
                best_rank = 64

    # Verify best
    if not verify(best_rank, best["U"], best["V"], best["W"]):
        # Fallback to schoolbook
        r = n * m * p
        U = [[0]*r for _ in range(n*m)]
        V = [[0]*r for _ in range(m*p)]
        W = [[0]*r for _ in range(p*n)]
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][c] = 1
                    V[j*p+k][c] = 1
                    W[k*n+i][c] = 1
                    c += 1
        best = {"U": U, "V": V, "W": W}
        best_rank = r

    return best
