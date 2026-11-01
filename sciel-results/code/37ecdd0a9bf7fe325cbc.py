import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 4, 6

    def build_strassen_42():
        rank = 42
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]

        strassen_terms = [
            {'A': [(0,0,1), (1,1,1)], 'B': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)]},
            {'A': [(1,0,1), (1,1,1)], 'B': [(0,0,1)], 'W': [(1,0,1), (1,1,-1)]},
            {'A': [(0,0,1)], 'B': [(0,1,1), (1,1,-1)], 'W': [(0,1,1), (1,1,1)]},
            {'A': [(1,1,1)], 'B': [(1,0,1), (0,0,-1)], 'W': [(0,0,1), (1,0,1)]},
            {'A': [(0,0,1), (0,1,1)], 'B': [(1,1,1)], 'W': [(0,0,-1), (0,1,1)]},
            {'A': [(1,0,1), (0,0,-1)], 'B': [(0,0,1), (0,1,1)], 'W': [(1,1,1)]},
            {'A': [(0,1,1), (1,1,-1)], 'B': [(1,0,1), (1,1,1)], 'W': [(0,0,1)]},
        ]

        col = 0
        for j_block in range(2):
            for k_block in range(3):
                for term in strassen_terms:
                    r = col
                    for (ar, ac, sign) in term['A']:
                        u_idx = ar * m + (j_block * 2 + ac)
                        U[u_idx][r] += sign
                    for (br, bc, sign) in term['B']:
                        v_idx = (j_block * 2 + br) * p + (k_block * 2 + bc)
                        V[v_idx][r] += sign
                    for (or_, oc, sign) in term['W']:
                        w_idx = (k_block * 2 + oc) * n + or_
                        W[w_idx][r] += sign
                    col += 1

        assert col == rank, f"Expected {rank} columns, got {col}"
        return {"U": U, "V": V, "W": W}

    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True

    # Build the rank-42 fallback
    fallback = build_strassen_42()

    if time.time() > deadline - 5:
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}

    # Try to find a lower-rank solution using numerical optimization
    # The tensor rank for 2x4x6 is at most 42 (Strassen) and at least 24 (naive)
    # Let's try to find a rank-24 or rank-28 solution
    
    # For rank r, we have r unknowns in each of U, V, W
    # The equations are: for all (i,j,k), sum_t U[i*m+j,t] * V[j*p+k,t] * W[k*n+i,t] = 1
    # That's n*m*p = 48 equations.
    # Unknowns: r * (n*m + m*p + p*n) = r * (8 + 24 + 12) = 44r
    # For r = 24: 48 equations, 1056 unknowns. Very underdetermined.
    
    # We'll use scipy.optimize.least_squares to minimize the error
    try:
        from scipy.optimize import least_squares
    except ImportError:
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}

    def flatten(U, V, W):
        return np.concatenate([np.array(U).flatten(), np.array(V).flatten(), np.array(W).flatten()])

    def unflatten(x):
        u_size = n * m
        v_size = m * p
        w_size = p * n
        U = x[:u_size].reshape(n, m, -1)
        V = x[u_size:u_size+v_size].reshape(m, p, -1)
        W = x[u_size+v_size:].reshape(p, n, -1)
        return U, V, W

    def objective(x, r):
        U, V, W = unflatten(x)
        errors = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i, j, t] * V[j, k, t] * W[k, i, t]
                    errors.append(s - 1)
        return errors

    # Try rank 24 first
    best_result = None
    for target_rank in [24, 28, 32, 36]:
        if time.time() > deadline - 10:
            break
            
        r = target_rank
        # Initialize with small random values
        np.random.seed(42)
        x0 = np.random.randn(r * (n*m + m*p + p*n)) * 0.1
        
        try:
            result = least_squares(objective, x0, args=(r,), method='lm', max_nfev=1000)
            if result.cost < 1e-10:
                U, V, W = unflatten(result.x)
                # Round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                U_list = U_rounded.tolist()
                V_list = V_rounded.tolist()
                W_list = W_rounded.tolist()
                
                if verify(U_list, V_list, W_list):
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    break
        except Exception:
            continue

    if best_result is not None:
        return best_result
    else:
        return {"U": fallback["U"], "V": fallback["V"], "W": fallback["W"]}

if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")
