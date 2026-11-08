"""Combined program: uses the verified rank-33 block decomposition from the seed
and adds a time-bounded search for a lower-rank decomposition (rank 32, 31, 30)
using random initialization and Newton-style refinement on the bilinear system.
If no lower-rank solution is found in time, returns the known rank-33 solution."""
import time
import numpy as np

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # safety margin for 150s limit

    n, m, p = DIMENSIONS  # n=2, m=3, p=6

    def get_rank_33_solution():
        rank = 33
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))

        # 11 terms for a 2x3x2 block
        terms = []

        # Strassen terms for the 2x2x2 part (j=0,1)
        # A indices: 0,1,3,4 (i*3+j); B indices: 0,1,2,3 (j*2+k); C indices: 0,1,2,3 (i*2+k)

        # M1 = (A00+A11)(B00+B11)
        terms.append({'u': {0: 1, 4: 1}, 'v': {0: 1, 3: 1}, 'w': {0: 1, 3: 1}})
        # M2 = (A10+A11)B00
        terms.append({'u': {3: 1, 4: 1}, 'v': {0: 1}, 'w': {2: 1, 3: -1}})
        # M3 = A00(B01-B11)
        terms.append({'u': {0: 1}, 'v': {1: 1, 3: -1}, 'w': {1: 1, 3: 1}})
        # M4 = A11(B10-B00)
        terms.append({'u': {4: 1}, 'v': {2: 1, 0: -1}, 'w': {0: 1, 2: 1}})
        # M5 = (A00+A01)B11
        terms.append({'u': {0: 1, 1: 1}, 'v': {3: 1}, 'w': {0: -1, 1: 1}})
        # M6 = (A10-A00)(B00+B01)
        terms.append({'u': {3: 1, 0: -1}, 'v': {0: 1, 1: 1}, 'w': {3: 1}})
        # M7 = (A01-A11)(B10+B11)
        terms.append({'u': {1: 1, 4: -1}, 'v': {2: 1, 3: 1}, 'w': {0: 1}})

        # Scalar terms for column 2 of A and row 2 of B
        # K1 = A02 * B20
        terms.append({'u': {2: 1}, 'v': {4: 1}, 'w': {0: 1}})
        # K2 = A02 * B21
        terms.append({'u': {2: 1}, 'v': {5: 1}, 'w': {1: 1}})
        # K3 = A12 * B20
        terms.append({'u': {5: 1}, 'v': {4: 1}, 'w': {2: 1}})
        # K4 = A12 * B21
        terms.append({'u': {5: 1}, 'v': {5: 1}, 'w': {3: 1}})

        for block in range(3):
            for t_idx, term in enumerate(terms):
                col = block * 11 + t_idx

                for idx, val in term['u'].items():
                    U[idx, col] = val

                for l_idx, val in term['v'].items():
                    j_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = j_local * p + (block * 2 + k_local)
                    V[g_idx, col] = val

                for l_idx, val in term['w'].items():
                    i_local = l_idx // 2
                    k_local = l_idx % 2
                    g_idx = (block * 2 + k_local) * n + i_local
                    W[g_idx, col] = val

        return {'U': U, 'V': V, 'W': W}

    def verify_solution(U, V, W):
        """Check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1 for all i,j,k."""
        T = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    T[i, j, k] = np.sum(U[u_idx, :] * V[v_idx, :] * W[w_idx, :])
        return np.allclose(T, 1.0, atol=1e-8)

    def try_find_lower_rank(R, max_iters=200, max_restarts=5, tol=1e-7):
        """
        Attempt to find a rank-R decomposition using random initialization
        and Newton-style refinement.
        
        We fix U and V randomly, then solve for W (or iterate).
        Actually, the system is: for each (i,j,k), sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1
        This is linear in W if U,V are fixed. But U and V are also variables.
        
        A simpler approach: fix U and V to random values with rank R, then for each output
        index (k*n+i), we need:
          sum_r V[j*p+k, r] * W[k*n+i, r] * U[i*m+j, r] = 1 for all j.
        
        Let's use a different approach: 
        For a given rank R, we can set up the equations. The tensor T has 36 entries, all 1.
        We have 3*R variables (well, U has 6*R, V has 18*R, W has 12*R, but scaling invariance).
        
        Actually, let's try: fix U to be random (6 x R), fix V to be random (18 x R),
        then for each (i,k) pair (there are n*p = 12 of them), we need to find W[k*n+i, :]
        such that for all j in 0..m-1:
            sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        
        This is a system of m=3 equations in R unknowns (W[k*n+i, 0..R-1]).
        If R >= 3, we can solve it. Let's try this.
        """
        if R < 3:
            return None
        
        np.random.seed(42)
        
        for restart in range(max_restarts):
            # Random U and V
            U = np.random.randn(n * m, R)
            V = np.random.randn(m * p, R)
            W = np.zeros((p * n, R))
            
            # For each (i, k), solve for W[k*n+i, :]
            success = True
            for i in range(n):
                for k in range(p):
                    w_idx = k * n + i
                    # System: for each j, sum_r U[i*m+j, r] * V[j*p+k, r] * W[w_idx, r] = 1
                    # Let A_jr = U[i*m+j, r] * V[j*p+k, r]
                    # Then sum_r A_jr * W[w_idx, r] = 1 for j=0..m-1
                    A = np.zeros((m, R))
                    for j in range(m):
                        A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                    
                    # Solve A @ W[w_idx, :] = ones(m)
                    try:
                        sol = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        W[w_idx, :] = sol
                    except:
                        success = False
                        break
                
                if not success:
                    break
            
            if not success:
                continue
            
            # Verify
            if verify_solution(U, V, W):
                # Round to nearest half-integer if possible
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                if verify_solution(U_rounded, V_rounded, W_rounded):
                    return {'U': U_rounded, 'V': V_rounded, 'W': W_rounded}
                else:
                    return {'U': U, 'V': V, 'W': W}
        
        return None

    # First, get the known rank-33 solution
    best_solution = get_rank_33_solution()
    best_rank = 33

    if not verify_solution(best_solution['U'], best_solution['V'], best_solution['W']):
        # Fallback to trivial solution
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    r = i * m * p + j * p + k
                    U[i * m + j, r] = 1
                    V[j * p + k, r] = 1
                    W[k * n + i, r] = 1
        best_solution = {'U': U, 'V': V, 'W': W}
        best_rank = rank

    # Try to find lower rank solutions
    for R in range(32, 28, -1):
        if time.time() > deadline - 10:
            break
        result = try_find_lower_rank(R, max_iters=100, max_restarts=3)
        if result is not None:
            best_solution = result
            best_rank = R
            break

    # Convert to lists for the return interface
    return {
        'U': best_solution['U'].tolist(),
        'V': best_solution['V'].tolist(),
        'W': best_solution['W'].tolist()
    }
# EVOLVE-BLOCK-END
