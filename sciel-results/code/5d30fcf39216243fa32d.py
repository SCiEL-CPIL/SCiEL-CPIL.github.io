import numpy as np
import time

DIMENSIONS = [2, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100  # Safety margin before 150s kill

    # Best known decomposition: rank 22 (Strassen blocks + outer product)
    # We first construct this as a fallback, then try to find something better.
    
    def build_rank_22():
        rank = 22
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        # A indices: 00:0, 01:1, 10:3, 11:4
        # B indices: 00:0, 01:1, 10:4, 11:5
        # C indices (W): 00:0, 01:2, 10:1, 11:3
        
        # M1: (A00+A11)(B00+B11) -> C00+C11
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        # M2: (A10+A11)B00 -> C10-C11
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        # M3: A00(B01-B11) -> C01+C11
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        # M4: A11(B10-B00) -> C00+C10
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        # M5: (A00+A01)B11 -> -C00+C01
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        # M6: (A10-A00)(B00+B01) -> C11
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        # M7: (A01-A11)(B10+B11) -> C00
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        # B indices: 02:2, 03:3, 12:6, 13:7
        # C indices (W): 02:4, 03:6, 12:5, 13:7
        
        # M1': (A00+A11)(B02+B13) -> C02+C13
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        # M2': (A10+A11)B02 -> C12-C13
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        # M3': A00(B03-B13) -> C03+C13
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        # M4': A11(B12-B02) -> C02+C12
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        # M5': (A00+A01)B13 -> -C02+C03
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        # M6': (A10-A00)(B02+B03) -> C13
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        # M7': (A01-A11)(B12+B13) -> C02
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer Product: A[0:2, 2:3] * B[2:3, 0:4]
        # A indices: 02:2, 12:5
        # B indices: 20:8, 21:9, 22:10, 23:11
        # C indices: 00:0, 01:2, 02:4, 03:6 for i=0
        #            10:1, 11:3, 12:5, 13:7 for i=1
        
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    def verify(U, V, W, rank):
        """Check if the decomposition is exact."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for t in range(rank):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def to_lists(U, V, W):
        """Convert numpy arrays to lists of lists."""
        U_list = [[round(float(x), 10) for x in row] for row in U]
        V_list = [[round(float(x), 10) for x in row] for row in V]
        W_list = [[round(float(x), 10) for x in row] for row in W]
        return U_list, V_list, W_list

    def snap_to_half_integer(arr, tol=1e-6):
        """Snap values to nearest half-integer if close."""
        snapped = np.zeros_like(arr)
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                val = arr[i, j]
                # Check if close to integer
                nearest_int = round(val)
                if abs(val - nearest_int) < tol:
                    snapped[i, j] = nearest_int
                else:
                    # Check if close to half-integer
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) < tol:
                        snapped[i, j] = nearest_half
                    else:
                        snapped[i, j] = val
        return snapped

    def try_als(rank, max_iter=200, tol=1e-12, seed=42):
        """Alternating least squares to find rank decomposition."""
        rng = np.random.RandomState(seed)
        
        # Initialize
        U = rng.randn(n * m, rank) * 0.1
        V = rng.randn(m * p, rank) * 0.1
        W = np.zeros((p * n, rank))
        
        for iteration in range(max_iter):
            if time.time() > deadline:
                return None
            
            # Solve for W: fix U, V
            W_new = np.zeros((p * n, rank))
            for k in range(p):
                for i in range(n):
                    w = k * n + i
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        W_new[w, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        W_new[w, :] = rng.randn(rank) * 0.1
            
            # Solve for U: fix V, W
            U_new = np.zeros((n * m, rank))
            for i in range(n):
                for j in range(m):
                    u = i * m + j
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    try:
                        U_new[u, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        U_new[u, :] = rng.randn(rank) * 0.1
            
            # Solve for V: fix U, W
            V_new = np.zeros((m * p, rank))
            for j in range(m):
                for k in range(p):
                    v = j * p + k
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        V_new[v, :], _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    except np.linalg.LinAlgError:
                        V_new[v, :] = rng.randn(rank) * 0.1
            
            # Check convergence
            diff = np.max(np.abs(W_new - W)) + np.max(np.abs(U_new - U)) + np.max(np.abs(V_new - V))
            W, U, V = W_new, U_new, V_new
            
            if diff < tol:
                # Verify with numpy
                U_list, V_list, W_list = to_lists(U, V, W)
                if verify(U_list, V_list, W_list, rank):
                    # Snap to half-integers
                    U_snap = snap_to_half_integer(U)
                    V_snap = snap_to_half_integer(V)
                    W_snap = snap_to_half_integer(W)
                    U_list, V_list, W_list = to_lists(U_snap, V_snap, W_snap)
                    if verify(U_list, V_list, W_list, rank):
                        return U_list, V_list, W_list
                    else:
                        return U_list, V_list, W_list
        
        # Final check
        U_list, V_list, W_list = to_lists(U, V, W)
        if verify(U_list, V_list, W_list, rank):
            U_snap = snap_to_half_integer(U)
            V_snap = snap_to_half_integer(V)
            W_snap = snap_to_half_integer(W)
            U_list, V_list, W_list = to_lists(U_snap, V_snap, W_snap)
            if verify(U_list, V_list, W_list, rank):
                return U_list, V_list, W_list
            else:
                return U_list, V_list, W_list
        
        return None

    # Try to find rank 21 or lower
    for rank in range(15, 22):
        if time.time() > deadline:
            break
        
        for seed in range(10):
            if time.time() > deadline:
                break
            result = try_als(rank, max_iter=150, tol=1e-12, seed=seed)
            if result is not None:
                U, V, W = result
                return {"U": U, "V": V, "W": W}

    # Fallback to rank 22
    U, V, W, rank = build_rank_22()
    return {"U": U, "V": V, "W": W}
