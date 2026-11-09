"""
Solution for 4x4 matrix multiplication - attempts rank < 49 via search,
falls back to rank-49 two-level Strassen decomposition.
"""
import time
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = 145.0  # Leave 5s margin for 150s limit

    n, m, p = 4, 4, 4

    # --- 2x2 Strassen Coefficients ---
    U_elem = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    V_elem = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    W_elem = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 0], [0, 0]]
    ]

    A_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (0, 0, -1)],
        [(0, 1, 1), (1, 1, -1)]
    ]
    B_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(0, 0, 1)],
        [(0, 1, 1), (1, 1, -1)],
        [(1, 0, 1), (0, 0, -1)],
        [(1, 1, 1)],
        [(0, 0, 1), (0, 1, 1)],
        [(1, 0, 1), (1, 1, 1)]
    ]
    C_blocks = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]

    def build_strassen_49():
        """Build the rank-49 two-level Strassen decomposition."""
        rank = 49
        U = np.zeros((16, rank), dtype=int)
        V = np.zeros((16, rank), dtype=int)
        W = np.zeros((16, rank), dtype=int)

        for s in range(7):
            for t in range(7):
                col = s * 7 + t
                for (bi, bj, c_a) in A_blocks[s]:
                    for ei in range(2):
                        for ej in range(2):
                            row_a = (bi * 2 + ei) * 4 + (bj * 2 + ej)
                            U[row_a, col] += c_a * U_elem[t][ei * 2 + ej]
                for (bj, bk, c_b) in B_blocks[s]:
                    for ej in range(2):
                        for ek in range(2):
                            row_b = (bj * 2 + ej) * 4 + (bk * 2 + ek)
                            V[row_b, col] += c_b * V_elem[t][ej * 2 + ek]
                for (bi, bk, c_c) in C_blocks[s]:
                    for ei in range(2):
                        for ek in range(2):
                            row_c = (bk * 2 + ek) * 4 + (bi * 2 + ei)
                            W[row_c, col] += c_c * W_elem[t][ei][ek]
        return U, V, W

    def verify_decomp(U, V, W, n, m, p, rank, num_tests=3):
        """Exact verification using integer arithmetic."""
        if time.time() - start_time > deadline - 5:
            return False
        
        U_arr = np.asarray(U, dtype=int)
        V_arr = np.asarray(V, dtype=int)
        W_arr = np.asarray(W, dtype=int)
        
        for test_idx in range(num_tests):
            np.random.seed(1000 + test_idx)
            A_test = np.random.randint(-5, 5, (n, m))
            B_test = np.random.randint(-5, 5, (m, p))
            AB_true = A_test @ B_test
            
            AB_decomp = np.zeros((n, p), dtype=int)
            A_flat = A_test.flatten()
            B_flat = B_test.flatten()
            
            for r in range(rank):
                u_val = int(np.dot(U_arr[:, r], A_flat))
                v_val = int(np.dot(V_arr[:, r], B_flat))
                scalar = u_val * v_val
                if scalar != 0:
                    AB_decomp += (W_arr[:, r] * scalar).reshape((p, n)).T
            
            if not np.array_equal(AB_true, AB_decomp):
                return False
        return True

    def try_reduce_rank(U, V, W, target_rank, time_limit):
        """
        Try to find a valid decomposition of rank target_rank by removing
        (target_rank - target_rank) terms from the rank-49 decomposition.
        Uses greedy elimination: try removing one term at a time and check
        if the remaining terms still span the tensor.
        """
        if time.time() - start_time > deadline - 10:
            return None
        
        rank = U.shape[1]
        if rank <= target_rank:
            return (U[:, :target_rank].copy(), V[:, :target_rank].copy(), W[:, :target_rank].copy())
        
        # Build the multiplication tensor T[i,j,k] = delta_{i,k}
        # T is 4x4x4, entry (i,j,k) is 1 if i==k else 0
        T = np.zeros((n, m, p), dtype=int)
        for i in range(n):
            for k in range(p):
                if i == k:
                    for j in range(m):
                        T[i, j, k] = 1
        
        # For each candidate removal, check if the remaining terms still
        # produce the correct tensor. We do this by checking the tensor
        # reconstruction: T[i,j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
        
        # Precompute for efficiency: for each r, the "slice" contribution
        # Actually, let's just do direct verification for subsets
        
        # Strategy: try removing each single term and check
        # If removing term r makes the decomposition invalid, keep it
        # Then try removing pairs, etc.
        
        # Simpler approach: greedily eliminate terms
        # Start with all 49 terms, try to remove one at a time
        
        active = list(range(rank))
        
        for _ in range(rank - target_rank):
            if time.time() - start_time > deadline - 10:
                break
            
            removed = False
            for idx, r in enumerate(active):
                # Try removing term r
                candidate = [x for x in active if x != r]
                
                # Check if candidate terms still reproduce the tensor
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for rr in candidate:
                                s += int(U[i*m+j, rr]) * int(V[j*p+k, rr]) * int(W[k*n+i, rr])
                            expected = 1 if i == k else 0
                            if s != expected:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                    if not valid:
                        break
                
                if valid:
                    active = candidate
                    removed = True
                    break
            
            if not removed:
                break
        
        if len(active) <= target_rank:
            U_new = U[:, active].copy()
            V_new = V[:, active].copy()
            W_new = W[:, active].copy()
            return (U_new, V_new, W_new)
        
        return None

    # Build the rank-49 decomposition
    U, V, W = build_strassen_49()
    
    # Verify rank 49
    if not verify_decomp(U, V, W, n, m, p, 49, num_tests=3):
        # Fallback to schoolbook rank 64
        rank_fb = 64
        U_fb = np.zeros((16, rank_fb), dtype=int)
        V_fb = np.zeros((16, rank_fb), dtype=int)
        W_fb = np.zeros((16, rank_fb), dtype=int)
        col_fb = 0
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    U_fb[i*4+j, col_fb] = 1
                    V_fb[j*4+k, col_fb] = 1
                    W_fb[k*4+i, col_fb] = 1
                    col_fb += 1
        return {"U": U_fb.tolist(), "V": V_fb.tolist(), "W": W_fb.tolist()}
    
    # Try to reduce rank below 49
    best_U, best_V, best_W = U, V, W
    best_rank = 49
    
    # Try rank 48
    if time.time() - start_time < deadline - 15:
        result = try_reduce_rank(U, V, W, 48, deadline - 15)
        if result is not None:
            U48, V48, W48 = result
            if verify_decomp(U48, V48, W48, n, m, p, 48, num_tests=5):
                best_U, best_V, best_W = U48, V48, W48
                best_rank = 48
    
    # Try rank 47 if we have time
    if best_rank == 48 and time.time() - start_time < deadline - 15:
        result = try_reduce_rank(U, V, W, 47, deadline - 15)
        if result is not None:
            U47, V47, W47 = result
            if verify_decomp(U47, V47, W47, n, m, p, 47, num_tests=5):
                best_U, best_V, best_W = U47, V47, W47
                best_rank = 47
    
    # Try rank 46 if we have time
    if best_rank == 47 and time.time() - start_time < deadline - 15:
        result = try_reduce_rank(U, V, W, 46, deadline - 15)
        if result is not None:
            U46, V46, W46 = result
            if verify_decomp(U46, V46, W46, n, m, p, 46, num_tests=5):
                best_U, best_V, best_W = U46, V46, W46
                best_rank = 46
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
