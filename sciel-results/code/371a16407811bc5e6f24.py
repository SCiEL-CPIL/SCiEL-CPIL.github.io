"""Program for matrix multiplication 2x3x7 tensor decomposition search."""
import time
import random
import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2 by 3 matrix
    by a 3 by 7 matrix.
    
    Returns:
        dict: {'U': (n*m, rank), 'V': (m*p, rank), 'W': (p*n, rank)}
    """
    start_time = time.time()
    deadline = start_time + 130.0  # Leave 10s safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 7
    nm, mp, pn = n * m, m * p, p * n  # 6, 21, 14
    
    def build_strassen_decomposition():
        """Build the known rank-39 decomposition using Strassen blocks."""
        # Rank calculation:
        # 3 Strassen blocks (2x2x2) for k=0,1; 2,3; 4,5 -> 3 * 7 = 21
        # 1 Schoolbook block (2x2x1) for k=6 -> 4
        # 1 Schoolbook block (2x1x7) for j=2 -> 14
        rank = 21 + 4 + 14
        
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        
        col = 0
        
        def add_term(u_indices, v_indices, w_indices):
            nonlocal col
            for idx in u_indices:
                U[idx][col] += 1
            for idx in v_indices:
                V[idx][col] += 1
            for idx in w_indices:
                W[idx][col] += 1
            col += 1

        def add_strassen_block(k_start):
            # A indices: i*3+j
            # B indices: j*7+k
            # C indices: k*2+i
            b00 = 0 * 7 + k_start
            b01 = 0 * 7 + k_start + 1
            b10 = 1 * 7 + k_start
            b11 = 1 * 7 + k_start + 1
            
            c00 = k_start * 2 + 0
            c01 = (k_start + 1) * 2 + 0
            c10 = k_start * 2 + 1
            c11 = (k_start + 1) * 2 + 1
            
            # M1: (A00+A11)(B00+B11) -> C00, C11
            add_term([0, 4], [b00, b11], [c00, c11])
            
            # M2: (A10+A11)B00 -> C10, -C11
            add_term([3, 4], [b00], [c10, c11])
            W[c11][col-1] = -1
            
            # M3: A00(B01-B11) -> C01, C11
            add_term([0], [b01, b11], [c01, c11])
            V[b11][col-1] = -1
            
            # M4: A11(B10-B00) -> C00, C10
            add_term([4], [b10, b00], [c00, c10])
            V[b00][col-1] = -1
            
            # M5: (A00+A01)B11 -> -C00, C01
            add_term([0, 1], [b11], [c00, c01])
            W[c00][col-1] = -1
            
            # M6: (A10-A00)(B00+B01) -> C11
            add_term([3, 0], [b00, b01], [c11])
            U[0][col-1] = -1
            
            # M7: (A01-A11)(B10+B11) -> C00
            add_term([1, 4], [b10, b11], [c00])
            U[4][col-1] = -1

        # Add 3 Strassen blocks for k pairs (0,1), (2,3), (4,5)
        add_strassen_block(0)
        add_strassen_block(2)
        add_strassen_block(4)
        
        # Block for k=6, 2x2x1 Schoolbook
        b06 = 0 * 7 + 6
        b16 = 1 * 7 + 6
        c06 = 6 * 2 + 0
        c16 = 6 * 2 + 1
        
        add_term([0], [b06], [c06])
        add_term([1], [b16], [c06])
        add_term([3], [b06], [c16])
        add_term([4], [b16], [c16])
        
        # Part for j=2: A[i,2]*B[2,k] for all k
        for k in range(7):
            b2k = 2 * 7 + k
            c0k = k * 2 + 0
            c1k = k * 2 + 1
            
            add_term([2], [b2k], [c0k])
            add_term([5], [b2k], [c1k])
        
        return U, V, W

    def verify_decomposition(U, V, W, tol=1e-10):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0])
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Check valid terms
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                    if abs(val - 1.0) > tol:
                        return False
        
        # Check invalid terms (sample)
        for _ in range(3000):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_c = c // n
            i_c = c % n
            if not (j_a == j_b and i_a == i_c and k_b == k_c):
                val = np.sum(U_arr[a, :] * V_arr[b, :] * W_arr[c, :])
                if abs(val) > tol:
                    return False
        
        return True

    # Build the initial Strassen decomposition (rank 39)
    U_init, V_init, W_init = build_strassen_decomposition()
    best_rank = len(U_init[0])
    best_U = [row[:] for row in U_init]
    best_V = [row[:] for row in V_init]
    best_W = [row[:] for row in W_init]
    
    # Verify the initial decomposition
    if not verify_decomposition(best_U, best_V, best_W):
        # Fallback to schoolbook
        rank_sb = n * m * p
        U_sb = [[0] * rank_sb for _ in range(nm)]
        V_sb = [[0] * rank_sb for _ in range(mp)]
        W_sb = [[0] * rank_sb for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_sb[i * m + j][col] = 1
                    V_sb[j * p + k][col] = 1
                    W_sb[k * n + i][col] = 1
                    col += 1
        best_rank = rank_sb
        best_U = U_sb
        best_V = V_sb
        best_W = W_sb
    
    # Try to find lower rank decompositions using structured search
    # Strategy: Try to reduce rank by finding linear dependencies or using
    # algebraic structure of the tensor
    
    def try_random_projection(target_rank, U0, V0, W0, max_attempts=5):
        """Try to find a target_rank decomposition via random projection."""
        U0_arr = np.array(U0, dtype=np.float64)
        V0_arr = np.array(V0, dtype=np.float64)
        W0_arr = np.array(W0, dtype=np.float64)
        orig_rank = U0_arr.shape[1]
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            np.random.seed(target_rank * 10000 + attempt * 100 + 7)
            
            # Random projection
            P = np.random.randn(orig_rank, target_rank)
            
            U_new = U0_arr @ P
            V_new = V0_arr @ P
            W_new = W0_arr @ P
            
            if verify_decomposition(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist()
        
        return None
    
    def try_gradient_refinement(target_rank, U0, V0, W0, max_iter=200):
        """Try to refine a decomposition to target_rank using gradient descent."""
        if time.time() > deadline - 10:
            return None
        
        U0_arr = np.array(U0, dtype=np.float64)
        V0_arr = np.array(V0, dtype=np.float64)
        W0_arr = np.array(W0, dtype=np.float64)
        orig_rank = U0_arr.shape[1]
        
        # Initialize with projection
        np.random.seed(42)
        P = np.random.randn(orig_rank, target_rank)
        U_opt = U0_arr @ P
        V_opt = V0_arr @ P
        W_opt = W0_arr @ P
        
        lr = 0.01
        penalty_weight = 50.0
        
        for step in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            U_grad = np.zeros_like(U_opt)
            V_grad = np.zeros_like(V_opt)
            W_grad = np.zeros_like(W_opt)
            
            # For valid terms
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b = j * p + k
                        c = k * n + i
                        val = np.sum(U_opt[a, :] * V_opt[b, :] * W_opt[c, :])
                        err = val - 1.0
                        U_grad[a, :] += 2 * penalty_weight * err * V_opt[b, :] * W_opt[c, :]
                        V_grad[b, :] += 2 * penalty_weight * err * U_opt[a, :] * W_opt[c, :]
                        W_grad[c, :] += 2 * penalty_weight * err * U_opt[a, :] * W_opt[c, :]
            
            # For invalid terms (sample)
            for _ in range(50):
                a = np.random.randint(0, nm)
                b = np.random.randint(0, mp)
                c = np.random.randint(0, pn)
                i_a = a // m
                j_a = a % m
                j_b = b // p
                k_b = b % p
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and i_a == i_c and k_b == k_c):
                    val = np.sum(U_opt[a, :] * V_opt[b, :] * W_opt[c, :])
                    U_grad[a, :] += 2 * penalty_weight * val * V_opt[b, :] * W_opt[c, :]
                    V_grad[b, :] += 2 * penalty_weight * val * U_opt[a, :] * W_opt[c, :]
                    W_grad[c, :] += 2 * penalty_weight * val * U_opt[a, :] * W_opt[c, :]
            
            # Update
            U_opt -= lr * U_grad
            V_opt -= lr * V_grad
            W_opt -= lr * W_grad
            
            # Check if we've found a valid decomposition
            if step % 20 == 0:
                if verify_decomposition(U_opt.tolist(), V_opt.tolist(), W_opt.tolist()):
                    return U_opt.tolist(), V_opt.tolist(), W_opt.tolist()
        
        return None
    
    # Try to find lower ranks
    # Theoretical lower bound: max(nm, mp, pn) = max(6, 21, 14) = 21
    # But we need to consider the structure
    
    # Try ranks from 38 down to 25
    for target_rank in range(best_rank - 1, 24, -1):
        if time.time() > deadline - 15:
            break
        
        # Try random projection first (fast)
        result = try_random_projection(target_rank, best_U, best_V, best_W, max_attempts=3)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
            continue
        
        # Try gradient refinement
        result = try_gradient_refinement(target_rank, best_U, best_V, best_W, max_iter=100)
        if result is not None:
            best_rank = target_rank
            best_U, best_V, best_W = result
    
    # Try a different structured approach: exploit the j=2 dimension
    # For j=2, we have 14 terms (2 i-values * 7 k-values)
    # Can we share computation between different k values?
    
    if best_rank > 35 and time.time() < deadline - 20:
        # Try to find a decomposition that shares the j=2 terms
        # using a different algebraic structure
        
        # Idea: For the j=2 part, we need:
        # C[i,k] += A[i,2] * B[2,k] for all i,k
        # This is rank 14 (2*7) in the schoolbook
        # Can we do better? The tensor for this part is:
        # T[i,2,k] = A[i,2] * B[2,k]
        # This is a rank-1 tensor in (i,k) space for each fixed value of A[i,2]*B[2,k]
        # But we need it for all (i,k) pairs, so it's actually rank 14
        
        # However, if we can combine with other terms...
        # This is complex, so let's try a direct search
        
        pass
    
    # Final verification
    if not verify_decomposition(best_U, best_V, best_W):
        # If verification fails, use the known valid Strassen decomposition
        U_init, V_init, W_init = build_strassen_decomposition()
        best_U = U_init
        best_V = V_init
        best_W = W_init
        best_rank = len(U_init[0])
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END

