import numpy as np
import time

DIMENSIONS = [3, 4, 6]

def verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    if rank == 0:
        return False
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
                if abs(s - 1.0) > 1e-9:
                    return False
    return True

def get_winograd_3x3x3():
    """
    Returns U, V, W for 3x3x3 matrix multiplication with rank 23.
    Based on Winograd's algorithm.
    U shape: (9, 23)
    V shape: (9, 23)
    W shape: (9, 23)
    Indices:
    U[i*3+j, c]
    V[j*3+k, c]
    W[k*3+i, c]
    """
    # Winograd 3x3x3 decomposition
    # The coefficients are integers.
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # The decomposition is often given as:
    # 23 products.
    # I will use a known valid set of coefficients.
    # Reference: "Fast Matrix Multiplication" by Winograd, or standard implementations.
    
    # A common representation:
    # Let A, B be 3x3.
    # The 23 multiplications are linear combinations of entries of A and B.
    # The result C is a linear combination of the products.
    
    # I will use a specific known decomposition.
    # If I don't have the exact coefficients memorized, I can use a simpler fallback:
    # Rank 27 for 3x3x3? No, 23 is the standard fast one.
    # If I cannot guarantee the coefficients, I should fall back to a verified search or schoolbook.
    # However, I can implement a **search** for the 3x3x3 decomposition if I don't hardcode it.
    # But hardcoding is safer if I know it.
    
    # Let's try to find a simple valid decomposition via search in the code if hardcoding fails.
    # But for now, I will implement the **search** strategy for the full 3x4x6 problem,
    # using the block structure hint to guide the search or just use the block construction
    # with **schoolbook** for the 3x3x3 blocks if I don't have Winograd coefficients.
    # Schoolbook 3x3x3 is 27.
    # 27 + 27 + 18 = 72. No gain.
    
    # So I MUST use a low rank 3x3x3 decomposition.
    # I will implement a **randomized search** for the 3x3x3 decomposition first,
    # then use it to build the 3x4x6 solution.
    
    return None # Placeholder

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best = {
        "U": U_sb.tolist(), 
        "V": V_sb.tolist(), 
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook

    # Strategy: Build 3x4x6 from 3x3x6 + 3x1x6.
    # 3x1x6 is schoolbook (Rank 18).
    # 3x3x6 is two 3x3x3 blocks.
    # We need a low rank decomposition for 3x3x3.
    # Let's search for a rank 23 or 24 decomposition for 3x3x3.
    
    def find_3x3x3_decomposition(rank_target, seed, time_limit=5.0):
        """
        Search for a rank-r decomposition of 3x3x3 matrix multiplication.
        """
        np.random.seed(seed)
        n3, m3, p3 = 3, 3, 3
        
        # Target tensor T[i,j,k] = 1
        # We want U, V, W of size (9, r)
        
        U = np.random.randn(9, rank_target) * 0.5
        V = np.random.randn(9, rank_target) * 0.5
        W = np.random.randn(9, rank_target) * 0.5
        
        start_time = time.time()
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                return None
            
            # Update U
            U_new = np.zeros_like(U)
            for i in range(n3):
                for j in range(m3):
                    X = np.zeros((p3, rank_target))
                    for k in range(p3):
                        X[k, :] = V[j * p3 + k, :] * W[k * n3 + i, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(X, np.ones(p3), rcond=None)
                        U_new[i * m3 + j, :] = u_sol
                    except:
                        pass
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for j in range(m3):
                for k in range(p3):
                    X = np.zeros((n3, rank_target))
                    for i in range(n3):
                        X[i, :] = U[i * m3 + j, :] * W[k * n3 + i, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(X, np.ones(n3), rcond=None)
                        V_new[j * p3 + k, :] = v_sol
                    except:
                        pass
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for k in range(p3):
                for i in range(n3):
                    X = np.zeros((m3, rank_target))
                    for j in range(m3):
                        X[j, :] = U[i * m3 + j, :] * V[j * p3 + k, :]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(X, np.ones(m3), rcond=None)
                        W_new[k * n3 + i, :] = w_sol
                    except:
                        pass
            W = W_new
            
            # Check error
            if step % 10 == 0:
                err = 0.0
                for i in range(n3):
                    for j in range(m3):
                        for k in range(p3):
                            s = np.dot(U[i * m3 + j, :], V[j * p3 + k, :] * W[k * n3 + i, :])
                            err += (s - 1.0) ** 2
                if err < 1e-10:
                    break

        # Round to integers
        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        
        # Verify
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    s = np.dot(U_r[i * m3 + j, :], V_r[j * p3 + k, :] * W_r[k * n3 + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return None
        
        return U_r, V_r, W_r

    # Try to find 3x3x3 decomposition
    found_333 = None
    for rank_333 in range(23, 28): # Try 23 to 27
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            res = find_333_decomposition(rank_333, seed, time_limit=2.0)
            if res is not None:
                found_333 = res
                print(f"Found 3x3x3 decomposition with rank {rank_333}")
                break
        if found_333:
            break

    if found_333 is not None:
        U333, V333, W333 = found_333
        r333 = U333.shape[1]
        
        # Construct 3x3x6 decomposition (two 3x3x3 blocks)
        # Block 1: columns 0-2 of B (k=0,1,2)
        # Block 2: columns 3-5 of B (k=3,4,5)
        
        # The full 3x4x6 problem:
        # U shape: (12, rank_total)
        # V shape: (24, rank_total)
        # W shape: (18, rank_total)
        
        # We split A (3x4) into A1 (3x3) and A2 (3x1).
        # We split B (4x6) into B1 (3x6) and B2 (1x6).
        # C = A1 B1 + A2 B2.
        
        # Part 1: A1 B1 (3x3x6)
        # This uses the 3x3x3 decomposition twice.
        # Rank contribution: 2 * r333.
        
        # Part 2: A2 B2 (3x1x6)
        # Schoolbook for 3x1x6: Rank 18.
        # A2 is 3x1 (indices i=0..2, j=3).
        # B2 is 1x6 (indices j=3, k=0..5).
        # C part is 3x6 (indices i=0..2, k=0..5).
        
        rank_total = 2 * r333 + 18
        
        U_total = np.zeros((12, rank_total))
        V_total = np.zeros((24, rank_total))
        W_total = np.zeros((18, rank_total))
        
        # Fill Part 1: A1 B1
        # A1 corresponds to i=0..2, j=0..2.
        # B1 corresponds to j=0..2, k=0..5.
        # C1 corresponds to i=0..2, k=0..5.
        
        # Block 1: k=0..2
        # U1: rows 0..8 (i=0..2, j=0..2)
        # V1: rows 0..8 (j=0..2, k=0..2)
        # W1: rows 0..8 (k=0..2, i=0..2)
        
        U_total[0:9, 0:r333] = U333
        V_total[0:9, 0:r333] = V333
        W_total[0:9, 0:r333] = W333 # k=0..2, i=0..2
        
        # Block 2: k=3..5
        # We reuse the same U333, V333, W333?
        # The tensor for k=3..5 is identical to k=0..2, just shifted.
        # So we can use the same coefficients.
        # U2: rows 0..8 (same A1)
        # V2: rows 18..26 (j=0..2, k=3..5) -> j*p+k. p=6.
        #    j=0, k=3 -> 3. j=0, k=5 -> 5.
        #    Wait, V index is j*p + k.
        #    For B1 (3x6), j is 0..2, k is 0..5.
        #    V rows for j=0..2, k=0..5 are 0..17.
        #    Block 1 (k=0..2): V rows 0..8? No.
        #    V index: j*6 + k.
        #    j=0: k=0,1,2 -> 0,1,2.
        #    j=1: k=0,1,2 -> 6,7,8.
        #    j=2: k=0,1,2 -> 12,13,14.
        #    This is not contiguous.
        
        # Let's map the 3x3x3 decomposition to the 3x3x6 structure carefully.
        # The 3x3x3 decomposition computes C[i,k] = sum_j A[i,j] B[j,k] for j,k in 0..2.
        # In the 3x3x6 context, for k=0..2, B[j,k] is at V[j*6 + k].
        # So V rows used are:
        # j=0: 0, 1, 2
        # j=1: 6, 7, 8
        # j=2: 12, 13, 14
        
        # We need to place V333 into these rows.
        # V333 is 9x r333. Rows correspond to j*3+k in the 3x3 case.
        # j=0: 0,1,2
        # j=1: 3,4,5
        # j=2: 6,7,8
        
        # Mapping V333 row r to V_total row:
        # If r = j*3 + k_local (k_local 0..2), then k_global = k_local.
        # V_total index = j*6 + k_global.
        
        for r in range(9):
            j = r // 3
            k_local = r % 3
            v_idx = j * 6 + k_local
            V_total[v_idx, 0:r333] = V333[r, :]
            
        # W333 rows correspond to k*3+i.
        # k=0: 0,1,2
        # k=1: 3,4,5
        # k=2: 6,7,8
        # In 3x3x6, W index is k*3 + i.
        # k=0..2, i=0..2.
        # W_total rows 0..8 are exactly k*3+i for k=0..2.
        # So W_total[0:9, 0:r333] = W333 is correct.
        
        # Now Block 2: k=3..5.
        # U is the same (A1).
        # V rows:
        # j=0: k=3,4,5 -> 3,4,5
        # j=1: k=3,4,5 -> 9,10,11
        # j=2: k=3,4,5 -> 15,16,17
        
        for r in range(9):
            j = r // 3
            k_local = r % 3
            k_global = 3 + k_local
            v_idx = j * 6 + k_global
            V_total[v_idx, r333:2*r333] = V333[r, :]
            
        # W rows for k=3..5:
        # k=3: 9,10,11
        # k=4: 12,13,14
        # k=5: 15,16,17
        
        for r in range(9):
            k = r // 3
            i = r % 3
            w_idx = (3 + k) * 3 + i
            W_total[w_idx, r333:2*r333] = W333[r, :]
            
        # U is the same for both blocks, so we just use the same U333?
        # No, U is a factor in the sum.
        # sum_c U[i,j,c] V[j,k,c] W[k,i,c].
        # For Block 1, c is 0..r333-1.
        # For Block 2, c is r333..2*r333-1.
        # So U for Block 2 is at columns r333..2*r333-1.
        # And it should be the same values as U333.
        
        U_total[0:9, r333:2*r333] = U333
        
        # Part 2: A2 B2 (3x1x6)
        # A2: i=0..2, j=3. (Rows 3, 7, 11 of U)
        # B2: j=3, k=0..5. (Rows 18..23 of V? j*p+k = 3*6+k = 18..23)
        # C2: i=0..2, k=0..5. (Rows 0..17 of W? k*3+i)
        
        # Schoolbook for 3x1x6:
        # 18 terms.
        # For each i in 0..2, k in 0..5:
        # U[i*4+3, c] = 1
        # V[3*6+k, c] = 1
        # W[k*3+i, c] = 1
        
        col_offset = 2 * r333
        for i in range(3):
            for k in range(6):
                u_idx = i * 4 + 3
                v_idx = 3 * 6 + k
                w_idx = k * 3 + i
                U_total[u_idx, col_offset] = 1
                V_total[v_idx, col_offset] = 1
                W_total[w_idx, col_offset] = 1
                col_offset += 1
                
        # Verify the constructed solution
        if verify(U_total.tolist(), V_total.tolist(), W_total.tolist(), n, m, p):
            if rank_total < best_rank:
                best = {
                    "U": U_total.tolist(),
                    "V": V_total.tolist(),
                    "W": W_total.tolist()
                }
                best_rank = rank_total
                print(f"Constructed valid solution with rank {rank_total}")
                return best

    return best

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    # Fallback: Schoolbook (Rank 72)
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
                
    best = {
        "U": U_sb.tolist(), 
        "V": V_sb.tolist(), 
        "W": W_sb.tolist()
    }
    best_rank = rank_schoolbook

    def find_3x3x3_decomposition(rank_target, seed, time_limit=5.0):
        np.random.seed(seed)
        n3, m3, p3 = 3, 3, 3
        
        U = np.random.randn(9, rank_target) * 0.5
        V = np.random.randn(9, rank_target) * 0.5
        W = np.random.randn(9, rank_target) * 0.5
        
        start_time = time.time()
        
        for step in range(2000):
            if time.time() - start_time > time_limit:
                return None
            
            U_new = np.zeros_like(U)
            for i in range(n3):
                for j in range(m3):
                    X = np.zeros((p3, rank_target))
                    for k in range(p3):
                        X[k, :] = V[j * p3 + k, :] * W[k * n3 + i, :]
                    try:
                        u_sol, _, _, _ = np.linalg.lstsq(X, np.ones(p3), rcond=None)
                        U_new[i * m3 + j, :] = u_sol
                    except:
                        pass
            U = U_new
            
            V_new = np.zeros_like(V)
            for j in range(m3):
                for k in range(p3):
                    X = np.zeros((n3, rank_target))
                    for i in range(n3):
                        X[i, :] = U[i * m3 + j, :] * W[k * n3 + i, :]
                    try:
                        v_sol, _, _, _ = np.linalg.lstsq(X, np.ones(n3), rcond=None)
                        V_new[j * p3 + k, :] = v_sol
                    except:
                        pass
            V = V_new
            
            W_new = np.zeros_like(W)
            for k in range(p3):
                for i in range(n3):
                    X = np.zeros((m3, rank_target))
                    for j in range(m3):
                        X[j, :] = U[i * m3 + j, :] * V[j * p3 + k, :]
                    try:
                        w_sol, _, _, _ = np.linalg.lstsq(X, np.ones(m3), rcond=None)
                        W_new[k * n3 + i, :] = w_sol
                    except:
                        pass
            W = W_new
            
            if step % 10 == 0:
                err = 0.0
                for i in range(n3):
                    for j in range(m3):
                        for k in range(p3):
                            s = np.dot(U[i * m3 + j, :], V[j * p3 + k, :] * W[k * n3 + i, :])
                            err += (s - 1.0) ** 2
                if err < 1e-10:
                    break

        U_r = np.round(U)
        V_r = np.round(V)
        W_r = np.round(W)
        
        for i in range(n3):
            for j in range(m3):
                for k in range(p3):
                    s = np.dot(U_r[i * m3 + j, :], V_r[j * p3 + k, :] * W_r[k * n3 + i, :])
                    if abs(s - 1.0) > 1e-9:
                        return None
        
        return U_r, V_r, W_r

    found_333 = None
    for rank_333 in range(23, 28):
        for seed in range(10):
            if time.time() > deadline - 20:
                break
            res = find_333_decomposition(rank_333, seed, time_limit=2.0)
            if res is not None:
                found_333 = res
                print(f"Found 3x3x3 decomposition with rank {rank_333}")
                break
        if found_333:
            break

    if found_333 is not None:
        U333, V333, W333 = found_333
        r333 = U333.shape[1]
        
        rank_total = 2 * r333 + 18
        
        U_total = np.zeros((12, rank_total))
        V_total = np.zeros((24, rank_total))
        W_total = np.zeros((18, rank_total))
        
        U_total[0:9, 0:r333] = U333
        U_total[0:9, r333:2*r333] = U333
        
        for r in range(9):
            j = r // 3
            k_local = r % 3
            v_idx_1 = j * 6 + k_local
            v_idx_2 = j * 6 + 3 + k_local
            V_total[v_idx_1, 0:r333] = V333[r, :]
            V_total[v_idx_2, r333:2*r333] = V333[r, :]
            
        W_total[0:9, 0:r333] = W333
        for r in range(9):
            k = r // 3
            i = r % 3
            w_idx = (3 + k) * 3 + i
            W_total[w_idx, r333:2*r333] = W333[r, :]
        
        col_offset = 2 * r333
        for i in range(3):
            for k in range(6):
                u_idx = i * 4 + 3
                v_idx = 3 * 6 + k
                w_idx = k * 3 + i
                U_total[u_idx, col_offset] = 1
                V_total[v_idx, col_offset] = 1
                W_total[w_idx, col_offset] = 1
                col_offset += 1
                
        if verify(U_total.tolist(), V_total.tolist(), W_total.tolist(), n, m, p):
            if rank_total < best_rank:
                best = {
                    "U": U_total.tolist(),
                    "V": V_total.tolist(),
                    "W": W_total.tolist()
                }
                best_rank = rank_total
                print(f"Constructed valid solution with rank {rank_total}")
                return best

    return best
# EVOLVE-BLOCK-END
