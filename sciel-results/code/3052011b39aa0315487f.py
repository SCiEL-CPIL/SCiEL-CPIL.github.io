import numpy as np
import time
from itertools import product

def construct():
    """
    Rank-minimal exact tensor decomposition for 3x3x6 matrix multiplication.
    
    Strategy:
    1. Decompose 3x3x6 as three independent 3x3x2 blocks.
    2. For 3x3x2, try known algebraic constructions (Strassen-like) at various ranks.
    3. Verify each candidate exactly before accepting.
    4. Fallback to schoolbook rank 54.
    
    The 3x3x6 tensor T[i,j,k] = delta_{i,i'}delta_{j,j'}delta_{k,k'}
    splits into 3 copies of the 3x3x2 tensor (k in {0,1} for each of 3 blocks).
    A rank-R decomposition of 3x3x2 gives rank 3R for 3x3x6.
    """
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin

    n, m, p = 3, 3, 6
    n2, m2, p2 = 3, 3, 2  # dimensions for one block

    def generate_schoolbook():
        """Rank-54 schoolbook decomposition for 3x3x6."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def verify_3x3x2(U, V, W, R):
        """
        Verify that U (9xR), V (6xR), W (6xR) give exact 3x3x2 matrix mult.
        T[i,j,k] = sum_r U[i*3+j, r] * V[j'*2+k, r] * W[k'*3+i', r]
        Should equal delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(3):
            for ip in range(3):
                for j in range(3):
                    for jp in range(3):
                        for k in range(2):
                            for kp in range(2):
                                expected = 1.0 if (i == ip and j == jp and k == kp) else 0.0
                                actual = 0.0
                                for r in range(R):
                                    actual += U_np[i * 3 + j, r] * V_np[jp * 2 + k, r] * W_np[kp * 3 + ip, r]
                                if abs(actual - expected) > 1e-8:
                                    return False
        return True

    def extend_to_3x3x6(U_core, V_core, W_core, R):
        """Extend a 3x3x2 decomposition to 3x3x6 by 3 independent blocks."""
        total_rank = 3 * R
        U_tot = [[0] * total_rank for _ in range(n * m)]
        V_tot = [[0] * total_rank for _ in range(m * p)]
        W_tot = [[0] * total_rank for _ in range(p * n)]
        
        for block in range(3):
            offset = block * R
            for i in range(3):
                for j in range(3):
                    for r in range(R):
                        U_tot[i * 3 + j][offset + r] = U_core[i * 3 + j][r]
            for j in range(3):
                for k in range(2):
                    for r in range(R):
                        V_tot[j * 6 + block * 2 + k][offset + r] = V_core[j * 2 + k][r]
            for i in range(3):
                for k in range(2):
                    for r in range(R):
                        W_tot[(block * 2 + k) * 3 + i][offset + r] = W_core[k * 3 + i][r]
        
        return {"U": U_tot, "V": V_tot, "W": W_tot}

    def solve_for_W(U, V, R):
        """
        Given U (9xR) and V (6xR) with half-integer entries, solve for W (6xR).
        For each (i, k) pair, solve linear system:
        sum_r U[i,j,r] * V[j',k,r] * W[k,i,r] = delta_{j,j'}
        W has entries W[k*3+i, r] for k in 0..1, i in 0..2.
        """
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W = np.zeros((6, R), dtype=float)
        
        for i in range(3):
            for k in range(2):
                # Build system: for each j, j' pair
                # M[j*m2+jp] = U[i*3+j, :] * V[jp*2+k, :]
                M = np.zeros((9, R))
                rhs = np.zeros(9)
                for j in range(3):
                    for jp in range(3):
                        M[j * 3 + jp, :] = U_np[i * 3 + j, :] * V_np[jp * 2 + k, :]
                        if j == jp:
                            rhs[j * 3 + jp] = 1.0
                
                # Solve M * w_vec = rhs, where w_vec = W[k*3+i, :]
                try:
                    w_vec, residuals, rank_m, s = np.linalg.lstsq(M, rhs, rcond=None)
                except:
                    return None
                
                # Check if w_vec is half-integer
                doubled = w_vec * 2.0
                rounded = np.round(doubled)
                if np.max(np.abs(doubled - rounded)) > 1e-6:
                    return None
                
                W[k * 3 + i, :] = rounded / 2.0
        
        return W.tolist()

    def search_3x3x2(R, max_attempts=10000, deadline=deadline):
        """
        Search for rank-R decomposition of 3x3x2.
        Randomly initialize U and V with small integers, solve for W, verify.
        """
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random small integers for U and V
            U_r = np.random.randint(-2, 3, size=(9, R)).tolist()
            V_r = np.random.randint(-2, 3, size=(6, R)).tolist()
            
            # Skip if too many zeros (degenerate)
            if np.count_nonzero(np.array(U_r)) < R * 2:
                continue
            if np.count_nonzero(np.array(V_r)) < R * 2:
                continue
            
            # Solve for W
            W_r = solve_for_W(U_r, V_r, R)
            if W_r is None:
                continue
            
            # Verify
            if verify_3x3x2(U_r, V_r, W_r, R):
                return (U_r, V_r, W_r)
        
        return None

    def structured_search_3x3x2(R, max_attempts=5000, deadline=deadline):
        """
        Structured search: use specific patterns for U and V that are more likely
        to yield valid W with half-integer entries.
        """
        # Pattern 1: U has entries from {-1, 0, 1}, V similar
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                return None
            
            # Random sign patterns with sparsity
            U_r = np.zeros((9, R))
            V_r = np.zeros((6, R))
            
            for r in range(R):
                # U column: pick 3-5 non-zero entries with values +/-1
                nz_u = np.random.choice(9, size=np.random.randint(3, 6), replace=False)
                for idx in nz_u:
                    U_r[idx, r] = np.random.choice([-1, 1])
                
                # V column: pick 2-4 non-zero entries
                nz_v = np.random.choice(6, size=np.random.randint(2, 5), replace=False)
                for idx in nz_v:
                    V_r[idx, r] = np.random.choice([-1, 1])
            
            U_list = U_r.astype(int).tolist()
            V_list = V_r.astype(int).tolist()
            
            W_r = solve_for_W(U_list, V_list, R)
            if W_r is None:
                continue
            
            if verify_3x3x2(U_list, V_list, W_r, R):
                return (U_list, V_list, W_r)
        
        return None

    # Try ranks from low to high for 3x3x2
    # Known lower bound: tensor rank of 3x3x2 is at least 9 (from support)
    # Strassen for 3x3x2 should be around 13-15
    best_decomp = None
    best_R = None
    
    for R in [13, 14, 15, 16]:
        if time.time() > deadline - 15:
            break
        
        print(f"Searching for 3x3x2 rank {R}...")
        
        # Try structured search first (more efficient)
        decomp = structured_search_3x3x2(R, max_attempts=3000, deadline=deadline)
        
        if decomp is None:
            # Try random search
            decomp = search_3x3x2(R, max_attempts=5000, deadline=deadline)
        
        if decomp is not None:
            best_decomp = decomp
            best_R = R
            print(f"Found rank-{R} decomposition for 3x3x2!")
            break
        else:
            print(f"No rank-{R} decomposition found for 3x3x2.")
    
    if best_decomp is not None:
        U_core, V_core, W_core = best_decomp
        result = extend_to_3x3x6(U_core, V_core, W_core, best_R)
        print(f"Final rank: {3 * best_R}")
        return result
    else:
        print("Falling back to schoolbook rank 54.")
        return generate_schoolbook()
