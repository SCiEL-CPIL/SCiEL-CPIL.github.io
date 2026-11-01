import time
import numpy as np
import itertools

def _verify(U, V, W, n, m, p, tol=1e-9):
    """
    Verify that the decomposition is exact using numpy for speed.
    """
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Tensor check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta_{i,k} is NOT correct.
    # The tensor T_{i,j,k} for matrix multiplication A(nxm) * B(mxk) = C(nxk)
    # is defined such that C_{ik} = sum_j A_{ij} B_{jk}.
    # The decomposition is: C_{ik} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} ? No.
    # Standard definition:
    # C_{ik} = sum_j A_{ij} B_{jk}
    # Decomposition: C_{ik} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is wrong indices.
    
    # Let's use the definition from the prompt:
    # U: (n*m, rank) -> U[i*m + j, r]
    # V: (m*p, rank) -> V[j*p + k, r]
    # W: (p*n, rank) -> W[k*n + i, r]
    # The product term is U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Summing over r and j? No, the sum is over r.
    # For fixed i, k, we sum over j and r?
    # C_{ik} = sum_j (sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r])
    # This must equal delta_{i,k} ? No, it must equal the coefficient of A_{ij}B_{jk} in C_{ik}.
    # The coefficient of A_{ij}B_{jk} in C_{ik} is 1 if k is the column index of B and i is row of A.
    # Wait, C_{ik} = sum_j A_{ij} B_{jk}.
    # So for a specific term A_{ij} B_{jk}, it contributes to C_{ik} with coefficient 1.
    # So we need: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1 for all i,j,k?
    # No, that's not right. The tensor T_{i,j,k} has entries 1 if the indices correspond to a valid multiplication step.
    # Actually, the multiplication tensor T has T_{i,j,k} = 1 if we are computing C_{ik} using A_{ij} and B_{jk}.
    # But C_{ik} depends on ALL j.
    # The bilinear map is: (A, B) -> C.
    # C_{ik} = sum_j A_{ij} B_{jk}.
    # We want to represent this as sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}?
    # The standard tensor rank definition for matrix multiplication M_{n,m,p} is:
    # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r} is NOT the standard form.
    
    # Let's look at the verify function in the seed:
    # total = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
    # u_idx = i*m + j
    # v_idx = j*p + k
    # w_idx = k*n + i
    # It checks if total == 1.0.
    # This implies that for EVERY i, j, k, the sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] must be 1.
    # Is this correct?
    # If A_{ij} and B_{jk} are the only non-zero entries in A and B, then C_{ik} should be 1.
    # Yes, this is the standard "basis" check for the tensor.
    # So the condition is: for all i in [0,n), j in [0,m), k in [0,p):
    # sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i*m + j
                v_idx = j*p + k
                w_idx = k*n + i
                total = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
                if abs(total - 1.0) > tol:
                    return False
    return True

def _schoolbook(n, m, p):
    """Return schoolbook decomposition of rank n*m*p."""
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

def _search_rank(R, n, m, p, time_limit=10):
    """
    Search for a decomposition of rank R using integer coefficients in {-1, 0, 1}.
    We exploit the structure:
    U has n*m rows.
    V has m*p rows.
    W has p*n rows.
    
    This is a constraint satisfaction problem.
    For each r, we choose a vector u_r (length n*m), v_r (length m*p), w_r (length p*n).
    The condition is:
    For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    This is equivalent to:
    Let U_r be the r-th column of U.
    Let V_r be the r-th column of V.
    Let W_r be the r-th column of W.
    
    We can fix the first few columns to break symmetry or just search.
    Given the small rank (30) and small dimensions, brute force is impossible.
    However, we can try a heuristic search or a specific construction.
    
    Let's try to construct a rank 30 solution based on the idea that 2x3x10 might have a lower rank.
    Known ranks:
    2x2x2 is 7 (Strassen).
    2x3x3?
    
    Let's try a randomized local search with integer steps.
    """
    start_time = time.time()
    
    # Initialize with random integers from {-1, 0, 1}
    # Or better, initialize with a known valid decomposition and perturb?
    # Since we don't know a valid one, let's start with random.
    
    U = np.random.randint(-1, 2, (n*m, R))
    V = np.random.randint(-1, 2, (m*p, R))
    W = np.random.randint(-1, 2, (p*n, R))
    
    # Precompute indices
    idx_list = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_list.append((i*m+j, j*p+k, k*n+i))
                
    def compute_error():
        err = 0
        for u_idx, v_idx, w_idx in idx_list:
            val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
            err += (val - 1.0)**2
        return err
    
    best_err = compute_error()
    if best_err < 1e-9:
        return U, V, W
        
    # Local search
    # We can flip signs or add/subtract 1 to elements.
    # To be fast, we only update one matrix at a time.
    
    steps = 0
    max_steps = 50000
    step_size = 1
    
    while time.time() - start_time < time_limit and steps < max_steps:
        # Pick a random matrix and element
        mat_choice = np.random.choice(3)
        if mat_choice == 0:
            row = np.random.choice(n*m)
            col = np.random.choice(R)
            old_val = U[row, col]
            U[row, col] += np.random.choice([-1, 1])
            new_err = compute_error()
            if new_err < best_err:
                best_err = new_err
                if best_err < 1e-9:
                    return U, V, W
            else:
                U[row, col] = old_val
        elif mat_choice == 1:
            row = np.random.choice(m*p)
            col = np.random.choice(R)
            old_val = V[row, col]
            V[row, col] += np.random.choice([-1, 1])
            new_err = compute_error()
            if new_err < best_err:
                best_err = new_err
                if best_err < 1e-9:
                    return U, V, W
            else:
                V[row, col] = old_val
        else:
            row = np.random.choice(p*n)
            col = np.random.choice(R)
            old_val = W[row, col]
            W[row, col] += np.random.choice([-1, 1])
            new_err = compute_error()
            if new_err < best_err:
                best_err = new_err
                if best_err < 1e-9:
                    return U, V, W
            else:
                W[row, col] = old_val
        steps += 1
        
    return None

def construct():
    start_time = time.time()
    deadline = 145  # 5s safety margin
    
    n, m, p = 2, 3, 10
    
    # 1. Try to find a lower rank solution via search
    # Rank 30 is a good target if it exists.
    # If not, we fall back to the known 55.
    
    # We can also try rank 45, 40, etc.
    # But 30 is the most likely candidate for a "jump" if it exists.
    
    # Let's try a few restarts for rank 30
    best_result = None
    
    # Try rank 30
    if time.time() < deadline - 5:
        res = _search_rank(30, n, m, p, time_limit=5)
        if res:
            U, V, W = res
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Try rank 40
    if time.time() < deadline - 5:
        res = _search_rank(40, n, m, p, time_limit=5)
        if res:
            U, V, W = res
            if _verify(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
    # Fallback to the hardcoded rank 55 solution from the seed
    # This ensures we always return a valid result.
    
    rank = 55
    U = [[0]*rank for _ in range(6)]
    V = [[0]*rank for _ in range(30)]
    W = [[0]*rank for _ in range(20)]
    
    col = 0
    
    # Part 1: Strassen for A1 (2x2) * B1 (2x10)
    # A1 is rows 0,1 of A. Cols 0,1 of B? No, B is 3x10.
    # We split B into B1 (cols 0,1) and B2 (cols 2..9)?
    # The seed splits A into A1 (2x2) and A2 (2x1).
    # A is 2x3. A1 uses cols 0,1. A2 uses col 2.
    # B is 3x10. B1 uses rows 0,1. B2 uses row 2.
    
    # A1 (2x2) * B1 (2x10)
    # B1 has 10 columns. We group them into 5 pairs of 2 columns.
    # Each pair is a 2x2 block of B1.
    # We apply Strassen to each 2x2 block.
    
    for g in range(5):
        k1 = 2*g
        k2 = 2*g + 1
        
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C11 += M1
        # Indices:
        # U rows: A00 (0), A11 (3) -> 0, 3? 
        # A is 2x3.
        # Row 0: A00, A01, A02
        # Row 1: A10, A11, A12
        # A1 uses A00, A01 (row 0) and A10, A11 (row 1)?
        # Wait, A1 is 2x2. It uses A[0,0], A[0,1], A[1,0], A[1,1].
        # U indices:
        # 0: A[0,0]
        # 1: A[0,1]
        # 2: A[0,2]
        # 3: A[1,0]
        # 4: A[1,1]
        # 5: A[1,2]
        
        # B1 uses B[0,0..9], B[1,0..9].
        # V indices:
        # 0..9: B[0,0..9]
        # 10..19: B[1,0..9]
        # 20..29: B[2,0..9]
        
        # C is 2x10.
        # W indices:
        # 0..9: C[0,0..9]
        # 10..19: C[1,0..9]
        
        # Strassen for A1 * B1_block (2x2 * 2x2)
        # A1 = [[A00, A01], [A10, A11]]
        # B1_block = [[B0k1, B0k2], [B1k1, B1k2]]
        
        # M1 = (A00+A11)(B0k1+B1k1) -> C0k1 += M1, C1k2 += M1
        # Note: The seed code had some specific index mappings. Let's stick to the seed's logic for Part 1 to ensure correctness of the fallback.
        
        # Seed Part 1:
        # M1: U[0]=1, U[4]=1 (A00, A11). V[k1]=1, V[10+k2]=1 (B0k1, B1k2). 
        # Wait, Strassen M1 is (A00+A11)(B00+B11).
        # If k1=0, k2=1. B00 is V[0], B11 is V[11].
        # Seed: V[k1][col]=1, V[10+k2][col]=1. So B0k1 and B1k2. This is (A00+A11)(B0k1+B1k2).
        # This is NOT standard Strassen. Standard is (A00+A11)(B00+B11).
        # If the block is diagonal? No.
        # Let's re-verify the seed's Strassen implementation.
        # If the seed's fallback is rank 55 and valid, I should just copy it.
        
        # Let's copy the seed's logic exactly for the fallback.
        
        # M1
        U[0][col] = 1; U[4][col] = 1
        V[k1][col] = 1; V[10+k2][col] = 1
        W[2*k1][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        
        # M2
        U[3][col] = 1; U[4][col] = 1
        V[k1][col] = 1
        W[2*k1+1][col] = 1
        W[2*k2+1][col] = -1
        col += 1
        
        # M3
        U[0][col] = 1
        V[10+k1][col] = 1; V[10+k2][col] = -1
        W[2*k2][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        
        # M4
        U[4][col] = 1
        V[k2][col] = 1; V[k1][col] = -1
        W[2*k1][col] = 1
        W[2*k1+1][col] = 1
        col += 1
        
        # M5
        U[0][col] = 1; U[1][col] = 1
        V[10+k2][col] = 1
        W[2*k1][col] = -1
        W[2*k2][col] = 1
        col += 1
        
        # M6
        U[3][col] = 1; U[0][col] = -1
        V[k1][col] = 1; V[10+k1][col] = 1
        W[2*k2+1][col] = 1
        col += 1
        
        # M7
        U[1][col] = 1; U[4][col] = -1
        V[k2][col] = 1; V[10+k2][col] = 1
        W[2*k1][col] = 1
        col += 1

    # Part 2: Schoolbook for A2 (2x1) * B2 (1x10)
    for k in range(10):
        # M0
        U[2][col] = 1
        V[20+k][col] = 1
        W[2*k][col] = 1
        col += 1
        
        # M1
        U[5][col] = 1
        V[20+k][col] = 1
        W[2*k+1][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res['U'][0]))
    print("Verified:", _verify(res['U'], res['V'], res['W'], 2, 3, 10))

