import time
import sys
import numpy as np

def _verify(U, V, W, n=3, m=3, p=6):
    """Verify that the factorization exactly decomposes the matrix multiplication tensor.
    Returns True if valid, False otherwise."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    rm, rank = U.shape
    sm, rank2 = V.shape
    tm, rank3 = W.shape
    
    if rank != rank2 or rank != rank3:
        return False
    if rm != n * m:
        return False
    if sm != m * p:
        return False
    if tm != p * n:
        return False
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Check on-diagonal entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    
    # Check off-diagonal entries
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    val = np.sum(U[a, :] * V[b, :] * W[c, :])
                    if abs(val) > 1e-9:
                        return False
    
    return True


def _compute_error(U, V, W, n=3, m=3, p=6):
    """Compute the total squared error of the tensor decomposition."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the decomposed tensor
    # T[a,b,c] = sum_d U[a,d] * V[b,d] * W[c,d]
    T = np.einsum('ar,br,cr->abc', U, V, W)
    
    err = 0.0
    
    # On-diagonal: should be 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                err += (T[a, b, c] - 1.0) ** 2
    
    # Off-diagonal: should be 0
    for a in range(nm):
        i_a = a // m
        j_a = a % m
        for b in range(mp):
            j_b = b // p
            k_b = b % p
            for c in range(pn):
                k_c = c // n
                i_c = c % n
                if not (j_a == j_b and k_b == k_c and i_c == i_a):
                    err += T[a, b, c] ** 2
    
    return err


def _try_structured_decomposition(n, m, p, rank, deadline):
    """Try to find a low-rank decomposition using structured patterns.
    
    Key idea: The 3x3x6 tensor has structure. We can exploit:
    1. Block structure: split B into two 3x3 blocks
    2. Symmetries in the index patterns
    3. Known low-rank constructions for small matrices
    
    We use a parameterized family of decompositions and optimize over the parameters.
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Strategy: Use the fact that C[i,k] = sum_j A[i,j]*B[j,k]
    # We can write this as a sum of rank-1 terms in the tensor sense.
    # 
    # For each "pattern" of indices, we create a term.
    # 
    # Let's try a decomposition based on the following:
    # For the 3x3 core, use a rank-19 style decomposition (Carmaker)
    # But that's complex. Instead, let's try to find a decomposition
    # that exploits the rectangular structure.
    #
    # Alternative: Use the identity that for 3x3x3, we can use
    # rank 19 (Carmaker 1986). For 3x3x6, we can potentially do better
    # by sharing structure.
    #
    # Let's try: decompose B as [B1 | B2] where B1, B2 are 3x3
    # C = A*B1 + A*B2 (concatenated)
    # If we have a rank-r decomposition for A*B1 and A*B2 separately,
    # total rank is 2r. But we can share the A-factor.
    #
    # Actually, let's try a different approach:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # 
    # Define: for each j, let u_j be the j-th column of A (as a vector in R^3)
    # and v_j be the j-th row of B (as a vector in R^6)
    # Then C = sum_j u_j \otimes v_j (outer product giving 3x6 matrix)
    # 
    # In tensor form: T = sum_j (e_u_j \otimes e_v_j \otimes e_w)
    # where e_w depends on the output indices.
    #
    # This is just the schoolbook with rank = m = 3? No, that's not right.
    # The issue is that the output index (i,k) depends on both i and k,
    # and the input indices are (i,j) and (j,k).
    #
    # Let me think about this more carefully.
    # 
    # The tensor T has T[a,b,c] = 1 iff a=i*m+j, b=j*p+k, c=k*n+i
    # 
    # For n=3, m=3, p=6:
    # a in [0,9), b in [0,18), c in [0,18)
    # 
    # The "on" entries are determined by (i,j,k) with i in [0,3), j in [0,3), k in [0,6)
    # That's 3*3*6 = 54 entries.
    #
    # A rank-r decomposition means we find vectors U_c, V_c, W_c for c=0..r-1
    # such that T = sum_c U_c \otimes V_c \otimes W_c
    #
    # This is equivalent to: for all (a,b,c), sum_d U[a,d]*V[b,d]*W[c,d] = T[a,b,c]
    #
    # Let's try to find such a decomposition with r < 54.
    #
    # Key insight: The tensor has a "matrix multiplication" structure.
    # We can use known results from algebraic complexity theory.
    #
    # For 3x3x3, the minimum rank is 19 (Carmaker 1986).
    # For 3x3x6, we can view it as two 3x3x3 tensors sharing the A-factor.
    # If we use the same U-factors for both, we save rank.
    #
    # Specifically: 
    # C1 = A * B1 (3x3 result)
    # C2 = A * B2 (3x3 result)
    # 
    # If we have U_1, V_1, W_1 for C1 = A*B1 with rank r1
    # and U_2, V_2, W_2 for C2 = A*B2 with rank r2
    # where U_1 and U_2 share some structure, we can combine.
    #
    # But this is getting complicated. Let's just try numerical optimization
    # with a better initialization and more careful local search.

    # Parameterized family:
    # U[i*m+j, c] = f_c(i, j)
    # V[j*p+k, c] = g_c(j, k)
    # W[k*n+i, c] = h_c(k, i)
    #
    # We can parameterize f, g, h as polynomials or linear combinations
    # of basis functions.
    
    # Let's try: f_c(i,j) = sum_{a,b} alpha_c[a,b] * i^a * j^b
    # with a, b in {0, 1, 2} (since i, j in {0,1,2})
    # That's 9 parameters per c.
    
    # Similarly for g and h.
    
    # Total parameters: rank * 9 * 3 = 27*rank
    # For rank=27, that's 729 parameters. Too many for random search.
    
    # Let's try a simpler parameterization:
    # f_c(i,j) = sum_{a=0}^{2} sum_{b=0}^{2} alpha_c[a,b] * (i-a)*(j-b) / ((i-a+0.5)*(j-b+0.5))
    # This is a rational function that's 1 at (i,j)=(a,b) and 0 elsewhere.
    
    # Actually, let's use Lagrange polynomials:
    # f_c(i,j) = sum_{a=0}^{2} sum_{b=0}^{2} alpha_c[a,b] * L_a(i) * L_b(j)
    # where L_a(i) is the Lagrange polynomial that's 1 at i=a and 0 at i=0,1,2 (a != i)
    
    # For i in {0,1,2}:
    # L_0(i) = (i-1)(i-2)/((0-1)(0-2)) = (i-1)(i-2)/2
    # L_1(i) = (i-0)(i-2)/((1-0)(1-2)) = i(i-2)/(-1) = -i(i-2) = i(2-i)
    # L_2(i) = (i-0)(i-1)/((2-0)(2-1)) = i(i-1)/2
    
    # So f_c(i,j) = sum_{a,b} alpha_c[a,b] * L_a(i) * L_b(j)
    # This is a 9-parameter family per c.
    
    # Let's try this approach with random alpha values and local search.
    
    def lagrange_3(i, a):
        """Lagrange polynomial L_a(i) for i in {0,1,2}, a in {0,1,2}"""
        if a == 0:
            return (i - 1) * (i - 2) / 2.0
        elif a == 1:
            return i * (2 - i)
        else:  # a == 2
            return i * (i - 1) / 2.0
    
    # Precompute Lagrange values
    L = np.zeros((3, 3))
    for i in range(3):
        for a in range(3):
            L[i, a] = lagrange_3(i, a)
    
    # Now, U[i*m+j, c] = sum_{a=0}^{2} sum_{b=0}^{2} alpha_c[a,b] * L[i,a] * L[j,b]
    # = (L @ alpha_c @ L.T)[i, j] where alpha_c is 3x3
    
    # Similarly for V and W.
    
    # Let's parameterize:
    # alpha_c is 3x3 for U, beta_c is 3x6 for V (since k in {0..5}), gamma_c is 6x3 for W
    
    # Wait, for V[j*p+k, c] = g_c(j, k) where j in {0,1,2}, k in {0..5}
    # g_c(j,k) = sum_{a=0}^{2} sum_{b=0}^{5} beta_c[a,b] * L_j[a] * M_k[b]
    # where M_k[b] is Lagrange for k in {0..5}, b in {0..5}
    
    # That's 3*6 = 18 parameters per c for V.
    # And 6*3 = 18 parameters per c for W.
    # And 3*3 = 9 parameters per c for U.
    # Total: 45 parameters per c.
    
    # For rank r, total parameters = 45*r.
    # For r=20, that's 900 parameters. Still too many.
    
    # Let's simplify: use a smaller parameterization.
    # 
    # Alternative: Use the fact that the tensor is "separable" in some sense.
    # 
    # Let's just do random initialization with integer/half-integer values
    # and use a more efficient local search.
    
    import random
    
    def random_entry(rng, scale=2):
        """Generate a random entry from a small set of values."""
        options = [-scale, -scale+0.5, -scale+1, -scale+1.5,
                   -scale+2, -scale+2.5,
                   -1.5, -1, -0.5, 0, 0.5, 1, 1.5,
                   scale-1.5, scale-1, scale-0.5, scale]
        return rng.choice(options)
    
    best_err = float('inf')
    best_U = None
    best_V = None
    best_W = None
    
    for attempt in range(20):
        if time.time() > deadline - 10:
            break
        
        rng = random.Random(attempt * 12345 + 678)
        
        U_cand = np.array([[random_entry(rng) for _ in range(rank)] for _ in range(nm)])
        V_cand = np.array([[random_entry(rng) for _ in range(rank)] for _ in range(mp)])
        W_cand = np.array([[random_entry(rng) for _ in range(rank)] for _ in range(pn)])
        
        err = _compute_error(U_cand, V_cand, W_cand, n, m, p)
        
        if err < best_err:
            best_err = err
            best_U = U_cand.copy()
            best_V = V_cand.copy()
            best_W = W_cand.copy()
        
        if err < 1e-9:
            break
        
        # Local search: try to improve by changing one entry at a time
        for step in range(100):
            if time.time() > deadline - 10:
                break
            
            # Pick a random matrix, row, column
            mat_idx = rng.randint(0, 2)
            if mat_idx == 0:
                mat = U_cand
                nrows = nm
            elif mat_idx == 1:
                mat = V_cand
                nrows = mp
            else:
                mat = W_cand
                nrows = pn
            
            row = rng.randint(0, nrows - 1)
            col = rng.randint(0, rank - 1)
            old_val = mat[row, col]
            
            # Try neighbor values
            candidates = set()
            for delta in [-2, -1.5, -1, -0.5, 0.5, 1, 1.5, 2]:
                candidates.add(old_val + delta)
            candidates.add(0)
            candidates.add(0.5)
            candidates.add(-0.5)
            candidates.add(1)
            candidates.add(-1)
            
            best_cand_err = err
            best_cand_val = old_val
            for cand in candidates:
                mat[row, col] = cand
                new_err = _compute_error(U_cand, V_cand, W_cand, n, m, p)
                if new_err < best_cand_err:
                    best_cand_err = new_err
                    best_cand_val = cand
                mat[row, col] = old_val
            
            if best_cand_val != old_val:
                mat[row, col] = best_cand_val
                err = best_cand_err
                if err < best_err:
                    best_err = err
                    best_U = U_cand.copy()
                    best_V = V_cand.copy()
                    best_W = W_cand.copy()
                if err < 1e-9:
                    break
        
        if best_err < 1e-9:
            break
    
    if best_err < 1e-9:
        return best_U, best_V, best_W, best_err
    
    return None, None, None, best_err


def construct():
    """Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns: {'U', 'V', 'W'}: factor matrices with integer/half-integer entries.
    """
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    n, m, p = 3, 3, 6
    nm, mp, pn = n * m, m * p, p * n  # 9, 18, 18
    
    # Start with schoolbook as baseline
    rank_schoolbook = n * m * p  # 54
    U = np.zeros((nm, rank_schoolbook))
    V = np.zeros((mp, rank_schoolbook))
    W = np.zeros((pn, rank_schoolbook))
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank_schoolbook
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve with structured decomposition search
    # Try ranks from low to high
    ranks_to_try = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 32, 36, 40, 45, 50]
    
    for r in sorted(ranks_to_try):
        if time.time() > deadline - 20:
            break
        if r >= best_rank:
            continue
        
        print(f"Trying rank {r}...", file=sys.stderr)
        U_cand, V_cand, W_cand, err = _try_structured_decomposition(n, m, p, r, deadline)
        
        if U_cand is not None and err < 1e-9:
            # Verify
            if _verify(U_cand, V_cand, W_cand, n, m, p):
                print(f"Found valid rank {r}!", file=sys.stderr)
                best_rank = r
                best_result = {
                    "U": U_cand.tolist(),
                    "V": V_cand.tolist(),
                    "W": W_cand.tolist()
                }
                break  # Found a valid lower rank, stop searching
            else:
                print(f"Rank {r}: error {err} but verification failed", file=sys.stderr)
        else:
            print(f"Rank {r}: no solution found (best err: {err:.6f})", file=sys.stderr)
    
    return best_result
