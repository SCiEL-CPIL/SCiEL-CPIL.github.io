"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
import random
import itertools
from fractions import Fraction
import numpy as np

def _build_target_tensor(n, m, p):
    """Build the target multiplication tensor as a dict of non-zero entries."""
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[(a, b, c)] = 1
    return T

def _verify_decomposition(U, V, W, n, m, p, tol=1e-10):
    """Verify that U, V, W exactly decompose the multiplication tensor."""
    T = _build_target_tensor(n, m, p)
    rank = len(U[0]) if U else 0
    for (a, b, c) in T:
        s = 0
        for r in range(rank):
            s += U[a][r] * V[b][r] * W[c][r]
        if abs(s - 1) > tol:
            return False
    nm, mp, pn = n * m, m * p, p * n
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if (a, b, c) not in T:
                    s = 0
                    for r in range(rank):
                        s += U[a][r] * V[b][r] * W[c][r]
                    if abs(s) > tol:
                        return False
    return True

def _schoolbook(n, m, p):
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

def _compute_rational_rank(matrix):
    """Compute the exact rank of a matrix over the rationals using Fraction."""
    M = [[Fraction(x) for x in row] for row in matrix]
    rows = len(M)
    if rows == 0:
        return 0
    cols = len(M[0])
    
    rank = 0
    for col in range(cols):
        # Find pivot row
        pivot = -1
        for row in range(rank, rows):
            if M[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap rows
        M[rank], M[pivot] = M[pivot], M[rank]
        
        # Eliminate below
        for row in range(rank + 1, rows):
            factor = M[row][col] / M[rank][col]
            for c in range(col, cols):
                M[row][c] -= factor * M[rank][c]
        
        rank += 1
    
    return rank

def _rank_factorization_decomposition(n, m, p, deadline=None):
    """
    Use rank factorization technique:
    1. Flatten tensor into matrix A of shape (n*m) x (m*p)
    2. Compute exact rational rank r
    3. Construct decomposition using that rank
    """
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the flattening matrix: A[a, b] where a = i*m+j, b = j*p+k
    # A[i*m+j, j*p+k] = 1 if the (i,j,k) term exists, else 0
    # Actually, for matrix multiplication C = A*B, the tensor is:
    # T[i,j,k] = A[i,j] * B[j,k]
    # Flattened: for each (i,j,k), we have a term at position (i*m+j, j*p+k, k*n+i)
    
    # The matrix flattening: M[a, b] where a indexes (i,j) and b indexes (j,k)
    # M[i*m+j, j*p+k] = 1 if this is a valid multiplication term
    # Note: j must be the same in both indices
    
    M = [[0] * mp for _ in range(nm)]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                M[a][b] = 1
    
    # Compute exact rank over rationals
    r = _compute_rational_rank(M)
    
    if deadline and time.time() > deadline:
        return None
    
    # Now construct the decomposition
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # sum_r U[a,r] * V[b,r] * W[c,r] = T[a,b,c]
    
    # Strategy: Use the rank factorization M = U * V^T where rank(M) = r
    # Then for each (a,b) with M[a,b] != 0, we have T[a,b,c] = 1 for c = k*n+i
    # where k is determined by b = j*p+k
    
    # Get the rank factorization via row reduction
    M_frac = [[Fraction(x) for x in row] for row in M]
    rows = len(M_frac)
    cols = len(M_frac[0])
    
    # Row reduce and track row operations
    pivot_cols = []
    pivot_rows = []
    row_ops = [[Fraction(0)] * rows for _ in range(rows)]  # row_ops[i] shows how row i is formed from original rows
    
    for i in range(rows):
        row_ops[i][i] = 1
    
    current_rank = 0
    for col in range(cols):
        # Find pivot
        pivot = -1
        for row in range(current_rank, rows):
            if M_frac[row][col] != 0:
                pivot = row
                break
        if pivot == -1:
            continue
        
        # Swap
        M_frac[current_rank], M_frac[pivot] = M_frac[pivot], M_frac[current_rank]
        row_ops[current_rank], row_ops[pivot] = row_ops[pivot], row_ops[current_rank]
        
        # Eliminate
        for row in range(current_rank + 1, rows):
            if M_frac[row][col] != 0:
                factor = M_frac[row][col] / M_frac[current_rank][col]
                for c in range(col, cols):
                    M_frac[row][c] -= factor * M_frac[current_rank][c]
                for c in range(rows):
                    row_ops[row][c] -= factor * row_ops[current_rank][c]
        
        pivot_cols.append(col)
        pivot_rows.append(current_rank)
        current_rank += 1
    
    r = current_rank
    if r == 0:
        return None
    
    # U matrix: nm x r, where U[a, i] = M_frac[pivot_rows[i], a] for the pivot rows
    # Actually, we need to be more careful. Let's use a different approach.
    
    # Simpler approach: Use numpy SVD to get a numerical rank factorization,
    # then snap to rational values.
    
    M_np = np.array(M, dtype=float)
    U_np, s_vals, Vt_np = np.linalg.svd(M_np, full_matrices=False)
    
    # Determine numerical rank
    tol_svd = 1e-10 * max(s_vals) if len(s_vals) > 0 else 1e-10
    num_rank = sum(1 for s in s_vals if s > tol_svd)
    
    if num_rank == 0:
        return None
    
    # Use the numerical rank
    r_num = num_rank
    
    # Get the factorization: M ≈ U_s * diag(s) * Vt_s
    U_s = U_np[:, :r_num]
    s_s = s_vals[:r_num]
    Vt_s = Vt_np[:r_num, :]
    
    # M ≈ (U_s * diag(s_s)) * Vt_s
    # Let U_mat = U_s * diag(s_s) and V_mat = Vt_s
    # Then M ≈ U_mat @ V_mat
    
    U_mat = U_s * s_s[np.newaxis, :]  # nm x r_num
    V_mat = Vt_s  # r_num x mp
    
    # Now we need to construct W such that the full tensor matches
    # For each (a, b, c) in the tensor:
    # sum_r U_mat[a, r] * V_mat[r, b] * W[c, r] should equal T[a,b,c]
    
    # The issue is that W depends on c, and c = k*n + i where i comes from a and k comes from b
    # This is not straightforward with a simple rank factorization
    
    # Alternative: Use the exact rational rank and construct a decomposition
    # based on the pivot structure
    
    # Let's try a direct construction using the pivot columns
    # For each pivot column col_i, we have a basis vector
    
    # Actually, let's use a simpler approach:
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1
    # We can write this as a sum over "modes"
    
    # For 2x3*3x6, let's try to find a structured decomposition
    
    # Try ranks from the computed lower bound up
    for r_try in range(r, n * m * p + 1):
        if deadline and time.time() > deadline:
            break
        
        # Try to construct a decomposition of rank r_try
        sol = _construct_rank_r(n, m, p, r_try, deadline)
        if sol is not None:
            return sol
    
    return None

def _construct_rank_r(n, m, p, r, deadline=None):
    """Try to construct a rank-r decomposition using structured patterns."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Try different structured patterns
    for seed in range(50):
        if deadline and time.time() > deadline:
            return None
        
        random.seed(seed)
        
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        
        # Generate random structured patterns
        for idx in range(r):
            # Choose a random (i, j, k) pattern
            i = random.randint(0, n - 1)
            j = random.randint(0, m - 1)
            k = random.randint(0, p - 1)
            
            a = i * m + j
            b = j * p + k
            c = k * n + i
            
            # Assign values
            val_u = random.choice([0, 0.5, 1, 1.5, 2])
            val_v = random.choice([0, 0.5, 1, 1.5, 2])
            val_w = random.choice([0, 0.5, 1, 1.5, 2])
            
            U[a][idx] = val_u
            V[b][idx] = val_v
            W[c][idx] = val_w
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_binary_search(r, n, m, p, deadline=None):
    """Search for binary (0/1) decomposition using combinatorial patterns."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    # Try structured binary patterns based on index relationships
    for pattern_type in range(5):
        if deadline and time.time() > deadline:
            return None
        
        U = [[0] * r for _ in range(nm)]
        V = [[0] * r for _ in range(mp)]
        W = [[0] * r for _ in range(pn)]
        
        for idx in range(r):
            if deadline and time.time() > deadline:
                return None
            
            # Different pattern generators
            if pattern_type == 0:
                # Simple modular pattern
                i = idx % n
                j = (idx // n) % m
                k = (idx // (n * m)) % p
            elif pattern_type == 1:
                # Interleaved pattern
                i = (idx * 7) % n
                j = (idx * 11) % m
                k = (idx * 13) % p
            elif pattern_type == 2:
                # Diagonal pattern
                i = idx % n
                j = (idx // n) % m
                k = (idx + i) % p
            elif pattern_type == 3:
                # Cross pattern
                i = idx % n
                j = (idx % m)
                k = (idx // (n * m)) % p
            else:
                # Random with seed
                random.seed(idx * 12345 + 67890)
                i = random.randint(0, n - 1)
                j = random.randint(0, m - 1)
                k = random.randint(0, p - 1)
            
            a = i * m + j
            b = j * p + k
            c = k * n + i
            U[a][idx] = 1
            V[b][idx] = 1
            W[c][idx] = 1
        
        if _verify_decomposition(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    return None

def _try_half_integer_search(r, n, m, p, deadline=None):
    """Search for half-integer decomposition using local refinement."""
    nm, mp, pn = n * m, m * p, p * n
    T = _build_target_tensor(n, m, p)
    
    for attempt in range(30):
        if deadline and time.time() > deadline:
            return None
        
        # Initialize with random half-integers
        U = [[random.choice([0, 0.5, 1, 1.5]) for _ in range(r)] for _ in range(nm)]
        V = [[random.choice([0, 0.5, 1, 1.5]) for _ in range(r)] for _ in range(mp)]
        W = [[random.choice([0, 0.5, 1, 1.5]) for _ in range(r)] for _ in range(pn)]
        
        # Local search to refine
        best_res = 999
        for it in range(100):
            if deadline and time.time() > deadline:
                return None
            
            # Compute residual
            max_res = 0
            for (a, b, c) in T:
                s = 0
                for rr in range(r):
                    s += U[a][rr] * V[b][rr] * W[c][rr]
                max_res = max(max_res, abs(s - 1))
            for _ in range(10):
                a = random.randint(0, nm - 1)
                b = random.randint(0, mp - 1)
                c = random.randint(0, pn - 1)
                if (a, b, c) not in T:
                    s = 0
                    for rr in range(r):
                        s += U[a][rr] * V[b][rr] * W[c][rr]
                    max_res = max(max_res, abs(s))
            
            if max_res < best_res:
                best_res = max_res
                if best_res == 0:
                    return {"U": U, "V": V, "W": W}
            
            # Perturb one entry
            mat = random.choice(['U', 'V', 'W'])
            if mat == 'U':
                row = random.randint(0, nm - 1)
                arr = U
            elif mat == 'V':
                row = random.randint(0, mp - 1)
                arr = V
            else:
                row = random.randint(0, pn - 1)
                arr = W
            col = random.randint(0, r - 1)
            old_val = arr[row][col]
            new_val = old_val + random.choice([-0.5, 0.5, -1, 1])
            arr[row][col] = new_val
        
        if best_res == 0:
            return {"U": U, "V": V, "W": W}
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try rank factorization approach first
    sol = _rank_factorization_decomposition(n, m, p, deadline)
    if sol is not None:
        best = sol
        best_rank = len(sol["U"][0])
        return best
    
    # Try ranks from lower bound up
    for r in range(18, best_rank + 1):
        if time.time() > deadline:
            break
        
        # First try binary patterns
        sol = _try_binary_search(r, n, m, p, deadline)
        if sol is not None:
            best = sol
            best_rank = r
            break
        
        # Then try half-integer search
        if time.time() < deadline - 10:
            sol = _try_half_integer_search(r, n, m, p, deadline)
            if sol is not None:
                best = sol
                best_rank = r
                break
    
    return best
# EVOLVE-BLOCK-END
