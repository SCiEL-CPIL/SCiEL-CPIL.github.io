import numpy as np
import time
import itertools
from functools import lru_cache

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank_max = n * m * p  # 72
    deadline = time.time() + 145  # safety margin

    # Schoolbook fallback
    U_fb = [[0] * rank_max for _ in range(n * m)]
    V_fb = [[0] * rank_max for _ in range(m * p)]
    W_fb = [[0] * rank_max for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i * m + j][col] = 1
                V_fb[j * p + k][col] = 1
                W_fb[k * n + i][col] = 1
                col += 1

    best_result = {"U": U_fb, "V": V_fb, "W": W_fb}
    best_rank = rank_max

    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition is exact."""
        r = U.shape[1]
        U_t = np.array(U).reshape(n, m, r)
        V_t = np.array(V).reshape(m, p, r)
        W_t = np.array(W).reshape(p, n, r)
        R = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.allclose(R, np.ones((n, m, p)), atol=tol)

    def compute_residual(U, V, W):
        U_t = np.array(U).reshape(n, m, -1)
        V_t = np.array(V).reshape(m, p, -1)
        W_t = np.array(W).reshape(p, n, -1)
        R = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
        return np.ones((n, m, p)) - R

    def norm_sq(res):
        return np.sum(res * res)

    # Finite field arithmetic for Gröbner-like search
    def modinv(a, p):
        return pow(a, p - 2, p)

    def gf_solve(A, b, p):
        """Solve linear system over finite field GF(p). Returns None if no unique solution."""
        A = np.array(A, dtype=object)
        b = np.array(b, dtype=object)
        n_rows = A.shape[0]
        n_cols = A.shape[1]
        
        # Gaussian elimination
        M = np.column_stack([A, b])
        rank = 0
        for col in range(n_cols):
            # Find pivot
            pivot = -1
            for row in range(rank, n_rows):
                if M[row, col] % p != 0:
                    pivot = row
                    break
            if pivot == -1:
                continue
            # Swap rows
            if pivot != rank:
                M[[rank, pivot]] = M[[pivot, rank]]
            # Normalize pivot row
            inv = modinv(int(M[rank, col]), p)
            M[rank, :] = (M[rank, :] * inv) % p
            # Eliminate other rows
            for row in range(n_rows):
                if row != rank:
                    factor = int(M[row, col])
                    if factor != 0:
                        M[row, :] = (M[row, :] - factor * M[rank, :]) % p
            rank += 1
        
        # Check for consistency
        for row in range(rank, n_rows):
            if M[row, n_cols] % p != 0:
                return None
        
        # Extract solution (assuming unique solution)
        if rank < n_cols:
            # Underdetermined - try to find a simple solution
            # Set free variables to 0
            x = [0] * n_cols
            for i in range(rank):
                # Find the column that is the pivot for row i
                for j in range(n_cols):
                    if M[i, j] % p != 0:
                        x[j] = int(M[i, n_cols])
                        break
            return x
        else:
            x = [int(M[i, n_cols]) for i in range(n_cols)]
            return x

    # Try to find decomposition using finite field search
    def search_gf(rank, primes):
        """Search for decomposition over finite fields, then lift to rationals."""
        for prime in primes:
            if time.time() > deadline - 5:
                break
            
            # We'll use a randomized approach: try random sparse vectors over GF(prime)
            # and check if they satisfy the tensor equations.
            
            max_attempts = 50
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Initialize with random sparse vectors
                U = np.zeros((n * m, rank), dtype=object)
                V = np.zeros((m * p, rank), dtype=object)
                W = np.zeros((p * n, rank), dtype=object)
                
                for c in range(rank):
                    # Random sparse support
                    nz_u = np.random.choice(n * m, size=min(np.random.randint(1, 4), n * m), replace=False)
                    nz_v = np.random.choice(m * p, size=min(np.random.randint(1, 4), m * p), replace=False)
                    nz_w = np.random.choice(p * n, size=min(np.random.randint(1, 4), p * n), replace=False)
                    
                    for idx in nz_u:
                        U[idx, c] = np.random.randint(1, prime)
                    for idx in nz_v:
                        V[idx, c] = np.random.randint(1, prime)
                    for idx in nz_w:
                        W[idx, c] = np.random.randint(1, prime)
                
                # Check if this satisfies the tensor equations mod prime
                # R[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] mod prime
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0
                            for c in range(rank):
                                s = (s + int(U[i*m+j, c]) * int(V[j*p+k, c]) * int(W[k*n+i, c])) % prime
                            if s % prime != 1:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    # Found a solution mod prime! Now try to lift to rationals.
                    # For small primes, the entries are already small integers.
                    U_int = np.array([[int(U[i, c]) for c in range(rank)] for i in range(n * m)])
                    V_int = np.array([[int(V[i, c]) for c in range(rank)] for i in range(m * p)])
                    W_int = np.array([[int(W[i, c]) for c in range(rank)] for i in range(p * n)])
                    
                    # Verify over reals
                    if verify(U_int, V_int, W_int):
                        return U_int, V_int, W_int, rank
                    else:
                        # The mod p solution doesn't lift directly. Try to find a rational solution.
                        # This is complex; for now, just return the integer solution if it's close.
                        residual = compute_residual(U_int, V_int, W_int)
                        if norm_sq(residual) < 1e-6:
                            # Round to nearest half-integer
                            U_round = np.round(U_int * 2) / 2
                            V_round = np.round(V_int * 2) / 2
                            W_round = np.round(W_int * 2) / 2
                            if verify(U_round, V_round, W_round):
                                return U_round, V_round, W_round, rank
                else:
                    # Not valid mod prime. Try to fix by solving linear equations.
                    # For each (i,j,k), we need: sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] = 1
                    # This is bilinear, so we can't directly use linear algebra.
                    # Skip for now.
                    pass
        
        return None

    # Try ranks from 24 upward
    primes_to_try = [97, 101, 103, 107, 109, 113]
    
    for r in range(24, 41):
        if time.time() > deadline - 10:
            break
        
        # Try finite field search
        result = search_gf(r, primes_to_try)
        if result is not None:
            U_found, V_found, W_found, rank_found = result
            best_result = {
                "U": U_found.tolist(),
                "V": V_found.tolist(),
                "W": W_found.tolist()
            }
            best_rank = rank_found
            break
    
    # If GF search didn't find anything, try a local search approach
    if best_rank == rank_max:
        np.random.seed(42)
        
        for r in range(24, 41):
            if time.time() > deadline - 5:
                break
            
            max_attempts = 100
            for attempt in range(max_attempts):
                if time.time() > deadline - 5:
                    break
                
                # Initialize with random sparse values from {-1, -0.5, 0.5, 1}
                U_c = np.zeros((n * m, r))
                V_c = np.zeros((m * p, r))
                W_c = np.zeros((p * n, r))
                
                for c in range(r):
                    nz_u = np.random.choice(n * m, size=np.random.randint(2, 5), replace=False)
                    nz_v = np.random.choice(m * p, size=np.random.randint(2, 5), replace=False)
                    nz_w = np.random.choice(p * n, size=np.random.randint(2, 5), replace=False)
                    
                    U_c[nz_u, c] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nz_u))
                    V_c[nz_v, c] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nz_v))
                    W_c[nz_w, c] = np.random.choice([-1, -0.5, 0.5, 1], size=len(nz_w))
                
                current_res = compute_residual(U_c, V_c, W_c)
                current_norm = norm_sq(current_res)
                
                if current_norm < 1e-9:
                    best_result = {
                        "U": U_c.tolist(),
                        "V": V_c.tolist(),
                        "W": W_c.tolist()
                    }
                    best_rank = r
                    break
                
                # Local search
                improved = True
                while improved and time.time() < deadline - 2:
                    improved = False
                    c = np.random.randint(r)
                    
                    old_U = U_c[:, c].copy()
                    old_V = V_c[:, c].copy()
                    old_W = W_c[:, c].copy()
                    
                    # Perturb with half-integer values
                    U_c[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=n * m)
                    V_c[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=m * p)
                    W_c[:, c] = np.random.choice([-1, -0.5, 0, 0.5, 1], size=p * n)
                    
                    new_res = compute_residual(U_c, V_c, W_c)
                    new_norm = norm_sq(new_res)
                    
                    if new_norm < current_norm:
                        current_norm = new_norm
                        current_res = new_res
                        improved = True
                    else:
                        U_c[:, c] = old_U
                        V_c[:, c] = old_V
                        W_c[:, c] = old_W
                    
                    if current_norm < 1e-9:
                        best_result = {
                            "U": U_c.tolist(),
                            "V": V_c.tolist(),
                            "W": W_c.tolist()
                        }
                        best_rank = r
                        break
            
            if best_rank < rank_max:
                break
    
    return best_result
# EVOLVE-BLOCK-END
