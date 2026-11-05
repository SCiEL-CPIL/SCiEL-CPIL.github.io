import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [2, 3, 4]
_START_TIME = time.time()
TIME_LIMIT = 140.0

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    total = n*m*p

    if time.time() - _START_TIME > TIME_LIMIT - 5:
        rank = total
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        return {"U": U, "V": V, "W": W}

    # Build the 24-dimensional schoolbook solution as baseline
    def build_schoolbook():
        rank = total
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1
        return U, V, W, rank

    # Build the 22-rank Strassen-based solution
    def build_strassen():
        rank = 22
        U = [[0]*rank for _ in range(nm)]
        V = [[0]*rank for _ in range(mp)]
        W = [[0]*rank for _ in range(pn)]
        
        def add_term(u_indices, v_indices, w_indices, col):
            for idx, val in u_indices:
                U[idx][col] += val
            for idx, val in v_indices:
                V[idx][col] += val
            for idx, val in w_indices:
                W[idx][col] += val

        # Block 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2]
        add_term([(0, 1), (4, 1)], [(0, 1), (5, 1)], [(0, 1), (3, 1)], 0)
        add_term([(3, 1), (4, 1)], [(0, 1)], [(1, 1), (3, -1)], 1)
        add_term([(0, 1)], [(1, 1), (5, -1)], [(2, 1), (3, 1)], 2)
        add_term([(4, 1)], [(4, 1), (0, -1)], [(0, 1), (1, 1)], 3)
        add_term([(0, 1), (1, 1)], [(5, 1)], [(0, -1), (2, 1)], 4)
        add_term([(3, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)], 5)
        add_term([(1, 1), (4, -1)], [(4, 1), (5, 1)], [(0, 1)], 6)

        # Block 2: Strassen for A[0:2, 0:2] * B[0:2, 2:4]
        add_term([(0, 1), (4, 1)], [(2, 1), (7, 1)], [(4, 1), (7, 1)], 7)
        add_term([(3, 1), (4, 1)], [(2, 1)], [(5, 1), (7, -1)], 8)
        add_term([(0, 1)], [(3, 1), (7, -1)], [(6, 1), (7, 1)], 9)
        add_term([(4, 1)], [(6, 1), (2, -1)], [(4, 1), (5, 1)], 10)
        add_term([(0, 1), (1, 1)], [(7, 1)], [(4, -1), (6, 1)], 11)
        add_term([(3, 1), (0, -1)], [(2, 1), (3, 1)], [(7, 1)], 12)
        add_term([(1, 1), (4, -1)], [(6, 1), (7, 1)], [(4, 1)], 13)

        # Outer products for A[0:2, 2:3] * B[2:3, 0:4]
        add_term([(2, 1)], [(8, 1)], [(0, 1)], 14)
        add_term([(2, 1)], [(9, 1)], [(2, 1)], 15)
        add_term([(2, 1)], [(10, 1)], [(4, 1)], 16)
        add_term([(2, 1)], [(11, 1)], [(6, 1)], 17)
        
        add_term([(5, 1)], [(8, 1)], [(1, 1)], 18)
        add_term([(5, 1)], [(9, 1)], [(3, 1)], 19)
        add_term([(5, 1)], [(10, 1)], [(5, 1)], 20)
        add_term([(5, 1)], [(11, 1)], [(7, 1)], 21)

        return U, V, W, rank

    # Verify a decomposition
    def verify(U, V, W, rank):
        # Convert to numpy
        Un = np.array(U, dtype=float)
        Vn = np.array(V, dtype=float)
        Wn = np.array(W, dtype=float)
        
        # For each (i, j, k), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # Actually, the tensor T[i,j,k] = delta_ij * delta_jk... wait no.
        # The matrix multiplication tensor T[(i,j), (j,k)] = delta for correct indices.
        # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
        # T[a, b, c] = 1 if a=(i,j), b=(j,k), c=(k,i) for the same i,j,k, else 0
        # So T[(i,j),(j,k),(k,i)] = 1, and 0 otherwise.
        
        # For a decomposition: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # We need: for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
        # And for all other (a,b,c): sum_r U[a,r] * V[b,r] * W[c,r] = 0
        
        # Check the 24 non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Un[i*m+j], Vn[j*p+k] * Wn[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        
        # Check a sample of zero entries to be thorough
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, nm-1)
            b = random.randint(0, mp-1)
            c = random.randint(0, pn-1)
            # Skip if this is a non-zero entry
            i = a // m
            j = a % m
            j2 = b // p
            k = b % p
            k2 = c // n
            i2 = c % n
            if i == i2 and j == j2 and j == k2:
                continue
            val = np.dot(Un[a], Vn[b] * Wn[c])
            if abs(val) > 1e-9:
                return False
        
        return True

    # Try to find a rank-21 or lower solution using structured random search
    # Start from Strassen rank-22 and try to merge terms or find better combinations
    
    # First, return the known rank-22 solution
    U22, V22, W22, rank22 = build_strassen()
    
    # Try a few random perturbations to see if we can get rank 21
    # This is a heuristic approach
    
    best_U, best_V, best_W, best_rank = U22, V22, W22, rank22
    
    # Try rank 21 with random initialization
    for attempt in range(10):
        if time.time() - _START_TIME > TIME_LIMIT - 10:
            break
        
        rank = 21
        rng = np.random.RandomState(attempt * 123 + 456)
        
        U = rng.randn(nm, rank)
        V = rng.randn(mp, rank)
        W = rng.randn(pn, rank)
        
        # Scale to make the decomposition more likely
        # Normalize each column
        for r in range(rank):
            un = np.linalg.norm(U[:, r])
            vn = np.linalg.norm(V[:, r])
            wn = np.linalg.norm(W[:, r])
            if un > 1e-10 and vn > 1e-10 and wn > 1e-10:
                scale = (un * vn * wn) ** (1.0/3)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        # Check if this random init is close
        Un = U
        Vn = V
        Wn = W
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(Un[i*m+j], Vn[j*p+k] * Wn[k*n+i])
                    error += (val - 1.0) ** 2
        
        # This random approach is unlikely to work, so we'll just use the known solution
        break
    
    return {"U": best_U, "V": best_V, "W": best_W}
