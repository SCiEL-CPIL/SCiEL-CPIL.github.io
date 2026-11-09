import time
import numpy as np
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

def _validate(U, V, W):
    """Check if U, V, W form a valid exact decomposition."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
        return False
    
    # Convert to numpy for fast check
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # For each (i, j, k), check: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for each (i, j, k, i', j', k') with (i,j,k) != (i',j',k'), the sum should be 0
    # Actually the condition is: the tensor T[i,j,k] = sum_c U[i*m+j,c] V[j*p+k,c] W[k*n+i,c]
    # must equal 1 if (i,j,k) corresponds to the "diagonal" entry, but wait...
    
    # Actually the matrix multiplication tensor is: (A * B)[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # No wait, that's not right either.
    
    # Let me reconsider. The matrix multiplication tensor T has dimensions (n, m, p) where
    # T[i,j,k] = 1 if we're computing (A*B)[i,k] using A[i,j]*B[j,k].
    # Actually T[i,j,k] = 1 for all valid (i,j,k) - it's the "all ones" tensor!
    # No, that's also not right.
    
    # The matrix multiplication tensor: T is a 3-tensor of size (n*m) x (m*p) x (p*n)
    # where T[i*m+j, j*p+k, k*n+i] = 1 and all other entries are 0.
    
    # So for a decomposition U (n*m x r), V (m*p x r), W (p*n x r):
    # sum_c U[a,c] * V[b,c] * W[c_idx,c] should equal T[a,b,c_idx]
    # where T[a,b,c_idx] = 1 if a=i*m+j, b=j*p+k, c_idx=k*n+i for some i,j,k
    # and 0 otherwise.
    
    # Let's just check all entries
    T = np.zeros((n*m, m*p, p*n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Compute the reconstructed tensor
    # Reconstructed[a,b,c_idx] = sum_c U[a,c] * V[b,c] * W[c_idx,c]
    # This is an einsum: 'ac,bc,dc->abd'
    recon = np.einsum('ac,bc,dc->abd', U_arr, V_arr, W_arr)
    
    if not np.allclose(recon, T, atol=1e-9):
        return False
    
    # Check that entries are integer or half-integer
    for mat in [U, V, W]:
        for row in mat:
            for val in row:
                if isinstance(val, Fraction):
                    v = float(val)
                else:
                    v = float(val)
                # Check if v is integer or half-integer
                if abs(v * 2 - round(v * 2)) > 1e-9:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # Safety margin before 150s limit
    
    # Start with the schoolbook decomposition
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][column] = 1
                V_school[j * p + k][column] = 1
                W_school[k * n + i][column] = 1
                column += 1
    
    best = (rank_school, U_school, V_school, W_school)
    
    # Try to find lower rank decompositions using random search + local optimization
    # For small matrices, we can try rank values below the schoolbook rank
    
    # Lower bounds: max(n*m, m*p, p*n) = max(6, 21, 14) = 21
    # Also, the rank of matrix multiplication tensor is at least max(n*m, m*p, p*n)
    # For 2x3 * 3x7: lower bound is max(6, 21, 14) = 21
    
    lb = max(n * m, m * p, p * n)
    
    # Try ranks from lb upward, trying to find a valid decomposition
    for rank in range(lb, rank_school):
        if time.time() > deadline - 5:
            break
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if time.time() > deadline - 5:
                break
            
            # Random initialization with integer/half-integer values
            U = np.zeros((n * m, rank), dtype=float)
            V = np.zeros((m * p, rank), dtype=float)
            W = np.zeros((p * n, rank), dtype=float)
            
            # Initialize with small random values
            for c in range(rank):
                for a in range(n * m):
                    if random.random() < 0.7:
                        U[a, c] = random.choice([0, 0.5, 1, -0.5, -1])
                for b in range(m * p):
                    if random.random() < 0.7:
                        V[b, c] = random.choice([0, 0.5, 1, -0.5, -1])
                for d in range(p * n):
                    if random.random() < 0.7:
                        W[d, c] = random.choice([0, 0.5, 1, -0.5, -1])
            
            # Local search: try to reduce the error
            best_err = np.inf
            for _ in range(200):
                if time.time() > deadline - 5:
                    break
                
                # Compute current error
                T = np.zeros((n * m, m * p, p * n), dtype=float)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T[i * m + j, j * p + k, k * n + i] = 1.0
                recon = np.einsum('ac,bc,dc->abd', U, V, W)
                err = np.sum((recon - T) ** 2)
                
                if err < 1e-6:
                    # Convert to exact fractions and validate
                    U_frac = [[Fraction(U[a, c]).limit_denominator(2) for c in range(rank)] for a in range(n * m)]
                    V_frac = [[Fraction(V[b, c]).limit_denominator(2) for c in range(rank)] for b in range(m * p)]
                    W_frac = [[Fraction(W[d, c]).limit_denominator(2) for c in range(rank)] for d in range(p * n)]
                    if _validate(U_frac, V_frac, W_frac):
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
                    break
                
                if err < best_err:
                    best_err = err
                
                # Pick a random entry to modify
                mat_choice = random.random()
                if mat_choice < 0.333:
                    mat = 'U'
                    idx = random.randint(0, n * m - 1)
                elif mat_choice < 0.667:
                    mat = 'V'
                    idx = random.randint(0, m * p - 1)
                else:
                    mat = 'W'
                    idx = random.randint(0, p * n - 1)
                
                col = random.randint(0, rank - 1)
                old_val = U[idx, col] if mat == 'U' else (V[idx, col] if mat == 'V' else W[idx, col])
                
                # Try a few perturbations
                candidates = [0, 0.5, 1, -0.5, -1, 1.5, -1.5, 2, -2]
                candidates = [v for v in candidates if v != old_val]
                if not candidates:
                    candidates = [old_val + random.choice([-0.5, 0.5])]
                
                new_val = random.choice(candidates)
                
                if mat == 'U':
                    U[idx, col] = new_val
                elif mat == 'V':
                    V[idx, col] = new_val
                else:
                    W[idx, col] = new_val
                
                # Compute new error
                recon_new = np.einsum('ac,bc,dc->abd', U, V, W)
                err_new = np.sum((recon_new - T) ** 2)
                
                if err_new >= err:
                    # Revert
                    if mat == 'U':
                        U[idx, col] = old_val
                    elif mat == 'V':
                        V[idx, col] = old_val
                    else:
                        W[idx, col] = old_val
                else:
                    # Accept
                    pass
        
        # After local search, check if we got a valid solution
        if best_err < 1e-6:
            U_frac = [[Fraction(U[a, c]).limit_denominator(2) for c in range(rank)] for a in range(n * m)]
            V_frac = [[Fraction(V[b, c]).limit_denominator(2) for c in range(rank)] for b in range(m * p)]
            W_frac = [[Fraction(W[d, c]).limit_denominator(2) for c in range(rank)] for d in range(p * n)]
            if _validate(U_frac, V_frac, W_frac):
                return {"U": U_frac, "V": V_frac, "W": W_frac}
    
    # Return the best solution found (or schoolbook if nothing better)
    _, U_best, V_best, W_best = best
    return {"U": U_best, "V": V_best, "W": W_best}
