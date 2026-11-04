"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import numpy as np
from itertools import product
import time

def _schoolbook():
    n, m, p = DIMENSIONS
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

def _verify(U, V, W):
    n, m, p = DIMENSIONS
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    # Build tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # For matrix multiplication: T should equal 1 if this triple contributes to C[i,k]
    # Actually, the tensor decomposition means:
    # C[i,k] = sum_{j} A[i,j] * B[j,k]
    # = sum_r (sum_j A[i,j]*U[i*m+j,r]) * (sum_k B[j,k]*V[j*p+k,r]) ... no
    # The standard form: C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
    # More precisely: for each r, we have a rank-1 term.
    # The multiplication tensor M has entries M[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # Actually the tensor is: M_{i,j,k} where the multiplication is C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor decomposition: M_{i,j,k} = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # Wait, that's not right either. Let me think again.
    # The matrix multiplication tensor T has indices (i, j, k) where:
    # T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B contribute to C[i,k]
    # Actually, the standard way: the tensor for C = AB is:
    # T[i, j, k] = delta_{j,j} ... no.
    # Let me use the convention from the task: W[k*n+i, r]
    # The decomposition is: for each r, the rank-1 tensor is:
    #   U[:, r] * V[:, r] * W[:, r] (outer product)
    # where U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
    # The full tensor T has shape (n*m, m*p, p*n) and:
    # T[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    # For matrix multiplication C = A*B where A is n x m, B is m x p, C is n x p:
    # C[i, k] = sum_j A[i, j] * B[j, k]
    # The tensor T should satisfy: T[i*m+j, j*p+k, k*n+i] = 1 for all valid i,j,k
    # And T[a, b, c] = 0 otherwise (for the exact decomposition to work)
    # Actually, for the decomposition to exactly reproduce matrix multiplication:
    # C[i,k] = sum_{j} A[i,j] B[j,k] = sum_r (sum_j A[i,j] U[i*m+j, r]) * (sum_k B[j,k] V[j*p+k, r]) * W[k*n+i, r]
    # No, that's not how it works. The correct form is:
    # C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
    # Let me just verify by checking the tensor entries directly.
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # The multiplication tensor has T[i*m+j, j*p+k, k*n+i] = 1
    # (note: the j in b must match the j in a, and k in c must match k in b)
    # For a valid decomposition, we need:
    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    # And for invalid triples (where indices don't align), it should be 0.
    
    # Check all valid triples
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    return False
    # Check that invalid triples give 0
    # For efficiency, check a sample of invalid triples
    for _ in range(200):
        i = np.random.randint(n)
        j = np.random.randint(m)
        k = np.random.randint(p)
        # Pick a different j' or k' to make it invalid
        j2 = (j + 1) % m
        if j2 == j:
            j2 = (j + 2) % m
        a = i * m + j
        b = j2 * p + k
        c = k * n + i
        val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
        if abs(val) > 1e-9:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin
    
    n, m, p = DIMENSIONS
    best_result = _schoolbook()
    best_rank = n * m * p
    
    # Try structured low-rank decompositions
    # Strategy: try ranks from a lower bound up to schoolbook
    # Lower bound for matrix multiplication rank is at least max(n*m, m*p, p*n) = max(9, 18, 18) = 18
    # But known results suggest higher. Let's try from 18 upward.
    
    for target_rank in range(18, best_rank):
        if time.time() > deadline - 5:
            break
        
        # Try to find a decomposition of this rank
        # Use a greedy/structured approach
        # For matrix multiplication, one approach is to use the fact that
        # we need T[i*m+j, j*p+k, k*n+i] = 1
        
        # Try a parametric family: U, V, W with small integer entries
        # This is a hard problem, so let's try a few structured patterns
        
        # Pattern 1: Diagonal-like structure
        # Try to find if a rank-r decomposition exists using random initialization + verification
        # This is heuristic and may not find anything, but worth trying
        
        if target_rank < 30:  # Only try small ranks with random search
            np.random.seed(target_rank)
            for attempt in range(50):
                if time.time() > deadline - 5:
                    break
                # Random integer matrices with entries in {-1, 0, 1}
                U = np.random.randint(-1, 2, size=(n*m, target_rank))
                V = np.random.randint(-1, 2, size=(m*p, target_rank))
                W = np.random.randint(-1, 2, size=(p*n, target_rank))
                
                # Check if this works
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            a = i * m + j
                            b = j * p + k
                            c = k * n + i
                            val = np.dot(U[a], np.multiply(V[b], W[c]))
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                    if not valid:
                        break
                if valid:
                    # Verify invalid triples
                    valid = True
                    for _ in range(100):
                        i = np.random.randint(n)
                        j = np.random.randint(m)
                        k = np.random.randint(p)
                        j2 = (j + 1) % m
                        a = i * m + j
                        b = j2 * p + k
                        c = k * n + i
                        val = np.dot(U[a], np.multiply(V[b], W[c]))
                        if abs(val) > 1e-9:
                            valid = False
                            break
                    if valid:
                        best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                        best_rank = target_rank
                        break
        else:
            # For larger ranks, just return schoolbook (we know it works)
            break
    
    return best_result
# EVOLVE-BLOCK-END
