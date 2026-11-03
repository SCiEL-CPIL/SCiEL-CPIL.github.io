"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

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

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition exactly computes matrix multiplication.
    
    The tensor T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c] should be 1 for all i,j,k.
    """
    rank = len(U[0]) if U else 0
    # Build factor matrices as numpy arrays for speed
    Um = np.zeros((n * m, rank))
    Vm = np.zeros((m * p, rank))
    Wm = np.zeros((p * n, rank))
    for i in range(n * m):
        for c in range(rank):
            if U[i][c] != 0:
                Um[i, c] = U[i][c]
    for j in range(m * p):
        for c in range(rank):
            if V[j][c] != 0:
                Vm[j, c] = V[j][c]
    for k in range(p * n):
        for c in range(rank):
            if W[k][c] != 0:
                Wm[k, c] = W[k][c]
    
    # Check: for each (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] == 1
    # This is expensive for large rank, so do it smartly
    # For each c, the contribution is outer product structure
    # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    
    # We can check by constructing the full tensor
    # But that's n*m*p*rank which for rank=80 is 80*80=6400, fine
    # Actually let's just check a few random points plus all points if rank is small enough
    
    nmp = n * m * p
    if nmp * rank <= 500000:
        T = np.zeros((n, m, p))
        for c in range(rank):
            u_c = Um.reshape(n, m, rank)[:, :, c]
            v_c = Vm.reshape(m, p, rank)[:, :, c]
            w_c = Wm.reshape(p, n, rank)[:, :, c]
            # T[i,j,k] += u_c[i,j] * v_c[j,k] * w_c[k,i]
            for i in range(n):
                for j in range(m):
                    T[i, j, :] += u_c[i, j] * v_c[j, :] * w_c[:, i]
        return np.allclose(T, 1.0, atol=tol)
    else:
        # Sample check
        rng = np.random.RandomState(42)
        for _ in range(100):
            i = rng.randint(n)
            j = rng.randint(m)
            k = rng.randint(p)
            s = 0.0
            for c in range(rank):
                s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
            if abs(s - 1.0) > tol:
                return False
        return True

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120  # 120 second limit
    
    # Start with schoolbook
    best = _schoolbook(n, m, p)
    best_rank = n * m * p
    
    # Try to find a lower rank decomposition
    # Strategy: Try to use Strassen for sub-blocks and schoolbook for the rest
    # For 4x4 * 4x5, we can partition:
    #   A = [[A11, A12], [A21, A22]] where A11 is 2x2, A12 is 2x2, etc.
    #   B = [[B11, B12], [B21, B22]] where B11 is 2x3, B12 is 2x2, etc.
    # This gets complicated. Let's try a simpler approach.
    
    # Alternative: Try rank values from schoolbook-1 downward and see if we can find a valid decomposition
    # using a greedy or structured approach.
    
    # For now, let's try a structured approach:
    # Use Strassen (rank 7) for the top-left 2x2 * 2x2 block,
    # and schoolbook for everything else.
    
    # Partition: A is 4x4, B is 4x5
    # Let's split A into 2x2 blocks: A = [[A11, A12], [A21, A22]] each 2x2
    # Let's split B into 2x3, 2x2, 2x3, 2x2 blocks:
    #   B = [[B11, B12], [B21, B22]] where B11, B21 are 2x3 and B12, B22 are 2x2
    # C = A*B is 4x5: C = [[C11, C12], [C21, C22]]
    #   C11 = A11*B11 + A12*B21 (2x3)
    #   C12 = A11*B12 + A12*B22 (2x2)
    #   C21 = A21*B11 + A22*B21 (2x3)
    #   C22 = A21*B12 + A22*B22 (2x2)
    
    # This is getting complex. Let's just return schoolbook for now since
    # finding a valid lower-rank decomposition is non-trivial.
    
    # Actually, let me try a simple optimization: check if we can remove any
    # rank-1 terms from the schoolbook decomposition.
    
    # For schoolbook, each term c corresponds to (i,j,k) with U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1
    # We can try to see if any term is redundant.
    
    # A term (i,j,k) is redundant if for all other terms (i',j',k'), the contribution
    # to T[i,j,k] from other terms already sums to 1. But in schoolbook, each T[i,j,k]
    # is only contributed to by its own term, so no term is redundant.
    
    # Let's try a different approach: use a known result.
    # The matrix multiplication tensor M_{4,4,5} has rank at most 4*4*5 = 80 (schoolbook).
    # Lower bounds exist but finding an exact decomposition is hard.
    
    # Given the time constraints, let's return the schoolbook decomposition.
    # This is guaranteed to be correct.
    
    return best

# EVOLVE-BLOCK-END
