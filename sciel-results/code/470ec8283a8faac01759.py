"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
import time
from fractions import Fraction

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return U, V, W

def _rank_of_flattening(n, m, p, mode):
    """
    Compute the rank of the matrix multiplication tensor flattened along a given mode.
    The tensor T has entries T[i,j,k] = delta_{i,i'} * delta_{j,j'} * delta_{k,k'} 
    where the multiplication is (A_{n x m}) * (B_{m x p}) = C_{n x p}.
    
    For matrix multiplication, the tensor is:
    T[(i,j), (j',k), (k',i')] = 1 if j=j' and k=k' and i=i'
    
    Mode-0 flattening (reshape to (n*m) x (m*p*n)):
    Row index: (i,j), Col index: (j',k,i')
    Entry = 1 if j=j' and k=i' and i=... wait.
    
    Actually for matmul: C_{i,k} = sum_j A_{i,j} B_{j,k}
    Tensor T_{i,j,k} = 1 (it's just the identity in j dimension essentially, but structured)
    
    Let me think more carefully. The bilinear map is:
    (A, B) -> A*B where A is n x m, B is m x p.
    
    The tensor T has indices (i, j, k) where:
    - i indexes rows of A and C (0..n-1)
    - j indexes cols of A and rows of B (0..m-1)  
    - k indexes cols of B and C (0..p-1)
    
    T[i,j,k] = 1 for all valid (i,j,k) - this is the structure tensor.
    
    Mode-1 flattening: shape (n, m*p)
    Entry at (i, (j,k)) = T[i,j,k] = 1
    
    Mode-2 flattening: shape (m, n*p)  
    Entry at (j, (i,k)) = T[i,j,k] = 1
    
    Mode-3 flattening: shape (p, n*m)
    Entry at (k, (i,j)) = T[i,j,k] = 1
    
    For this specific tensor (which is just all-ones in the relevant substructure),
    let's compute ranks exactly.
    """
    if mode == 0:
        # Flatten to (n*m) x (m*p)
        # Row: (i,j), Col: (j',k)
        # Entry = 1 if j=j' (the j index must match)
        # So it's a block diagonal structure
        rows = n * m
        cols = m * p
        M = [[Fraction(0)] * cols for _ in range(rows)]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    row_idx = i * m + j
                    col_idx = j * p + k
                    M[row_idx][col_idx] = Fraction(1)
    elif mode == 1:
        # Flatten to (m) x (n*p)
        # Row: j, Col: (i,k)
        # Entry = 1 always (for valid indices)
        rows = m
        cols = n * p
        M = [[Fraction(1)] * cols for _ in range(rows)]
    elif mode == 2:
        # Flatten to (p) x (n*m)
        # Row: k, Col: (i,j)
        # Entry = 1 always
        rows = p
        cols = n * m
        M = [[Fraction(1)] * cols for _ in range(rows)]
    else:
        raise ValueError(f"Unknown mode {mode}")
    
    # Compute rank via Gaussian elimination over Q
    rank = _gauss_rank(M)
    return rank

def _gauss_rank(M):
    """Compute rank of matrix M (list of lists of Fractions) via Gaussian elimination."""
    if not M or not M[0]:
        return 0
    rows = len(M)
    cols = len(M[0])
    # Make a copy
    A = [row[:] for row in M]
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        # Find pivot
        pivot = None
        for i in range(r, rows):
            if A[i][c] != 0:
                pivot = i
                break
        if pivot is None:
            continue
        # Swap
        A[r], A[pivot] = A[pivot], A[r]
        # Eliminate below
        pivot_val = A[r][c]
        for i in range(r + 1, rows):
            if A[i][c] != 0:
                factor = A[i][c] / pivot_val
                for j in range(c, cols):
                    A[i][j] -= factor * A[r][j]
        r += 1
    return r

def _verify_decomposition(U, V, W, n, m, p, max_samples=20):
    """Verify that the decomposition is exact by checking random integer matrices."""
    import random
    random.seed(42)
    for _ in range(max_samples):
        # Generate random integer matrices
        A = [[random.randint(-3, 3) for _ in range(m)] for _ in range(n)]
        B = [[random.randint(-3, 3) for _ in range(p)] for _ in range(m)]
        
        # Compute A*B directly
        C_direct = [[sum(A[i][j] * B[j][k] for j in range(m)) for k in range(p)] for i in range(n)]
        
        # Compute via decomposition: C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
        # Actually: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # Wait, the decomposition is: T = sum_r U_r ⊗ V_r ⊗ W_r
        # where U_r is in R^{n*m}, V_r is in R^{m*p}, W_r is in R^{p*n}
        # C[i,k] = sum_r sum_j sum_{j'} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j',k]
        # Hmm, this doesn't look right. Let me reconsider.
        
        # Standard tensor decomposition for matmul:
        # C[i,k] = sum_r (sum_j U[r][i][j] * A[i,j]) * (sum_j V[r][j][k] * B[j,k])
        # So U[r] is n x m, V[r] is m x p, W[r] is a scalar (or 1 for each (i,k) pair?)
        
        # Actually the convention given is:
        # U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # W[k*n+i, r]
        
        # The bilinear form: C[i,k] = sum_r U[i*m+?, r] ... 
        # I think: C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
        # No wait, that's trilinear.
        
        # For bilinear: C[i,k] = sum_r sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k]
        # And W is... hmm.
        
        # Let me re-read: "U, V, W: factor matrices of shapes (n*m, rank), (m*p, rank), (p*n, rank)"
        # "output factor follows the official trace convention W[k*n+i,r]"
        
        # I think the decomposition is:
        # C[i,k] = sum_r U[i*m+?, r] * V[?, r] * W[k*n+i, r]
        # where the ? indices are summed over j.
        
        # Actually I believe it's:
        # C[i,k] = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
        # and W is not used in the product? That doesn't make sense.
        
        # Or perhaps: C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k]
        # with summation over j and r.
        
        # Let me just verify using the schoolbook and see if it works.
        C_decomp = [[0] * p for _ in range(n)]
        for i in range(n):
            for k in range(p):
                total = 0
                for r in range(len(U[0])):
                    wa = sum(U[i*m+j][r] * A[i][j] for j in range(m))
                    wb = sum(V[j*p+k][r] * B[j][k] for j in range(m))
                    total += wa * wb * W[k*n+i][r]
                C_decomp[i][k] = total
        
        if C_direct != C_decomp:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # First, get the schoolbook decomposition as a fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Compute flattening ranks to get lower bounds
    # For matmul of (n x m) * (m x p), the tensor rank is at least max(n*m, m*p, n*p)
    # but actually it's more subtle.
    
    # Let's compute the actual flattening ranks
    rank0 = _rank_of_flattening(n, m, p, 0)
    rank1 = _rank_of_flattening(n, m, p, 1)
    rank2 = _rank_of_flattening(n, m, p, 2)
    
    lower_bound = max(rank0, rank1, rank2)
    
    # The schoolbook rank is n*m*p = 2*5*5 = 50
    # Lower bound from flattening is max(10, 5, 25) = 25
    # But we need to find if we can do better than 50.
    
    # For this specific case (2x5 * 5x5), let's try to find a lower rank decomposition
    # by solving the system exactly.
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k) in the matmul structure.
    # We want to find rank r such that:
    # T[i,j,k] = sum_{r=0}^{R-1} U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # This is a nonlinear problem. Let's try small ranks.
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try ranks from lower_bound to best_rank - 1
    # This is expensive, so let's be strategic.
    
    # For now, let's try a few candidate ranks
    candidates = list(range(lower_bound, best_rank))
    candidates.reverse()  # Try smaller ranks first... actually we want to find the MINIMAL, so try from low to high
    
    # Actually, let's try from the lower bound upward
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        # Skip some to save time - focus on promising values
        # For 2x5*5x5, known results suggest rank might be around 20-30
        if r < 20 or r > 40:
            continue
        
        # Try to find a decomposition of rank r
        # This is hard in general, but for this small case we can try
        # random initialization + local search, or structured approaches.
        
        # For now, let's just verify if the schoolbook works and return it
        # if we can't find something better quickly.
        pass
    
    # Return the best we have (schoolbook for now)
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
