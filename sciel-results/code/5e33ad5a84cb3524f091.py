"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W form a valid tensor decomposition for n×m×p matrix mult."""
    rank = len(U[0]) if U else 0
    # Check shapes
    if len(U) != n * m or len(V) != m * p or len(W) != p * n:
        return False
    if any(len(row) != rank for row in U):
        return False
    if any(len(row) != rank for row in V):
        return False
    if any(len(row) != rank for row in W):
        return False
    
    # Verify: for each (i, j, k), sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
    # And for all other combinations, the sum should be 0
    # Actually, the tensor T[i,j,k] for matmul is: T[i,j,k] = 1 if we're computing C[i,k] from A[i,j]*B[j,k]
    # The decomposition requires: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # For matrix multiplication, T[i,j,k] = 1 for all i,j,k (since every product contributes)
    # Wait, that's not right. Let me reconsider.
    
    # For matrix multiplication C = A * B where A is n×m and B is m×p:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] which equals 1 for all valid i,j,k (each term A[i,j]*B[j,k] contributes)
    # So we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for c in range(rank):
                    s += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                if s != 1:
                    return False
    return True

def _build_schoolbook(n, m, p):
    """Build the schoolbook decomposition of rank n*m*p."""
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
    return {"U": U, "V": V, "W": W}

def _build_block_diagonal(n, m, p, block_ranks):
    """
    Build a block-diagonal style decomposition.
    block_ranks: list of (n_i, m_i, p_i, rank_i) for each block
    This assumes we can decompose into independent sub-problems.
    """
    total_rank = sum(r for _, _, _, r in block_ranks)
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    for (ni, mi, pi, ri) in block_ranks:
        # For each block, we need a decomposition of ni×mi×pi matrix mult of rank ri
        # Use schoolbook within each block if ri == ni*mi*pi
        for local_i in range(ni):
            for local_j in range(mi):
                for local_k in range(pi):
                    if offset + local_i * mi + local_j >= n * m:
                        break
                    # Map local indices to global
                    gi = local_i  # assume blocks are contiguous in i
                    gj = local_j
                    gk = local_k
                    if gi < n and gj < m and gk < p:
                        U[gi * m + gj][offset] = 1
                        V[gj * p + gk][offset] = 1
                        W[gk * n + gi][offset] = 1
                    offset += 1
    
    return {"U": U, "V": V, "W": W}

def _search_low_rank(n, m, p, max_rank, deadline):
    """
    Search for a low-rank decomposition using structured patterns.
    """
    start_time = time.time()
    best_result = None
    best_rank = n * m * p
    
    # Try various structured decompositions
    # Strategy: use Kronecker products and structured matrices
    
    # First, try to see if we can use a smaller rank by exploiting linearity
    # For 3x4x5, the schoolbook rank is 60.
    # Known lower bounds: max(n*m, m*p, p*n) = max(12, 20, 15) = 20
    # Upper bounds from various methods...
    
    # Try a systematic search over small ranks with random initialization and verification
    # Since we need exact integer/half-integer entries, we'll work with fractions
    
    random.seed(42)
    
    for target_rank in range(max(20, best_rank - 20), best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try a few random structured initializations
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize with small random values from {-1, 0, 1} and {0, 1/2}
            U = [[random.choice([0, 0, 0, 1, -1, Fraction(1, 2), Fraction(-1, 2)]) for _ in range(target_rank)] for _ in range(n * m)]
            V = [[random.choice([0, 0, 0, 1, -1, Fraction(1, 2), Fraction(-1, 2)]) for _ in range(target_rank)] for _ in range(m * p)]
            W = [[random.choice([0, 0, 0, 1, -1, Fraction(1, 2), Fraction(-1, 2)]) for _ in range(target_rank)] for _ in range(p * n)]
            
            if _verify_decomposition(U, V, W, n, m, p):
                return {"U": U, "V": V, "W": W}
        
        # If we found a valid decomposition, update best
        # (The verification above returns immediately on success)
    
    return None

def _try_kronecker_based(n, m, p, deadline):
    """
    Try to build a decomposition using Kronecker product structures.
    For 3x4x5, note that 4 = 2*2, so we might be able to use Strassen-like tricks.
    """
    start_time = time.time()
    
    # 3x4x5: We can think of the 4 as 2x2 blocks
    # If we decompose the middle dimension into 2x2, we get 3*(2x2)*5
    # Each 2x2 multiplication can be done with rank 7 (Strassen) instead of 4
    
    # So total rank could be: 3 * 5 * 7 = 105? No, that's worse.
    # Actually, we need to be more careful.
    
    # Let's try a different approach: use the fact that some entries can be shared
    # For now, just return the schoolbook as fallback
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin
    
    # Start with schoolbook
    schoolbook = _build_schoolbook(n, m, p)
    best_result = schoolbook
    best_rank = n * m * p
    
    # Try to find better decompositions
    # Attempt 1: Search for low-rank with structured patterns
    result = _search_low_rank(n, m, p, best_rank, deadline)
    if result is not None:
        rank = len(result["U"][0])
        if rank < best_rank:
            best_result = result
            best_rank = rank
    
    # Attempt 2: Try Kronecker-based approaches
    if time.time() < deadline - 5:
        result = _try_kronecker_based(n, m, p, deadline)
        if result is not None:
            rank = len(result["U"][0])
            if rank < best_rank:
                best_result = result
                best_rank = rank
    
    # Convert fractions to appropriate format if needed
    # The evaluator expects integer/half-integer entries
    # Fractions like Fraction(1, 2) should be fine, but let's convert to float if needed
    # Actually, let's keep them as fractions for exactness
    
    return best_result
# EVOLVE-BLOCK-END
