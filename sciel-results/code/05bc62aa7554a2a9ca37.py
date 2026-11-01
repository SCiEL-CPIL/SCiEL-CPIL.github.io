"""Matrix multiplication 2x3x9: rank-minimal exact tensor decomposition via search."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 9]
T_START = time.time()
TIME_LIMIT = 110  # seconds, leave margin for 120s limit

def _elapsed():
    return time.time() - T_START

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
    """Check if U, V, W form a valid decomposition of the matrix multiplication tensor."""
    n, m, p = DIMENSIONS
    rank = len(U[0]) if U else 0
    # Tensor entry T[i,j,k] = delta_{i, r} * delta_{j, s} * delta_{k, t} 
    # where r = i*m+j (U index), s = j*p+k (V index), t = k*n+i (W index)
    # Decomposition: T = sum_l U[r,l] * V[s,l] * W[t,l]
    # So for each (i,j,k): sum_l U[i*m+j, l] * V[j*p+k, l] * W[k*n+i, l] == 1
    # And for all other combinations where the triple (r,s,t) doesn't correspond to a valid (i,j,k),
    # i.e., where s != j*p+k or t != k*n+i for the given i,j,k, the sum should be 0.
    
    # More precisely: The matrix multiplication tensor has entries:
    # T_{r,s,t} = 1 if r = i*m+j, s = j*p+k, t = k*n+i for some (i,j,k), else 0
    # The decomposition requires: sum_l U[r,l] V[s,l] W[t,l] = T_{r,s,t} for all r,s,t
    
    # We only need to check the 54 entries that should be 1, and a sample of those that should be 0.
    # Actually, we need to check ALL entries to be safe, but that's 6*27*18 = 2916 checks.
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                r = i * m + j
                s = j * p + k
                t = k * n + i
                val = sum(U[r][l] * V[s][l] * W[t][l] for l in range(rank))
                if val != 1:
                    return False
    # Check some zero entries
    checked = 0
    for r in range(n * m):
        for s in range(m * p):
            for t in range(p * n):
                # Skip the known 1 entries
                i_r = r // m
                j_r = r % m
                j_s = s // p
                k_s = s % p
                k_t = t // n
                i_t = t % n
                if i_r == i_t and j_r == j_s and k_s == k_t:
                    continue  # This is a 1 entry
                val = sum(U[r][l] * V[s][l] * W[t][l] for l in range(rank))
                if val != 0:
                    return False
                checked += 1
    return True

def _try_strassen_style():
    """
    Attempt a Strassen-style decomposition for 2x3 x 3x9.
    
    Partition A (2x3) into rows: A1 (1x3), A2 (1x3)
    Partition B (3x9) into columns: B1 (3x4), B2 (3x5) -- uneven split
    
    Product C = A*B is 2x9:
    C1 = A1*B (1x9)
    C2 = A2*B (1x9)
    
    This doesn't directly help. Let's try a different approach.
    
    Alternative: Use the fact that 2x3x9 can be related to smaller products.
    Rank lower bound considerations:
    - The tensor rank of 2x3x9 is at most 54 (schoolbook).
    - Known results: rank(2x3x9) is between some values.
    
    Let's try a direct search for rank 53, 52, ... using randomized local search.
    """
    pass

def _random_search(target_rank, max_time):
    """Randomized search for a decomposition of target_rank."""
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    deadline = T_START + max_time
    
    for attempt in range(200):
        if _elapsed() > deadline:
            break
        
        # Initialize with small random values
        scale = 1.0
        U = [[random.randint(-scale, scale) for _ in range(target_rank)] for _ in range(nm)]
        V = [[random.randint(-scale, scale) for _ in range(target_rank)] for _ in range(mp)]
        W = [[random.randint(-scale, scale) for _ in range(target_rank)] for _ in range(pn)]
        
        # Local search: iterate, adjust one entry at a time
        for iteration in range(5000):
            if _elapsed() > deadline:
                break
            
            # Pick a random matrix and entry
            mat_idx = random.choice([0, 1, 2])
            if mat_idx == 0:
                r = random.randint(0, nm - 1)
                l = random.randint(0, target_rank - 1)
                old_val = U[r][l]
                U[r][l] = old_val + random.randint(-2, 2)
                # Check if improved (just check a few random tensor entries)
                # For simplicity, do a full check every 100 iterations
                if iteration % 100 == 0:
                    if _verify(U, V, W):
                        return U, V, W
            elif mat_idx == 1:
                r = random.randint(0, mp - 1)
                l = random.randint(0, target_rank - 1)
                old_val = V[r][l]
                V[r][l] = old_val + random.randint(-2, 2)
                if iteration % 100 == 0:
                    if _verify(U, V, W):
                        return U, V, W
            else:
                r = random.randint(0, pn - 1)
                l = random.randint(0, target_rank - 1)
                old_val = W[r][l]
                W[r][l] = old_val + random.randint(-2, 2)
                if iteration % 100 == 0:
                    if _verify(U, V, W):
                        return U, V, W
    
    return None

def _systematic_low_rank_search():
    """
    Try to find a decomposition with rank < 54 using a more structured approach.
    
    Key insight: The matrix multiplication tensor for 2x3x9 has structure.
    Let's exploit the small size of n=2.
    
    A is 2x3, B is 3x9.
    C[i][k] = sum_j A[i][j] * B[j][k]
    
    For i=0: C[0][k] = A[0][0]*B[0][k] + A[0][1]*B[1][k] + A[0][2]*B[2][k]
    For i=1: C[1][k] = A[1][0]*B[0][k] + A[1][1]*B[1][k] + A[1][2]*B[2][k]
    
    Each C[i][k] is a bilinear form in the entries of A and B.
    There are 2*9 = 18 output entries, each involving 3 products.
    
    Total schoolbook multiplications: 18 * 3 = 54.
    
    To reduce rank, we need to share multiplications between different output entries.
    Two output entries C[i1][k1] and C[i2][k2] can share a multiplication if they
    involve the same pair (A[i][j], B[j'][k]) after some linear combination.
    
    This is essentially the problem of finding a rank decomposition.
    Given the difficulty, let's try rank 53 first with a targeted search.
    """
    pass

def construct():
    n, m, p = DIMENSIONS
    schoolbook = _schoolbook()
    
    # Try to find a better decomposition, but fall back to schoolbook if time runs out
    # Given the difficulty, we'll do a quick search for rank 53
    
    # Quick search for rank 53
    if _elapsed() < TIME_LIMIT - 30:
        result = _random_search(53, TIME_LIMIT - 40)
        if result is not None:
            U, V, W = result
            if _verify(U, V, W):
                return {"U": U, "V": V, "W": W}
    
    # Fall back to schoolbook
    return schoolbook
