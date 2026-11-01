"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 4, 4]
n, m, p = DIMENSIONS
N = n * m  # 8
M = m * p  # 16
P = p * n  # 8
SCHOOLBOOK = n * m * p  # 32

# Deadline guard: keep a safety margin before the 150s wall-clock kill.
_DEADLINE = 135.0  # seconds
_START = time.time()


def _elapsed():
    return time.time() - _START


def _remaining():
    return _DEADLINE - _elapsed()


def _verify(U, V, W):
    """Check exact decomposition: sum_c U[:,c] V[:,c]^T W[:,c]^T == tensor."""
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    rank = Ua.shape[1]
    # Compute tensor T[i,j,k] = sum_c U[i,j,c]*V[j,k,c]*W[k,i,c]
    # U index: i*m+j, V index: j*p+k, W index: k*n+i
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                if abs(s - 1.0) > 1e-6:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            break
    return ok


def _make_schoolbook():
    """Return the schoolbook rank-32 decomposition."""
    rank = SCHOOLBOOK
    U = [[0.0] * rank for _ in range(N)]
    V = [[0.0] * rank for _ in range(M)]
    W = [[0.0] * rank for _ in range(P)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1.0
                V[j * p + k][col] = 1.0
                W[k * n + i][col] = 1.0
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(rank):
    """
    Attempt to find a rank-`rank` decomposition via structured search.
    Uses the fact that for matrix multiplication, we can try to find
    a low-rank bilinear form. We use a randomized approach with verification.
    """
    if rank >= SCHOOLBOOK:
        return _make_schoolbook()

    # Strategy: try to find a decomposition using a structured approach.
    # For 2x4 * 4x4, the tensor has 32 non-zero entries.
    # We'll try a few structured patterns.

    # Approach 1: Try to exploit the rectangular structure.
    # The 2x4 matrix A has entries A[i,j], the 4x4 matrix B has B[j,k].
    # Output C[i,k] = sum_j A[i,j] B[j,k].
    
    # For rank r, we need U (8 x r), V (16 x r), W (8 x r) such that
    # C[i,k] = sum_c U[i*m, c] ... wait, let me re-read the indexing.
    
    # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    # C[i,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] summed over j too?
    # No, the tensor entry is T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # And T[i,j,k] should be 1 if this corresponds to the (i,j,k) term in the product,
    # which is always 1 for valid (i,j,k).
    
    # So we need: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is a system of 32 equations in 32*r unknowns (with the constraint that
    # the entries should be "nice" like integers or half-integers, but the problem
    # says integer/half-integer entries).
    
    # Let's try a structured approach: assume U, V, W have a specific pattern.
    # For the 2x4*4x4 case, we can try to use the fact that n=2 is small.
    
    # Let's try a random search with verification for small ranks.
    # But first, let's try some known low-rank constructions.
    
    # Actually, for matrix multiplication, the rank is at least max(n*m, m*p, p*n) / something...
    # The lower bound is max(n*m, m*p, p*n) for the "obvious" bound, but that's not tight.
    # Actually, the rank of the matrix multiplication tensor M_{n,m,p} is at least
    # max(n*m, m*p, p*n) / ... no, the lower bound is more subtle.
    
    # For our small case, let's just try to find any valid decomposition with rank < 32.
    # We'll use a simple iterative refinement approach.
    
    # Given the time constraints, let's try a few structured guesses and a short random search.
    
    # Structured guess: Try to use the 2x4 structure.
    # Let's try rank = 20, 16, 12, etc.
    
    # For now, return schoolbook as a fallback. We'll try to improve in the main loop.
    return _make_schoolbook()


def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    Returns {'U', 'V', 'W'} with the best rank found.
    """
    best_result = _make_schoolbook()
    best_rank = SCHOOLBOOK

    # Try a few structured approaches
    # Approach: Use the fact that for M_{2,4,4}, we can potentially do better than 32.
    
    # Let's try a systematic search for lower ranks using a simple heuristic.
    # We'll try ranks from 24 down to 16, and for each, do a short random search.
    
    # Given the 135s deadline, we need to be efficient.
    # Let's try a few specific structured decompositions first.
    
    # Actually, let's just return the schoolbook decomposition for now,
    # but structure the code so that if we find something better, we use it.
    
    # For the 2x4*4x4 case, let's try to see if rank 24 is achievable.
    # This is a research problem, so let's use a simple approach.
    
    # Try a few random initializations and see if we can verify them.
    # But verification is expensive, so let's be careful.
    
    # For now, return the schoolbook.
    return best_result


# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 * 4x4 matrix multiplication.
    """
    best_result = _make_schoolbook()
    best_rank = SCHOOLBOOK

    # Try to find a better decomposition using structured search.
    # For M_{2,4,4}, let's try a few approaches.
    
    # Approach: Exploit the small n=2.
    # We can write C[i,k] = sum_j A[i,j] B[j,k] for i=0,1.
    # C[0,k] = A[0,0]B[0,k] + A[0,1]B[1,k] + A[0,2]B[2,k] + A[0,3]B[3,k]
    # C[1,k] = A[1,0]B[0,k] + A[1,1]B[1,k] + A[1,2]B[2,k] + A[1,3]B[3,k]
    
    # The tensor has 2*4*4 = 32 entries, all equal to 1.
    
    # Let's try a random search for lower ranks.
    # We'll try ranks from 28 down to 20, with a limited number of attempts.
    
    for rank in [28, 26, 24, 22, 20]:
        if _remaining() < 10:
            break
        # Try a few random initializations
        for attempt in range(5):
            if _remaining() < 10:
                break
            # Generate random U, V, W with small integer entries
            U = np.random.randint(-2, 3, size=(N, rank))
            V = np.random.randint(-2, 3, size=(M, rank))
            W = np.random.randint(-2, 3, size=(P, rank))
            # Verify
            if _verify(U.tolist(), V.tolist(), W.tolist()):
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = rank
                break
        if best_rank < SCHOOLBOOK:
            break

    return best_result
# EVOLVE-BLOCK-END
