"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import numpy as np

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    # Build the matrix multiplication tensor T[i,j,k] = 1
    # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
    # such that T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    #
    # The schoolbook rank is n*m*p = 48.
    # We try to find a lower rank decomposition.
    #
    # Strategy: Try rank values below 48 and use randomized search / structured ansatz.
    # For 3x4x4, the best known upper bound might be lower than 48.
    # Let's try a structured approach: use the fact that we can share factors.

    # First, let's try the schoolbook as fallback
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    c = i * m * p + j * p + k
                    U[i * m + j, c] = 1
                    V[j * p + k, c] = 1
                    W[k * n + i, c] = 1
        return U, V, W

    U_sb, V_sb, W_sb = schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p

    # Try to find lower rank decompositions
    # Use a simple iterative approach: start with random rank-r factors and try to fit
    # But exact decomposition is hard. Let's try structured reductions.
    
    # For matrix multiplication, there are known results:
    # 2x2x2 has rank 7 (Strassen)
    # We can potentially use Strassen-like tricks.
    
    # Let's try rank 47, 46, ... down to some lower bound
    # The tensor rank lower bound for 3x4x4 is at least max(n*m, m*p, p*n) = max(12, 16, 12) = 16
    # But realistically it's much higher.
    
    # Let's try a few specific ranks with random initialization and gradient-free search
    # Actually, for exact decomposition, we need rational/integer solutions.
    # Let's try to use the fact that small entries (0, 1, -1) often work.
    
    # Try rank 47: can we merge two schoolbook terms?
    # Two terms (i,j,k) and (i',j',k') can be merged if they share structure.
    # This is complex. Let's just try random search for small rank reductions.
    
    # Actually, let's try a different approach: use the known 2x2x2 rank 7 decomposition
    # and embed it into the 3x4x4 problem.
    
    # For now, return the schoolbook if we can't find better quickly
    if time.time() > deadline:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

    # Try to find rank 47 decomposition by modifying schoolbook
    # Pick two terms and try to combine them with a new term
    # This is a heuristic search
    
    # Let's try a randomized approach for rank r = best_rank - 1
    np.random.seed(42)
    
    for target_rank in range(best_rank - 1, max(15, best_rank - 5), -1):
        if time.time() > deadline - 10:
            break
        
        # Try random initialization with small integers
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.randint(-2, 3, size=(n * m, target_rank))
            V = np.random.randint(-2, 3, size=(m * p, target_rank))
            W = np.random.randint(-2, 3, size=(p * n, target_rank))
            
            # Check if this decomposes the tensor
            # T[i,j,k] should equal sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            success = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                        if val != 1:
                            success = False
                            break
                    if not success:
                        break
                if not success:
                    break
            if success:
                best_U, best_V, best_W = U, V, W
                best_rank = target_rank
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
