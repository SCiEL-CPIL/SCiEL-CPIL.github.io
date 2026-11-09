"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]
import numpy as np
import random
import time
from itertools import product

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 130
    n, m, p = DIMENSIONS

    def validate_full(U, V, W):
        """Check that sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1 for all (i,j,k)."""
        r = len(U[0])
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = int(np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i]))
                    if s != 1:
                        return False
        return True

    def make_schoolbook():
        """Standard rank n*m*p decomposition."""
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    assert validate_full(best_U, best_V, best_W)

    def try_decomposition(U, V, W):
        """Try to reduce rank by removing redundant columns."""
        r = len(U[0])
        Ua = np.array(U, dtype=np.int64)
        Va = np.array(V, dtype=np.int64)
        Wa = np.array(W, dtype=np.int64)
        
        # Try removing each column and checking validity
        for c in range(r):
            if time.time() > deadline:
                return None
            # Remove column c
            mask = [x for x in range(r) if x != c]
            U_new = Ua[:, mask]
            V_new = Va[:, mask]
            W_new = Wa[:, mask]
            if validate_full(U_new.tolist(), V_new.tolist(), W_new.tolist()):
                return U_new.tolist(), V_new.tolist(), W_new.tolist(), r - 1
        return None

    # Try to reduce the schoolbook decomposition
    current_U, current_V, current_W = best_U, best_V, best_W
    current_rank = best_rank
    reduced = True
    while reduced and time.time() < deadline:
        reduced = False
        for _ in range(50):  # Try multiple times
            result = try_decomposition(current_U, current_V, current_W)
            if result is not None:
                current_U, current_V, current_W, current_rank = result
                reduced = True
                best_U, best_V, best_W, best_rank = current_U, current_V, current_W, current_rank
                break
            if time.time() > deadline:
                break

    # Try structured low-rank decompositions
    # For 2x3x6, known lower bounds suggest rank might be lower than 36
    # Try rank values from 24 down to 20
    for target_rank in range(best_rank - 1, max(18, best_rank - 15), -1):
        if time.time() > deadline:
            break
        found = False
        # Try several random initializations
        for attempt in range(100):
            if time.time() > deadline:
                break
            U = np.zeros((n*m, target_rank), dtype=np.int64)
            V = np.zeros((m*p, target_rank), dtype=np.int64)
            W = np.zeros((p*n, target_rank), dtype=np.int64)
            
            # Initialize with small random integers
            for r in range(target_rank):
                # Sparse initialization
                nzu_u = random.randint(2, min(4, n*m))
                nzu_v = random.randint(2, min(4, m*p))
                nzu_w = random.randint(2, min(4, p*n))
                uidx = random.sample(range(n*m), nzu_u)
                vidx = random.sample(range(m*p), nzu_v)
                widx = random.sample(range(p*n), nzu_w)
                for a in uidx:
                    U[a, r] = random.choice([-1, 1, 2])
                for a in vidx:
                    V[a, r] = random.choice([-1, 1, 2])
                for a in widx:
                    W[a, r] = random.choice([-1, 1, 2])
            
            if validate_full(U.tolist(), V.tolist(), W.tolist()):
                best_U, best_V, best_W, best_rank = U.tolist(), V.tolist(), W.tolist(), target_rank
                found = True
                break
        
        if found:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
