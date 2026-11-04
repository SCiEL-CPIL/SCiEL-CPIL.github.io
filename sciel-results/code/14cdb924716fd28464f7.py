import numpy as np
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 5, 5]

def _verify_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast vectorized verification."""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    rank = U_arr.shape[1]
    
    # Reshape W to (p, n, rank)
    W_reshaped = W_arr.reshape(p, n, rank)  # W_reshaped[k, i, r]
    
    # For each (a,b,c,d), check if sum_r U[a*m+b,r]*V[c*p+d,r]*W[k*n+i,r] == delta
    # We'll check all (a,b,c,d) combinations
    
    for a in range(n):
        for b in range(m):
            u_vec = U_arr[a * m + b, :]  # (rank,)
            for c in range(m):
                if b != c:
                    # Should be 0 for all (d,i,k)
                    for d in range(p):
                        v_val = V_arr[c * p + d, :]
                        uv = u_vec * v_val
                        if np.any(np.abs(uv) > tol):
                            # Need to check if W cancels it
                            M = W_reshaped.transpose(1, 0, 2) @ uv  # (n, p)
                            if np.max(np.abs(M)) > tol:
                                return False
                else:
                    # b == c: should give delta_{a,i}*delta_{d,k}
                    for d in range(p):
                        v_val = V_arr[c * p + d, :]
                        uv = u_vec * v_val
                        M = W_reshaped.transpose(1, 0, 2) @ uv  # (n, p)
                        expected = np.zeros((n, p))
                        expected[a, d] = 1.0
                        if np.max(np.abs(M - expected)) > tol:
                            return False
    return True

def _make_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _try_structured_pairing(n, m, p, R, rng, seed_offset=0):
    """Try structured decomposition where each rank-1 term is assigned to a (b,d) pair
    with specific U and W patterns. For n=2, we can share terms between rows."""
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Strategy: for each (b,d) pair, we need to decompose the identity in (a,i) and (d,k)
    # For n=2, we can use 2 terms per (b,d) or share cleverly
    
    # Assign terms to (b,d) pairs
    num_bd = m * p  # 25
    terms_per_bd = R // num_bd
    remainder = R % num_bd
    
    term_idx = 0
    for b in range(m):
        for d in range(p):
            count = terms_per_bd + (1 if b * p + d < remainder else 0)
            for t in range(count):
                if term_idx >= R:
                    break
                r = term_idx
                
                # V: activate (b,d)
                V[b * p + d, r] = 1.0
                
                # Choose pattern for U and W
                # We need: for this term, U[a*m+b,r]*W[k*n+i,r] should contribute
                # to delta_{a,i}*delta_{d,k} when summed over all terms for this (b,d)
                
                # Simple approach: use basis-like patterns
                if t == 0:
                    # Term for i=0, k=d
                    U[0 * m + b, r] = 1.0
                    W[d * n + 0, r] = 1.0
                elif t == 1:
                    # Term for i=1, k=d
                    U[1 * m + b, r] = 1.0
                    W[d * n + 1, r] = 1.0
                else:
                    # Additional terms - try to cancel or add
                    # Use random signs
                    c0 = rng.choice([-1.0, 1.0])
                    c1 = rng.choice([-1.0, 1.0])
                    U[0 * m + b, r] = c0
                    U[1 * m + b, r] = c1
                    W[d * n + 0, r] = c0
                    W[d * n + 1, r] = c1
                
                term_idx += 1
    
    return U, V, W

def _try_random_sparse(n, m, p, R, rng):
    """Random sparse decomposition with structured V."""
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Each term assigned to a (b,d) pair
    for r in range(R):
        b = rng.randint(0, m - 1)
        d = rng.randint(0, p - 1)
        
        V[b * p + d, r] = rng.choice([-1.0, 1.0])
        
        # U and W patterns
        pattern = rng.randint(0, 3)
        if pattern == 0:
            U[0 * m + b, r] = rng.choice([-1.0, 1.0])
            W[d * n + 0, r] = rng.choice([-1.0, 1.0])
        elif pattern == 1:
            U[1 * m + b, r] = rng.choice([-1.0, 1.0])
            W[d * n + 1, r] = rng.choice([-1.0, 1.0])
        elif pattern == 2:
            c = rng.choice([-1.0, 1.0])
            U[0 * m + b, r] = c
            U[1 * m + b, r] = c
            W[d * n + 0, r] = c
            W[d * n + 1, r] = c
        else:
            c0 = rng.choice([-1.0, 1.0])
            c1 = rng.choice([-1.0, 1.0])
            U[0 * m + b, r] = c0
            U[1 * m + b, r] = c1
            W[d * n + 0, r] = c0
            W[d * n + 1, r] = c1
    
    return U, V, W

def _try_row_pairing(n, m, p, R, rng):
    """Try to pair terms between row 0 and row 1 to save rank."""
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # For each (b,d), we need at least 2 terms (one for each row) unless we can share
    # Try: use R/25 terms per (b,d), with some shared
    
    num_bd = m * p
    terms_per_bd = R // num_bd
    remainder = R % num_bd
    
    term_idx = 0
    for b in range(m):
        for d in range(p):
            count = terms_per_bd + (1 if b * p + d < remainder else 0)
            for t in range(count):
                if term_idx >= R:
                    break
                r = term_idx
                
                V[b * p + d, r] = 1.0
                
                # Try different patterns
                if count >= 2:
                    if t == 0:
                        U[0 * m + b, r] = 1.0
                        W[d * n + 0, r] = 1.0
                    elif t == 1:
                        U[1 * m + b, r] = 1.0
                        W[d * n + 1, r] = 1.0
                    else:
                        # Extra terms
                        c0 = rng.choice([-1.0, 1.0])
                        c1 = rng.choice([-1.0, 1.0])
                        U[0 * m + b, r] = c0
                        U[1 * m + b, r] = c1
                        W[d * n + 0, r] = c0
                        W[d * n + 1, r] = c1
                else:
                    # Only 1 term - try to handle both rows
                    c0 = rng.choice([-1.0, 1.0])
                    c1 = rng.choice([-1.0, 1.0])
                    U[0 * m + b, r] = c0
                    U[1 * m + b, r] = c1
                    W[d * n + 0, r] = c0
                    W[d * n + 1, r] = c1
                
                term_idx += 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145
    
    def elapsed():
        return time.time() - start_time
    
    def still_going():
        return elapsed() < deadline
    
    # Start with schoolbook
    best = _make_schoolbook(n, m, p)
    best_rank = n * m * p
    
    rng = random.Random(42)
    
    # Try ranks from 25 to best_rank-1
    for R in range(25, best_rank):
        if not still_going():
            break
        
        # Try different strategies
        max_attempts = 3000 if R <= 32 else 1000 if R <= 40 else 300
        
        for attempt in range(max_attempts):
            if not still_going():
                break
            
            # Choose strategy based on attempt
            if attempt % 3 == 0:
                U, V, W = _try_random_sparse(n, m, p, R, rng)
            elif attempt % 3 == 1:
                U, V, W = _try_structured_pairing(n, m, p, R, rng)
            else:
                U, V, W = _try_row_pairing(n, m, p, R, rng)
            
            if _verify_fast(U.tolist(), V.tolist(), W.tolist(), n, m, p):
                best_rank = R
                best = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                break
    
    return best
