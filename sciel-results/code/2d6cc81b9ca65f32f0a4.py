"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import product, combinations
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p  # 42
    
    def to_list(mat):
        return mat.tolist()
    
    def verify_fast(U, V, W, tolerance=1e-9):
        """Fast verification using float, for initial screening."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = 0.0
                    for r in range(rank):
                        actual += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    if abs(actual - 1.0) > tolerance:
                        return False
        
        # Check all zero entries (small enough for n=2,m=3,p=7)
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    if not (j1 == j2 and i1 == i2 and k1 == k2):
                        actual = 0.0
                        for r in range(rank):
                            actual += U[a, r] * V[b, r] * W[c, r]
                        if abs(actual) > tolerance:
                            return False
        return True
    
    def verify_exact(U, V, W):
        """Verify that U, V, W correctly decompose the tensor using exact arithmetic."""
        rank = U.shape[1]
        if rank != V.shape[1] or rank != W.shape[1]:
            return False
        
        # Check all n*m*p required entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    actual = Fraction(0)
                    for r in range(rank):
                        actual += Fraction(int(round(U[i * m + j, r] * 2)), 2) * \
                                   Fraction(int(round(V[j * p + k, r] * 2)), 2) * \
                                   Fraction(int(round(W[k * n + i, r] * 2)), 2)
                    if actual != Fraction(1):
                        return False
        
        # Check all zero entries
        for a in range(n * m):
            i1, j1 = divmod(a, m)
            for b in range(m * p):
                j2, k1 = divmod(b, p)
                for c in range(p * n):
                    k2, i2 = divmod(c, n)
                    if not (j1 == j2 and i1 == i2 and k1 == k2):
                        actual = Fraction(0)
                        for r in range(rank):
                            actual += Fraction(int(round(U[a, r] * 2)), 2) * \
                                       Fraction(int(round(V[b, r] * 2)), 2) * \
                                       Fraction(int(round(W[c, r] * 2)), 2)
                        if actual != Fraction(0):
                            return False
        return True
    
    # Build schoolbook decomposition
    U_sb = np.zeros((n * m, rank_schoolbook), dtype=np.float64)
    V_sb = np.zeros((m * p, rank_schoolbook), dtype=np.float64)
    W_sb = np.zeros((p * n, rank_schoolbook), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    def try_structured_search():
        """Try structured rank reductions using algebraic properties."""
        if time.time() > deadline:
            return None
        
        # Strategy: For each j, try to find rank-1 terms that cover multiple (i,k) pairs
        # Key insight: if we can find U, V, W such that for fixed j:
        # sum over some structure gives the right values
        
        # Try: for each j, use p terms that are "diagonal" in k
        # U[i*m+j, r] depends on i, V[j*p+k, r] depends on k, W[k*n+i, r] depends on both
        
        # Let's try a specific construction:
        # For each j, use n*p terms but try to merge some
        
        # Actually, let's try the known upper bound construction:
        # rank <= n*m + m*p + n*p - m = 6 + 21 + 14 - 3 = 38
        
        # This comes from: for each (i,j) pair, use one term for the "output" part
        # and combine with input terms
        
        # Let me try a different approach: use the fact that for m=3,
        # we can potentially use bilinear forms
        
        # Try rank 41 first (easiest to find)
        # Then 40, 39, 38
        
        half_ints = [Fraction(x, 2) for x in range(-8, 9)]  # -4 to 4 in steps of 0.5
        
        np.random.seed(12345)
        
        # Try various ranks
        for target_rank in [41, 40, 39, 38]:
            if time.time() > deadline:
                break
            
            max_attempts = 500 if target_rank >= 40 else 300
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                
                U = np.zeros((n * m, target_rank), dtype=np.float64)
                V = np.zeros((m * p, target_rank), dtype=np.float64)
                W = np.zeros((p * n, target_rank), dtype=np.float64)
                
                # Initialize with schoolbook-like structure, then perturb
                for r_idx in range(min(target_rank, rank_schoolbook)):
                    # Map to schoolbook term
                    i = r_idx // (m * p)
                    rem = r_idx % (m * p)
                    j = rem // p
                    k = rem % p
                    
                    U[i * m + j, r_idx] = 1.0
                    V[j * p + k, r_idx] = 1.0
                    W[k * n + i, r_idx] = 1.0
                
                # For remaining ranks, add random terms
                for r_idx in range(rank_schoolbook, target_rank):
                    # Add a random sparse term
                    num_u = np.random.randint(1, 3)
                    num_v = np.random.randint(1, 3)
                    num_w = np.random.randint(1, 3)
                    
                    u_pos = np.random.choice(n * m, size=num_u, replace=False)
                    v_pos = np.random.choice(m * p, size=num_v, replace=False)
                    w_pos = np.random.choice(p * n, size=num_w, replace=False)
                    
                    for pos in u_pos:
                        U[pos, r_idx] = float(np.random.choice(half_ints))
                    for pos in v_pos:
                        V[pos, r_idx] = float(np.random.choice(half_ints))
                    for pos in w_pos:
                        W[pos, r_idx] = float(np.random.choice(half_ints))
                
                # Quick check
                if verify_fast(U, V, W):
                    # Exact check
                    if verify_exact(U, V, W):
                        return U, V, W
        
        return None
    
    result = try_structured_search()
    if result is not None:
        U, V, W = result
        return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
    
    # Return schoolbook
    return {
        "U": to_list(U_sb),
        "V": to_list(V_sb),
        "W": to_list(W_sb)
    }
# EVOLVE-BLOCK-END
