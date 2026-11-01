"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x3 matrix by a 3x4 matrix.
    Uses structured search over sparse factor matrices with small integer/half-integer entries.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=4
    nm, mp, pn = n * m, m * p, p * n  # 6, 12, 8
    
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    def validate(U, V, W, rank):
        """Check if U, V, W with given rank exactly decompose the multiplication tensor."""
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Check valid entries: T[i*m+j, j*p+k, k*n+i] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                    if abs(val - 1) > 1e-9:
                        return False
        
        # Check that T[a,b,c] = 0 when (a,b,c) doesn't correspond to valid multiplication
        for a in range(nm):
            i, j = divmod(a, m)
            for b in range(mp):
                j2, k = divmod(b, p)
                if j2 != j:
                    for c in range(pn):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if abs(val) > 1e-9:
                            return False
                else:
                    valid_c = k * n + i
                    for c in range(pn):
                        val = np.sum(U_arr[a] * V_arr[b] * W_arr[c])
                        if c == valid_c:
                            if abs(val - 1) > 1e-9:
                                return False
                        else:
                            if abs(val) > 1e-9:
                                return False
        return True
    
    def check_half_integer(U, V, W, tol=1e-9):
        """Check if all entries are integer or half-integer within tolerance."""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if abs(val) < 1e-12:
                        continue
                    two_val = 2 * val
                    nearest = round(two_val)
                    if abs(two_val - nearest) > tol:
                        return False
        return True
    
    # Create schoolbook solution of rank 24
    rank_schoolbook = n * m * p  # 24
    U_sb = [[0] * rank_schoolbook for _ in range(nm)]
    V_sb = [[0] * rank_schoolbook for _ in range(mp)]
    W_sb = [[0] * rank_schoolbook for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    best_solution = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = rank_schoolbook
    
    random.seed(42)
    
    # Try to find lower rank decompositions using structured patterns
    # For each rank component, use a "star" pattern centered at a j-index
    candidates = [0.5, -0.5, 1, -1]
    
    for target_rank in range(12, 24):
        if time.time() > deadline - 5:
            break
        
        num_attempts = 200 if target_rank <= 15 else 100
        found = False
        
        for attempt in range(num_attempts):
            if time.time() > deadline - 3:
                break
            
            U = [[0.0] * target_rank for _ in range(nm)]
            V = [[0.0] * target_rank for _ in range(mp)]
            W = [[0.0] * target_rank for _ in range(pn)]
            
            for r in range(target_rank):
                # Choose a hub j for this component
                hub_j = random.randint(0, m - 1)
                
                # U: touch 1-2 (i, hub_j) positions
                num_u = random.randint(1, 2)
                u_is = random.sample(range(n), num_u)
                for i in u_is:
                    U[i * m + hub_j][r] = random.choice(candidates)
                
                # V: touch 1-2 (hub_j, k) positions
                num_v = random.randint(1, 2)
                v_ks = random.sample(range(p), num_v)
                for k in v_ks:
                    V[hub_j * p + k][r] = random.choice(candidates)
                
                # W: touch 1-2 (i, k) positions
                num_w = random.randint(1, 2)
                w_pos = random.sample(range(pn), num_w)
                for pos in w_pos:
                    W[pos][r] = random.choice(candidates)
            
            if validate(U, V, W, target_rank):
                if check_half_integer(U, V, W):
                    best_solution = {"U": U, "V": V, "W": W}
                    best_rank = target_rank
                    found = True
                    break
        
        if found:
            break
    
    return best_solution
# EVOLVE-BLOCK-END
