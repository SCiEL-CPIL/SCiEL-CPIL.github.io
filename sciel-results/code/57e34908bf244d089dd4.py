"""
Matrix multiplication 2x3 * 3x9 tensor decomposition.
We search for a rank-minimal decomposition.
"""
import numpy as np
import time
import itertools
from fractions import Fraction

DIMENSIONS = [2, 3, 9]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = 115.0  # seconds
    
    # 1. Default schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_U, best_V, best_W = U, V, W

    def check_validity(rank, U_list, V_list, W_list):
        """
        Checks if the decomposition is valid using exact arithmetic.
        U: (n*m) x R
        V: (m*p) x R
        W: (p*n) x R
        Condition: For all (i, j, k), sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1
        """
        # Convert to numpy arrays of Fractions for speed? 
        # Numpy doesn't support Fractions well. 
        # Given small sizes, we can iterate.
        # Or use integers if possible.
        # The prompt allows integer/half-integer. We can scale by 2 to work with integers if needed, but Fractions are safest.
        
        # Optimized check using numpy with object dtype is slow.
        # Let's just loop, but break early if we find a mismatch.
        
        # Pre-process W for faster access?
        # W is (p*n) x R. Index is k*n+i.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    u_row = U_list[i*m + j]
                    v_row = V_list[j*p + k]
                    w_row = W_list[k*n + i]
                    for r in range(rank):
                        s += u_row[r] * v_row[r] * w_row[r]
                    if s != 1:
                        return False
        return True

    def check_validity_fast(rank, U_list, V_list, W_list):
        """
        Faster check using numpy with float64.
        If float check passes with high precision, we can try to verify with exact arithmetic only if close.
        However, for small ranks, exact is manageable.
        Let's use numpy for the initial screening.
        """
        U_np = np.array(U_list, dtype=np.float64)
        V_np = np.array(V_list, dtype=np.float64)
        W_np = np.array(W_list, dtype=np.float64)
        
        # We need to check all (i, j, k)
        # T[i, j, k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We can reshape and use einsum?
        # U is (n, m, R)
        # V is (m, p, R)
        # W is (p, n, R)
        
        U_r = U_np.reshape(n, m, rank)
        V_r = V_np.reshape(m, p, rank)
        W_r = W_np.reshape(p, n, rank)
        
        # T[i,j,k] = einsum('ir,jr,kr->ijk', U_r[:,:,i? no, U is (n,m,R), so U_r[i,j,r])
        # Let's do it in steps to save memory and time
        # For each k:
        #   For each i:
        #     Check sum_j sum_r U[i,j,r] V[j,k,r] W[k,i,r] == 1? No, the sum is over j as well in the product?
        #     Wait, the tensor entry is T_{i,j,k}.
        #     T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        
        # We can compute M[i, j, k] for all.
        # But checking 2*3*9 = 54 equations is small.
        
        # Loop over k
        for k in range(p):
            # W[k*n + i, :] is W_r[k, i, :]
            for i in range(n):
                w_row = W_r[k, i, :]
                for j in range(m):
                    u_row = U_r[i, j, :]
                    v_row = V_r[j, k, :]
                    val = np.dot(np.dot(u_row, v_row), w_row) # (u*v).w
                    if not np.isclose(val, 1.0, rtol=1e-10, atol=1e-12):
                        return False
        return True

    def try_rank(rank):
        nonlocal best_rank, best_U, best_V, best_W
        if rank >= best_rank:
            return False
        
        # Random search
        # Dimensions:
        # U: (n*m, rank)
        # V: (m*p, rank)
        # W: (p*n, rank)
        
        # We can fix one matrix to normalize (e.g., set some entries to 1) to avoid scaling ambiguity?
        # Or just search random rationals with small numerators/denominators.
        
        # Let's try small integer coefficients first: -1, 0, 1
        # And then -2, 2, -0.5, 0.5
        
        # Number of variables is (n*m + m*p + p*n) * rank
        # For rank 54: (6 + 27 + 18) * 54 = 51 * 54 = 2754 variables.
        # Too many for brute force.
        
        # We need a smarter search.
        # Maybe greedy?
        # Or use the structure.
        
        # Let's try a random walk / hill climbing on integer matrices.
        # Start with random integers in {-1, 0, 1}.
        # Check error.
        # If error is small, refine.
        
        # Since exact solution is required, random integer search is unlikely to find it unless rank is very close to schoolbook and structure is simple.
        # However, for 2x3x9, there might be a known lower bound.
        # Lower bound: max(n*m, m*p, p*n) = max(6, 27, 18) = 27.
        # Schoolbook is 54.
        # Is rank 27 possible?
        # If rank 27, U is 6x27, V is 27x27, W is 18x27.
        
        # Let's attempt to search for rank 27, 28, ... up to best_rank-1.
        # We will use a simple random restart with local search.
        
        for _ in range(20): # 20 restarts per rank
            if time.time() > deadline:
                return False
            
            U_list = [[0]*rank for _ in range(n*m)]
            V_list = [[0]*rank for _ in range(m*p)]
            W_list = [[0]*rank for _ in range(p*n)]
            
            # Initialize with small random integers
            for i in range(len(U_list)):
                for r in range(rank):
                    U_list[i][r] = np.random.randint(-1, 2)
            for i in range(len(V_list)):
                for r in range(rank):
                    V_list[i][r] = np.random.randint(-1, 2)
            for i in range(len(W_list)):
                for r in range(rank):
                    W_list[i][r] = np.random.randint(-1, 2)
            
            # Local search: flip random entries
            improved = False
            for step in range(50):
                if time.time() > deadline:
                    return False
                
                # Randomly pick a variable and change it
                mat_idx = np.random.randint(0, 3)
                if mat_idx == 0:
                    row = np.random.randint(0, n*m)
                    col = np.random.randint(0, rank)
                    # Try changing value
                    old_val = U_list[row][col]
                    new_val = np.random.randint(-2, 3)
                    if new_val == old_val:
                        continue
                    U_list[row][col] = new_val
                    
                    # Check if it improved or is valid
                    # This is expensive.
                    # We should compute the error tensor once.
                    
                    # Let's restructure: compute error, then do gradient-like steps?
                    # For integers, "gradient" is discrete.
                    
                    # To speed up, we can just check validity periodically.
                    if step % 10 == 0:
                        if check_validity_fast(rank, U_list, V_list, W_list):
                            # Verify with exact
                            if check_validity(rank, U_list, V_list, W_list):
                                best_rank = rank
                                best_U, best_V, best_W = U_list, V_list, W_list
                                return True
            # If not found, try again
            
        return False

    # Search for lower ranks
    # Start from max lower bound
    # Lower bound is max(n*m, m*p, p*n)
    lb = max(n*m, m*p, p*n)
    
    for r in range(lb, best_rank):
        if time.time() > deadline:
            break
        if try_rank(r):
            # We found a better rank
            # We can try even lower?
            # try_rank returns True if found.
            # We should continue searching lower?
            # The loop goes upwards, so we stop at first found if we want minimal.
            # But we want minimal, so we start from lb upwards.
            # If we find r, we can break?
            # Yes, because we are searching lb, lb+1, ...
            # So the first one found is the minimum found.
            # But maybe a lower one exists that we missed?
            # Unlikely if we search from bottom up.
            break
            
    return {"U": best_U, "V": best_V, "W": best_W}
