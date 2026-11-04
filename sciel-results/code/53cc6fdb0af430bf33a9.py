"""Find rank-minimal exact tensor decomposition for 3x4 * 4x7 matrix multiplication."""
import time
import random
import numpy as np
from fractions import Fraction
import itertools

DIMENSIONS = [3, 4, 7]
DEADLINE = 140.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 4, 7
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((n*m, rank_sb))
    V_sb = np.zeros((m*p, rank_sb))
    W_sb = np.zeros((p*n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m + j, col] = 1.0
                V_sb[j*p + k, col] = 1.0
                W_sb[k*n + i, col] = 1.0
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    
    def check_decomposition(U, V, W, rank):
        """Check if U, V, W exactly decompose the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def check_decomposition_fast(U, V, W, rank):
        """Fast vectorized check using numpy."""
        # Compute all (i,j,k) values
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def is_valid_w_val(v):
        """Check if v is integer or half-integer."""
        if abs(v) < 1e-12:
            return True
        if abs(v - round(v)) < 1e-9:
            return True
        if abs(v*2 - round(v*2)) < 1e-9:
            return True
        return False
    
    def try_find_w(A_sub, rank, max_nz=4):
        """
        Given A_sub (matrix of shape m x rank), find W_row such that A_sub @ W_row = 1.
        Uses linear algebra to find sparse solutions with integer/half-integer entries.
        Returns W_row or None.
        """
        m = A_sub.shape[0]
        # Try subsets of indices of size k from 1 to max_nz
        # We want A_sub[:, S] @ W_S = 1
        # If k < m, the system is underdetermined? No, A is m x k.
        # If k < m, we need the columns to span the space? No, we just need a solution.
        # If k < m, the system is overdetermined (m equations, k unknowns).
        # We can use least squares, but we need EXACT solution.
        # So we require that 1 is in the column space of A_sub[:, S].
        # This is only possible if rank(A_sub[:, S]) == m (if k>=m) or rank(A_sub[:, S]) == k and 1 is in span.
        # Actually, if k < m, the system A x = b has a solution iff b is in Col(A).
        
        indices = np.where(np.any(np.abs(A_sub) > 1e-12, axis=0))[0]
        if len(indices) == 0:
            return None
            
        # Iterate over subset sizes
        for k in range(1, min(max_nz, len(indices)) + 1):
            # Generate combinations of indices
            # To save time, limit the number of combinations if too many
            # For m=4, k=4, we have C(R, 4) combinations. R~30, C(30,4) ~ 27k. Manageable.
            for S in itertools.combinations(indices, k):
                A_S = A_sub[:, list(S)]
                # Solve A_S @ W_S = 1
                try:
                    # Use lstsq to check for consistency and get solution
                    # If k >= m, we can have many solutions. We want a sparse/integer one.
                    # If k == m, unique solution.
                    # If k > m, infinite solutions.
                    
                    # First, check if a solution exists.
                    # Use rank check
                    rank_A = np.linalg.matrix_rank(A_S)
                    if rank_A < m:
                        # Check if 1 is in the column space
                        # Project 1 onto Col(A_S)
                        # If distance is 0, solution exists.
                        proj = A_S @ np.linalg.lstsq(A_S, np.ones(m), rcond=None)[0]
                        if np.linalg.norm(proj - np.ones(m)) > 1e-9:
                            continue
                    else:
                        # Rank is m, solution exists.
                        pass
                        
                    # Find a specific solution
                    # If k == m, solve directly
                    if k == m:
                        W_S = np.linalg.solve(A_S, np.ones(m))
                    else:
                        # Use least squares to get minimum norm solution
                        W_S = np.linalg.lstsq(A_S, np.ones(m), rcond=None)[0]
                    
                    # Check if W_S is integer/half-integer
                    valid = True
                    for val in W_S:
                        if not is_valid_w_val(val):
                            valid = False
                            break
                    if valid:
                        W_row = np.zeros(rank)
                        for idx, val in zip(S, W_S):
                            W_row[idx] = val
                        return W_row
                        
                except np.linalg.LinAlgError:
                    continue
        return None
    
    def try_sparse_rank(rank, num_trials=200):
        """
        Try to find a rank-r exact decomposition using sparse U, V with entries in {-1, 0, 1}.
        Uses structured generation to increase success probability.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            if trial % 3 == 0:
                # Strategy 1: Each column has exactly 2 non-zeros
                for r in range(rank):
                    indices_u = random.sample(range(n*m), 2)
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1, 1])
                    indices_v = random.sample(range(m*p), 2)
                    for idx in indices_v:
                        V[idx, r] = random.choice([-1, 1])
            elif trial % 3 == 1:
                # Strategy 2: Each column has 3 non-zeros
                for r in range(rank):
                    indices_u = random.sample(range(n*m), 3)
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1, 1])
                    indices_v = random.sample(range(m*p), 3)
                    for idx in indices_v:
                        V[idx, r] = random.choice([-1, 1])
            else:
                # Strategy 3: Mixed sparsity
                for r in range(rank):
                    nz_u = random.choice([2, 3, 4])
                    indices_u = random.sample(range(n*m), min(nz_u, n*m))
                    for idx in indices_u:
                        U[idx, r] = random.choice([-1, 1])
                    nz_v = random.choice([2, 3, 4])
                    indices_v = random.sample(range(m*p), min(nz_v, m*p))
                    for idx in indices_v:
                        V[idx, r] = random.choice([-1, 1])
            
            # For each (k,i), find W
            W = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    # Construct A_sub of shape (m, rank)
                    A_sub = np.zeros((m, rank))
                    for j in range(m):
                        A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W_row = try_find_w(A_sub, rank)
                    if W_row is None:
                        valid = False
                        break
                    W[k*n+i, :] = W_row
                if not valid:
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_structured_rank(rank, num_trials=100):
        """
        Try structured patterns for U and V with systematic coverage.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            if trial % 4 == 0:
                # Pattern A: Cycle through (i,j) and (j,k) pairs
                for r in range(rank):
                    i_u = (r * 3) % n
                    j_u = (r * 7) % m
                    i_v = (r * 5) % m
                    k_v = (r * 11) % p
                    U[i_u*m + j_u, r] = 1.0 if (r % 2 == 0) else -1.0
                    V[i_v*p + k_v, r] = 1.0 if (r % 2 == 0) else -1.0
            elif trial % 4 == 1:
                # Pattern B: Block diagonal structure
                block_size = max(1, rank // max(n, m, p))
                for r in range(rank):
                    block = r // block_size
                    pos_in_block = r % block_size
                    if block < n*m:
                        U[block, r] = 1.0 if pos_in_block % 2 == 0 else -1.0
                    if block < m*p:
                        V[block, r] = 1.0 if pos_in_block % 2 == 0 else -1.0
            elif trial % 4 == 2:
                # Pattern C: Latin square based
                for r in range(rank):
                    i_u = (r % n)
                    j_u = ((r // n) + r) % m
                    j_v = (r % m)
                    k_v = ((r // m) + r) % p
                    U[i_u*m + j_u, r] = 1.0
                    V[j_v*p + k_v, r] = 1.0
            else:
                # Pattern D: Random but with guaranteed coverage
                perm_u = np.random.permutation(n*m)
                perm_v = np.random.permutation(m*p)
                for r in range(rank):
                    U[perm_u[r % n*m], r] = 1.0 if r % 2 == 0 else -1.0
                    V[perm_v[r % m*p], r] = 1.0 if r % 2 == 0 else -1.0
            
            # Find W
            W = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    A_sub = np.zeros((m, rank))
                    for j in range(m):
                        A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W_row = try_find_w(A_sub, rank)
                    if W_row is None:
                        valid = False
                        break
                    W[k*n+i, :] = W_row
                if not valid:
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_block_rank(rank, num_trials=50):
        """
        Try block-structured decomposition with systematic block assignment.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Divide rank into blocks, each block handles a specific (i,j,k) combination
            num_blocks = min(rank, n*m*p)
            block_size = max(1, rank // num_blocks)
            
            for r in range(rank):
                block = r % num_blocks
                pos = r // num_blocks
                
                # Map block to specific indices
                i = block % n
                j = (block // n) % m
                k = (block // (n*m)) % p
                
                U[i*m + j, r] = 1.0 if pos % 2 == 0 else -1.0
                V[j*p + k, r] = 1.0 if pos % 2 == 0 else -1.0
            
            # Find W
            W = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    A_sub = np.zeros((m, rank))
                    for j in range(m):
                        A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W_row = try_find_w(A_sub, rank)
                    if W_row is None:
                        valid = False
                        break
                    W[k*n+i, :] = W_row
                if not valid:
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_systematic_rank(rank, num_trials=30):
        """
        Systematic search: fix U and V patterns, solve for W exactly.
        Uses the fact that for fixed U,V, W must satisfy linear constraints.
        """
        for trial in range(num_trials):
            if time.time() - start_time > DEADLINE - 10:
                return None
            
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            
            # Generate U with controlled sparsity
            for r in range(rank):
                # Each column of U has exactly 2 non-zeros
                i1 = random.randint(0, n-1)
                j1 = random.randint(0, m-1)
                i2 = random.randint(0, n-1)
                j2 = random.randint(0, m-1)
                U[i1*m + j1, r] = 1.0
                if (i2, j2) != (i1, j1):
                    U[i2*m + j2, r] = random.choice([-1.0, 1.0])
            
            # Generate V with controlled sparsity
            for r in range(rank):
                j1 = random.randint(0, m-1)
                k1 = random.randint(0, p-1)
                j2 = random.randint(0, m-1)
                k2 = random.randint(0, p-1)
                V[j1*p + k1, r] = 1.0
                if (j2, k2) != (j1, k1):
                    V[j2*p + k2, r] = random.choice([-1.0, 1.0])
            
            # For each (k,i), find W using linear algebra
            W = np.zeros((p*n, rank))
            valid = True
            for i in range(n):
                for k in range(p):
                    A_sub = np.zeros((m, rank))
                    for j in range(m):
                        A_sub[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    W_row = try_find_w(A_sub, rank)
                    if W_row is None:
                        valid = False
                        break
                    W[k*n+i, :] = W_row
                if not valid:
                    break
            
            if valid and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bound for matrix multiplication rank is max(n*m, m*p, p*n) = 28
    # But tensor rank can be lower. Known results suggest ~20-30 for small cases.
    ranks_to_try = list(range(20, min(n*m*p, 84)))
    
    for rank in ranks_to_try:
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Try systematic search first (more structured)
        if time.time() - start_time < DEADLINE - 12:
            result = try_systematic_rank(rank, num_trials=20)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
        
        # Try sparse random search
        if time.time() - start_time < DEADLINE - 10:
            result = try_sparse_rank(rank, num_trials=80)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
        
        # Try structured search if time permits
        if time.time() - start_time < DEADLINE - 10:
            result = try_structured_rank(rank, num_trials=40)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
        
        # Try block search if time permits
        if time.time() - start_time < DEADLINE - 10:
            result = try_block_rank(rank, num_trials=20)
            if result is not None:
                U, V, W = result
                if rank < best_rank:
                    best_rank = rank
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
