"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [3, 4, 7]
DEADLINE = 145.0  # Safety margin before 150s limit

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook decomposition baseline (rank 84)
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
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
            for k in range(p):
                for j in range(m):
                    s = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    def compute_M(U, V, rank):
        """Compute M[k*n+i, r] = sum_j U[i*m+j, r] * V[j*p+k, r] vectorized."""
        M = np.zeros((pn, rank))
        for i in range(n):
            for k in range(p):
                row = np.zeros(rank)
                for j in range(m):
                    row += U[i*m + j, :] * V[j*p + k, :]
                M[k*n + i, :] = row
        return M
    
    def find_W_for_M(M, rank, max_entries=2, coeff_range=5):
        """Find W such that M[i,:] dot W[i,:] = 1 for all i, with sparse integer/half-integer W."""
        W = np.zeros((pn, rank))
        for idx in range(pn):
            m_row = M[idx, :]
            nonzero = np.nonzero(m_row)[0]
            if len(nonzero) == 0:
                return None
            
            # Try single entry
            for r in nonzero:
                w_val = 1.0 / m_row[r]
                if abs(w_val - round(w_val)) < 1e-9 or abs(w_val*2 - round(w_val*2)) < 1e-9:
                    W[idx, r] = w_val
                    break
            else:
                # Try two entries
                found = False
                for a_idx in range(len(nonzero)):
                    for b_idx in range(a_idx+1, len(nonzero)):
                        r1, r2 = nonzero[a_idx], nonzero[b_idx]
                        for a in range(-coeff_range, coeff_range+1):
                            for b in range(-coeff_range, coeff_range+1):
                                if a == 0 and b == 0:
                                    continue
                                if abs(a*m_row[r1] + b*m_row[r2] - 1.0) < 1e-9:
                                    W[idx, r1] = a
                                    W[idx, r2] = b
                                    found = True
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if not found:
                    return None
        return W
    
    def try_random_search(rank, num_trials=100, time_limit=None):
        """Random U,V with {-1,0,1} entries, then solve for W."""
        for trial in range(num_trials):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            density = random.choice([0.3, 0.4, 0.5, 0.6])
            U = np.random.choice([-1, 0, 1], size=(nm, rank), p=[density/2, 1-density, density/2])
            V = np.random.choice([-1, 0, 1], size=(mp, rank), p=[density/2, 1-density, density/2])
            
            M = compute_M(U, V, rank)
            W = find_W_for_M(M, rank)
            
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_permutation_structure(rank, time_limit=None):
        """
        Use permutation-based structure: assign each (i,j) to a unique rank position,
        each (j,k) to a unique rank position, ensuring M has simple structure.
        """
        for trial in range(50):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            # Random permutation assignment for U
            U = np.zeros((nm, rank))
            perm_U = list(range(rank))
            random.shuffle(perm_U)
            for idx in range(nm):
                if idx < rank:
                    U[idx, perm_U[idx]] = 1.0
                else:
                    # Multiple rows can share a position
                    U[idx, perm_U[idx % rank]] = 1.0
            
            V = np.zeros((mp, rank))
            perm_V = list(range(rank))
            random.shuffle(perm_V)
            for idx in range(mp):
                if idx < rank:
                    V[idx, perm_V[idx]] = 1.0
                else:
                    V[idx, perm_V[idx % rank]] = 1.0
            
            M = compute_M(U, V, rank)
            W = find_W_for_M(M, rank)
            
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_sparse_one_hot(rank, time_limit=None):
        """
        One-hot style: each row of U has exactly one 1, each row of V has exactly one 1.
        This makes M entries be small integers.
        """
        for trial in range(100):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            
            U = np.zeros((nm, rank))
            V = np.zeros((mp, rank))
            
            # Random one-hot assignment
            for i in range(nm):
                pos = random.randint(0, rank-1)
                U[i, pos] = 1.0
            for j in range(mp):
                pos = random.randint(0, rank-1)
                V[j, pos] = 1.0
            
            M = compute_M(U, V, rank)
            W = find_W_for_M(M, rank, coeff_range=3)
            
            if W is not None and check_decomposition(U, V, W, rank):
                return (U, V, W)
        
        return None
    
    def try_local_improvement(U, V, W, current_rank, time_limit=None):
        """
        Try to reduce rank by merging columns or zeroing out unused ones.
        """
        if time_limit is not None and time.time() - start_time > time_limit:
            return None
        
        # Check for zero columns
        U_copy = U.copy()
        V_copy = V.copy()
        W_copy = W.copy()
        
        # Try removing columns that are all zero
        for r in range(current_rank):
            if time_limit is not None and time.time() - start_time > time_limit:
                return None
            if np.all(U_copy[:, r] == 0) and np.all(V_copy[:, r] == 0):
                # This column is unused, try to remove it
                new_U = np.delete(U_copy, r, axis=1)
                new_V = np.delete(V_copy, r, axis=1)
                new_W = np.delete(W_copy, r, axis=1)
                new_rank = current_rank - 1
                if check_decomposition(new_U, new_V, new_W, new_rank):
                    return (new_U, new_V, new_W)
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Theoretical lower bound: max(n*p, n*m, m*p) = max(21, 12, 28) = 28
    # But for matrix multiplication, the known bounds are tighter
    for rank in range(25, min(n*m*p, 84)):
        if time.time() - start_time > DEADLINE - 5:
            break
        
        # Try different strategies
        result = None
        
        # Strategy 1: One-hot sparse (most structured, most likely to work)
        result = try_sparse_one_hot(rank, time_limit=DEADLINE)
        
        if result is None:
            # Strategy 2: Random with {-1,0,1}
            result = try_random_search(rank, num_trials=80, time_limit=DEADLINE)
        
        if result is None:
            # Strategy 3: Permutation structure
            result = try_permutation_structure(rank, time_limit=DEADLINE)
        
        if result is not None:
            U, V, W = result
            if rank < best_rank:
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
            
            # Try to improve further via local optimization
            improved = try_local_improvement(U, V, W, rank, time_limit=DEADLINE)
            if improved is not None:
                U2, V2, W2 = improved
                if len(U2[0]) < best_rank:
                    best_rank = len(U2[0])
                    best_U = U2
                    best_V = V2
                    best_W = W2
            
            break
    
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
