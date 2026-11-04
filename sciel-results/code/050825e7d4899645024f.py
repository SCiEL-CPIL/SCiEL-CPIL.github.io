import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    def get_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}, rank

    schoolbook_sol, schoolbook_rank = get_schoolbook()
    best_solution = schoolbook_sol
    best_rank = schoolbook_rank

    def is_valid(U, V, W, rank):
        # U, V, W are lists of lists of Fractions
        # Check if the tensor decomposition is exact
        # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # We need T[i,j,k] = 1 for all i,j,k
        
        # Build the full tensor from the decomposition
        T = np.zeros((n, m, p))
        for r in range(rank):
            u_col = U[:, r]
            v_col = V[:, r]
            w_col = W[:, r]
            
            # u_col has n*m entries, v_col has m*p entries, w_col has p*n entries
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        # U index: i*m + j
                        # V index: j*p + k
                        # W index: k*n + i
                        T[i, j, k] += u_col[i * m + j] * v_col[j * p + k] * w_col[k * n + i]
        
        # Check if T is all 1s
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True

    def is_valid_numeric(U, V, W, rank):
        # Quick numeric check for candidate generation
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U_arr[:, r]
            v = V_arr[:, r]
            w = W_arr[:, r]
            u_reshaped = u.reshape(n, m)
            v_reshaped = v.reshape(m, p)
            w_reshaped = w.reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u_reshaped, v_reshaped, w_reshaped, optimize=True)
        return np.allclose(Rec, 1.0, atol=1e-6)

    def search_exact_solutions():
        nonlocal best_solution, best_rank
        search_start = time.time()
        max_search_time = 120.0
        
        # We'll use a structured approach based on the algebraic structure
        # For 3x3x5, we can try to find solutions with small integer entries
        
        # First, let's try a more systematic search with small integers
        # We'll use random initialization but with more structured patterns
        
        import itertools
        
        # Try ranks from schoolbook down
        for rank in range(schoolbook_rank - 1, 20, -1):
            if time.time() > search_start + max_search_time:
                break
            if rank >= best_rank:
                continue
            
            # Try a few random configurations
            attempts = 0
            max_attempts = 500
            np.random.seed(42 + rank)
            
            while attempts < max_attempts:
                if time.time() > search_start + max_search_time:
                    break
                if time.time() > deadline:
                    return
                
                attempts += 1
                
                # Generate random sparse matrices with small integer entries
                U = np.random.randint(-2, 3, (n * m, rank))
                V = np.random.randint(-2, 3, (m * p, rank))
                W = np.random.randint(-2, 3, (p * n, rank))
                
                # Quick numeric check
                if is_valid_numeric(U, V, W, rank):
                    # Convert to Fractions for exact verification
                    U_frac = [[Fraction(x) for x in row] for row in U]
                    V_frac = [[Fraction(x) for x in row] for row in V]
                    W_frac = [[Fraction(x) for x in row] for row in W]
                    
                    if is_valid(U_frac, V_frac, W_frac, rank):
                        best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = rank
                        return True
            
            # If we didn't find a solution at this rank, try structured patterns
            # Pattern 1: Block diagonal structure
            # Pattern 2: Strassen-like block structure
            
            # Try block structure for 3x3x5
            # Split into 3x3x3 and 3x3x2 blocks
            if rank >= 27:  # 7 + 20 = 27 minimum for this split
                # Try to combine Strassen for 3x3x3 with schoolbook for 3x3x2
                # This is complex, so we'll skip for now and rely on random search
                pass
        
        return False

    # Try to find a better solution
    search_exact_solutions()

    def to_lists(mat):
        # Convert Fractions to their string representation or keep as numbers
        # The task says integer/half-integer entries, so we can return them as numbers
        result = []
        for row in mat:
            result.append([float(x) for x in row])
        return result

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
