import numpy as np
import time
from fractions import Fraction
import itertools

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

    def is_valid_exact(U, V, W, rank):
        """Exact validation using Fractions"""
        T = np.zeros((n, m, p), dtype=object)
        for r in range(rank):
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True

    def is_valid_numeric(U, V, W, rank, atol=1e-8):
        """Fast numeric validation"""
        U_arr = U.astype(float)
        V_arr = V.astype(float)
        W_arr = W.astype(float)
        
        Rec = np.zeros((n, m, p))
        for r in range(rank):
            u = U_arr[:, r].reshape(n, m)
            v = V_arr[:, r].reshape(m, p)
            w = W_arr[:, r].reshape(p, n)
            Rec += np.einsum('ij,jk,ki->ijk', u, v, w, optimize=True)
        return np.allclose(Rec, 1.0, atol=atol)

    def strassen_3x3x3():
        """Return Strassen decomposition for 3x3x3 (rank 7)"""
        # Strassen's algorithm for 3x3 matrices
        # Using the standard 7-product decomposition
        rank = 7
        U = np.zeros((9, rank))
        V = np.zeros((9, rank))
        W = np.zeros((9, rank))
        
        # Strassen products for 2x2 blocks, extended to 3x3
        # We'll use a known rank-7 decomposition for 3x3x3
        # This is based on the standard Strassen algorithm
        
        # Define the 7 products
        # P1 = (A11)(B11 + B22)
        # P2 = (A11 + A22)(B22)
        # P3 = (A12)(B21)
        # P4 = (A21)(B12)
        # P5 = (A11 + A22)(B11 + B22)
        # P6 = (A11 - A22)(B12 + B22)
        # P7 = (A12 + A22)(B21 - B11)
        
        # For 3x3, we need to adapt. Let's use a simpler approach:
        # Just return a valid rank-7 solution if we can find one,
        # otherwise fall back to a higher rank.
        
        # Actually, let's construct a valid decomposition using known patterns
        # For now, return None to indicate we need a different approach
        
        return None

    def try_structured_decomposition(rank):
        """Try structured patterns for given rank"""
        if time.time() > deadline:
            return None
        
        # Try to build a decomposition using Kronecker products and structured patterns
        # For 3x3x5, we can try to combine smaller decompositions
        
        # Pattern 1: Use Strassen for the 3x3 part and schoolbook for the extra column
        # This would give rank = 7 + 3*3*2 = 7 + 18 = 25
        
        if rank == 25:
            # Try to construct a rank-25 solution
            # First 7 columns: Strassen-like for first 3 columns
            # Last 18 columns: schoolbook for last 2 columns
            
            U = np.zeros((9, rank))
            V = np.zeros((15, rank))
            W = np.zeros((15, rank))
            
            # For the first 3 columns (k=0,1,2), use a rank-7 pattern
            # For the last 2 columns (k=3,4), use schoolbook (rank 18)
            
            # Schoolbook part for k=3,4
            col = 7
            for i in range(3):
                for j in range(3):
                    for k in [3, 4]:
                        U[i * 3 + j, col] = 1
                        V[j * 5 + k, col] = 1
                        W[k * 3 + i, col] = 1
                        col += 1
            
            # Now we need to fill the first 7 columns with a valid pattern
            # that handles k=0,1,2 with rank 7
            
            # Let's try a systematic search for the first 7 columns
            # We need: sum_{r=0}^{6} U[i,j,r] * V[j,k,r] * W[k,i,r] = 1 for k in {0,1,2}
            # and 0 for k in {3,4}
            
            # This is complex. Let's try random search with constraints.
            
            np.random.seed(42)
            for attempt in range(10000):
                if time.time() > deadline:
                    return None
                
                # Random initialization for the first 7 columns
                U[:, :7] = np.random.randint(-2, 3, (9, 7))
                V[:, :7] = np.random.randint(-2, 3, (15, 7))
                W[:, :7] = np.random.randint(-2, 3, (15, 7))
                
                # Set V and W to 0 for k=3,4 in the first 7 columns
                # V indices: j*5+k, so for k=3: j*5+3 for j in 0,1,2 -> indices 3, 8, 13
                # For k=4: j*5+4 for j in 0,1,2 -> indices 4, 9, 14
                V[[3, 8, 13, 4, 9, 14], :7] = 0
                
                # W indices: k*3+i, so for k=3: 3*3+i = 9,10,11
                # For k=4: 4*3+i = 12,13,14
                W[[9, 10, 11, 12, 13, 14], :7] = 0
                
                # Check if this works
                if is_valid_numeric(U, V, W, rank):
                    # Convert to Fractions and verify exactly
                    U_frac = np.array(U, dtype=object)
                    V_frac = np.array(V, dtype=object)
                    W_frac = np.array(W, dtype=object)
                    for r in range(rank):
                        for idx in range(U_frac.shape[0]):
                            U_frac[idx, r] = Fraction(int(U[idx, r]))
                        for idx in range(V_frac.shape[0]):
                            V_frac[idx, r] = Fraction(int(V[idx, r]))
                        for idx in range(W_frac.shape[0]):
                            W_frac[idx, r] = Fraction(int(W[idx, r]))
                    
                    if is_valid_exact(U_frac, V_frac, W_frac, rank):
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        # Pattern 2: Try rank 24 with a different structure
        if rank == 24:
            np.random.seed(43)
            for attempt in range(5000):
                if time.time() > deadline:
                    return None
                
                U = np.random.randint(-2, 3, (9, rank))
                V = np.random.randint(-2, 3, (15, rank))
                W = np.random.randint(-2, 3, (15, rank))
                
                if is_valid_numeric(U, V, W, rank, atol=1e-6):
                    U_frac = np.array(U, dtype=object)
                    V_frac = np.array(V, dtype=object)
                    W_frac = np.array(W, dtype=object)
                    for r in range(rank):
                        for idx in range(U_frac.shape[0]):
                            U_frac[idx, r] = Fraction(int(U[idx, r]))
                        for idx in range(V_frac.shape[0]):
                            V_frac[idx, r] = Fraction(int(V[idx, r]))
                        for idx in range(W_frac.shape[0]):
                            W_frac[idx, r] = Fraction(int(W[idx, r]))
                    
                    if is_valid_exact(U_frac, V_frac, W_frac, rank):
                        return {"U": U_frac, "V": V_frac, "W": W_frac}
        
        return None

    def search_systematic():
        """Systematic search for lower rank decompositions"""
        nonlocal best_solution, best_rank
        search_start = time.time()
        
        # Try structured decompositions first
        for rank in [25, 24, 23, 22, 21, 20]:
            if time.time() > deadline:
                break
            if rank >= best_rank:
                continue
            
            result = try_structured_decomposition(rank)
            if result is not None:
                best_solution = result
                best_rank = rank
                return True
        
        # Try more random search with better heuristics
        for rank in range(20, 25):
            if time.time() > deadline:
                break
            if rank >= best_rank:
                continue
            
            np.random.seed(100 + rank)
            max_attempts = 20000
            
            for attempt in range(max_attempts):
                if time.time() > deadline:
                    break
                if time.time() - search_start > 100:  # Leave margin
                    break
                
                # Generate sparse random matrices
                density = 0.3
                U = np.zeros((9, rank))
                V = np.zeros((15, rank))
                W = np.zeros((15, rank))
                
                for r in range(rank):
                    # U: 9 entries, sparse
                    num_nonzero = max(1, int(9 * density))
                    indices = np.random.choice(9, num_nonzero, replace=False)
                    U[indices, r] = np.random.choice([-2, -1, 1, 2], num_nonzero)
                    
                    # V: 15 entries, sparse
                    num_nonzero = max(1, int(15 * density))
                    indices = np.random.choice(15, num_nonzero, replace=False)
                    V[indices, r] = np.random.choice([-2, -1, 1, 2], num_nonzero)
                    
                    # W: 15 entries, sparse
                    num_nonzero = max(1, int(15 * density))
                    indices = np.random.choice(15, num_nonzero, replace=False)
                    W[indices, r] = np.random.choice([-2, -1, 1, 2], num_nonzero)
                
                if is_valid_numeric(U, V, W, rank, atol=1e-7):
                    U_frac = np.array(U, dtype=object)
                    V_frac = np.array(V, dtype=object)
                    W_frac = np.array(W, dtype=object)
                    for r in range(rank):
                        for idx in range(U_frac.shape[0]):
                            U_frac[idx, r] = Fraction(int(U[idx, r]))
                        for idx in range(V_frac.shape[0]):
                            V_frac[idx, r] = Fraction(int(V[idx, r]))
                        for idx in range(W_frac.shape[0]):
                            W_frac[idx, r] = Fraction(int(W[idx, r]))
                    
                    if is_valid_exact(U_frac, V_frac, W_frac, rank):
                        best_solution = {"U": U_frac, "V": V_frac, "W": W_frac}
                        best_rank = rank
                        return True
        
        return False

    # Try to find a better solution
    search_systematic()

    def to_lists(mat):
        """Convert matrix of Fractions to list of lists of floats"""
        result = []
        for i in range(mat.shape[0]):
            row = []
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if isinstance(val, Fraction):
                    row.append(float(val))
                else:
                    row.append(float(val))
            result.append(row)
        return result

    return {
        "U": to_lists(best_solution["U"]),
        "V": to_lists(best_solution["V"]),
        "W": to_lists(best_solution["W"])
    }
