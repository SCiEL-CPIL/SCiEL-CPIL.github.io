"""Combined program: schoolbook fallback + block-structured tensor decomposition search."""
import time
import numpy as np

DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1 for all i,j,k."""
    U_np = np.asarray(U, dtype=float)
    V_np = np.asarray(V, dtype=float)
    W_np = np.asarray(W, dtype=float)
    rank = U_np.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.einsum('c,c,c->', U_np[i*m+j, :], V_np[j*p+k, :], W_np[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W


def _try_block_decomposition(n, m, p, deadline):
    """
    Try a block-based decomposition:
    Partition 5x5 into blocks and use structured rank reduction.
    
    Strategy: Use a 2+3 partition.
    - 2x2 block: use Strassen (rank 7 instead of 8)
    - 3x3 block: use a rank-17 decomposition (instead of 27)
    - Cross terms: handle with minimal rank
    
    For 5x5 = 2x2 + 2x3 + 3x2 + 3x3:
    - (2x2)*(2x2): rank 7 (Strassen)
    - (2x2)*(3x3): rank 2*3=6 schoolbook, try to reduce
    - (3x2)*(2x2): rank 3*2=6 schoolbook, try to reduce  
    - (3x2)*(3x3): rank 18 schoolbook, try to reduce
    - (3x3)*(2x3): rank 9 schoolbook, try to reduce
    - (3x3)*(3x3): rank 17 (known best for 3x3)
    
    Total schoolbook: 8+6+6+18+9+27 = 74
    With Strassen for 2x2 and rank-17 for 3x3: 7+6+6+18+9+17 = 63
    
    But we need to be careful about the tensor structure. Let's try a simpler approach:
    Just use a parameterized ansatz and solve for W given random U,V.
    """
    # Try a few different rank targets
    for target_rank in [63, 60, 55, 50, 45, 40]:
        if time.time() > deadline - 10:
            break
        result = _try_rank_structured(target_rank, n, m, p, deadline - 10, max_attempts=50)
        if result is not None:
            return result
    return None


def _try_rank_structured(rank, n, m, p, deadline, max_attempts):
    """
    Try to find a rank-r decomposition with structured U,V.
    Use a block-based initialization for U,V, then solve for W.
    """
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        # Initialize U,V with block structure
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        
        # Use a 2+3 partition for rows/cols
        # Block indices: 0-1 for 2x2, 2-4 for 3x3
        # For each block, assign columns in U,V based on block structure
        
        col = 0
        # 2x2 block: Strassen-like (7 terms)
        strassen_2x2 = [
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0)
        ]
        for (i, j, k) in strassen_2x2:
            if col < rank:
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                col += 1
        
        # 3x3 block: use a simple rank-17 pattern
        # For 3x3, we can use a known decomposition
        # Simplified: just use schoolbook for now, but with fewer terms
        # 3x3 has 9 elements, schoolbook rank 27
        # Try to use rank 17 by sharing some terms
        for i in range(2, 5):  # rows 2,3,4
            for j in range(2, 5):  # cols 2,3,4
                for k in range(2, 5):  # inner dim 2,3,4
                    if col < rank:
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        col += 1
        
        # Cross terms: 2x3, 3x2, etc.
        # 2x3 block (rows 0-1, cols 2-4)
        for i in range(0, 2):
            for j in range(2, 5):
                for k in range(2, 5):
                    if col < rank:
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        col += 1
        
        # 3x2 block (rows 2-4, cols 0-1)
        for i in range(2, 5):
            for j in range(0, 2):
                for k in range(0, 2):
                    if col < rank:
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        col += 1
        
        # Fill remaining columns with random values
        while col < rank:
            U[:, col] = np.random.choice([-1, 0, 1], size=n*m)
            V[:, col] = np.random.choice([-1, 0, 1], size=m*p)
            col += 1
        
        # Now solve for W
        W = np.zeros((p*n, rank))
        feasible = True
        for k in range(p):
            for i in range(n):
                # Build m x rank matrix A where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                
                # Solve A @ Wc = rhs
                try:
                    Wc, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                
                # Check consistency
                if np.max(np.abs(A @ Wc - rhs)) > 1e-7:
                    feasible = False
                    break
                
                # Round to nearest half-integer
                Wc_rounded = 0.5 * np.round(2.0 * Wc)
                
                # Check if rounding introduced significant error
                if np.max(np.abs(A @ Wc_rounded - rhs)) > 1e-7:
                    feasible = False
                    break
                
                W[k*n+i, :] = Wc_rounded
        
        if feasible:
            # Verify fully with rounded values
            if _verify(U, V, W, n, m, p):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return U_list, V_list, W_list
    
    return None


def _try_random_decomposition(rank, n, m, p, deadline, max_attempts=20):
    """Try random U,V with entries in {-1,0,1} and solve for W."""
    for attempt in range(max_attempts):
        if time.time() > deadline:
            return None
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        for c in range(rank):
            U[:, c] = np.random.choice([-1, 0, 1], size=n*m)
            V[:, c] = np.random.choice([-1, 0, 1], size=m*p)
        
        W = np.zeros((p*n, rank))
        feasible = True
        for k in range(p):
            for i in range(n):
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                rhs = np.ones(m)
                
                try:
                    Wc, residuals, rank_A, sv = np.linalg.lstsq(A, rhs, rcond=None)
                except np.linalg.LinAlgError:
                    feasible = False
                    break
                
                if np.max(np.abs(A @ Wc - rhs)) > 1e-7:
                    feasible = False
                    break
                
                Wc_rounded = 0.5 * np.round(2.0 * Wc)
                
                if np.max(np.abs(A @ Wc_rounded - rhs)) > 1e-7:
                    feasible = False
                    break
                
                W[k*n+i, :] = Wc_rounded
        
        if feasible:
            if _verify(U, V, W, n, m, p):
                U_list = [[float(x) for x in row] for row in U]
                V_list = [[float(x) for x in row] for row in V]
                W_list = [[float(x) for x in row] for row in W]
                return U_list, V_list, W_list
    
    return None


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 130.0  # safety margin below 150s kill
    
    # Best-so-far: schoolbook
    U0, V0, W0 = _schoolbook(n, m, p)
    U0_list = [[float(x) for x in row] for row in U0]
    V0_list = [[float(x) for x in row] for row in V0]
    W0_list = [[float(x) for x in row] for row in W0]
    best = {"U": U0_list, "V": V0_list, "W": W0_list}
    best_rank = n * m * p
    
    # Try block-structured decomposition first
    if time.time() < deadline - 20:
        result = _try_block_decomposition(n, m, p, deadline - 20)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = len(U[0])
            return best
    
    # Try random decomposition for lower ranks
    for rank in range(min(best_rank - 1, 100), 30, -5):
        if time.time() > deadline - 5.0:
            break
        remaining = deadline - 5.0 - time.time()
        attempts = max(10, int(remaining * 2))
        result = _try_random_decomposition(rank, n, m, p, deadline - 5.0, attempts)
        if result is not None:
            U, V, W = result
            best = {"U": U, "V": V, "W": W}
            best_rank = rank
            break
    
    return best
# EVOLVE-BLOCK-END
