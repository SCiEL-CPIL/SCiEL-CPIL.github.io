"""Find rank-minimal exact tensor decomposition for 3x5x6 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    n, m, p = DIMENSIONS
    
    # Start with schoolbook rank 90
    best_rank = n * m * p
    best_result = _schoolbook(n, m, p)
    
    # Try to find lower rank via optimization
    # The matrix multiplication tensor T[i,j,k] = sum over (a,b) of A[i,a]*B[a,k]
    # We need U (nm x r), V (mp x r), W (pn x r) such that:
    # T = sum_c U[:,c] * V[:,c] * W[:,c] (outer product sum)
    # In index form: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # Strategy: try ranks from a lower bound upward
    # Lower bound: max(n*m, m*p, p*n) = max(15, 30, 18) = 30
    # But realistically, for 3x5x6, we expect something between 40-70
    
    # Try Winograd-style approach for sub-blocks
    # 3x5x6 can be seen as combining 2x2x2 Strassen blocks with padding
    
    # First, try a structured approach:
    # Split 3x5 into (2x2 + 1x3) and 5x6 into (2x4 + 3x2) etc.
    
    # Let's try a direct optimization for various ranks
    # Use the fact that entries should be simple (integers or half-integers)
    
    # Try rank values that are plausible
    candidate_ranks = [78, 76, 75, 74, 72, 70, 68, 66, 64, 63, 60, 56, 54, 52, 50, 48, 45, 42, 40, 38, 36, 34, 32, 30]
    
    for target_rank in candidate_ranks:
        if time.time() > deadline - 30:
            break
        
        result = _try_rank(n, m, p, target_rank, deadline)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
    
    # If no better found, also try the structured fast-multiplication approach
    if time.time() < deadline - 20:
        structured = _try_structured_fast(n, m, p, deadline)
        if structured is not None:
            r = len(structured['U'][0])
            if r < best_rank:
                best_rank = r
                best_result = structured
    
    return best_result


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _try_rank(n, m, p, target_rank, deadline):
    """Try to find an exact decomposition of given rank using optimization."""
    # The tensor T has dimensions n x m x p
    # T[i,j,k] = 1 if we're computing (AB)[i,k] from A[i,j]*B[j,k]
    # Actually, the multiplication tensor is:
    # T[i,j,k] = delta (it's 1 for all valid (i,j,k) in the bilinear form)
    # The bilinear form is: (AB)[i,k] = sum_j A[i,j] B[j,k]
    # Tensor form: T[i,j,k] = 1 for the coefficient of A[i,j]*B[j,k] in (AB)[i,k]
    # So T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to output (i,k)
    # Which it always does, so T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    
    # We need: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    
    # This is a 3-way tensor decomposition problem.
    # Use a greedy/randomized approach with integer/half-integer constraints.
    
    # Simplified approach: try to build the decomposition column by column
    # Each column c contributes U_col * V_col * W_col (outer product)
    # The sum of these must equal T
    
    # For small enough ranks, we can try a structured approach
    # Let's use the fact that we can pad to use known fast algorithms
    
    # 3x5x6: Try to embed in a larger problem with known fast algorithm
    # Or try direct search for small ranks
    
    # Given time constraints, try a few random restarts with structured initialization
    import random
    random.seed(42)
    
    for attempt in range(5):
        if time.time() > deadline - 25:
            break
        
        result = _randomized_search(n, m, p, target_rank, deadline, attempt)
        if result is not None:
            return result
    
    return None


def _randomized_search(n, m, p, target_rank, deadline, seed):
    """Randomized search for exact decomposition."""
    import random
    random.seed(seed * 1000 + target_rank)
    
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Build the target tensor T[i,j,k] = 1
    # We need sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all i,j,k
    
    # Try structured initialization based on known patterns
    # For 3x5x6, try using a combination of:
    # - 2x2x2 Strassen (rank 7) for a 2x2x2 subblock
    # - Simple multiplication for remaining parts
    
    # Let's try a block-structured approach
    # Split rows: 3 = 2 + 1
    # Split cols: 5 = 4 + 1  
    # Split outputs: 6 = 4 + 2
    
    # Actually, let's try a more systematic approach
    # Use the fact that for 2x2x2, Strassen gives rank 7 vs 8 schoolbook
    # For 3x3x3, best known is 23 (Landsberg)
    
    # For 3x5x6, let's try to construct explicitly
    # This is complex, so let's try a different approach:
    # Use numpy to set up the linear system and solve
    
    # The constraint is: for each (i,j,k):
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # This is nonlinear (trilinear). Let's try fixing two matrices and solving for the third.
    
    # Strategy: 
    # 1. Initialize U, V with random values from {-1, 0, 1}
    # 2. For each (i,k), we need: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all j
    #    This gives m equations for W[k*n+i, c] across c
    # 3. Solve the resulting linear system
    
    # Actually, for fixed U and V, the constraint on W is:
    # For each (i,j,k): W[k*n+i, c] appears in exactly one equation per c
    # Let's rewrite: sum_c (U[i*m+j,c] * V[j*p+k,c]) * W[k*n+i,c] = 1
    # For fixed (i,k), as j varies, we get m equations:
    # sum_c A_jc * W_c = 1, where A_jc = U[i*m+j,c] * V[j*p+k,c]
    # This is a system of m equations in r unknowns (W[k*n+i, c] for c=0..r-1)
    # If m <= r, we can solve it
    
    # Let's try this approach
    for trial in range(3):
        if time.time() > deadline - 20:
            break
        
        # Initialize U and V
        U = np.zeros((nm, target_rank))
        V = np.zeros((mp, target_rank))
        
        # Use a structured initialization
        # For each column c, assign it to a particular (i,j,k) pattern
        # Or use random {-1, 0, 1}
        
        for c in range(target_rank):
            # Assign random non-zero entries
            num_nonzero_u = random.randint(1, 3)
            indices_u = random.sample(range(nm), num_nonzero_u)
            for idx in indices_u:
                U[idx, c] = random.choice([-1, 1])
            
            num_nonzero_v = random.randint(1, 3)
            indices_v = random.sample(range(mp), num_nonzero_v)
            for idx in indices_v:
                V[idx, c] = random.choice([-1, 1])
        
        # Now solve for W
        W = np.zeros((pn, target_rank))
        success = True
        
        for i in range(n):
            for k in range(p):
                # Build system: for each j, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                # A @ W[k*n+i, :] = 1 (vector of m ones)
                A = np.zeros((m, target_rank))
                for j in range(m):
                    for c in range(target_rank):
                        A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                
                # Solve A @ w = ones(m)
                try:
                    # Use least squares if overdetermined, or direct solve
                    if m <= target_rank:
                        # Underdetermined - find one solution
                        w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    else:
                        w, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    
                    # Check if solution is close to integer/half-integer
                    W[k*n+i, :] = w
                except:
                    success = False
                    break
            if not success:
                break
        
        if not success:
            continue
        
        # Verify the solution
        if _verify(n, m, p, U, V, W, tol=1e-6):
            # Round to nearest half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if _verify(n, m, p, U_rounded, V_rounded, W_rounded, tol=1e-9):
                # Convert to lists
                return _to_lists(U_rounded, V_rounded, W_rounded)
            else:
                # Try with original values
                return _to_lists(U, V, W)
    
    return None


def _verify(n, m, p, U, V, W, tol=1e-6):
    """Verify that the decomposition is exact."""
    nm = n * m
    mp = m * p
    pn = p * n
    rank = U.shape[1]
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _to_lists(U, V, W):
    """Convert numpy arrays to lists of lists."""
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    return {"U": U_list, "V": V_list, "W": W_list}


def _try_structured_fast(n, m, p, deadline):
    """Try a structured fast multiplication approach."""
    # For 3x5x6, try to use block structure
    # This is a heuristic approach
    
    # One approach: use the fact that 3x5x6 can be decomposed as:
    # (2x2 + 1x3) * (2x4 + 3x2) with appropriate Strassen-like tricks
    
    # For now, try a few known configurations
    # 2x2x2 Strassen: rank 7
    # 3x3x3: best known 23
    
    # Try to build 3x5x6 from blocks
    # This is complex, so return None for now and rely on the randomized search
    
    return None

# EVOLVE-BLOCK-END
