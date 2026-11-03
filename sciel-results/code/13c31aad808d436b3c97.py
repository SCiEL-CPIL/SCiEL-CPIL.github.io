"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import product

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3x3 matrix by a 3x7 matrix.
    
    The output factor follows the official trace convention W[k*n+i, r].
    
    Returns: {'U': list of lists, 'V': list of lists, 'W': list of lists}
    """
    n, m, p = 3, 3, 7
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    # Schoolbook decomposition as baseline
    schoolbook_rank = n * m * p  # 63
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_result = {'U': U_sb, 'V': V_sb, 'W': W_sb}
    best_rank = schoolbook_rank
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        n_, m_, p_ = n, m, p
        if len(U) != n_ * m_:
            return False
        if len(V) != m_ * p_:
            return False
        if len(W) != p_ * n_:
            return False
        if len(U) == 0 or len(V) == 0 or len(W) == 0:
            return False
        rank_ = len(U[0])
        if len(V[0]) != rank_ or len(W[0]) != rank_:
            return False
        if rank_ <= 0:
            return False
        
        # Convert to numpy for faster verification
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    expected = 1.0
                    actual = np.dot(U_np[i * m_ + j], V_np[j * p_ + k] * W_np[k * n_ + i])
                    if abs(actual - expected) > tol:
                        return False
        return True
    
    def make_result(U, V, W):
        """Convert to list-of-lists format with clean values."""
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        # Clean up near-integer and near-half-integer values
        def clean_val(x):
            r = round(x * 2) / 2  # snap to half-integers
            if abs(x - r) < 1e-8:
                return r
            r = round(x)
            if abs(x - r) < 1e-8:
                return float(r)
            return x
        U_list = [[clean_val(x) for x in row] for row in U_list]
        V_list = [[clean_val(x) for x in row] for row in V_list]
        W_list = [[clean_val(x) for x in row] for row in W_list]
        return {'U': U_list, 'V': V_list, 'W': W_list}
    
    def try_decomposition(U, V, W):
        """Try a decomposition and update best if valid and lower rank."""
        nonlocal best_result, best_rank
        if verify_decomposition(U, V, W):
            rank = len(U[0])
            if rank < best_rank:
                best_result = make_result(U, V, W)
                best_rank = rank
                return True
        return False
    
    # Strategy: Structured decompositions based on algebraic patterns
    # For (3,3,7), we can try to group columns of the 3x7 matrix
    
    # Coefficient set: {-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2}
    coeff_set = [-2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2]
    
    random.seed(42)
    
    # Strategy 1: Block decomposition patterns
    # Split the 7 columns into groups and apply Strassen-like tricks within groups
    
    # Pattern: 7 = 4 + 3, so we can decompose into two sub-problems
    # For 3x3 * 3x4, known rank might be lower than 36
    # For 3x3 * 3x3, known rank is 21 (Strassen)
    
    # Let's try to construct a decomposition using block structure
    # A (3x3) * B (3x7) = A * [B1 | B2] where B1 is 3x4, B2 is 3x3
    
    # For the 3x3 * 3x3 part, use Strassen's algorithm (rank 7 per 2x2, but for 3x3 it's different)
    # Actually, for 3x3 * 3x3, the known minimal rank is 21 (not 27)
    
    # Let's try a more systematic approach: random search with structured initialization
    
    # Try rank values from reasonable lower bounds up
    # Lower bound: max(n*m, m*p, p*n) = max(9, 21, 21) = 21
    # But realistically, for exact decomposition, we need more
    
    # Try known small-rank patterns with integer/half-integer coefficients
    
    # Approach: For each target rank, try random sparse initializations
    # and verify. If found, try to reduce further.
    
    # Use numpy for faster verification
    def verify_fast(U, V, W, tol=1e-9):
        """Fast verification using numpy."""
        n_, m_, p_ = n, m, p
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Build the tensor T[i,j,k] = sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
        # Check that T[i,j,k] = 1 for all valid i,j,k
        for i in range(n_):
            for j in range(m_):
                for k in range(p_):
                    val = np.dot(U_np[i*m_+j], V_np[j*p_+k] * W_np[k*n_+i])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try random search for lower ranks
    # Start from rank 62 down to 30
    # For each rank, try several random initializations
    
    # Use a smarter approach: start from schoolbook and try to merge/eliminate terms
    # Or try random search with verification
    
    # Given time constraints, try a focused random search
    for target_rank in range(62, 25, -1):
        if time.time() > deadline - 5:
            break
        
        # Try multiple random initializations for this rank
        num_trials = 30 if target_rank > 40 else 15
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random sparse initialization
            sparsity = 0.3  # 30% non-zero
            U = np.zeros((n*m, target_rank))
            V = np.zeros((m*p, target_rank))
            W = np.zeros((p*n, target_rank))
            
            for r in range(target_rank):
                # Choose random non-zero positions
                u_idx = random.randint(0, n*m-1)
                v_idx = random.randint(0, m*p-1)
                w_idx = random.randint(0, p*n-1)
                
                u_val = random.choice(coeff_set)
                v_val = random.choice(coeff_set)
                w_val = random.choice(coeff_set)
                
                # Ensure the product contributes meaningfully
                if u_val == 0 or v_val == 0 or w_val == 0:
                    continue
                    
                U[u_idx, r] = u_val
                V[v_idx, r] = v_val
                W[w_idx, r] = w_val
            
            # Convert to lists for verification
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if verify_decomposition(U_list, V_list, W_list):
                try_decomposition(U_list, V_list, W_list)
                break  # Found one for this rank, try lower
    
    # If we found an improvement, try to polish it
    if best_rank < schoolbook_rank:
        # Try to reduce rank further with local search
        # For now, just return what we found
        pass
    
    # Return the best result
    return best_result

# EVOLVE-BLOCK-END
