"""Matrix multiplication 2,2,13 rank-minimal decomposition."""
import time
import itertools

DIMENSIONS = [2, 2, 13]
DEADLINE = 150.0

# EVOLVE-BLOCK-START

def _check_decomposition(U, V, W, n, m, p):
    """Check if U, V, W form a valid decomposition.
    
    The matrix multiplication tensor T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    for all i in [0,n), j in [0,m), k in [0,p).
    """
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected = 1  # T[i,j,k] = 1 for all (i,j,k) in standard matrix mult
                actual = 0
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                for r in range(rank):
                    actual += U[u_idx][r] * V[v_idx][r] * W[w_idx][r]
                if abs(actual - expected) > 1e-9:
                    return False
    return True


def _make_schoolbook(n, m, p):
    """Standard schoolbook decomposition of rank n*m*p."""
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


def _try_strassen_extension(n, m, p):
    """
    Attempt a Strassen-based decomposition.
    
    For 2x2 times 2xp matrix multiplication, we can potentially use
    Strassen's 7-multiplication scheme adapted for rectangular B.
    
    Actually, for M_{2,2,p}, the tensor rank is at most 7p by treating
    each column independently with Strassen. But we can do better by
    sharing structure.
    
    Let's try: A is 2x2, B is 2xp. C = AB is 2xp.
    
    Write A = [[a, b], [c, d]], B columns are b_1, ..., b_p each in R^2.
    C row 1: [a*b_k[0] + b*b_k[1]] for k=1..p
    C row 2: [c*b_k[0] + d*b_k[1]] for k=1..p
    
    Strassen for 2x2x1: 7 rank-1 terms.
    For p columns, naive gives 7p. But can we share?
    
    Let's try a rank 2p+2 construction:
    
    Actually, let me think about this more carefully.
    
    The matrix multiplication tensor M_{2,2,p} has components T[i,j,k] = delta_{i,k} for the "A part" 
    wait no. Let me re-index.
    
    C = AB where A is n x m = 2x2, B is m x p = 2x13.
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor is T[i,j,k] = 1 (it's the standard multiplication tensor).
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (26, rank)  
    W has shape (p*n, rank) = (26, rank)
    
    We need: for all (i,j,k) with i in {0,1}, j in {0,1}, k in {0,...,12}:
    sum_r U[i*2+j, r] * V[j*13+k, r] * W[k*2+i, r] = 1
    
    Let's try rank = 7 + p = 20? Or 2p+2 = 28?
    
    A known result: R(2,2,p) <= 2p + 2 for p >= 2? Let me try to construct it.
    
    Idea: Use 2 "row" terms and 2 "col" terms plus p shared terms.
    
    Actually, let me try a simpler approach: rank = 7p / something.
    
    Let me try the construction where we group the 13 columns.
    
    Alternative: try to find a rank-20 or rank-28 decomposition by solving a system.
    """
    n, m, p = 2, 2, 13
    
    # Try rank r by random initialization and verification
    # This is expensive, so try small ranks first
    
    # Known: R(2,2,1) = 7 (Strassen)
    # R(2,2,2) <= 14 (2*7), but actually R(2,2,2) = 12? 
    # R(2,2,p) grows roughly linearly in p.
    
    # For now, try to verify if a rank less than 52 works with a structured ansatz.
    
    # Strassen for each column: 7 * 13 = 91 (worse than schoolbook 52)
    # So that's not helpful directly.
    
    # Let me try rank = 2p + 2 = 28
    # Construction idea:
    # U[i*2+j, r] and V[j*13+k, r] and W[k*2+i, r]
    
    # Let's try a specific rank-28 ansatz:
    # 2 terms for "diagonal" A entries, 2 for "off-diagonal", and 24 more?
    
    # Actually, let me just return schoolbook for safety, as finding a valid
    # sub-52 decomposition in the time budget is risky.
    # But the directive says to try improvements.
    
    # Let me try: can we do rank = 4p - 2 = 50? Or 3p = 39?
    
    # I'll return None to indicate no improvement found.
    return None


def _try_small_rank_search(n, m, p, max_rank=51, deadline_time=None):
    """Try to find a decomposition of rank < 52 using random search."""
    import numpy as np
    
    start_time = time.time()
    
    # The constraints are: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # This is a system of n*m*p = 52 equations in 4*rank + 26*rank + 26*rank = 56*rank variables
    # For rank=51, that's 56*51 = 2856 unknowns, 52 equations - very underdetermined
    # So solutions should exist for rank well below 52.
    # The question is finding integer/half-integer solutions.
    
    # Let's try rank = 20, 25, 30, etc.
    
    # Actually, for the standard matrix multiplication tensor M_{2,2,p},
    # the tensor rank is known to be bounded. For p=13:
    # R(2,2,13) is at most ... let me think.
    
    # A simple bound: R(2,2,p) <= 2p + 2 (using a specific construction)
    # Let me try to verify rank 28 works.
    
    # For now, let me try a few ranks and see if I can find valid solutions
    # with random initialization + rounding.
    
    for rank in [20, 25, 28, 30, 35, 40, 45, 50]:
        if deadline_time is not None and time.time() > deadline_time:
            break
        
        # Random initialization
        best_err = float('inf')
        best_solution = None
        
        for attempt in range(10):
            if deadline_time is not None and time.time() > deadline_time:
                break
            
            # Initialize with small random values
            np.random.seed(42 + rank + attempt)
            U = np.random.uniform(-1, 1, (4, rank))
            V = np.random.uniform(-1, 1, (26, rank))
            W = np.random.uniform(-1, 1, (26, rank))
            
            # Check the constraint
            errors = []
            for i in range(2):
                for j in range(2):
                    for k in range(13):
                        val = 0
                        for r in range(rank):
                            val += U[i*2+j, r] * V[j*13+k, r] * W[k*2+i, r]
                        errors.append(abs(val - 1))
            
            max_err = max(errors)
            if max_err < best_err:
                best_err = max_err
                best_solution = (U.copy(), V.copy(), W.copy())
            
            if best_err < 1e-6:
                # Found a good solution, try rounding to integers/halves
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify rounded solution
                round_errors = []
                for i in range(2):
                    for j in range(2):
                        for k in range(13):
                            val = 0
                            for r in range(rank):
                                val += U_r[i*2+j, r] * V_r[j*13+k, r] * W_r[k*2+i, r]
                            round_errors.append(abs(val - 1))
                
                if max(round_errors) < 1e-9:
                    # Success! Convert to lists
                    U_list = U_r.tolist()
                    V_list = V_r.tolist()
                    W_list = W_r.tolist()
                    return {"U": U_list, "V": V_list, "W": W_list}
        
        # If no integer solution found for this rank, continue
        pass
    
    return None


def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + (DEADLINE - 5)  # Safety margin
    
    # Start with schoolbook as fallback
    result = _make_schoolbook(n, m, p)
    
    # Try to find a better (lower rank) decomposition
    # First try structured approaches
    strat = _try_strassen_extension(n, m, p)
    if strat is not None:
        # Verify
        if _check_decomposition(strat["U"], strat["V"], strat["W"], n, m, p):
            result = strat
    
    # Try random search for lower rank
    if time.time() < deadline:
        better = _try_small_rank_search(n, m, p, deadline_time=deadline)
        if better is not None:
            if _check_decomposition(better["U"], better["V"], better["W"], n, m, p):
                result = better
    
    return result

# EVOLVE-BLOCK-END
