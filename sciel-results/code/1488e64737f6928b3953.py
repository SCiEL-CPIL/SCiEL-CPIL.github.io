"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
import time
_start_time = time.time()
DEADLINE = 145.0
SAFETY_MARGIN = 5.0
import numpy as np
from itertools import product

def _check_decomposition(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _strassen_2x2():
    """Standard Strassen decomposition for 2x2 * 2x2."""
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (A00 + A11)(B00 + B11)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[2, 0] = 1; W[1, 0] = 1; W[3, 0] = 1
    
    # M2 = (A10 + A11)B00
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = 1
    
    # M3 = A00(B10 - B00)
    U[0, 2] = 1
    V[2, 2] = 1; V[0, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # M4 = A11(B01 - B10)
    U[3, 3] = 1
    V[1, 3] = 1; V[2, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # M5 = (A00 + A01)B11
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = 1; W[1, 4] = 1
    
    # M6 = (A10 - A00)(B00 + B01)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1; W[3, 5] = 1
    
    # M7 = (A11 - A10)(B10 + B11)
    U[3, 6] = 1; U[2, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1; W[1, 6] = 1
    
    return U, V, W

def _verify_strassen(U, V, W):
    """Verify Strassen decomposition for 2x2 * 2x2."""
    n, m, p = 2, 2, 2
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True

def _construct_strassen_hybrid():
    """Construct a rank-18 decomposition using two Strassen blocks + 4 schoolbook."""
    n, m, p = 2, 2, 5
    rank = 18
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    U1, V1, W1 = _strassen_2x2()
    
    # First Strassen block: columns 0,1 of B
    for c in range(7):
        for r in range(4):
            U[r, c] = U1[r, c]
        for r in range(4):
            j = r // 2
            k = r % 2
            V[j * 5 + k, c] = V1[r, c]
        for r in range(4):
            W[r, c] = W1[r, c]
    
    # Second Strassen block: columns 2,3 of B
    U2, V2, W2 = _strassen_2x2()
    offset = 7
    for c in range(7):
        for r in range(4):
            U[r, offset + c] = U2[r, c]
        for r in range(4):
            j = r // 2
            k_local = r % 2
            global_k = 2 + k_local
            V[j * 5 + global_k, offset + c] = V2[r, c]
        for r in range(4):
            k_local = r // 2
            i = r % 2
            global_k = 2 + k_local
            W[global_k * 2 + i, offset + c] = W2[r, c]
    
    # Schoolbook for column 4 of B
    offset = 14
    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
        c = offset + idx
        U[i*2+j, c] = 1
        V[j*5+4, c] = 1
        W[4*2+i, c] = 1
        
    return U.tolist(), V.tolist(), W.tolist()

def _try_reduce_rank(n, m, p, start_rank, target_rank, max_iters=500):
    """
    Try to reduce from start_rank to target_rank using iterative refinement.
    """
    if time.time() > DEADLINE - SAFETY_MARGIN:
        return None
    
    # Get initial solution
    if start_rank == 18:
        U_init, V_init, W_init = _construct_strassen_hybrid()
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
    else:
        U = np.random.randn(n*m, start_rank)
        V = np.random.randn(m*p, start_rank)
        W = np.random.randn(p*n, start_rank)
    
    current_rank = start_rank
    
    for _ in range(max_iters):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        if current_rank <= target_rank:
            break
        
        # Pick a random column to try to eliminate
        col_to_remove = np.random.randint(0, current_rank)
        
        # Set that column to zero
        U[:, col_to_remove] = 0
        V[:, col_to_remove] = 0
        W[:, col_to_remove] = 0
        
        # Now re-solve the remaining columns using ALS
        active_cols = [c for c in range(current_rank) if c != col_to_remove]
        
        converged = False
        for it in range(100):
            # Solve for U
            for i in range(n):
                for j in range(m):
                    A_x = np.zeros((p, len(active_cols)))
                    b = np.ones(p)
                    for k_idx, k in enumerate(range(p)):
                        for c_idx, c in enumerate(active_cols):
                            A_x[k_idx, c_idx] = V[j*p+k, c] * W[k*n+i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_x, b, rcond=None)
                        for c_idx, c in enumerate(active_cols):
                            U[i*m+j, c] = sol[c_idx]
                    except:
                        pass
            
            # Solve for V
            for j in range(m):
                for k in range(p):
                    A_x = np.zeros((n, len(active_cols)))
                    b = np.ones(n)
                    for i_idx, i in enumerate(range(n)):
                        for c_idx, c in enumerate(active_cols):
                            A_x[i_idx, c_idx] = U[i*m+j, c] * W[k*n+i, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_x, b, rcond=None)
                        for c_idx, c in enumerate(active_cols):
                            V[j*p+k, c] = sol[c_idx]
                    except:
                        pass
            
            # Solve for W
            for k in range(p):
                for i in range(n):
                    A_x = np.zeros((m, len(active_cols)))
                    b = np.ones(m)
                    for j_idx, j in enumerate(range(m)):
                        for c_idx, c in enumerate(active_cols):
                            A_x[j_idx, c_idx] = U[i*m+j, c] * V[j*p+k, c]
                    try:
                        sol, _, _, _ = np.linalg.lstsq(A_x, b, rcond=None)
                        for c_idx, c in enumerate(active_cols):
                            W[k*n+i, c] = sol[c_idx]
                    except:
                        pass
            
            # Check convergence
            if _check_decomposition(U, V, W, n, m, p):
                converged = True
                break
        
        if converged:
            # Successfully removed a column
            U = U[:, active_cols]
            V = V[:, active_cols]
            W = W[:, active_cols]
            current_rank = len(active_cols)
            
            # Try to round
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            if _check_decomposition(U_rounded, V_rounded, W_rounded, n, m, p):
                U, V, W = U_rounded, V_rounded, W_rounded
            
            if current_rank <= target_rank:
                return U.tolist(), V.tolist(), W.tolist()
        else:
            break
    
    return None

def _direct_search_rank(n, m, p, rank, max_attempts=3000):
    """Direct search for rank-r decomposition with integer/half-integer entries."""
    if time.time() > DEADLINE - SAFETY_MARGIN:
        return None
    
    candidates = [-2, -1, -0.5, 0, 0.5, 1, 2]
    
    for attempt in range(max_attempts):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        
        for c in range(rank):
            n_u = np.random.randint(1, 4)
            n_v = np.random.randint(1, 4)
            n_w = np.random.randint(1, 4)
            
            u_idx = np.random.choice(n*m, size=n_u, replace=False)
            v_idx = np.random.choice(m*p, size=n_v, replace=False)
            w_idx = np.random.choice(p*n, size=n_w, replace=False)
            
            U[u_idx, c] = np.random.choice(candidates, size=n_u)
            V[v_idx, c] = np.random.choice(candidates, size=n_v)
            W[w_idx, c] = np.random.choice(candidates, size=n_w)
        
        if _check_decomposition(U, V, W, n, m, p):
            return U.tolist(), V.tolist(), W.tolist()
    
    return None

def _structured_search(n, m, p, rank, max_attempts=2000):
    """Structured search: use Strassen-like patterns for subsets of columns."""
    if time.time() > DEADLINE - SAFETY_MARGIN:
        return None
    
    # Try various block decompositions
    # Split p columns into groups and apply Strassen or schoolbook to each
    from itertools import combinations_with_replacement
    
    # For p=5, try splits like (2,2,1), (2,3), (3,2), (1,4), (4,1), (5)
    splits = [(2,2,1), (2,3), (3,2), (1,4), (4,1), (1,1,1,1,1), (2,1,2), (1,2,2)]
    
    for split in splits:
        if time.time() > DEADLINE - SAFETY_MARGIN:
            return None
        
        # Calculate required rank for this split
        required_rank = 0
        for s in split:
            if s == 1:
                required_rank += n * m  # schoolbook for 1 column
            elif s == 2:
                required_rank += 7  # Strassen
            elif s == 3:
                required_rank += 12  # best known for 2x2x3
            else:
                required_rank += n * m * s  # schoolbook
        
        if required_rank <= rank:
            # Try to construct this decomposition
            U = np.zeros((n*m, rank))
            V = np.zeros((m*p, rank))
            W = np.zeros((p*n, rank))
            
            col_offset = 0
            term_offset = 0
            
            for s in split:
                if s == 1:
                    # Schoolbook for 1 column
                    for idx, (i, j) in enumerate([(0,0), (0,1), (1,0), (1,1)]):
                        c = term_offset + idx
                        if c < rank:
                            U[i*m+j, c] = 1
                            V[j*p + col_offset, c] = 1
                            W[col_offset*n + i, c] = 1
                    term_offset += n * m
                elif s == 2:
                    # Strassen for 2 columns
                    U1, V1, W1 = _strassen_2x2()
                    for c in range(7):
                        if term_offset + c < rank:
                            for r in range(4):
                                U[r, term_offset + c] = U1[r, c]
                                j = r // 2
                                k = r % 2
                                V[j*p + col_offset + k, term_offset + c] = V1[r, c]
                                W[(col_offset + k)*n + r%n, term_offset + c] = W1[r, c]
                    term_offset += 7
                else:
                    # Schoolbook for s columns
                    for idx in range(n * m * s):
                        c = term_offset + idx
                        if c < rank:
                            i = idx // (m * s)
                            rem = idx % (m * s)
                            j = rem // s
                            k = rem % s
                            U[i*m+j, c] = 1
                            V[j*p + col_offset + k, c] = 1
                            W[(col_offset + k)*n + i, c] = 1
                    term_offset += n * m * s
                
                col_offset += s
            
            if _check_decomposition(U, V, W, n, m, p):
                return U.tolist(), V.tolist(), W.tolist()
    
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Start with rank-18 Strassen hybrid
    U18, V18, W18 = _construct_strassen_hybrid()
    if _check_decomposition(U18, V18, W18, n, m, p):
        best_result = (U18, V18, W18)
        best_rank = 18
    else:
        # Fallback to schoolbook
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    # Try structured search first
    for target_rank in range(13, best_rank):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        result = _structured_search(n, m, p, target_rank, max_attempts=1000)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    # Try to reduce rank using iterative refinement
    for target_rank in range(13, best_rank):
        if time.time() > DEADLINE - SAFETY_MARGIN:
            break
        
        result = _try_reduce_rank(n, m, p, best_rank, target_rank, max_iters=500)
        if result is not None:
            best_result = result
            best_rank = target_rank
            break
    
    # If still at 18, try direct search for lower ranks
    if best_rank == 18:
        for target_rank in range(13, 18):
            if time.time() > DEADLINE - SAFETY_MARGIN:
                break
            
            result = _direct_search_rank(n, m, p, target_rank, max_attempts=3000)
            if result is not None:
                best_result = result
                best_rank = target_rank
                break
    
    U, V, W = best_result
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
