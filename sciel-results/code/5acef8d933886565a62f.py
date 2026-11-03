"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import itertools
import random
from fractions import Fraction

DIMENSIONS = [3, 5, 6]

def _verify(U, V, W, n, m, p):
    """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank):
                    s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if s != 1:
                    return False
    return True

def _verify_full(U, V, W, n, m, p):
    """Full verification using Fraction for exactness."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = Fraction(0)
                for r in range(rank):
                    s += Fraction(U[i * m + j][r]) * Fraction(V[j * p + k][r]) * Fraction(W[k * n + i][r])
                if s != 1:
                    return False
    return True

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s deadline, 10s safety margin
    
    n, m, p = DIMENSIONS  # 3, 5, 6
    rank_schoolbook = n * m * p  # 90
    
    # Build Strassen-based decomposition for rank 84
    # Partition:
    # n (rows of A): [2, 1]
    # m (cols of A / rows of B): [2, 2, 1]
    # p (cols of B): [2, 2, 2]
    
    n_parts = [2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 2]
    
    # Track offsets for each dimension
    n_offsets = [0]
    for i in range(len(n_parts) - 1):
        n_offsets.append(n_offsets[-1] + n_parts[i])
    
    m_offsets = [0]
    for i in range(len(m_parts) - 1):
        m_offsets.append(m_offsets[-1] + m_parts[i])
    
    p_offsets = [0]
    for i in range(len(p_parts) - 1):
        p_offsets.append(p_offsets[-1] + p_parts[i])
    
    terms = []  # each term: (list of (u_row, coeff), list of (v_row, coeff), list of (w_row, coeff))
    
    def add_strassen(a_rows, c_cols, d_cols):
        """Add 7 Strassen terms for a 2x2x2 block."""
        a0, a1 = a_rows
        c0, c1 = c_cols
        d0, d1 = d_cols
        
        def U(la, lc): return a_rows[la]*m + c_cols[lc]
        def V(lc, ld): return c_cols[lc]*p + d_cols[ld]
        def W(ld, la): return d_cols[ld]*n + a_rows[la]
        
        # M1 = (A[0,0]+A[1,1])*(B[0,0]+B[1,1])
        # C[0,0] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1), (U(1,1), 1)],
            [(V(0,0), 1), (V(1,1), 1)],
            [(W(0,0), 1), (W(1,1), 1)]
        ))
        
        # M2 = (A[1,0]+A[1,1])*B[0,0]
        # C[1,0] += 1, C[1,1] += -1
        terms.append((
            [(U(1,0), 1), (U(1,1), 1)],
            [(V(0,0), 1)],
            [(W(0,1), 1), (W(1,1), -1)]
        ))
        
        # M3 = A[0,0]*(B[0,1]-B[1,1])
        # C[0,1] += 1, C[1,1] += 1
        terms.append((
            [(U(0,0), 1)],
            [(V(0,1), 1), (V(1,1), -1)],
            [(W(1,0), 1), (W(1,1), 1)]
        ))
        
        # M4 = A[1,1]*(B[1,0]-B[0,0])
        # C[0,0] += 1, C[1,0] += 1
        terms.append((
            [(U(1,1), 1)],
            [(V(1,0), 1), (V(0,0), -1)],
            [(W(0,0), 1), (W(0,1), 1)]
        ))
        
        # M5 = (A[0,0]+A[0,1])*B[1,1]
        # C[0,0] += -1, C[0,1] += 1
        terms.append((
            [(U(0,0), 1), (U(0,1), 1)],
            [(V(1,1), 1)],
            [(W(0,0), -1), (W(1,0), 1)]
        ))
        
        # M6 = (A[1,0]-A[0,0])*(B[0,0]+B[0,1])
        # C[1,1] += 1
        terms.append((
            [(U(1,0), 1), (U(0,0), -1)],
            [(V(0,0), 1), (V(0,1), 1)],
            [(W(1,1), 1)]
        ))
        
        # M7 = (A[0,1]-A[1,1])*(B[1,0]+B[1,1])
        # C[0,0] += 1
        terms.append((
            [(U(0,1), 1), (U(1,1), -1)],
            [(V(1,0), 1), (V(1,1), 1)],
            [(W(0,0), 1)]
        ))
    
    def add_schoolbook(a_rows, c_cols, d_cols):
        """Add schoolbook terms for a block."""
        for la in range(len(a_rows)):
            for lc in range(len(c_cols)):
                for ld in range(len(d_cols)):
                    terms.append((
                        [(a_rows[la]*m + c_cols[lc], 1)],
                        [(c_cols[lc]*p + d_cols[ld], 1)],
                        [(d_cols[ld]*n + a_rows[la], 1)]
                    ))
    
    # Now we need to tile the 3x5x6 problem with these blocks
    # n=3: [2, 1], m=5: [2, 2, 1], p=6: [2, 2, 2]
    
    # For each combination of n-block, m-block, p-block:
    for ni in range(len(n_parts)):
        for mi in range(len(m_parts)):
            for pi in range(len(p_parts)):
                a_rows = list(range(n_offsets[ni], n_offsets[ni] + n_parts[ni]))
                c_cols = list(range(m_offsets[mi], m_offsets[mi] + m_parts[mi]))
                d_cols = list(range(p_offsets[pi], p_offsets[pi] + p_parts[pi]))
                
                if n_parts[ni] == 2 and m_parts[mi] == 2 and p_parts[pi] == 2:
                    add_strassen(a_rows, c_cols, d_cols)
                else:
                    add_schoolbook(a_rows, c_cols, d_cols)
    
    # Count total rank
    total_rank = len(terms)
    
    # Build U, V, W matrices
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for (row, coeff) in u_entries:
            U[row][r] = coeff
        for (row, coeff) in v_entries:
            V[row][r] = coeff
        for (row, coeff) in w_entries:
            W[row][r] = coeff
    
    # Verify the decomposition
    if not _verify(U, V, W, n, m, p):
        # Fall back to schoolbook if Strassen fails
        U = [[0] * rank_schoolbook for _ in range(n * m)]
        V = [[0] * rank_schoolbook for _ in range(m * p)]
        W = [[0] * rank_schoolbook for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W}
    
    best_U, best_V, best_W = U, V, W
    best_rank = total_rank
    
    # Now try to reduce rank further using randomized algebraic search
    # Strategy: initialize with the Strassen decomposition, then try to find
    # linear dependencies among the rank-1 tensors
    
    import numpy as np
    
    def try_reduce_rank(U, V, W, n, m, p, target_reduction, max_iters=100):
        """Try to reduce rank by target_reduction using random linear combinations."""
        current_rank = len(U[0])
        target_rank = current_rank - target_reduction
        
        if target_rank < max(n*m, m*p, n*p):
            return None  # Can't go below the lower bound
        
        # Build the tensor as a 3D array and try to find a lower-rank decomposition
        # This is a hard problem, so we use a heuristic approach
        
        # Convert to numpy arrays for faster computation
        U_np = np.array(U, dtype=float)
        V_np = np.array(V, dtype=float)
        W_np = np.array(W, dtype=float)
        
        # Try random column eliminations
        for attempt in range(max_iters):
            if time.time() > deadline:
                break
            
            # Try to eliminate one column by expressing it as a combination of others
            col_to_eliminate = random.randint(0, current_rank - 1)
            
            # Check if this column is linearly dependent on the others
            # This is a simplified check - we just try random coefficients
            success = False
            for _ in range(10):  # Try a few random coefficient sets
                if time.time() > deadline:
                    break
                
                # Try to express column col_to_eliminate as a combination of others
                # This is a very rough heuristic
                pass
            
            # If we can't easily eliminate, just continue
            pass
        
        return None
    
    # For now, return the Strassen decomposition
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
