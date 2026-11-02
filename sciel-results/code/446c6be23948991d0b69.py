import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    schoolbook_rank = n * m * p

    def _build_schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def _verify_exact(U, V, W):
        """Verify that sum_c U[i*m+j, c] * V[j*p+k, c] == W[k*n+i, c] for all i,j,k"""
        rank = len(U[0]) if U else 0
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check: for each c, U[:,c] outer V[:,c] should contribute to the tensor
        # Actually we need: the matrix multiplication tensor T[i,j,k] = delta
        # T is defined such that sum over c of U[i*m+j, c] * V[j*p+k, c] should equal W[k*n+i, c]... 
        # Wait, let me re-read the convention.
        # The tensor decomposition: T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[i,k,c]
        # Where T[i,j,k] = 1 if the (i,j) entry of A and (j,k) entry of B multiply to give (i,k) entry of C
        # For matrix multiplication, T[i,j,k] = 1 for all valid i,j,k (since C[i,k] = sum_j A[i,j]*B[j,k])
        # Wait no. The matrix multiplication tensor has T[i,j,k] = 1 if the product A[i,j]*B[j,k] contributes to C[i,k].
        # Actually for the standard matrix multiplication, the tensor is:
        # T[i,j,k] = 1 (it's the identity tensor in the sense that every (i,j,k) triple is used)
        # No wait. Let me think again.
        
        # The matrix multiplication can be written as:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor T has T[i,j,k] = 1 for all valid i, j, k.
        # The decomposition is: T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        
        # So we need: for all i, j, k: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] == 1
        
        # Let's check this
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(rank):
                        s += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True

    def _check_half_integer(U, V, W):
        """Check that all entries are integers or half-integers"""
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if isinstance(val, float):
                        if abs(val * 2 - round(val * 2)) > 1e-9:
                            return False
                    elif isinstance(val, int):
                        pass
                    else:
                        try:
                            v = float(val)
                            if abs(v * 2 - round(v * 2)) > 1e-9:
                                return False
                        except:
                            return False
        return True

    def _try_optimize():
        """Try to find a lower rank decomposition using optimization"""
        deadline = start_time + 110  # Leave 40s safety margin
        best_result = None
        best_rank = schoolbook_rank
        
        # Start from schoolbook and try to reduce rank by removing columns
        # First, let's try a simple approach: start with rank = schoolbook_rank and try to eliminate columns
        
        for target_rank in range(schoolbook_rank - 1, max(schoolbook_rank - 20, 0), -1):
            if time.time() > deadline:
                break
            if best_rank <= target_rank:
                continue
                
            # Try random restarts for this target rank
            for attempt in range(50):
                if time.time() > deadline:
                    break
                    
                # Initialize with schoolbook and randomly zero out some columns
                U = [[0] * target_rank for _ in range(nm)]
                V = [[0] * target_rank for _ in range(mp)]
                W = [[0] * target_rank for _ in range(pn)]
                
                # Start from schoolbook and keep only target_rank columns
                # Randomly select which columns to keep
                cols_to_keep = list(range(schoolbook_rank))
                np.random.shuffle(cols_to_keep)
                cols_to_keep = cols_to_keep[:target_rank]
                
                for new_col, old_col in enumerate(cols_to_keep):
                    i = old_col // (m * p)
                    j = (old_col // p) % m
                    k = old_col % p
                    U[i * m + j][new_col] = 1
                    V[j * p + k][new_col] = 1
                    W[k * n + i][new_col] = 1
                
                # Verify
                if _verify_exact(U, V, W) and _check_half_integer(U, V, W):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_result = {"U": U, "V": V, "W": W}
                    break  # Found valid decomposition for this rank
        
        # Try a more systematic approach: use the fact that we can combine terms
        # For matrix multiplication, there are known lower bounds
        # For 3x4 x 4x5, the known best is at most 3*4*5 = 60, but there might be better
        
        # Try rank 50-59 with more systematic search
        for target_rank in range(50, min(60, schoolbook_rank)):
            if time.time() > deadline:
                break
            if best_rank <= target_rank:
                continue
                
            # Use a structured approach: try to express some products as combinations
            # This is complex, so let's just try random perturbations of the schoolbook
            for attempt in range(100):
                if time.time() > deadline:
                    break
                    
                # Start from schoolbook, try to merge some columns
                U = [[0] * target_rank for _ in range(nm)]
                V = [[0] * target_rank for _ in range(mp)]
                W = [[0] * target_rank for _ in range(pn)]
                
                # Map each (i,j,k) triple to a column
                # If two triples map to the same column, their contributions add
                col_map = {}
                col_id = 0
                triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
                np.random.shuffle(triples)
                
                for idx, (i, j, k) in enumerate(triples):
                    if col_id < target_rank:
                        col_map[(i, j, k)] = col_id
                        col_id += 1
                    else:
                        # Try to merge with an existing column that has compatible indices
                        # This is a simplification - just assign to random existing column
                        # This won't work correctly, so skip
                        pass
                
                # This approach doesn't work because merging columns changes the tensor
                # Let's abandon this and just use the best we found
        
        return best_result, best_rank

    # First, try to optimize
    try:
        result, rank = _try_optimize()
        if result is not None:
            return result
    except Exception as e:
        pass
    
    # Fall back to schoolbook
    return _build_schoolbook()
# EVOLVE-BLOCK-END
