import numpy as np
import random
import time

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0  # 10 seconds buffer
    
    def _schoolbook_decomposition(n, m, p):
        """Return the standard schoolbook decomposition of rank n*m*p."""
        rank = n * m * p
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
    
    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        num_u = len(U)
        num_v = len(V)
        num_w = len(W)
        if num_u != n * m or num_v != m * p or num_w != p * n:
            return False
        
        rank = len(U[0])
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        # Check all entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r in range(rank):
                        val += U[a][r] * V[b][r] * W[c][r]
                    if val != 1:
                        return False
        return True

    def try_rank_randomized(r, n, m, p, max_attempts=200):
        """
        Try to find a rank-r decomposition using randomized approach.
        Fix U and V randomly, then solve linear system for W.
        """
        # Precompute indices for efficiency
        eq_indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    eq_indices.append((i, j, k, k * n + i))
        
        num_eq = n * m * p
        num_unk = p * n * r
        
        # We will build the matrix A once for the structure, but values change
        # Actually A depends on U and V values, so we must rebuild or update it.
        # Since r is small, building A is fast.
        
        for attempt in range(max_attempts):
            if time.time() > deadline:
                return None
            
            # Random U and V with small integers
            # Using {-1, 0, 1} is good for sparsity and small numbers
            U = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(n * m)]
            V = [[random.choice([-1, 0, 1]) for _ in range(r)] for _ in range(m * p)]
            
            # Build linear system for W
            # A[eq_idx, col_idx] = U[i*m+j, r_idx] * V[j*p+k, r_idx]
            # where eq_idx corresponds to (i,j,k) and col_idx = (k*n+i)*r + r_idx
            A = np.zeros((num_eq, num_unk), dtype=np.float64)
            b = np.ones(num_eq, dtype=np.float64)
            
            eq_idx = 0
            for i, j, k, c_idx in eq_indices:
                u_row = U[i * m + j]
                v_row = V[j * p + k]
                for r_idx in range(r):
                    col_idx = c_idx * r + r_idx
                    A[eq_idx, col_idx] = u_row[r_idx] * v_row[r_idx]
                eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_A, sv = np.linalg.lstsq(A, b, rcond=None)
            except np.linalg.LinAlgError:
                continue
            
            # Check if the solution is exact (residuals close to 0)
            if len(residuals) > 0 and residuals[0] > 1e-9:
                continue
            
            # Reshape W
            W = [[0.0] * r for _ in range(p * n)]
            for c_idx in range(p * n):
                for r_idx in range(r):
                    W[c_idx][r_idx] = W_flat[c_idx * r + r_idx]
            
            # Try to round to half-integers
            valid = True
            for c_idx in range(p * n):
                for r_idx in range(r):
                    val = W[c_idx][r_idx]
                    nearest_half = round(val * 2) / 2.0
                    if abs(val - nearest_half) > 1e-7:
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            # Round to half-integers
            for c_idx in range(p * n):
                for r_idx in range(r):
                    W[c_idx][r_idx] = round(W[c_idx][r_idx] * 2) / 2.0
            
            # Verify the decomposition with exact arithmetic (or careful float)
            # Since entries are half-integers, products are quarter-integers.
            # Sum of 42 quarter-integers is an integer.
            # We can verify with integers to avoid float issues.
            # U, V are integers. W is half-integers.
            # Let's convert W to integers by multiplying by 2.
            # Then check if sum U*V*(2*W) == 4 ? No.
            # Equation: sum U*V*W = 1.
            # If W = w/2, then sum U*V*w/2 = 1 => sum U*V*w = 2.
            
            W_int = [[int(round(W[c_idx][r_idx] * 2)) for r_idx in range(r)] for c_idx in range(p * n)]
            
            ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        a = i * m + j
                        b_idx = j * p + k
                        c = k * n + i
                        s = 0
                        for r_idx in range(r):
                            s += U[a][r_idx] * V[b_idx][r_idx] * W_int[c][r_idx]
                        if s != 2:
                            ok = False
                            break
                if not ok:
                    break
            
            if ok:
                # Convert back to float or keep as is? 
                # The interface expects integer/half-integer entries.
                # Python floats representing 0.5 are exact.
                return {"U": U, "V": V, "W": W}
        
        return None

    # Try ranks from 21 up to 41
    # We start from low ranks to find the minimal one.
    # But finding a low rank is harder.
    # Maybe start from 41 down? No, we want minimal.
    # However, if rank 21 doesn't exist, we waste time.
    # Let's try a few low ranks, then mid, then high.
    
    # Actually, for matrix multiplication, the rank is usually close to schoolbook.
    # Let's try 41, 40, 39... down to 21.
    # If we find 41, we return 41.
    # If we find 40, we return 40.
    # We want the MINIMAL rank. So we should search 21, 22, ...
    # But if 21 doesn't exist, we have to try 22, etc.
    
    # Given the time limit, let's try a mix.
    # Prioritize ranks close to the known bounds if any.
    # For 2x2x2, rank 7 (vs 8).
    # For 2x3x3, rank ?
    
    # Let's just iterate from 21 to 41.
    # If we find one, return it.
    
    for r in range(21, 42):
        if time.time() > deadline:
            break
        # Allocate more attempts for lower ranks? Or equal?
        # Lower ranks are more constrained, so maybe more attempts needed?
        # Or less likely to exist?
        # Let's give 100 attempts per rank.
        result = try_rank_randomized(r, n, m, p, max_attempts=100)
        if result is not None:
            return result

    # Fallback to schoolbook
    return _schoolbook_decomposition(n, m, p)
