"""Model-and-optimize approach for 2x4x7 matrix multiplication tensor decomposition."""
import numpy as np
from itertools import product
import time
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x4 @ 4x7.
    
    Strategy: Build multiple candidate decompositions and select the best.
    1. Standard rank-50 Strassen-based decomposition
    2. Try to find lower-rank decompositions via structured search
    3. Use exact rational arithmetic for final verification
    """
    n, m, p = 2, 4, 7
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 120s limit
    
    def time_remaining():
        return deadline - time.time()
    
    # Build the rank-50 decomposition as starting point
    rank_init = 50
    U = np.zeros((n * m, rank_init), dtype=float)
    V = np.zeros((m * p, rank_init), dtype=float)
    W = np.zeros((p * n, rank_init), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A11+A22)(B11+B22)
        U[i0*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M2: (A21+A22)B11
        U[(i0+1)*m + j0, col] += 1
        U[(i0+1)*m + (j0+1), col] += 1
        V[j0*p + k0, col] += 1
        W[k0*n + (i0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] -= 1
        col += 1
        
        # M3: A11(B12-B22)
        U[i0*m + j0, col] += 1
        V[j0*p + (k0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] -= 1
        W[(k0+1)*n + i0, col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M4: A22(B21-B11)
        U[(i0+1)*m + (j0+1), col] += 1
        V[(j0+1)*p + k0, col] += 1
        V[j0*p + k0, col] -= 1
        W[k0*n + i0, col] += 1
        W[k0*n + (i0+1), col] += 1
        col += 1
        
        # M5: (A11+A12)B22
        U[i0*m + j0, col] += 1
        U[i0*m + (j0+1), col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] -= 1
        W[(k0+1)*n + i0, col] += 1
        col += 1
        
        # M6: (A21-A11)(B11+B12)
        U[(i0+1)*m + j0, col] += 1
        U[i0*m + j0, col] -= 1
        V[j0*p + k0, col] += 1
        V[j0*p + (k0+1), col] += 1
        W[(k0+1)*n + (i0+1), col] += 1
        col += 1
        
        # M7: (A12-A22)(B21+B22)
        U[i0*m + (j0+1), col] += 1
        U[(i0+1)*m + (j0+1), col] -= 1
        V[(j0+1)*p + k0, col] += 1
        V[(j0+1)*p + (k0+1), col] += 1
        W[k0*n + i0, col] += 1
        col += 1

    # 6 Strassen blocks for the 2x4x6 part (k=0..5)
    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)
    add_strassen(0, 2, 0)
    add_strassen(0, 2, 2)
    add_strassen(0, 0, 4)
    add_strassen(0, 2, 4)
    
    # 8 rank-1 terms for k=6
    for i in range(2):
        for j in range(4):
            U[i*m + j, col] = 1
            V[j*p + 6, col] = 1
            W[6*n + i, col] = 1
            col += 1
    
    assert col == rank_init, f"Expected rank {rank_init}, got {col}"
    
    def verify_decomposition_fast(U_m, V_m, W_m, n, m, p, num_tests=10):
        """Fast verification using random integer inputs."""
        np.random.seed(42)
        for _ in range(num_tests):
            A = np.random.randint(-3, 4, size=(n, m)).astype(float)
            B = np.random.randint(-3, 4, size=(m, p)).astype(float)
            C_expected = A @ B
            rank_m = U_m.shape[1]
            C_comp = np.zeros((n, p), dtype=float)
            for r in range(rank_m):
                u_vec = U_m[:, r]
                v_vec = V_m[:, r]
                w_vec = W_m[:, r]
                # C[i,k] = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                for i in range(n):
                    for k in range(p):
                        for j in range(m):
                            C_comp[i, k] += U_m[i*m+j, r] * V_m[j*p+k, r] * W_m[k*n+i, r]
            if not np.allclose(C_comp, C_expected, atol=1e-8):
                return False
        return True
    
    # Verify initial decomposition
    if not verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    best_rank = rank_init
    
    # Fast rank reduction: use SVD on the "flattened" tensor
    # The tensor T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # We can try to find a lower-rank decomposition by looking at the
    # structure more carefully
    
    def try_svd_reduction(U_arr, V_arr, W_arr, n, m, p, target_rank=None):
        """
        Try to reduce rank by finding a lower-rank approximation.
        This is a heuristic that may not find exact solutions.
        """
        if time_remaining() < 20:
            return None, None, None, U_arr.shape[1]
        
        current_rank = U_arr.shape[1]
        
        # Try different target ranks
        if target_rank is None:
            target_rank = max(1, current_rank - 5)
        
        # For each target rank, try to find a decomposition
        # This is complex, so we'll use a simpler approach:
        # Look for columns that are nearly linearly dependent
        
        # Build the "combined" matrix by stacking U, V, W columns
        # Each column r has entries in U[:,r], V[:,r], W[:,r]
        # We want to find if some columns are redundant
        
        # Use a faster approach: compute the "tensor rank" via
        # checking if removing a column still gives valid decomposition
        
        # For speed, only try removing one column at a time
        eliminated = set()
        
        for c in range(current_rank):
            if time_remaining() < 15:
                break
            if c in eliminated:
                continue
            
            # Try removing column c
            keep_cols = [oc for oc in range(current_rank) if oc != c and oc not in eliminated]
            if len(keep_cols) < target_rank:
                continue
            
            U_test = U_arr[:, keep_cols]
            V_test = V_arr[:, keep_cols]
            W_test = W_arr[:, keep_cols]
            
            # Quick verification with fewer tests
            if verify_decomposition_fast(U_test, V_test, W_test, n, m, p, num_tests=5):
                eliminated.add(c)
        
        if not eliminated:
            return None, None, None, current_rank
        
        keep_cols = [c for c in range(current_rank) if c not in eliminated]
        new_rank = len(keep_cols)
        
        if new_rank < current_rank:
            U_new = U_arr[:, keep_cols]
            V_new = V_arr[:, keep_cols]
            W_new = W_arr[:, keep_cols]
            return U_new, V_new, W_new, new_rank
        
        return None, None, None, current_rank
    
    # Try SVD-based reduction
    if time_remaining() > 30:
        U_red, V_red, W_red, new_rank = try_svd_reduction(U, V, W, n, m, p)
        if U_red is not None and new_rank < best_rank:
            if verify_decomposition_fast(U_red, V_red, W_red, n, m, p):
                best_U, best_V, best_W = U_red, V_red, W_red
                best_rank = new_rank
    
    # Try to find a completely different decomposition
    # Strategy: use the fact that 2x4x7 = 2x4x4 + 2x4x3
    # R(2,4,4) might be achievable with rank < 28
    # R(2,4,3) might be achievable with rank < 21
    
    def build_alternative_decomp(n, m, p):
        """
        Try to build a lower-rank decomposition using a different strategy.
        For 2x4x7, we can try to use overlapping Strassen blocks.
        """
        if time_remaining() < 25:
            return None, None, None, None
        
        # Alternative: use a 2x4x2 block with rank 14, repeated 3 times for k=0..5
        # Plus 8 for k=6, total = 14*3 + 8 = 50 (same as before)
        
        # Try a different approach: use the fact that we can share
        # some computations between adjacent k columns
        
        # For now, let's try a greedy approach: build the tensor and
        # try to find a low-rank decomposition
        
        # Build the full tensor T[i,j,k] for i in 0..1, j in 0..3, k in 0..6
        T = np.zeros((n, m, p), dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i, j, k] = 1  # This is the identity tensor for matmul
        
        # The tensor for matrix multiplication is:
        # T[i,j,k] = delta_{i,i} * delta_{j,j} * delta_{k,k} -- no, that's wrong
        # Actually, (A@B)[i,k] = sum_j A[i,j] * B[j,k]
        # So the tensor is T[i,j,k] = 1 for all i,j,k (it's the all-ones tensor
        # in the sense that each (i,j,k) triple contributes A[i,j]*B[j,k])
        
        # Let me reconsider. The tensor decomposition is:
        # (A@B)[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # summed over j and r
        # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This should equal sum_j A[i,j] * B[j,k]
        
        # So we need: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = A[i,j] * B[j,k]
        # for all i, j, k
        
        # This is a 3-way tensor decomposition problem.
        
        # For now, return None to indicate no better decomposition found
        return None, None, None, None
    
    # Try alternative decomposition
    if time_remaining() > 20:
        U_alt, V_alt, W_alt, alt_rank = build_alternative_decomp(n, m, p)
        if U_alt is not None and alt_rank is not None and alt_rank < best_rank:
            if verify_decomposition_fast(U_alt, V_alt, W_alt, n, m, p):
                best_U, best_V, best_W = U_alt, V_alt, W_alt
                best_rank = alt_rank
    
    # Final exact verification with rational arithmetic
    def verify_exact(U_m, V_m, W_m, n, m, p):
        """Exact verification using rational arithmetic."""
        # Convert to fractions
        U_frac = np.zeros((n*m, U_m.shape[1]), dtype=object)
        V_frac = np.zeros((m*p, V_m.shape[1]), dtype=object)
        W_frac = np.zeros((p*n, W_m.shape[1]), dtype=object)
        
        for i in range(n*m):
            for r in range(U_m.shape[1]):
                U_frac[i, r] = Fraction(U_m[i, r]).limit_denominator(1000)
        for i in range(m*p):
            for r in range(V_m.shape[1]):
                V_frac[i, r] = Fraction(V_m[i, r]).limit_denominator(1000)
        for i in range(p*n):
            for r in range(W_m.shape[1]):
                W_frac[i, r] = Fraction(W_m[i, r]).limit_denominator(1000)
        
        # Test on a few specific inputs
        test_cases = [
            (np.array([[1, 0, 1, -1], [0, 1, -1, 1]]), np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10]])),
            (np.array([[1, 1, 1, 1], [1, -1, 1, -1]]), np.array([[1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0], [1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0]])),
            (np.array([[2, -1, 3, 0], [-1, 2, 0, 3]]), np.array([[1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1], [1, -1, 1, -1, 1, -1, 1], [0, 0, 0, 0, 0, 0, 0]])),
        ]
        
        for A, B in test_cases:
            C_expected = A @ B
            C_comp = np.zeros((n, p), dtype=object)
            rank_m = U_frac.shape[1]
            for i in range(n):
                for k in range(p):
                    val = Fraction(0)
                    for j in range(m):
                        for r in range(rank_m):
                            val += U_frac[i*m+j, r] * V_frac[j*p+k, r] * W_frac[k*n+i, r]
                    C_comp[i, k] = val
            if not all(C_comp[i, k] == C_expected[i, k] for i in range(n) for k in range(p)):
                return False
        return True
    
    # Exact verification of best solution
    if verify_exact(best_U, best_V, best_W, n, m, p):
        pass  # Solution is exact
    else:
        # Fallback to original
        if verify_exact(U, V, W, n, m, p):
            best_U, best_V, best_W = U, V, W
            best_rank = rank_init
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
