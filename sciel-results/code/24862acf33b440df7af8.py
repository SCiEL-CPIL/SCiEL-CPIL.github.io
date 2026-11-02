"""Matrix multiplication 3x4x5 - attempt to find a lower rank decomposition."""

import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 4, 5]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 3 by 4 matrix
    by a 4 by 5 matrix.
    
    Returns:
        dict with 'U', 'V', 'W' factor matrices
    """
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback (rank = n*m*p = 60)
    rank_schoolbook = n * m * p
    U_school = [[0] * rank_schoolbook for _ in range(n * m)]
    V_school = [[0] * rank_schoolbook for _ in range(m * p)]
    W_school = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    # Build the target tensor for verification
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = i'*n+k'
    # T[a, b, c] = 1 if i=i', j=j', k=k' else 0
    def build_tensor(n, m, p):
        T = np.zeros((n*m, m*p, p*n))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    T[a, b, c] = 1
        return T
    
    T = build_tensor(n, m, p)
    nm, mp, pn = n*m, m*p, p*n
    
    def verify_decomp(U, V, W, tol=1e-9):
        """Check if decomposition is exact for all entries."""
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # Check all non-zero entries
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    c = k * n + i
                    val = 0
                    for r_idx in range(r):
                        val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(val - 1) > tol:
                        return False
        
        # Check all zero entries (sample if too many)
        # Total entries = nm * mp * pn = 12 * 20 * 15 = 3600
        # Non-zero = 60, so 3540 zero entries
        # We can check all of them since 3600 is small
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    i = a // m
                    j = a % m
                    jp = b // p
                    kk = b % p
                    ii = c // n
                    kk2 = c % n
                    if i == ii and j == jp and kk == kk2:
                        continue
                    val = 0
                    for r_idx in range(r):
                        val += U_mat[a, r_idx] * V_mat[b, r_idx] * W_mat[c, r_idx]
                    if abs(val) > tol:
                        return False
        return True
    
    def snap_to_halfint(mat, tol=0.01):
        """Snap values close to half-integers to exact half-integers."""
        mat = mat.copy()
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                # Check if close to 0, 0.5, 1, 1.5, 2, -0.5, -1, etc.
                candidates = []
                for k in range(-4, 5):
                    candidates.append(k / 2.0)
                best = min(candidates, key=lambda x: abs(v - x))
                if abs(v - best) < tol:
                    mat[i, j] = best
        return mat
    
    def polish_decomposition(U, V, W, max_iters=100, tol=1e-12):
        """
        Refine a near-exact decomposition using local search.
        Tries to snap to half-integers and verify.
        """
        r = len(U[0]) if U else 0
        U_mat = np.array(U, dtype=float)
        V_mat = np.array(V, dtype=float)
        W_mat = np.array(W, dtype=float)
        
        # First, try snapping to half-integers
        U_snapped = snap_to_halfint(U_mat)
        V_snapped = snap_to_halfint(V_mat)
        W_snapped = snap_to_halfint(W_mat)
        
        if verify_decomp(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
            return {"U": U_snapped.tolist(), "V": V_snapped.tolist(), "W": W_snapped.tolist()}
        
        # Try local perturbation search
        rng = np.random.RandomState(42)
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Randomly perturb one entry
            mat_name = rng.randint(0, 3)
            if mat_name == 0:
                mat = U_mat
            elif mat_name == 1:
                mat = V_mat
            else:
                mat = W_mat
            
            idx_i = rng.randint(0, mat.shape[0])
            idx_j = rng.randint(0, mat.shape[1])
            
            # Try changing to a nearby half-integer
            current = mat[idx_i, idx_j]
            step = rng.choice([-0.5, 0.5])
            new_val = current + step
            
            # Apply
            old_val = mat[idx_i, idx_j]
            mat[idx_i, idx_j] = new_val
            
            # Check if improved (simplified: just check if it's closer to a valid solution)
            # For now, just accept if it snaps to half-integers and passes a quick check
            if mat_name == 0:
                U_mat = mat
            elif mat_name == 1:
                V_mat = mat
            else:
                W_mat = mat
            
            # Periodically check full validity
            if it % 10 == 0:
                U_s = snap_to_halfint(U_mat)
                V_s = snap_to_halfint(V_mat)
                W_s = snap_to_halfint(W_mat)
                if verify_decomp(U_s.tolist(), V_s.tolist(), W_s.tolist()):
                    return {"U": U_s.tolist(), "V": V_s.tolist(), "W": W_s.tolist()}
        
        return None
    
    # Strategy: Divide and conquer by partitioning the inner dimension m=4 into two parts of size 2
    # This reduces the problem to two 3x2 by 2x5 multiplications
    # However, the tensor structure is more complex. Let's use a different approach.
    
    # Better strategy: Use known low-rank algorithms for small matrix multiplication.
    # For 3x3x3, rank is 23 (Strassen-like). For 3x4x5, we can try to construct
    # from known patterns or use a more systematic search.
    
    # Let's try a structured search:
    # 1. Start with schoolbook and try to merge columns
    # 2. Use random initialization with optimization
    # 3. Use divide-and-conquer if applicable
    
    # First, try a more systematic approach:
    # The matrix multiplication tensor M_{n,m,p} can be decomposed using
    # Kronecker products and known lower-rank decompositions.
    
    # For now, let's try random search with local optimization for ranks 45-59
    best_result = None
    
    # Try ranks from 45 to 59
    for r in range(45, 60):
        if time.time() > deadline - 10:
            break
        
        # Try a few random initializations
        for attempt in range(3):
            if time.time() > deadline - 10:
                break
            
            rng = np.random.RandomState(r * 100 + attempt)
            
            # Initialize with sparse random values
            U_mat = np.zeros((nm, r))
            V_mat = np.zeros((mp, r))
            W_mat = np.zeros((pn, r))
            
            # For each column, randomly assign values from {-1, 0, 0.5, 1}
            for c in range(r):
                for i in range(nm):
                    if rng.random() < 0.2:
                        U_mat[i, c] = rng.choice([-1, -0.5, 0.5, 1])
                for j in range(mp):
                    if rng.random() < 0.2:
                        V_mat[j, c] = rng.choice([-1, -0.5, 0.5, 1])
                for k in range(pn):
                    if rng.random() < 0.2:
                        W_mat[k, c] = rng.choice([-1, -0.5, 0.5, 1])
            
            # Check if this works (unlikely, but worth trying)
            if verify_decomp(U_mat.tolist(), V_mat.tolist(), W_mat.tolist()):
                best_result = {"U": U_mat.tolist(), "V": V_mat.tolist(), "W": W_mat.tolist()}
                break
            
            # Try polishing
            polished = polish_decomposition(U_mat.tolist(), V_mat.tolist(), W_mat.tolist(), max_iters=20)
            if polished is not None:
                best_result = polished
                break
        
        if best_result is not None:
            break
    
    # If no result found, try a more structured approach:
    # Use the fact that M_{3,4,5} can be embedded in M_{4,4,4} or similar
    # Or use a greedy column merging approach
    
    if best_result is None:
        # Try greedy merging: start with schoolbook and try to combine columns
        # This is complex, so fall back to schoolbook
        return {"U": U_school, "V": V_school, "W": W_school}
    else:
        return best_result

if __name__ == "__main__":
    result = construct()
    rank = len(result["U"][0]) if result["U"] else 0
    print(f"Found decomposition with rank: {rank}")
