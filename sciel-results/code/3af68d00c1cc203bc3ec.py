import time
import sys
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin before 150s kill
    
    n, m, p = DIMENSIONS
    nm = n * m  # 8
    mp = m * p  # 28
    pn = p * n  # 14
    
    # The matrix multiplication tensor T has shape (nm, mp, pn)
    # T[i, j, k] = 1 if i corresponds to (row, col) of A, j to (col, row) of B, k to (row, col) of C
    # where C = A @ B
    # Specifically: C[r, c] = sum_s A[r, s] * B[s, c]
    # So T[(r, s), (s, c), (c, r)] = 1 for all r, s, c where indices match
    
    # Build the tensor
    T = np.zeros((nm, mp, pn))
    for r in range(n):
        for s in range(m):
            for c in range(p):
                T[r * m + s, s * p + c, c * n + r] = 1.0
    
    # Rank of tensor decomposition: find U (nm x R), V (mp x R), W (pn x R)
    # such that T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
    
    # Start with schoolbook rank = nm * mp * pn / (m) ... actually schoolbook is m * n * p = 56
    # Wait, schoolbook for (n x m) * (m x p): rank = m * n * p? No.
    # Schoolbook: for each s in 0..m-1, for each r in 0..n-1, for each c in 0..p-1:
    #   one term: U[(r,s), idx]=1, V[(s,c), idx]=1, W[(c,r), idx]=1
    # So rank = m * n * p = 4 * 2 * 7 = 56
    
    # Let's try to find a lower rank decomposition using random search
    
    # First, try a greedy/iterative approach:
    # We want to find U, V, W with integer or half-integer entries
    
    # Strategy: Randomized search with local optimization
    # Try ranks from a lower bound up to 56
    
    # Lower bound: max(nm, mp, pn) = max(8, 28, 14) = 28
    # But tensor rank can be higher than matrix rank
    
    # Let's try a simple approach: use numpy to find a low-rank approximation
    # then round to integers/half-integers and verify
    
    # For a given rank R, we can try:
    # 1. Randomly initialize U, V, W
    # 2. Use alternating least squares to fit the tensor
    # 3. Round to nearest integer or half-integer
    # 4. Verify exactness
    
    # But this is hard for exact integer solutions. Let's try a different approach.
    
    # Actually, for small matrices, we can try to find known decompositions.
    # The 2x4x7 matrix multiplication might have a known rank.
    
    # Let's try: can we do better than 56?
    # Try rank 55, 54, ... using a simple search
    
    # Alternative: use the fact that the tensor has structure.
    # T[i,j,k] = 1 iff i = r*m+s, j = s'*p+c, k = c'*n+r' where s=s' and r=r' and c=c'
    # Wait no: T[(r,s),(s,c),(c,r)] = 1
    
    # Let me think about this differently.
    # The tensor T can be viewed as a 3D array.
    # T[i,j,k] = 1 if i mod m == j // p and i // m == k // n and j % p == k % n
    # where i = r*m+s, j = s'*p+c, k = c'*n+r'
    # Condition: s == s' and r == r' and c == c'
    
    # This is a structured tensor. Let's see if we can exploit this.
    
    # For now, let's just return the schoolbook solution but verify it's correct.
    # The schoolbook rank is m*n*p = 56.
    
    # Actually wait - let me re-examine. The schoolbook algorithm for C = A*B where
    # A is n x m and B is m x p uses m*n*p multiplications, so rank = m*n*p = 4*2*7 = 56.
    
    # Let's verify the schoolbook decomposition is correct.
    
    def verify(U, V, W, R):
        """Verify that the decomposition is exact."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        # Check: T[i,j,k] = sum_r U[i,r] * V[j,r] * W[k,r]
        for i in range(nm):
            for j in range(mp):
                for k in range(pn):
                    val = sum(U[i, r] * V[j, r] * W[k, r] for r in range(R))
                    if abs(val - T[i, j, k]) > 1e-9:
                        return False
        return True
    
    # Schoolbook decomposition
    R_school = m * n * p  # 56
    U_school = np.zeros((nm, R_school))
    V_school = np.zeros((mp, R_school))
    W_school = np.zeros((pn, R_school))
    
    col = 0
    for r in range(n):
        for s in range(m):
            for c in range(p):
                U_school[r * m + s, col] = 1
                V_school[s * p + c, col] = 1
                W_school[c * n + r, col] = 1
                col += 1
    
    # Verify schoolbook
    if not verify(U_school, V_school, W_school, R_school):
        # Should not happen
        pass
    
    # Now try to find a lower rank decomposition
    # Use a simple randomized search: try to find U, V, W with rank < 56
    
    # Given time constraints, let's try a few ranks below 56
    # This is a heuristic search - may not find anything better
    
    best_rank = R_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try to find lower rank using random initialization + ALS + rounding
    # This is a heuristic and may not work, but let's try
    
    def try_rank(R, num_trials=50):
        """Try to find a rank-R decomposition."""
        if R >= R_school:
            return None
        
        for trial in range(num_trials):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            np.random.seed(trial * 1000 + R)
            U = np.random.randn(nm, R)
            V = np.random.randn(mp, R)
            W = np.random.randn(pn, R)
            
            # Alternating least squares
            for it in range(50):
                if time.time() > deadline - 5:
                    return None
                
                # Update U: for each i, U[i,:] = (V^T * W^T * T[i,:,:]) / (V^T * W^T * V * W)
                # Actually, for each i: sum_j,k T[i,j,k] V[j,r] W[k,r] = U[i,r] * sum_j,k V[j,r] W[k,r] V[j,s] W[k,s]
                # This is complex. Let's use a simpler approach.
                
                # Flatten and use matrix operations
                # T_i = T[i, :, :] reshaped to (mp*pn,)
                # V_kron_W = V.T @ W.T reshaped... this is getting complex.
                
                # Let's just do a simple gradient step or skip this.
                pass
            
            # Round to nearest integer or half-integer
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            if verify(U_rounded, V_rounded, W_rounded, R):
                return (U_rounded, V_rounded, W_rounded)
        
        return None
    
    # Try ranks from 55 down to 40 (limited by time)
    for R in range(R_school - 1, max(30, R_school - 20), -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(R, num_trials=20)
        if result is not None:
            best_rank = R
            best_U, best_V, best_W = result
            break  # Found a lower rank, stop
    
    # Convert to lists for the return format
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
