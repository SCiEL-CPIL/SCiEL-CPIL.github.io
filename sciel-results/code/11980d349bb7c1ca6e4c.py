import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Schoolbook decomposition (rank 63) as fallback
    rank_school = n * m * p
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    def to_matrix(lst):
        return [list(row) for row in lst]
    
    def verify(U, V, W, n, m, p):
        rank = len(U[0]) if U else 0
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if any(len(row) != rank for row in U) or any(len(row) != rank for row in V) or any(len(row) != rank for row in W):
            return False
        
        # Check if entries are integers or half-integers
        for mat in [U, V, W]:
            for row in mat:
                for val in row:
                    if isinstance(val, (int, float)):
                        v = float(val)
                        if v != int(v) and abs(v * 2 - int(v * 2)) > 1e-9:
                            return False
                    elif isinstance(val, Fraction):
                        if val.denominator not in (1, 2):
                            return False
        
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        return True
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    if verify(U_school, V_school, W_school, n, m, p):
        best_rank = rank_school
        best_U = U_school
        best_V = V_school
        best_W = W_school
    
    # Strategy: Try to find lower rank by exploiting linear structure
    # The tensor T_{i,j,k} = 1 for all i,j,k can be decomposed as sum of rank-1 tensors
    # We'll use a randomized approach: fix U and V randomly, then solve for W
    
    def try_rank(r, n, m, p, max_iters=100):
        """Try to find a rank-r decomposition"""
        rng = np.random.RandomState(42 + r)
        
        for attempt in range(max_iters):
            # Random initialization with small values
            U = rng.randn(n * m, r) * 0.1
            V = rng.randn(m * p, r) * 0.1
            
            # For each (i,k) pair, we need:
            # sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is a linear system in W!
            
            # Build the linear system: for each (i,k), the coefficient of W[k*n+i, r] 
            # is sum_j U[i*m+j, r] * V[j*p+k, r]
            
            # Actually, let's think differently. For fixed U and V, we need to find W such that:
            # For all i,j,k: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            
            # This is nonlinear in W because W appears multiplied by U*V terms.
            # But if we fix the pattern, we can use iterative methods.
            
            # Alternative: use the fact that we can scale. Let's try a different approach.
            # Initialize W randomly and use gradient descent or fixed-point iteration.
            
            W = rng.randn(p * n, r) * 0.1
            
            # Fixed-point iteration: update W to minimize error
            for it in range(50):
                if time.time() > deadline - 20:
                    return None
                
                # Compute current tensor values
                # T_computed[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                
                # Update W using a simple rule: for each (i,k,r), 
                # W[k*n+i, r] should be adjusted to help satisfy the equations
                # where U[i*m+j, r] * V[j*p+k, r] != 0
                
                W_new = W.copy()
                for i in range(n):
                    for k in range(p):
                        for r in range(r):
                            # Sum over j of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            # should contribute to the total for each j
                            # This is complex; let's use a simpler heuristic
                            pass
                
                # This approach is getting complicated. Let's try a different strategy.
                # Use the schoolbook structure but try to merge terms.
                break
            
            # Simpler approach: try to find rank r by solving a linear system
            # Assume W has a specific structure. For example, if we set W[k*n+i, r] = 1/r
            # for all k,i,r, then we need sum_r U[i*m+j, r] * V[j*p+k, r] / r = 1
            # This is still nonlinear.
            
            # Let's try yet another approach: use the known result that R(3,3,3) <= 23
            # and see if we can extend it to p=7.
            # For now, return None to indicate failure
            return None
    
    # Try a more systematic approach: use the structure of the problem
    # C_{ik} = sum_j A_{ij} B_{jk}
    # We can write this as C = A * B
    # The tensor is the identity bilinear map.
    
    # Key insight: we can use the fact that for fixed i,k, the equation is
    # sum_j A_{ij} B_{jk} = C_{ik}
    # This is a sum of m terms. We can try to represent each product A_{ij} * B_{jk}
    # using fewer than m terms by exploiting linearity.
    
    # Actually, the simplest improvement is to note that we don't need all 63 terms.
    # Let's try rank 50 and see if we can find a valid decomposition.
    
    # Use a greedy approach: start with schoolbook and try to merge/remove terms
    # while maintaining validity.
    
    def try_greedy_reduction(n, m, p, target_rank, U_init, V_init, W_init):
        """Try to reduce rank by removing terms that don't affect the result"""
        U = [row[:] for row in U_init]
        V = [row[:] for row in V_init]
        W = [row[:] for row in W_init]
        current_rank = len(U[0])
        
        # Try to remove each term one by one and check if the decomposition still holds
        for term_to_remove in range(current_rank - 1, target_rank - 1, -1):
            if time.time() > deadline - 30:
                break
            
            # Temporarily remove this term
            U_test = [row[:term_to_remove] + row[term_to_remove+1:] for row in U]
            V_test = [row[:term_to_remove] + row[term_to_remove+1:] for row in V]
            W_test = [row[:term_to_remove] + row[term_to_remove+1:] for row in W]
            
            # Check if this still works
            U_arr = np.array(U_test, dtype=float)
            V_arr = np.array(V_test, dtype=float)
            W_arr = np.array(W_test, dtype=float)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(term_to_remove):
                            s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                        if abs(s - 1.0) > 1e-9:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                # Keep the removal
                U = U_test
                V = V_test
                W = W_test
                current_rank = term_to_remove
        
        return U, V, W, current_rank
    
    # Try greedy reduction from schoolbook
    if time.time() < deadline - 40:
        U_red, V_red, W_red, rank_red = try_greedy_reduction(n, m, p, 40, U_school, V_school, W_school)
        if verify(U_red, V_red, W_red, n, m, p) and rank_red < best_rank:
            best_rank = rank_red
            best_U = U_red
            best_V = V_red
            best_W = W_red
    
    # Try to construct a better decomposition using block structure
    # Split the 7 columns into groups and use known results for smaller matrices
    
    # For 3x3x3, we know R <= 23. Let's try to hardcode a known rank-23 decomposition.
    # This is complex, so we'll try a randomized search for rank 50.
    
    def randomized_search(rank, n, m, p, max_attempts=20):
        """Randomized search for a rank-r decomposition"""
        rng = np.random.RandomState(123 + rank)
        
        for attempt in range(max_attempts):
            if time.time() > deadline - 20:
                return None
            
            # Initialize with small random values
            U = rng.randn(n * m, rank) * 0.05
            V = rng.randn(m * p, rank) * 0.05
            W = rng.randn(p * n, rank) * 0.05
            
            # Use iterative refinement: for each (i,j,k), adjust W to reduce error
            for iteration in range(100):
                if time.time() > deadline - 15:
                    break
                
                # Compute errors for all (i,j,k)
                max_error = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            s = 0.0
                            for r in range(rank):
                                s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                            error = 1.0 - s
                            max_error = max(max_error, abs(error))
                
                if max_error < 1e-6:
                    # Found a solution!
                    return U.tolist(), V.tolist(), W.tolist()
                
                # Update W: for each (i,k,r), the contribution to error for (i,j,k) is
                # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                # We want to adjust W[k*n+i, r] to reduce the total error.
                # Simple update: W[k*n+i, r] += eta * sum_j U[i*m+j, r] * V[j*p+k, r] * error(i,j,k)
                eta = 0.1
                for i in range(n):
                    for k in range(p):
                        for r in range(rank):
                            # Sum over j of U[i*m+j, r] * V[j*p+k, r] * error(i,j,k)
                            grad = 0.0
                            for j in range(m):
                                s = 0.0
                                for rr in range(rank):
                                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                error_jk = 1.0 - s
                                grad += U[i*m+j, r] * V[j*p+k, r] * error_jk
                            W[k*n+i, r] += eta * grad
                
                # Also update U and V similarly
                for i in range(n):
                    for j in range(m):
                        for r in range(rank):
                            grad = 0.0
                            for k in range(p):
                                s = 0.0
                                for rr in range(rank):
                                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                error_ik = 1.0 - s
                                grad += V[j*p+k, r] * W[k*n+i, r] * error_ik
                            U[i*m+j, r] += eta * grad
                
                for j in range(m):
                    for k in range(p):
                        for r in range(rank):
                            grad = 0.0
                            for i in range(n):
                                s = 0.0
                                for rr in range(rank):
                                    s += U[i*m+j, rr] * V[j*p+k, rr] * W[k*n+i, rr]
                                error_jk = 1.0 - s
                                grad += U[i*m+j, r] * W[k*n+i, r] * error_jk
                            V[j*p+k, r] += eta * grad
            
            # Check final result
            U_arr = np.array(U, dtype=float)
            V_arr = np.array(V, dtype=float)
            W_arr = np.array(W, dtype=float)
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(rank):
                            s += U_arr[i*m+j, r] * V_arr[j*p+k, r] * W_arr[k*n+i, r]
                        if abs(s - 1.0) > 1e-5:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return U.tolist(), V.tolist(), W.tolist()
        
        return None
    
    # Try ranks from 55 down to 45
    for target_rank in [55, 52, 50, 48, 45]:
        if time.time() > deadline - 25:
            break
        
        result = randomized_search(target_rank, n, m, p, max_attempts=5)
        if result is not None:
            U_new, V_new, W_new = result
            # Convert to half-integers if possible
            if verify(U_new, V_new, W_new, n, m, p) and target_rank < best_rank:
                best_rank = target_rank
                best_U = U_new
                best_V = V_new
                best_W = W_new
                break
    
    if time.time() > deadline - 10:
        return {"U": to_matrix(best_U), "V": to_matrix(best_V), "W": to_matrix(best_W)}
    
    return {"U": to_matrix(best_U), "V": to_matrix(best_V), "W": to_matrix(best_W)}
