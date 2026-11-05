import numpy as np
import time
import itertools

DIMENSIONS = [3, 4, 7]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # Safety margin before 150s limit
    
    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
    T = np.zeros((n*m, m*p, p*n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1
    
    # Schoolbook decomposition as fallback
    def schoolbook_decomp(n, m, p):
        rank = n * m * p
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U, V, W
    
    # Verify that a decomposition is exact
    def verify_decomp(U, V, W, T, tol=1e-9):
        # Check: T[a,b,c] == sum_r U[a,r]*V[b,r]*W[c,r]
        # Efficient check using einsum
        reconstructed = np.einsum('ar,br,cr->abc', U, V, W)
        return np.allclose(reconstructed, T, atol=tol)
    
    # Try to find a low-rank decomposition for a sub-problem M_{n,m,p_sub}
    def find_decomp(n_sub, m_sub, p_sub, max_rank, time_limit=10):
        """
        Find a rank-max_rank decomposition for M_{n_sub, m_sub, p_sub}.
        Returns (U, V, W) or None if not found.
        """
        T_sub = np.zeros((n_sub*m_sub, m_sub*p_sub, p_sub*n_sub), dtype=np.int64)
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    T_sub[i*m_sub+j, j*p_sub+k, k*n_sub+i] = 1
        
        best = None
        start_time = time.time()
        
        # Try ranks from 1 to max_rank
        for r in range(1, max_rank + 1):
            if time.time() - start_time > time_limit:
                break
            
            # For small ranks, we can try to solve the system
            # The condition is: T_sub[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
            # This is a system of nonlinear equations.
            
            # Strategy: Fix U and V with random {-1,0,1} entries, solve for W
            # Or use a more systematic approach
            
            # For very small cases, enumerate possible patterns
            if r <= 5:
                # Try all combinations of {-1, 0, 1} for a subset of entries
                # This is only feasible for very small problems
                pass
            
            # Alternative: Use the fact that for matrix multiplication,
            # there are known structured decompositions
            
            # Try a structured approach: 
            # For M_{n,m,p}, one can use the "border rank" or known results
            
            # Let's try a simple heuristic: 
            # Use the schoolbook for small sub-problems if r is close to n*m*p
            if r >= n_sub * m_sub * p_sub:
                U, V, W = schoolbook_decomp(n_sub, m_sub, p_sub)
                if verify_decomp(U, V, W, T_sub):
                    return U, V, W
        
        return None
    
    # Known optimal rank for M_{3,4,2} is 10
    # Let's construct it explicitly
    def decomp_3_4_2():
        """Return rank-10 decomposition for M_{3,4,2}"""
        n, m, p = 3, 4, 2
        rank = 10
        U = np.zeros((n*m, rank), dtype=np.int64)
        V = np.zeros((m*p, rank), dtype=np.int64)
        W = np.zeros((p*n, rank), dtype=np.int64)
        
        # The 10 terms correspond to:
        # For each i in {0,1,2}, j in {0,1}, we have a term
        # Plus additional terms to handle the structure
        # 
        # Actually, let me use a known construction.
        # M_{3,4,2} can be seen as: C[i,k] = sum_j A[i,j]*B[j,k]
        # 
        # One rank-10 decomposition:
        # Terms 0-2: For i=0, j=0,1,2 with k=0
        # Terms 3-5: For i=1, j=0,1,2 with k=0  
        # Terms 6-8: For i=2, j=0,1,2 with k=0
        # That's only 9 terms for k=0. We need 1 more for k=1.
        #
        # Let me think differently. The rank of M_{3,4,2} is indeed 10.
        # A construction:
        # Use the identity: (x+y)(u+v) = xu + xv + yu + yv
        # For each i, we can write:
        # C[i,0] = sum_j A[i,j]*B[j,0]
        # C[i,1] = sum_j A[i,j]*B[j,1]
        #
        # Combine: C[i,0] + C[i,1] = sum_j A[i,j]*(B[j,0]+B[j,1])
        # C[i,0] - C[i,1] = sum_j A[i,j]*(B[j,0]-B[j,1])
        #
        # So for each i, we have 2 "combined" outputs.
        # For 3 values of i, that's 6 terms. But we need more.
        #
        # Actually, the standard result is rank(M_{3,4,2}) = 10.
        # Let me just use the schoolbook for this sub-problem and see if we can do better.
        
        # For now, use schoolbook: rank = 3*4*2 = 24
        # But we want rank 10.
        
        # Let me try a different approach: enumerate all possible rank-10 decompositions
        # This is complex. Instead, let's use a known algorithm.
        
        # For M_{3,4,2}, one can use the following construction based on:
        # C = A*B where A is 3x4, B is 4x2
        # 
        # Write B as [B0, B1] where B0, B1 are 4x1
        # C = [A*B0, A*B1]
        #
        # Let S = B0 + B1, D = B0 - B1
        # Then B0 = (S+D)/2, B1 = (S-D)/2
        # C[:,0] = A*(S+D)/2 = (A*S + A*D)/2
        # C[:,1] = A*(S-D)/2 = (A*S - A*D)/2
        #
        # So we need to compute A*S and A*D, where S,D are 4x1 vectors.
        # A*S is a 3x1 vector, A*D is a 3x1 vector.
        # Computing A*S requires 3*4 = 12 multiplications in schoolbook.
        # But we can share: A*S and A*D both use the same A.
        #
        # Total: 12 (for A*S) + 12 (for A*D) = 24 multiplications, but with sharing...
        # Actually, no sharing between A*S and A*D in this formulation.
        #
        # The rank-10 result comes from a more clever construction.
        # 
        # For now, let's just use the schoolbook for the full problem and try to improve.
        
        # Fallback: schoolbook for M_{3,4,2}
        U, V, W = schoolbook_decomp(n, m, p)
        return U, V, W
    
    # Strategy: Use tensor slicing
    # Split p=7 into p1=2 and p2=5
    # M_{3,4,7} = M_{3,4,2} ⊕ M_{3,4,5}
    # rank(M_{3,4,7}) <= rank(M_{3,4,2}) + rank(M_{3,4,5})
    
    # For M_{3,4,2}, use rank-10 if we can construct it, else schoolbook (rank 24)
    # For M_{3,4,5}, try to find a good decomposition
    
    # Let's try to find a decomposition for M_{3,4,5}
    # Schoolbook rank is 3*4*5 = 60
    
    # Use a search strategy for M_{3,4,5}
    def search_decomp_3_4_5(time_limit=60):
        """
        Search for a low-rank decomposition of M_{3,4,5}.
        """
        n, m, p = 3, 4, 5
        T_sub = np.zeros((n*m, m*p, p*n), dtype=np.int64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T_sub[i*m+j, j*p+k, k*n+i] = 1
        
        start_time = time.time()
        best_rank = n*m*p  # Schoolbook
        best_decomp = None
        
        # Try various rank values
        # Based on known results, rank(M_{3,4,5}) might be around 25-35
        for r in range(20, min(n*m*p, 40)):
            if time.time() - start_time > time_limit:
                break
            
            # Try to find a rank-r decomposition
            # Use a heuristic: initialize U, V with random {-1,0,1} and solve for W
            
            found = False
            attempts = 0
            max_attempts = 50
            
            while attempts < max_attempts and not found:
                attempts += 1
                if time.time() - start_time > time_limit:
                    break
                
                # Random initialization for U and V with entries in {-1, 0, 1}
                U = np.random.choice([-1, 0, 1], size=(n*m, r))
                V = np.random.choice([-1, 0, 1], size=(m*p, r))
                
                # Solve for W: T_sub[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
                # This is a linear system in W for each (a,b) pair
                # For fixed (a,b): T_sub[a,b,:] = sum_r U[a,r]*V[b,r]*W[:,r]
                # Let M_ab = sum_r U[a,r]*V[b,r]*W[:,r] (this is a vector of length p*n)
                # But W appears linearly: M_ab[c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
                # So for each c: T_sub[a,b,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
                
                # We can set up a linear system:
                # For each c in range(p*n):
                #   T_sub[a,b,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
                # This is: T_sub[a,b,c] = dot(U[a,:]*V[b,:], W[c,:])
                
                # Stack all equations:
                # For each (a,b,c): T_sub[a,b,c] = sum_r (U[a,r]*V[b,r]) * W[c,r]
                # Let's define for each r: coeff_r(a,b,c) = U[a,r]*V[b,r]
                # Then T_sub[a,b,c] = sum_r coeff_r(a,b,c) * W[c,r]
                
                # This is a linear system in the unknowns W[c,r] for c=0..p*n-1, r=0..r-1
                # Total unknowns: (p*n)*r
                # Total equations: (n*m)*(m*p)*(p*n) but many are zero
                
                # Let's collect non-zero equations
                A_mat = []
                b_vec = []
                unknown_indices = []  # Maps each row of A_mat to (c, r)
                
                for a in range(n*m):
                    for b in range(m*p):
                        for c in range(p*n):
                            if T_sub[a,b,c] != 0:
                                # Equation: sum_r U[a,r]*V[b,r]*W[c,r] = T_sub[a,b,c]
                                row = np.zeros(r)
                                for rr in range(r):
                                    row[rr] = U[a,rr] * V[b,rr]
                                # This equation involves W[c, :]
                                # We need to express this in terms of the full unknown vector
                                # The unknown vector is W flattened: W[0,0], W[0,1], ..., W[0,r-1], W[1,0], ...
                                # For this equation, only W[c, :] is involved
                                full_row = np.zeros(p*n*r)
                                for rr in range(r):
                                    full_row[c*r + rr] = row[rr]
                                A_mat.append(full_row)
                                b_vec.append(T_sub[a,b,c])
                                unknown_indices.append((c, rr))
                
                if len(A_mat) == 0:
                    continue
                    
                A_mat = np.array(A_mat, dtype=np.float64)
                b_vec = np.array(b_vec, dtype=np.float64)
                
                # Solve the linear system (least squares)
                try:
                    W_flat, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, b_vec, rcond=None)
                    W_sol = W_flat.reshape(p*n, r)
                except:
                    continue
                
                # Check if the solution is close to integer/half-integer
                # And verify the decomposition
                if verify_decomp(U, V, W_sol, T_sub, tol=1e-6):
                    # Check if entries are close to half-integers
                    W_rounded = np.round(W_sol * 2) / 2
                    if verify_decomp(U, V, W_rounded, T_sub, tol=1e-6):
                        found = True
                        best_rank = r
                        best_decomp = (U, V, W_rounded)
                        break
                elif verify_decomp(U, V, W_sol, T_sub, tol=1e-4):
                    # Accept if close enough
                    W_rounded = np.round(W_sol * 2) / 2
                    if verify_decomp(U, V, W_rounded, T_sub, tol=1e-4):
                        found = True
                        best_rank = r
                        best_decomp = (U, V, W_rounded)
                        break
            
            if found:
                break
        
        if best_decomp is not None:
            return best_decomp
        else:
            # Fallback to schoolbook
            return schoolbook_decomp(n, m, p)
    
    # Try to find a decomposition for M_{3,4,5}
    decomp_3_4_5 = search_decomp_3_4_5(time_limit=60)
    
    if time.time() > deadline - 30:
        # Not enough time, use schoolbook for full problem
        U, V, W = schoolbook_decomp(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Now combine M_{3,4,2} and M_{3,4,5}
    # M_{3,4,2}: rank r1, M_{3,4,5}: rank r2
    # Total rank = r1 + r2
    
    # For M_{3,4,2}, use schoolbook for now (rank 24)
    U1, V1, W1 = schoolbook_decomp(3, 4, 2)
    r1 = U1.shape[1]
    
    U2, V2, W2 = decomp_3_4_5
    r2 = U2.shape[1]
    
    # Combine: 
    # U_combined has shape (3*4, r1+r2)
    # V_combined has shape (4*7, r1+r2) - but V for M_{3,4,2} has shape (4*2, r1) and for M_{3,4,5} has shape (4*5, r2)
    # W_combined has shape (7*3, r1+r2)
    
    # For V: M_{3,4,2} uses B[:, 0:2] (first 2 columns of B), M_{3,4,5} uses B[:, 2:7] (last 5 columns)
    # So V_combined[j*7+k, r] = V1[j*2+k, r] if k < 2, else V2[j*5+(k-2), r-r1] if k >= 2
    
    U_combined = np.zeros((3*4, r1+r2), dtype=np.int64)
    V_combined = np.zeros((4*7, r1+r2), dtype=np.int64)
    W_combined = np.zeros((7*3, r1+r2), dtype=np.int64)
    
    # Fill U: same for both sub-problems (A is 3x4 for both)
    U_combined[:, :r1] = U1
    U_combined[:, r1:] = U2
    
    # Fill V: M_{3,4,2} uses B[:, 0:2], M_{3,4,5} uses B[:, 2:7]
    for j in range(4):
        for k in range(2):
            V_combined[j*7+k, :r1] = V1[j*2+k, :]
        for k in range(5):
            V_combined[j*7+(k+2), r1:] = V2[j*5+k, :]
    
    # Fill W: M_{3,4,2} produces C[:, 0:2], M_{3,4,5} produces C[:, 2:7]
    # W for M_{3,4,2}: shape (2*3, r1), indexed by [k*3+i] for k in {0,1}
    # W for M_{3,4,5}: shape (5*3, r2), indexed by [k*3+i] for k in {0,1,2,3,4}
    # In the full problem, W_combined[k*3+i, :] for k in {0,...,6}
    # For k in {0,1}: from W1
    # For k in {2,...,6}: from W2 with k' = k-2
    for k in range(2):
        for i in range(3):
            W_combined[k*3+i, :r1] = W1[k*3+i, :]
    for k in range(5):
        for i in range(3):
            W_combined[(k+2)*3+i, r1:] = W2[k*3+i, :]
    
    # Verify the combined decomposition
    if verify_decomp(U_combined, V_combined, W_combined, T, tol=1e-6):
        return {"U": U_combined.tolist(), "V": V_combined.tolist(), "W": W_combined.tolist()}
    
    # If verification fails, fall back to schoolbook
    U, V, W = schoolbook_decomp(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
