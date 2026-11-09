import numpy as np
import time

DIMENSIONS = [3, 3, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    
    def check_solution(U, V, W, tol=1e-9):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > tol:
                        return False
        return True
    
    def round_to_half_int(x):
        return np.round(x * 2) / 2
    
    def solve_for_W(U, V):
        R = U.shape[1]
        W = np.zeros((p*n, R))
        for i in range(n):
            for k in range(p):
                A = np.zeros((m, R))
                b = np.ones(m)
                for j in range(m):
                    A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[k*n+i, :] = w
        return W
    
    def solve_for_V(U, W):
        R = U.shape[1]
        V = np.zeros((m*p, R))
        for j in range(m):
            for k in range(p):
                A = np.zeros((n, R))
                b = np.ones(n)
                for i in range(n):
                    A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[j*p+k, :] = v
        return V
    
    def solve_for_U(V, W):
        R = V.shape[1]
        U = np.zeros((n*m, R))
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, R))
                b = np.ones(p)
                for k in range(p):
                    A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                u, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[i*m+j, :] = u
        return U
    
    def build_winograd_3x3x3():
        """Build Winograd-style decomposition for 3x3x3 core."""
        # Winograd F(2,3) style: use 3x3 transforms
        # A = F^T * a * F, where F is 3x3 transform
        # For 3x3x3: rank can be reduced from 27 to ~15-16
        # Using structure: U depends on (i,j), V on (j,k), W on (k,i)
        
        # Try a structured approach with small integer entries
        # Winograd for 3x3x3 has known rank ~15
        # Let's construct using the idea that:
        # C[i,k] = sum_j A[i,j]*B[j,k]
        # Tensor: T[i,j,k] = delta[i,j,k] (1 if this triple appears)
        
        # Use the fact that for 3x3x3, we can use:
        # F_3 = [[1,0,0],[1,1,0],[1,1,1]] or similar
        # Actually, let's use a known construction
        
        # Winograd F(3,3): 
        # Transform A: a = F^T A F
        # Transform B: b = F^T B F  
        # Pointwise multiply: c = a * b
        # Inverse: C = F^T c F
        # This gives rank = 3x3 = 9 for the pointwise part, but with transforms...
        
        # Actually for matrix mult 3x3x3, the tensor rank is at most 27 (schoolbook)
        # Winograd reduces it. Let me use a different approach.
        
        # Use the structure: 
        # Let's define basis vectors for U, V, W
        # U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
        
        # For 3x3x3, try rank 15 construction
        # Using circulant-like structure
        
        # Simple approach: use the fact that we can share computations
        # Let's just build schoolbook for 3x3x3 and then extend
        
        rank_core = n * m * n  # 3*3*3 = 27 for 3x3x3
        U_core = np.zeros((n*m, rank_core))
        V_core = np.zeros((m*n, rank_core))
        W_core = np.zeros((n*n, rank_core))
        
        col = 0
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    U_core[i*m+j, col] = 1
                    V_core[j*n+k, col] = 1
                    W_core[k*n+i, col] = 1
                    col += 1
        
        return U_core, V_core, W_core
    
    # Build initial solution: schoolbook for 3x3x8
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    # Phase 1: Try to reduce by removing columns one at a time
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for r in range(rank):
            if time.time() > deadline:
                break
            
            mask = np.ones(rank, dtype=bool)
            mask[r] = False
            U_trial = U[:, mask]
            V_trial = V[:, mask]
            W_trial = W[:, mask]
            
            # Alternating solve
            for _ in range(3):
                W_trial = solve_for_W(U_trial, V_trial)
                V_trial = solve_for_V(U_trial, W_trial)
                U_trial = solve_for_U(V_trial, W_trial)
            
            # Round to half-integers
            U_rounded = round_to_half_int(U_trial)
            V_rounded = round_to_half_int(V_trial)
            W_rounded = round_to_half_int(W_trial)
            
            if check_solution(U_rounded, V_rounded, W_rounded):
                U, V, W = U_rounded, V_rounded, W_rounded
                rank -= 1
                improved = True
                break
    
    # Phase 2: Try structured Winograd-like constructions
    # For 3x3x8, we can think of it as 8 copies of 3x3x1 or use shared structure
    # Try rank values from known lower bounds upward
    
    # The theoretical lower bound for 3x3x8 matrix multiplication
    # is at least max(3*3, 3*8, 8*3) = 24, but likely higher
    # Schoolbook is 72. Winograd for 3x3x3 gives ~15-16 per "block"
    
    # Try to find a good rank by random restarts with structured initialization
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    if time.time() < deadline:
        # Try various target ranks
        for target_R in range(28, best_rank + 1, 2):
            if time.time() > deadline:
                break
            
            found = False
            for restart in range(15):
                if time.time() > deadline:
                    break
                
                np.random.seed(restart * 1000 + target_R)
                
                # Initialize with structured random
                U_trial = np.random.randn(n*m, target_R) * 0.3
                V_trial = np.random.randn(m*p, target_R) * 0.3
                W_trial = np.random.randn(p*n, target_R) * 0.3
                
                # Add some structure: make entries sparse-ish
                U_trial[np.abs(U_trial) < 0.2] = 0
                V_trial[np.abs(V_trial) < 0.2] = 0
                W_trial[np.abs(W_trial) < 0.2] = 0
                
                for iteration in range(100):
                    if time.time() > deadline:
                        break
                    
                    W_trial = solve_for_W(U_trial, V_trial)
                    V_trial = solve_for_V(U_trial, W_trial)
                    U_trial = solve_for_U(V_trial, W_trial)
                    
                    # Try rounding
                    U_rounded = round_to_half_int(U_trial)
                    V_rounded = round_to_half_int(V_trial)
                    W_rounded = round_to_half_int(W_trial)
                    
                    if check_solution(U_rounded, V_rounded, W_rounded):
                        U, V, W = U_rounded, V_rounded, W_rounded
                        best_rank = target_R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
                    
                    # Also try integer rounding
                    U_int = np.round(U_trial)
                    V_int = np.round(V_trial)
                    W_int = np.round(W_trial)
                    if check_solution(U_int, V_int, W_int):
                        U, V, W = U_int, V_int, W_int
                        best_rank = target_R
                        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
                        found = True
                        break
                
                if found:
                    break
            
            if found:
                break
    
    # Phase 3: Local search - try swapping/modifying individual columns
    if time.time() < deadline:
        for _ in range(50):
            if time.time() > deadline:
                break
            
            # Pick a random column to modify
            r = np.random.randint(0, best_rank)
            
            # Perturb
            U_trial = best_U.copy()
            V_trial = best_V.copy()
            W_trial = best_W.copy()
            
            U_trial[:, r] += np.random.randn(n*m) * 0.1
            V_trial[:, r] += np.random.randn(m*p) * 0.1
            W_trial[:, r] += np.random.randn(p*n) * 0.1
            
            # Re-solve
            W_trial = solve_for_W(U_trial, V_trial)
            V_trial = solve_for_V(U_trial, W_trial)
            U_trial = solve_for_U(V_trial, W_trial)
            
            U_rounded = round_to_half_int(U_trial)
            V_rounded = round_to_half_int(V_trial)
            W_rounded = round_to_half_int(W_trial)
            
            if check_solution(U_rounded, V_rounded, W_rounded):
                best_U, best_V, best_W = U_rounded, V_rounded, W_rounded
    
    U, V, W = best_U, best_V, best_W
    
    # Convert to lists
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    return {"U": U_list, "V": V_list, "W": W_list}
