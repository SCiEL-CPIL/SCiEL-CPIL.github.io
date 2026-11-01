import numpy as np
import time

def construct():
    n, m, p = 2, 3, 3
    # Schoolbook fallback
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 100 # 100s margin
    
    # Verification function using vectorized dot products
    def verify(U, V, W, tol=1e-6):
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    A = i * m + j
                    B = j * p + k
                    C = k * n + i
                    # sum_r U[A,r] * V[B,r] * W[C,r]
                    # Vectorize: dot(U[A], V[B] * W[C])
                    val = 0.0
                    for r in range(rank):
                        val += U[A][r] * V[B][r] * W[C][r]
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Border Rank / Deformation Strategy
    # We look for a decomposition of a tensor T_epsilon = T + epsilon * S
    # such that T_epsilon has a low rank decomposition, and the limit as epsilon -> 0
    # yields T.
    # In practice, we try to find U, V, W of rank r such that:
    # sum_r U[A,r] V[B,r] W[C,r] = T[A,B,C] + epsilon * S[A,B,C]
    # where S is a "noise" or "deformation" tensor.
    # If we can find such a decomposition with small epsilon, we might be able to
    # "clean" it up or it implies the border rank is <= r.
    # However, the task requires EXACT decomposition of T.
    # So, if we find U,V,W that satisfy the equation with S=0, we are done.
    # The "border rank" idea suggests that if the variety of rank-r tensors is not closed,
    # T might be in the closure.
    # A practical heuristic:
    # Try to solve for U,V,W of rank r such that the residual is minimized.
    # If the residual is very small, snap to integers/half-integers and check.
    # If it passes, return it.
    
    # A stronger approach:
    # Use the fact that the tensor is sparse (only n*m*p non-zero entries, all 1s).
    # Actually, the tensor T is defined by T[i*m+j, j*p+k, k*n+i] = 1.
    # All other entries are 0.
    # The number of non-zero entries is n*m*p = 18.
    # The total number of entries is (n*m)*(m*p)*(p*n) = 6*9*6 = 324.
    
    # Let's try a direct search for small ranks using the sparsity.
    # We can formulate this as a system of polynomial equations.
    # sum_r U[A,r] V[B,r] W[C,r] = T[A,B,C]
    
    # For r < 18, we have 324 equations and 3*r*(n*m + m*p + p*n) variables.
    # For r=17: 3*17*(6+9+6) = 51*17 = 867 variables.
    # This is underdetermined.
    
    # Let's try a randomized search with refinement.
    # Initialize U, V, W randomly.
    # Use ALS to minimize the error.
    # If error is small, snap and verify.
    
    def solve_als(r, max_iter=200, tol=1e-8, seed=42):
        np.random.seed(seed)
        # Initialize with small random values
        U = np.random.randn(n * m, r) * 0.1
        V = np.random.randn(m * p, r) * 0.1
        W = np.random.randn(p * n, r) * 0.1
        
        # Precompute indices
        indices = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    indices.append((i*m+j, j*p+k, k*n+i))
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Update U
            U_new = np.zeros_like(U)
            for A in range(n * m):
                # Equations involving U[A, :]
                # sum_r U[A,r] V[B,r] W[C,r] = T[A,B,C]
                # For fixed A, this is a linear system in U[A, :]
                # Let M be a matrix where M[row, r] = V[B,row, r] * W[C,row, r]
                # We need to identify all (B, C) such that T[A,B,C] is defined.
                # Actually, T is non-zero only for specific triples.
                # But the equation must hold for ALL A, B, C.
                # T[A,B,C] is 0 for most triples.
                # So we have:
                # sum_r U[A,r] V[B,r] W[C,r] = 0  for (A,B,C) not in indices
                # sum_r U[A,r] V[B,r] W[C,r] = 1  for (A,B,C) in indices
                
                # This is a large system. Let's just use all 324 equations for each A.
                # M is (324, r).
                # M[row, r] = V[B(row), r] * W[C(row), r]
                # b[row] = T[A, B(row), C(row)]
                
                M = np.zeros((324, r))
                b = np.zeros(324)
                row = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            A_idx = i*m+j
                            B_idx = j*p+k
                            C_idx = k*n+i
                            # This loop is over all non-zero entries of T.
                            # But we need equations for ALL A, B, C.
                            # This is too slow if done naively.
                            
                # Optimization:
                # For fixed A, the non-zero entries of T are those where A = i*m+j.
                # There are p such entries (varying k).
                # The zero entries are all other (B, C).
                # Total 324 equations.
                
                # Let's precompute the list of (B, C, T_val) for each A.
                # Actually, let's just build the full matrix once and slice it? No, it depends on V, W.
                
                # Let's iterate over all possible B, C.
                # There are (m*p) * (p*n) = 9 * 6 = 54 pairs of (B, C).
                # Wait, A is fixed. B ranges over m*p=9, C ranges over p*n=6.
                # So 54 equations for each A.
                
                M = np.zeros((54, r))
                b = np.zeros(54)
                row = 0
                for B in range(m * p):
                    for C in range(p * n):
                        # Check if (A, B, C) is a non-zero entry
                        # A = i*m+j, B = j*p+k, C = k*n+i
                        # We need to check if there exist i,j,k such that:
                        # i*m+j == A
                        # j*p+k == B
                        # k*n+i == C
                        # And T[A,B,C] = 1.
                        # Otherwise T[A,B,C] = 0.
                        
                        # Find i, j from A
                        i = A // m
                        j = A % m
                        
                        # Check B
                        if B // p != j:
                            t_val = 0.0
                        else:
                            k = B % p
                            # Check C
                            if C // n != k:
                                t_val = 0.0
                            else:
                                i_check = C % n
                                if i_check != i:
                                    t_val = 0.0
                                else:
                                    t_val = 1.0
                        
                        M[row, :] = V[B, :] * W[C, :]
                        b[row] = t_val
                        row += 1
                        
                # Solve M * U_A = b
                # Least squares
                U_A, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                U_new[A, :] = U_A
                
            U = U_new
            
            # Update V
            V_new = np.zeros_like(V)
            for B in range(m * p):
                M = np.zeros((54, r))
                b = np.zeros(54)
                row = 0
                for A in range(n * m):
                    for C in range(p * n):
                        # Find j, k from B
                        j = B // p
                        k = B % p
                        
                        # Check A
                        if A % m != j:
                            t_val = 0.0
                        else:
                            i = A // m
                            # Check C
                            if C // n != k:
                                t_val = 0.0
                            else:
                                i_check = C % n
                                if i_check != i:
                                    t_val = 0.0
                                else:
                                    t_val = 1.0
                            
                        M[row, :] = U[A, :] * W[C, :]
                        b[row] = t_val
                        row += 1
                        
                V_B, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                V_new[B, :] = V_B
                
            V = V_new
            
            # Update W
            W_new = np.zeros_like(W)
            for C in range(p * n):
                M = np.zeros((54, r))
                b = np.zeros(54)
                row = 0
                for A in range(n * m):
                    for B in range(m * p):
                        # Find k, i from C
                        k = C // n
                        i = C % n
                        
                        # Check A
                        if A // m != i:
                            t_val = 0.0
                        else:
                            j = A % m
                            # Check B
                            if B // p != j:
                                t_val = 0.0
                            else:
                                k_check = B % p
                                if k_check != k:
                                    t_val = 0.0
                                else:
                                    t_val = 1.0
                            
                        M[row, :] = U[A, :] * V[B, :]
                        b[row] = t_val
                        row += 1
                        
                W_C, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                W_new[C, :] = W_C
                
            W = W_new
            
            # Check error
            err = 0.0
            for A in range(n * m):
                for B in range(m * p):
                    for C in range(p * n):
                        # Calculate T_val
                        i = A // m
                        j = A % m
                        if B // p != j:
                            t_val = 0.0
                        else:
                            k = B % p
                            if C // n != k:
                                t_val = 0.0
                            else:
                                i_check = C % n
                                if i_check != i:
                                    t_val = 0.0
                                else:
                                    t_val = 1.0
                            
                        val = np.dot(U[A, :], V[B, :] * W[C, :])
                        err += (val - t_val) ** 2
                        
            if err < tol:
                break
                
        # Snap to half-integers
        U_snapped = np.round(U * 2) / 2
        V_snapped = np.round(V * 2) / 2
        W_snapped = np.round(W * 2) / 2
        
        # Verify snapped
        if verify(U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()):
            return U_snapped.tolist(), V_snapped.tolist(), W_snapped.tolist()
        else:
            return None

    # Try ranks from low to high
    for r in range(11, 18):
        if time.time() > deadline:
            break
        res = solve_als(r)
        if res:
            U, V, W = res
            return {"U": U, "V": V, "W": W}
            
    return schoolbook()
