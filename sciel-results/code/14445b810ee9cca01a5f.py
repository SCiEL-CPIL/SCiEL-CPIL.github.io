import numpy as np
import time

def construct():
    n, m, p = 2, 5, 6
    deadline = time.time() + 110
    
    # Precompute index mappings
    # Constraint indices: (i, j, jp, k, kp, ip) -> equation number
    # For each (i,j,jp,k), we have p*n = 12 equations
    
    # Schoolbook init
    r0 = n * m * p
    U = np.zeros((n*m, r0))
    V = np.zeros((m*p, r0))
    W = np.zeros((p*n, r0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j, col] = 1
                V[j*p+k, col] = 1
                W[k*n+i, col] = 1
                col += 1
    
    best_rank = r0
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    def verify(U_r, V_r, W_r, tol=1e-8):
        """Check if the decomposition is valid."""
        # For each (i,j,jp,k,kp,ip), check sum_r U[i*m+j,r]*V[jp*p+k,r]*W[kp*n+ip,r] = delta
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                val = np.dot(U_r[i*m+j], V_r[jp*p+k] * W_r[kp*n+ip])
                                target = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                if abs(val - target) > tol:
                                    return False
        return True
    
    def solve_for_W(U_r, V_r, r):
        """Given U and V, solve for W."""
        # Build A and b
        # Number of equations: n*m*m*p*p*n = 2*5*5*6*6*2 = 3600
        # Number of unknowns: p*n*r = 12*r
        num_eq = n * m * m * p * p * n
        num_unk = p * n * r
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                for r_idx in range(r):
                                    A[eq, (kp*n+ip)*r + r_idx] = U_r[i*m+j, r_idx] * V_r[jp*p+k, r_idx]
                                if i == ip and j == jp and k == kp:
                                    b[eq] = 1.0
                                eq += 1
        
        W_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return W_flat.reshape(p*n, r)
    
    def solve_for_V(U_r, W_r, r):
        num_eq = n * m * m * p * p * n
        num_unk = m * p * r
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                for r_idx in range(r):
                                    A[eq, (jp*p+k)*r + r_idx] = U_r[i*m+j, r_idx] * W_r[kp*n+ip, r_idx]
                                if i == ip and j == jp and k == kp:
                                    b[eq] = 1.0
                                eq += 1
        
        V_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return V_flat.reshape(m*p, r)
    
    def solve_for_U(V_r, W_r, r):
        num_eq = n * m * m * p * p * n
        num_unk = n * m * r
        A = np.zeros((num_eq, num_unk))
        b = np.zeros(num_eq)
        
        eq = 0
        for i in range(n):
            for j in range(m):
                for jp in range(m):
                    for k in range(p):
                        for kp in range(p):
                            for ip in range(n):
                                for r_idx in range(r):
                                    A[eq, (i*m+j)*r + r_idx] = V_r[jp*p+k, r_idx] * W_r[kp*n+ip, r_idx]
                                if i == ip and j == jp and k == kp:
                                    b[eq] = 1.0
                                eq += 1
        
        U_flat, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        return U_flat.reshape(n*m, r)
    
    # Try decreasing ranks
    for target_r in range(12, best_rank):
        if time.time() > deadline - 20:
            break
        
        success = False
        for trial in range(3):
            if time.time() > deadline - 20:
                break
            
            np.random.seed(42 + target_r * 100 + trial)
            U_r = np.random.randn(n*m, target_r)
            V_r = np.random.randn(m*p, target_r)
            W_r = np.random.randn(p*n, target_r)
            
            for it in range(50):
                if time.time() > deadline - 20:
                    break
                
                W_r = solve_for_W(U_r, V_r, target_r)
                V_r = solve_for_V(U_r, W_r, target_r)
                U_r = solve_for_V  # BUG: should be solve_for_U
                U_r = solve_for_U(V_r, W_r, target_r)
                
                # Check convergence
                W_check = solve_for_W(U_r, V_r, target_r)
                if np.max(np.abs(W_check - W_r)) < 1e-10:
                    break
            
            if verify(U_r, V_r, W_r, tol=1e-6):
                best_rank = target_r
                best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                success = True
                break
        
        if success:
            continue  # Try even lower
        else:
            # Can't achieve this rank, but also can't go lower easily
            # Continue trying lower ranks
            pass
    
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
