"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import numpy as np
import time

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    # Check: for all i,j,jp,k,kp: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[k*n+i',c] = delta(i,i')*delta(j,jp)*delta(k,kp)
    # More efficiently: for each (i,j), the vector V[:,c] weighted by U[i*m+j,c] should give 
    # a matrix where entry (jp,k) is 1 only when jp==j and k matches... 
    # Actually: sum_c U[i*m+j,c] * V[jp*p+k,c] * W[k'*n+i',c] = delta(i,i')*delta(j,jp)*delta(k,k')
    # Let's check: for fixed (i,j), define a_{k',i'} = sum_c U[i*m+j,c] * W[k'*n+i',c]
    # Then sum_k a_{k,i'} * V[jp*p+k,c]... no this isn't separable that way.
    
    # Direct check: construct the full tensor and compare
    # T_target[i,j,k] for output at (i,k) with input A[i,j], B[j,k]
    # The decomposition: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Wait, the output index in W is (k,i) meaning row i, col k of C.
    # So T[i,j,k] should equal 1 for all valid (i,j,k) since C[i,k] = sum_j A[i,j]*B[j,k]
    # No wait: T[i,j,k] is the coefficient of A[i,j]*B[j,k] in C[i,k], which is 1.
    # But the tensor equation is: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for all i,j,k
    # AND it should be 0 when indices don't match... 
    
    # Let me re-derive. The multiplication tensor T in (A^* x B^* x C) is:
    # T[(i,j), (j',k), (i',k')] = 1 if i=i', j=j', k=k' else 0
    # A rank-r decomposition: T = sum_c u_c ⊗ v_c ⊗ w_c
    # where u_c in R^{n*m}, v_c in R^{m*p}, w_c in R^{n*p}
    # So: sum_c U[i*m+j, c] * V[j'*p+k, c] * W[k'*n+i', c] = delta(i,i')*delta(j,j')*delta(k,k')
    
    # Check all combinations
    for i in range(n):
        for j in range(m):
            for jp in range(m):
                for k in range(p):
                    for kp in range(p):
                        for ip in range(n):
                            lhs = 0.0
                            for c in range(rank):
                                lhs += U[i*m+j][c] * V[jp*p+k][c] * W[kp*n+ip][c]
                            rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                            if abs(lhs - rhs) > 1e-6:
                                return False
    return True

def _try_rank(r, n, m, p, max_iters=2000, seed=42):
    """Try to find a rank-r decomposition using alternating least squares."""
    np.random.seed(seed)
    U = np.random.randn(n*m, r) * 0.5
    V = np.random.randn(m*p, r) * 0.5
    W = np.random.randn(p*n, r) * 0.5
    
    for it in range(max_iters):
        # Fix U, V: solve for W
        # For each (kp, ip): W[kp*n+ip, c] = sum over (i,j,jp,k) of ... 
        # The constraint: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        # For fixed (kp,ip): sum_c (sum_{i,j,jp,k} U[i*m+j,c]*V[jp*p+k,c]*delta(i,ip)*delta(j,jp)*delta(k,kp)) * W[kp*n+ip,c] = 1 if... 
        # This is getting complex. Let me use a different approach.
        
        # Build the linear system for W:
        # For each output index (kp, ip), and for each (i,j,jp,k):
        # sum_c U[i*m+j,c] * V[jp*p+k,c] * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        # 
        # Group by (kp,ip): for fixed (kp,ip), the equation becomes:
        # sum_c [sum_{i,j,jp,k} U[i*m+j,c]*V[jp*p+k,c]*delta(i,ip)*delta(j,jp)*delta(k,kp)] * W[kp*n+ip,c] = 1
        # But delta(i,ip)*delta(j,jp)*delta(k,kp) means i=ip, j=jp, k=kp, so:
        # sum_c U[ip*m+jp,c]*V[jp*p+kp,c]*W[kp*n+ip,c] = 1
        # 
        # Wait, that's only one equation per (kp,ip)! That means:
        # For each (kp, ip): sum_c U[ip*m+jp,c]*V[jp*p+kp,c]*W[kp*n+ip,c] = 1
        # But this must hold for ALL jp? No, jp is free...
        
        # I'm confusing myself. Let me be very careful.
        # The full constraint is:
        # sum_c U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        # for ALL i,j,jp,k,kp,ip in their respective ranges.
        
        # For fixed (i,j,jp,k,kp,ip), this is one scalar equation.
        # Total equations: n*m*m*p*p*n = (n*m)*(m*p)*(p*n) = 8*16*12 = 1536 for n=2,m=4,p=6
        # Unknowns: 3*r (entries of U, V, W)
        
        # This is a system of quadratic equations (bilinear in the variables).
        # Hard to solve directly. Let me use a different strategy.
        
        # Alternative: use the fact that for fixed U, the problem is linear in (V,W) in a tensor sense.
        # Or use random restarts with gradient descent on the loss:
        # L = sum_{i,j,jp,k,kp,ip} (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta(i,ip)*delta(j,jp)*delta(k,kp))^2
        
        # This is getting expensive. Let me just try small ranks with a simpler approach.
        pass
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 120  # 120 second deadline
    
    # Start with schoolbook
    best_rank = n * m * p
    best_result = None
    
    # Schoolbook
    U = [[0] * best_rank for _ in range(n * m)]
    V = [[0] * best_rank for _ in range(m * p)]
    W = [[0] * best_rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    best_result = {"U": U, "V": V, "W": W}
    
    # Try to find lower rank decompositions
    # For 2x4x6, try ranks from 47 down to some minimum
    # Use a tensor network / CP decomposition approach with alternating minimization
    
    def cp_decompose(r, max_iters=500, n_restarts=5):
        """Try CP decomposition of the multiplication tensor at rank r."""
        np.random.seed(0)
        # Target tensor: T[i,j,jp,k,kp,ip] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        # We want: T = sum_c u_c ⊗ v_c ⊗ w_c where
        # u_c indexed by (i,j), v_c indexed by (jp,k), w_c indexed by (kp,ip)
        # So: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
        
        for restart in range(n_restarts):
            if time.time() > deadline - 10:
                break
            np.random.seed(restart * 1000 + r)
            U = np.random.randn(n*m, r)
            V = np.random.randn(m*p, r)
            W = np.random.randn(p*n, r)
            
            for it in range(max_iters):
                if time.time() > deadline - 10:
                    break
                    
                # Fix U, V: solve for W
                # For each (kp, ip): W[kp*n+ip, :] = ?
                # The equations involving W[kp*n+ip, c]:
                # For all i,j,jp,k: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
                # This is a system: for fixed (kp,ip), we have n*m*m*p equations in r unknowns W[kp*n+ip, c]
                # But the RHS depends on whether (i,j,k) = (ip,jp,kp)
                # Actually, for fixed (kp,ip), the equations are:
                # For all i,j,jp,k: sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
                # If (i,j,k) != (ip,jp,kp), RHS = 0
                # If (i,j,k) = (ip,jp,kp), RHS = 1
                # But (i,j,k) ranges over all combinations, and (ip,jp,kp) is fixed...
                # So we have many equations. This is overdetermined in general.
                
                # Better approach: use least squares.
                # For fixed U, V: the problem is to find W minimizing
                # sum_{i,j,jp,k,kp,ip} (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta(i,ip)*delta(j,jp)*delta(k,kp))^2
                # 
                # This is a linear least squares problem in W!
                # Let me set it up properly.
                
                # For each (kp, ip), W[kp*n+ip, c] appears in equations for all (i,j,jp,k).
                # Equation for (i,j,jp,k,kp,ip): 
                #   sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
                # 
                # This couples all (kp,ip) through the same (i,j,jp,k).
                # Actually no, each equation involves exactly one (kp,ip) pair's W row.
                # So the system block-diagonal in (kp,ip)? No, because for fixed (i,j,jp,k),
                # the equation involves W[kp*n+ip,c] for ALL kp,ip.
                
                # Hmm, let me reconsider. The equation is:
                # sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
                # for each tuple (i,j,jp,k,kp,ip).
                # 
                # For fixed (i,j,jp,k), this gives p*n equations (one for each (kp,ip)):
                # sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
                # 
                # Let a_c = U[i*m+j,c]*V[jp*p+k,c] for c=0..r-1
                # Then: sum_c a_c * W[kp*n+ip,c] = delta(i,ip)*delta(j,jp)*delta(k,kp)
                # 
                # For fixed (i,j,jp,k), this is a system of p*n equations in p*n*r unknowns 
                # (all W entries), but it only constrains the dot product of a with each row of W.
                # 
                # This is still complex. Let me just use a numerical optimization approach.
                
                # Compute gradient of loss w.r.t. W, V, U and do gradient descent
                # Loss = sum_{i,j,jp,k,kp,ip} (sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c] - delta(i,ip)*delta(j,jp)*delta(k,kp))^2
                
                # This is O(n*m*m*p*p*n*r) per iteration which is 2*4*4*6*6*2*r = 1152*r
                # For r=40, that's ~46080 operations per iter, very fast.
                
                # Compute loss and gradients
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for jp in range(m):
                            for k in range(p):
                                for kp in range(p):
                                    for ip in range(n):
                                        # Compute sum_c U[i*m+j,c]*V[jp*p+k,c]*W[kp*n+ip,c]
                                        s = 0.0
                                        for c in range(r):
                                            s += U[i*m+j, c] * V[jp*p+k, c] * W[kp*n+ip, c]
                                        rhs = 1.0 if (i==ip and j==jp and k==kp) else 0.0
                                        err = s - rhs
                                        loss += err * err
                                        for c in range(r):
                                            dU[i*m+j, c] += 2*err*V[jp*p+k, c]*W[kp*n+ip, c]
                                            dV[jp*p+k, c] += 2*err*U[i*m+j, c]*W[kp*n+ip, c]
                                            dW[kp*n+ip, c] += 2*err*U[i*m+j, c]*V[jp*p+k, c]
                
                # Gradient descent with adaptive learning rate
                lr = 0.01
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
                
                if loss < 1e-8:
                    break
            
            if loss < 1e-6:
                # Convert to lists with rounding
                U_list = [[round(U[i,c], 6) for c in range(r)] for i in range(n*m)]
                V_list = [[round(V[i,c], 6) for c in range(r)] for i in range(m*p)]
                W_list = [[round(W[i,c], 6) for c in range(r)] for i in range(p*n)]
                # Verify
                if _verify(U_list, V_list, W_list, n, m, p):
                    return U_list, V_list, W_list
        return None
    
    # Try ranks from 47 down
    for r in range(best_rank - 1, 20, -1):
        if time.time() > deadline - 15:
            break
        result = cp_decompose(r, max_iters=300, n_restarts=3)
        if result is not None:
            U_list, V_list, W_list = result
            best_rank = r
            best_result = {"U": U_list, "V": V_list, "W": W_list}
            print(f"Found rank {r} decomposition")
            break
    
    return best_result
# EVOLVE-BLOCK-END
