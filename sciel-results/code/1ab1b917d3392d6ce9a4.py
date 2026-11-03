import time
import numpy as np
from itertools import product

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    nm, mp, pn = n * m, m * p, p * n  # 6, 30, 20
    
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1, else 0
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Build the rank-55 solution
    def build_rank55():
        rank = 55
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        c = 0
        # Part A: j=0 (first column of A times first row of B)
        for i in range(n):
            for k in range(p):
                U[i*m+0, c] = 1
                V[0*p+k, c] = 1
                W[k*n+i, c] = 1
                c += 1
        
        # Part B: A1*B1 + A2*B2 using Strassen for 2x2 blocks
        for b in range(5):
            k0 = 2*b
            k1 = 2*b+1
            
            u11, u12 = 0*m+1, 0*m+2
            u21, u22 = 1*m+1, 1*m+2
            
            v11, v12 = 1*p+k0, 1*p+k1
            v21, v22 = 2*p+k0, 2*p+k1
            
            w00, w01 = k0*n+0, k1*n+0
            w10, w11 = k0*n+1, k1*n+1
            
            # M1 = (a11+a22)(b11+b22)
            U[u11,c]+=1; U[u22,c]+=1; V[v11,c]+=1; V[v22,c]+=1; W[w00,c]+=1; W[w11,c]+=1; c+=1
            # M2 = (a21+a22)b11
            U[u21,c]+=1; U[u22,c]+=1; V[v11,c]+=1; W[w10,c]+=1; W[w11,c]+=-1; c+=1
            # M3 = a11(b12-b22)
            U[u11,c]+=1; V[v12,c]+=1; V[v22,c]+=-1; W[w01,c]+=1; W[w11,c]+=1; c+=1
            # M4 = a22(b21-b11)
            U[u22,c]+=1; V[v21,c]+=1; V[v11,c]+=-1; W[w00,c]+=1; W[w10,c]+=1; c+=1
            # M5 = (a11+a12)b22
            U[u11,c]+=1; U[u12,c]+=1; V[v22,c]+=1; W[w00,c]+=-1; W[w01,c]+=1; c+=1
            # M6 = (a21-a11)(b11+b12)
            U[u21,c]+=1; U[u11,c]+=-1; V[v11,c]+=1; V[v12,c]+=1; W[w11,c]+=1; c+=1
            # M7 = (a12-a22)(b21+b22)
            U[u12,c]+=1; U[u22,c]+=-1; V[v21,c]+=1; V[v22,c]+=1; W[w00,c]+=1; c+=1
        
        return U, V, W, rank
    
    # Verify a decomposition
    def verify(U, V, W, tol=1e-6):
        r = U.shape[1]
        # Check all non-zero entries of T
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        # Check some zero entries to ensure no spurious terms
        # Sample a few random zero positions
        np.random.seed(42)
        for _ in range(100):
            a = np.random.randint(0, nm)
            b = np.random.randint(0, mp)
            c = np.random.randint(0, pn)
            if T[a,b,c] == 0:
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val) > tol:
                    return False
        return True
    
    # ALS optimization to find lower-rank decompositions
    def als_optimize(U_init, V_init, W_init, target_rank, max_iters=200, tol=1e-8):
        """Alternating Least Squares to optimize tensor decomposition"""
        U = U_init[:, :target_rank].copy()
        V = V_init[:, :target_rank].copy()
        W = W_init[:, :target_rank].copy()
        
        for iteration in range(max_iters):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each row a, solve least squares
            for a in range(nm):
                # T[a,:,:] = U[a,:] * (V^T * W)
                # We need: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # Fix V and W, solve for U[a,:]
                # This is a linear system: for each (b,c), T[a,b,c] = sum_r U[a,r] * (V[b,r]*W[c,r])
                # Let M[a,r] = sum over... no, it's element-wise
                # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # For fixed a, b, c: this is linear in U[a,:]
                # We can vectorize: for fixed a, T[a,:] (flattened) = U[a,:] @ (V * W)^T flattened?
                # No. Let's think differently.
                # T[a,b,c] = sum_r U[a,r] * (V[b,r] * W[c,r])
                # Let P[b,c] = V[b,:] * W[c,:] (element-wise, shape mp x pn)
                # Then T[a,b,c] = sum_r U[a,r] * P[b,c,r]... no, P is 2D
                # Actually: T[a,b,c] = sum_r U[a,r] * V[b,r] * W[c,r]
                # This is: T[a] (as a 2D array) = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # So we have a linear system in U[a,:]:
                # T[a] = sum_r U[a,r] * outer(V[:,r], W[:,r])
                # Let A_r = outer(V[:,r], W[:,r]) (mp x pn)
                # Then T[a] = sum_r U[a,r] * A_r
                # Flatten: T[a].flatten() = sum_r U[a,r] * A_r.flatten()
                # So we solve: T[a].flatten() = B @ U[a,:]
                # where B is (mp*pn) x r, with B[mp*c + b, r] = V[b,r] * W[c,r]
                
                B = np.zeros((mp * pn, target_rank))
                for r in range(target_rank):
                    B[:, r] = (V[:, r][:, None] * W[:, r][None, :]).flatten()
                
                rhs = T[a, :, :].flatten()
                # Solve least squares: B @ u = rhs
                try:
                    u, residuals, rank_b, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    U[a, :] = u
                except:
                    pass
            
            # Update V similarly
            for b in range(mp):
                B = np.zeros((nm * pn, target_rank))
                for r in range(target_rank):
                    B[:, r] = (U[:, r][:, None] * W[:, r][None, :]).flatten()
                rhs = T[:, b, :].flatten()
                try:
                    v, residuals, rank_b, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    V[b, :] = v
                except:
                    pass
            
            # Update W similarly
            for c in range(pn):
                B = np.zeros((nm * mp, target_rank))
                for r in range(target_rank):
                    B[:, r] = (U[:, r][:, None] * V[:, r][None, :]).flatten()
                rhs = T[:, :, c].flatten()
                try:
                    w, residuals, rank_b, sv = np.linalg.lstsq(B, rhs, rcond=None)
                    W[c, :] = w
                except:
                    pass
            
            # Check convergence
            if iteration % 10 == 0:
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            err += abs(val - 1.0)
                if err < tol:
                    break
        
        return U, V, W
    
    # Get initial rank-55 solution
    U55, V55, W55, rank55 = build_rank55()
    assert verify(U55, V55, W55), "Rank-55 solution failed verification"
    
    best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Try to find lower-rank solutions using ALS
    for target_rank in range(rank55 - 1, 40, -1):
        if time.time() > deadline - 10:
            break
        
        # Multiple random restarts
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Initialize with random perturbation of the rank-55 solution
            if restart == 0:
                U_init = U55[:, :target_rank] + 0.01 * np.random.randn(nm, target_rank)
                V_init = V55[:, :target_rank] + 0.01 * np.random.randn(mp, target_rank)
                W_init = W55[:, :target_rank] + 0.01 * np.random.randn(pn, target_rank)
            else:
                # Use previous result as starting point
                U_init = best_U[:, :target_rank] + 0.01 * np.random.randn(nm, target_rank)
                V_init = best_V[:, :target_rank] + 0.01 * np.random.randn(mp, target_rank)
                W_init = best_W[:, :target_rank] + 0.01 * np.random.randn(pn, target_rank)
            
            U_opt, V_opt, W_opt = als_optimize(U_init, V_init, W_init, target_rank, max_iters=100)
            
            # Check if the solution is valid
            if verify(U_opt, V_opt, W_opt, tol=1e-4):
                best_U, best_V, best_W, best_rank = U_opt, V_opt, W_opt, target_rank
                break  # Found a valid solution at this rank, try lower
        
        if best_rank < rank55:
            # Found improvement, continue searching for even lower
            continue
    
    # Final verification
    if not verify(best_U, best_V, best_W, tol=1e-4):
        # Fallback to rank-55
        best_U, best_V, best_W, best_rank = U55, V55, W55, rank55
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
