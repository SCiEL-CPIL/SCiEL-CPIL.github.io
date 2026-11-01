import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Build target tensor
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                T[i*m+k, k*p+j, j*n+i] = 1.0
    
    # Strassen rank-35 solution
    rank35 = 35
    U35 = np.zeros((n*m, rank35))
    V35 = np.zeros((m*p, rank35))
    W35 = np.zeros((p*n, rank35))
    
    for q in range(5):
        r0 = 7*q
        U35[0, r0] += 1; U35[3, r0] += 1
        V35[2*q, r0] += 1; V35[11+2*q, r0] += 1
        W35[4*q, r0] += 1; W35[4*q+3, r0] += 1
        r = r0+1
        U35[2, r] += 1; U35[3, r] += 1
        V35[2*q, r] += 1
        W35[4*q+1, r] += 1; W35[4*q+3, r] += -1
        r = r0+2
        U35[0, r] += 1
        V35[2*q+1, r] += 1; V35[11+2*q, r] += -1
        W35[4*q+2, r] += 1; W35[4*q+3, r] += 1
        r = r0+3
        U35[3, r] += 1
        V35[10+2*q, r] += 1; V35[2*q, r] += -1
        W35[4*q, r] += 1; W35[4*q+1, r] += 1
        r = r0+4
        U35[0, r] += 1; U35[1, r] += 1
        V35[11+2*q, r] += 1
        W35[4*q, r] += -1; W35[4*q+2, r] += 1
        r = r0+5
        U35[2, r] += 1; U35[0, r] += -1
        V35[2*q, r] += 1; V35[2*q+1, r] += 1
        W35[4*q+3, r] += 1
        r = r0+6
        U35[1, r] += 1; U35[3, r] += -1
        V35[10+2*q, r] += 1; V35[11+2*q, r] += 1
        W35[4*q, r] += 1
    
    def tensor_error(U, V, W):
        err = 0.0
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    val = np.dot(U[a], V[b] * W[c])
                    err += (val - T[a, b, c])**2
        return err
    
    # Verify initial
    assert tensor_error(U35, V35, W35) < 1e-10
    
    best_U, best_V, best_W = U35, V35, W35
    best_rank = rank35
    
    # Try rank 34
    for target_rank in [34]:
        if time.time() > deadline:
            break
        
        for attempt in range(50):
            if time.time() > deadline:
                break
            
            # Initialize
            U = U35[:, :target_rank].copy() + np.random.randn(n*m, target_rank) * 0.5
            V = V35[:, :target_rank].copy() + np.random.randn(m*p, target_rank) * 0.5
            W = W35[:, :target_rank].copy() + np.random.randn(p*n, target_rank) * 0.5
            
            for it in range(1000):
                if time.time() > deadline:
                    break
                
                # Fix U, W; solve for V
                # For each b: sum_r V[b,r] * (U[a,r]*W[c,r]) = T[a,b,c] for all a,c
                # M_stack[r] = outer(U[:,r], W[:,r]).flatten(), shape (4*20, target_rank)
                M_V = np.zeros((n*m * p*n, target_rank))
                for r in range(target_rank):
                    M_V[:, r] = np.outer(U[:, r], W[:, r]).flatten()
                
                for b in range(m*p):
                    rhs = T[:, b, :].flatten()
                    V[b] = np.linalg.lstsq(M_V, rhs, rcond=None)[0]
                
                # Fix V, W; solve for U
                M_U = np.zeros((m*p * p*n, target_rank))
                for r in range(target_rank):
                    M_U[:, r] = np.outer(V[:, r], W[:, r]).flatten()
                
                for a in range(n*m):
                    rhs = T[a, :, :].flatten()
                    U[a] = np.linalg.lstsq(M_U, rhs, rcond=None)[0]
                
                # Fix U, V; solve for W
                M_W = np.zeros((n*m * m*p, target_rank))
                for r in range(target_rank):
                    M_W[:, r] = np.outer(U[:, r], V[:, r]).flatten()
                
                for c in range(p*n):
                    rhs = T[:, :, c].flatten()
                    W[c] = np.linalg.lstsq(M_W, rhs, rcond=None)[0]
                
                if it % 10 == 0:
                    err = tensor_error(U, V, W)
                    if err < 1e-8:
                        best_U, best_V, best_W = U, V, W
                        best_rank = target_rank
                        break
            
            if best_rank == target_rank:
                break
    
    # Convert to lists
    U_list = [[float(x) for x in row] for row in best_U]
    V_list = [[float(x) for x in row] for row in best_V]
    W_list = [[float(x) for x in row] for row in best_W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
