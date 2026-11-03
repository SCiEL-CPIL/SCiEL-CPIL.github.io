import time
import numpy as np

def construct():
    n, m, p = 2, 5, 6
    start_time = time.time()
    deadline = start_time + 130
    
    def find_2x2x2_rank7():
        """Find a rank-7 decomposition of the 2x2x2 all-ones tensor."""
        # T[i,j,k] = 1 for all i,j,k in {0,1}
        # Find u_r, v_r, w_r for r=0..6 such that sum_r u_r[i]*v_r[j]*w_r[k] = 1
        
        # Use gradient descent to find the decomposition
        np.random.seed(42)
        U = np.random.randn(2, 7)
        V = np.random.randn(2, 7)
        W = np.random.randn(2, 7)
        
        lr = 0.01
        for it in range(100000):
            # Compute T[i,j,k] = sum_r U[i,r]*V[j,r]*W[k,r]
            T = np.zeros((2,2,2))
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        T[i,j,k] = np.sum(U[i,:]*V[j,:]*W[k,:])
            
            residual = T - 1.0
            loss = np.sum(residual**2)
            if loss < 1e-12:
                break
            
            # Gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        dU[i,:] += 2*residual[i,j,k]*V[j,:]*W[k,:]
                        dV[j,:] += 2*residual[i,j,k]*U[i,:]*W[k,:]
                        dW[k,:] += 2*residual[i,j,k]*U[i,:]*V[j,:]
            
            U -= lr*dU
            V -= lr*dV
            W -= lr*dW
        
        return U, V, W
    
    def verify(U, V, W, rank):
        """Verify the decomposition."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, r]*V[j*p+k, r]*W[k*n+i, r] for r in range(rank))
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    # Find the 2x2x2 rank-7 decomposition
    U2, V2, W2 = find_2x2x2_rank7()
    
    # Construct the full 2x5x6 decomposition
    # Partition j into {0,1}, {2,3}, {4}
    # Partition k into {0,1}, {2,3}, {4,5}
    
    rank = 6*7 + 1  # 43
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # 6 blocks of 2x2x2
    for a in range(2):  # j-blocks: {0,1}, {2,3}
        for b in range(3):  # k-blocks: {0,1}, {2,3}, {4,5}
            for r in range(7):
                for i in range(2):
                    for jp in range(2):
                        U[i*m + (2*a+jp), col] = U2[i, r]
                for jp in range(2):
                    for kp in range(2):
                        V[(2*a+jp)*p + (2*b+kp), col] = V2[jp, r]
                for kp in range(2):
                    for i in range(2):
                        W[(2*b+kp)*n + i, col] = W2[kp, r]
                col += 1
    
    # 1 term for j=4, all k
    for i in range(2):
        U[i*m + 4, col] = 1.0
    for k in range(p):
        V[4*p + k, col] = 1.0
    for k in range(p):
        for i in range(2):
            W[k*n + i, col] = 1.0
    col += 1
    
    # Verify
    if not verify(U, V, W, rank):
        # Fallback to schoolbook
        rank = n*m*p
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
        return {"U": [[float(x) for x in row] for row in U],
                "V": [[float(x) for x in row] for row in V],
                "W": [[float(x) for x in row] for row in W]}
    
    return {"U": [[float(x) for x in row] for row in U],
            "V": [[float(x) for x in row] for row in V],
            "W": [[float(x) for x in row] for row in W]}
