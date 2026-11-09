import time
import numpy as np

DIMENSIONS = [3, 4, 6]

def construct():
    start_time = time.time()
    deadline = 115.0  # safety margin before 120s limit
    n, m, p = DIMENSIONS  # 3, 4, 6
    
    def make_schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank
    
    def strassen_2x2_block():
        """
        Return U, V, W for 2x2x2 Strassen with shapes (4,7), (4,7), (4,7).
        U indexed as U[i*2+j], V as V[j*2+k], W as W[k*2+i] for i,j,k in {0,1}.
        Must satisfy: for all (i,j,k) in {0,1}^3, sum_r U[i*2+j,r]*V[j*2+k,r]*W[k*2+i,r] = 1
        """
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen 7 products:
        # M1 = (A00+A11)(B00+B11) -> contributes to C00 and C11
        # M2 = (A10+A11)B00 -> contributes to C10 and C11
        # M3 = A00(B01-B11) -> contributes to C01 and C11
        # M4 = A11(B10-B00) -> contributes to C00 and C10
        # M5 = (A00+A01)B11 -> contributes to C00 and C01
        # M6 = (A10-A00)(B00+B01) -> contributes to C11
        # M7 = (A01-A11)(B10+B11) -> contributes to C00
        
        # For each r, the condition is:
        # sum_j U[i*2+j,r] * V[j*2+k,r] * W[k*2+i,r] = 1 for all i,j,k
        
        # M1: (A00+A11)(B00+B11)
        # U[:,0]: U[0,0]=1 (i=0,j=0), U[3,0]=1 (i=1,j=1)
        # V[:,0]: V[0,0]=1 (j=0,k=0), V[3,0]=1 (j=1,k=1)
        # W[:,0]: W[0,0]=1 (k=0,i=0), W[3,0]=1 (k=1,i=1)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2: (A10+A11)B00
        # U[:,1]: U[2,1]=1 (i=1,j=0), U[3,1]=1 (i=1,j=1)
        # V[:,1]: V[0,1]=1 (j=0,k=0)
        # W[:,1]: W[2,1]=1 (k=1,i=0), W[3,1]=-1 (k=1,i=1)
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1; W[3, 1] = -1
        
        # M3: A00(B01-B11)
        # U[:,2]: U[0,2]=1 (i=0,j=0)
        # V[:,2]: V[1,2]=1 (j=0,k=1), V[3,2]=-1 (j=1,k=1)
        # W[:,2]: W[1,2]=1 (k=0,i=1), W[3,2]=1 (k=1,i=1)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1; W[3, 2] = 1
        
        # M4: A11(B10-B00)
        # U[:,3]: U[3,3]=1 (i=1,j=1)
        # V[:,3]: V[2,3]=1 (j=1,k=0), V[0,3]=-1 (j=0,k=0)
        # W[:,3]: W[0,3]=1 (k=0,i=0), W[2,3]=1 (k=1,i=0)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # M5: (A00+A01)B11
        # U[:,4]: U[0,4]=1 (i=0,j=0), U[1,4]=1 (i=0,j=1)
        # V[:,4]: V[3,4]=1 (j=1,k=1)
        # W[:,4]: W[0,4]=-1 (k=0,i=0), W[1,4]=1 (k=0,i=1)
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # M6: (A10-A00)(B00+B01)
        # U[:,5]: U[2,5]=1 (i=1,j=0), U[0,5]=-1 (i=0,j=0)
        # V[:,5]: V[0,5]=1 (j=0,k=0), V[1,5]=1 (j=0,k=1)
        # W[:,5]: W[3,5]=1 (k=1,i=1)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7: (A01-A11)(B10+B11)
        # U[:,6]: U[1,6]=1 (i=0,j=1), U[3,6]=-1 (i=1,j=1)
        # V[:,6]: V[2,6]=1 (j=1,k=0), V[3,6]=1 (j=1,k=1)
        # W[:,6]: W[0,6]=1 (k=0,i=0)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1
        
        return U, V, W
    
    def verify_decomposition(U, V, W, rank, n, m, p):
        """Check: for all (i,j,k), sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] == 1"""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                    if abs(s - 1.0) > 1e-8:
                        return False
        return True
    
    def make_blocked_strassen():
        """
        Block structure for 3x4x6:
        A: 3x4, B: 4x6, C: 3x6
        
        Split A rows into [0:2] (2 rows) and [2:3] (1 row)
        Split A cols into [0:2] (2 cols) and [2:4] (2 cols)
        Split B rows into [0:2] (2 rows) and [2:4] (2 rows)
        Split B cols into [0:2], [2:4], [4:6] (3 blocks of 2 cols)
        
        Contributions:
        For A rows [0:2]:
          For each a_col in {[0:2], [2:4]}:
            For each b_col in {[0:2], [2:4], [4:6]}:
              C[0:2, b_col] += A[0:2, a_col] * B[a_col, b_col]
              This is a (2,2,2) block -> Strassen rank 7
        For A row [2:3]:
          For each a_col in {[0:2], [2:4]}:
            For each b_col in {[0:2], [2:4], [4:6]}:
              C[2:3, b_col] += A[2:3, a_col] * B[a_col, b_col]
              This is a (1,2,2) block -> schoolbook rank 4
        
        Total: 6 * 7 + 6 * 4 = 42 + 24 = 66
        """
        rank = 66
        U = np.zeros((n * m, rank))  # (12, 66)
        V = np.zeros((m * p, rank))  # (24, 66)
        W = np.zeros((p * n, rank))  # (18, 66)
        
        col = 0
        
        # Strassen blocks for A rows [0:2]
        for a_col_start in [0, 2]:  # A cols [0:2] and [2:4]
            for b_col_start in [0, 2, 4]:  # B cols [0:2], [2:4], [4:6]
                # This is a (2,2,2) Strassen block
                sU, sV, sW = strassen_2x2_block()
                
                # Map local indices to global
                # A local (i,j) -> global (i, a_col_start + j) where i in {0,1}
                # B local (j,k) -> global (a_col_start + j, b_col_start + k) where j in {0,1}, k in {0,1}
                # C local (i,k) -> global (i, b_col_start + k) where i in {0,1}, k in {0,1}
                
                for r in range(7):
                    # U: local (i, j) -> global (i * m + (a_col_start + j))
                    for i in range(2):
                        for j in range(2):
                            if sU[i * 2 + j, r] != 0:
                                U[i * m + (a_col_start + j), col] = sU[i * 2 + j, r]
                    
                    # V: local (j, k) -> global ((a_col_start + j) * p + (b_col_start + k))
                    for j in range(2):
                        for k in range(2):
                            if sV[j * 2 + k, r] != 0:
                                V[(a_col_start + j) * p + (b_col_start + k), col] = sV[j * 2 + k, r]
                    
                    # W: local (k, i) -> global ((b_col_start + k) * n + i)
                    for k in range(2):
                        for i in range(2):
                            if sW[k * 2 + i, r] != 0:
                                W[(b_col_start + k) * n + i, col] = sW[k * 2 + i, r]
                    
                    col += 1
        
        # Schoolbook blocks for A row [2:3]
        for a_col_start in [0, 2]:  # A cols [0:2] and [2:4]
            for b_col_start in [0, 2, 4]:  # B cols [0:2], [2:4], [4:6]
                # This is a (1,2,2) schoolbook block
                # A local (i,j) where i=0 only, j in {0,1}
                # B local (j,k) where j in {0,1}, k in {0,1}
                # C local (i,k) where i=0 only, k in {0,1}
                
                for j in range(2):
                    for k in range(2):
                        # U: local (0, j) -> global (2 * m + (a_col_start + j))
                        U[2 * m + (a_col_start + j), col] = 1.0
                        
                        # V: local (j, k) -> global ((a_col_start + j) * p + (b_col_start + k))
                        V[(a_col_start + j) * p + (b_col_start + k), col] = 1.0
                        
                        # W: local (k, 0) -> global ((b_col_start + k) * n + 2)
                        W[(b_col_start + k) * n + 2, col] = 1.0
                        
                        col += 1
        
        assert col == rank, f"Expected {rank} columns, got {col}"
        
        # Verify
        if not verify_decomposition(U, V, W, rank, n, m, p):
            raise ValueError("Blocked Strassen decomposition failed verification")
        
        return U, V, W, rank
    
    # Start with schoolbook
    best_U, best_V, best_W, best_rank = make_schoolbook()
    
    if time.time() - start_time > deadline - 10:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try blocked Strassen
    try:
        bU, bV, bW, bRank = make_blocked_strassen()
        if bRank < best_rank:
            best_U, best_V, best_W, best_rank = bU, bV, bW, bRank
    except Exception as e:
        pass
    
    if time.time() - start_time > deadline - 5:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Try to find even lower rank using optimization
    # We'll try rank values from best_rank - 1 downward
    # For each rank, try random initialization and gradient descent
    
    def try_random_rank(target_rank, max_iterations=500, lr=0.01):
        """Try to find a decomposition of given rank using random init and optimization."""
        if target_rank < 1:
            return None
        
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Compute loss: sum over all (i,j,k) of (sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] - 1)^2
        def compute_loss(U, V, W):
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                        loss += (s - 1.0) ** 2
            return loss
        
        initial_loss = compute_loss(U, V, W)
        
        for it in range(max_iterations):
            if time.time() - start_time > deadline - 5:
                return None
            
            # Compute gradients
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0.0
                        for r in range(target_rank):
                            s += W[k * n + i, r] * U[i * m + j, r] * V[j * p + k, r]
                        err = 2.0 * (s - 1.0)
                        for r in range(target_rank):
                            dU[i * m + j, r] += err * V[j * p + k, r] * W[k * n + i, r]
                            dV[j * p + k, r] += err * U[i * m + j, r] * W[k * n + i, r]
                            dW[k * n + i, r] += err * U[i * m + j, r] * V[j * p + k, r]
            
            # Gradient descent
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
            
            if it % 50 == 0:
                current_loss = compute_loss(U, V, W)
                if current_loss < 1e-8:
                    # Verify
                    if verify_decomposition(U, V, W, target_rank, n, m, p):
                        return U, V, W
        
        final_loss = compute_loss(U, V, W)
        if final_loss < 1e-6:
            if verify_decomposition(U, V, W, target_rank, n, m, p):
                return U, V, W
        
        return None
    
    # Try ranks from best_rank - 1 down to 50
    for target_rank in range(best_rank - 1, 49, -1):
        if time.time() - start_time > deadline - 10:
            break
        
        result = try_random_rank(target_rank, max_iterations=200, lr=0.05)
        if result is not None:
            U, V, W = result
            best_U, best_V, best_W, best_rank = U, V, W, target_rank
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
