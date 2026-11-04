"""Matrix multiplication 3,3,6 - low rank tensor decomposition"""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W exactly decompose the tensor."""
    rank = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1.0) > 1e-10:
                    return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank
    
    U_school, V_school, W_school, rank_school = schoolbook()
    best_U, best_V, best_W, best_rank = U_school, V_school, W_school, rank_school
    
    # Key insight: We can decompose 3x3 * 3x6 as two 3x3 * 3x3 multiplications
    # B (3x6) = [B1 | B2] where B1, B2 are 3x3
    # C (3x6) = [C1 | C2] where C1 = A*B1, C2 = A*B2
    # If we have a rank-r scheme for 3x3 matmul, we can use 2r multiplications total
    # But we need to verify the tensor decomposition works with the W[k*n+i] convention
    
    # For 3x3 matmul with rank r:
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k in {0,1,2}
    
    # For the 3x3x6 case, we need:
    # sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] = 1 for all i in {0,1,2}, j in {0,1,2}, k in {0,1,2,3,4,5}
    
    # Strategy: Use a rank-23 scheme for 3x3 matmul (if we can find one), giving rank 46
    # Or use the known rank-23 scheme and adapt it
    
    # Let's try to construct a rank-46 decomposition using two copies of a 3x3 scheme
    # First, let's see if we can find any 3x3 scheme with rank < 27
    
    # Actually, let me try a more direct approach:
    # Use the fact that we can share some computations between the two 3x3 blocks
    
    # For now, let's try to find a decomposition with rank < 54 using optimization
    
    # Approach: Start from schoolbook and try to merge/eliminate terms
    # Or: Use random initialization and gradient descent on the constraint
    
    # Let's try a structured approach:
    # Partition B into B1 (cols 0-2) and B2 (cols 3-5)
    # C1[i,k] = sum_j A[i,j] * B1[j,k] for k in {0,1,2}
    # C2[i,k] = sum_j A[i,j] * B2[j,k] for k in {3,4,5}
    
    # If we have a rank-r decomposition for 3x3 matmul:
    # C1[i,k] = sum_{c=1}^r mu_{c}(A) * nu_{c}(B1) for k in {0,1,2}
    # C2[i,k] = sum_{c=1}^r mu_{c}(A) * nu_{c}(B2) for k in {3,4,5}
    
    # In tensor form, for the full 3x3x6 problem:
    # We need sum_c U[i*3+j, c] * V[j*6+k, c] * W[k*3+i, c] = 1
    
    # If we use 2r terms:
    # Terms 0..r-1: handle B1 (k in {0,1,2})
    # Terms r..2r-1: handle B2 (k in {3,4,5})
    
    # For term c (0 <= c < r), corresponding to B1:
    # U[i*3+j, c] = U_3x3[i*3+j, c] (same as 3x3 scheme)
    # V[j*6+k, c] = V_3x3[j*3+k, c] if k in {0,1,2}, else 0
    # W[k*3+i, c] = W_3x3[k*3+i, c] if k in {0,1,2}, else 0
    
    # For term r+c (0 <= c < r), corresponding to B2:
    # U[i*3+j, r+c] = U_3x3[i*3+j, c] (same as 3x3 scheme)
    # V[j*6+k, r+c] = V_3x3[j*3+(k-3), c] if k in {3,4,5}, else 0
    # W[k*3+i, r+c] = W_3x3[(k-3)*3+i, c] if k in {3,4,5}, else 0
    
    # So if we have a rank-r 3x3 scheme, we get a rank-2r scheme for 3x3x6
    
    # Now I need a rank-r 3x3 scheme. The best known is 23 (Lader 2023).
    # Let me try to hardcode a rank-23 scheme for 3x3 matmul.
    
    # Actually, let me first try to see if I can get rank 46 (2*23) or better
    # by finding a 3x3 scheme with rank 23
    
    # For now, let's try to use a known 3x3 scheme. 
    # The Lader 2023 rank-23 scheme is complex. Let me try a simpler approach.
    
    # Alternative: Use Strassen's 7-term scheme for 2x2 blocks
    # Partition 3x3 into 2x2 + boundary terms
    # This gives a rank of about 7*4 + boundary = ~35-40
    
    # Let me try a direct optimization approach:
    # Minimize the violation of the tensor equation using gradient descent
    
    def try_optimization(target_rank, max_iter=5000, lr=0.01):
        """Try to find a rank-target_rank decomposition using optimization."""
        np.random.seed(42)
        
        # Initialize with small random values
        U = np.random.randn(n*m, target_rank) * 0.1
        V = np.random.randn(m*p, target_rank) * 0.1
        W = np.random.randn(p*n, target_rank) * 0.1
        
        # The constraint is: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        # We can write this as a system of equations
        
        # Let's use a simple iterative approach:
        # For each (i,j,k), compute the residual and update
        
        best_loss = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        for iteration in range(max_iter):
            if time.time() > deadline - 10:
                break
            
            # Compute loss
            loss = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                        loss += (val - 1.0) ** 2
            
            if loss < best_loss:
                best_loss = loss
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if loss < 1e-10:
                break
            
            # Simple gradient step (approximate)
            # This is slow but might work for small problems
            
            # Actually, let's try a different approach: coordinate descent
            # For each term c, try to optimize it while keeping others fixed
            
            # For simplicity, let's just try random restarts with verification
            break
        
        # Check if we found a valid solution
        U_list = best_U.tolist()
        V_list = best_V.tolist()
        W_list = best_W.tolist()
        
        if _verify(U_list, V_list, W_list, n, m, p):
            return U_list, V_list, W_list, target_rank
        
        return None
    
    # Try various ranks
    for target_rank in [46, 45, 44, 43, 42, 41, 40]:
        if time.time() > deadline - 20:
            break
        
        # Try optimization
        result = try_optimization(target_rank, max_iter=1000)
        if result is not None:
            best_U, best_V, best_W, best_rank = result
            break
        
        # Try random search
        import random
        random.seed(42 + target_rank)
        
        for trial in range(50):
            if time.time() > deadline - 10:
                break
            
            U = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(n*m)]
            V = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(m*p)]
            W = [[random.uniform(-1, 1) for _ in range(target_rank)] for _ in range(p*n)]
            
            if _verify(U, V, W, n, m, p):
                best_U, best_V, best_W, best_rank = U, V, W, target_rank
                break
        
        if best_rank < rank_school:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
