"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0  # Safety margin before 150s kill
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    
    def schoolbook():
        rank = rank_schoolbook
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
    
    best_result = schoolbook()
    best_rank = rank_schoolbook
    
    def check_decomposition(U, V, W):
        """Check if U, V, W form a valid decomposition using vectorized numpy ops."""
        rank = len(U[0])
        # Verify shapes
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        
        U_arr = np.array(U, dtype=np.float64)
        V_arr = np.array(V, dtype=np.float64)
        W_arr = np.array(W, dtype=np.float64)
        
        # Build tensor T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_arr.reshape(n, m, rank)
        V_reshaped = V_arr.reshape(m, p, rank)
        W_reshaped = W_arr.reshape(p, n, rank)
        
        # For each (i, j, k): sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # We can compute this by iterating over c and accumulating
        # Or use einsum: sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
        # This is a bit complex with einsum, let's do it in a loop over c
        # But that's O(rank * n * m * p) which is fine for rank ~ 20-50
        
        # Actually, let's use a more efficient approach:
        # For each c, compute outer product contributions
        # T += U[:, :, c] * V[:, :, c] * W[:, :, c] broadcast appropriately
        
        # U_reshaped[i,j,c], V_reshaped[j,k,c], W_reshaped[k,i,c]
        # For fixed c: U_reshaped[:,:,c] is (n,m), V_reshaped[:,:,c] is (m,p), W_reshaped[:,:,c] is (p,n)
        # We need: for each i,j,k: U[i,j,c] * V[j,k,c] * W[k,i,c]
        
        # This can be done as: sum_c (U[:,:,c] * V[:,:,c] * W[:,:,c]) with proper broadcasting
        # U[:,:,c] has shape (n,m), V[:,:,c] has shape (m,p), W[:,:,c] has shape (p,n)
        # We need to multiply them element-wise after broadcasting to (n,m,p)
        # U[i,j,c] * V[j,k,c] * W[k,i,c]
        # = U_reshaped[:,:,c][:, :, None] * V_reshaped[:,:,c][None, :, :] * W_reshaped[:,:,c][None, None, :]
        # Wait, W is indexed as W[k,i,c], so W_reshaped[:,:,c] is (p,n) where first index is k, second is i
        # So W[k,i,c] = W_reshaped[:,:,c][k, i]
        # We want to broadcast to (n,m,p):
        # U[i,j,c] -> U_reshaped[:,:,c][i,j] -> broadcast to (n,m,1)
        # V[j,k,c] -> V_reshaped[:,:,c][j,k] -> broadcast to (1,m,p)
        # W[k,i,c] -> W_reshaped[:,:,c][k,i] -> broadcast to (n,1,p)
        
        # So the product is: U_reshaped[:,:,c][:,:,None] * V_reshaped[:,:,c][None,:,:] * W_reshaped[:,:,c].T[None,:,:]
        # Wait, W_reshaped[:,:,c] is (p,n), we need (n,1,p) so we transpose to (n,p) then add axis
        # W_reshaped[:,:,c].T is (n,p), then [None,:,:] gives (1,n,p)
        # But we need (n,1,p) for broadcasting with (n,m,1) and (1,m,p)
        
        # Let me be more careful:
        # U_reshaped[i,j,c] -> shape (n,m) for fixed c
        # V_reshaped[j,k,c] -> shape (m,p) for fixed c
        # W_reshaped[k,i,c] -> shape (p,n) for fixed c, but we want W[k,i] so it's (p,n)
        
        # To get T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]:
        # For fixed c:
        #   U_c = U_reshaped[:,:,c]  # shape (n,m)
        #   V_c = V_reshaped[:,:,c]  # shape (m,p)
        #   W_c = W_reshaped[:,:,c]  # shape (p,n), where W_c[k,i] = W[k,i,c]
        
        #   T_c[i,j,k] = U_c[i,j] * V_c[j,k] * W_c[k,i]
        
        # To compute this:
        #   U_c[:, :, None]  # (n,m,1)
        #   V_c[None, :, :]  # (1,m,p)
        #   W_c.T[None, :, :]  # W_c.T is (n,p), so (1,n,p)
        
        # Product: (n,m,1) * (1,m,p) * (1,n,p) = (n,m,p)
        # This works!
        
        T = np.zeros((n, m, p), dtype=np.float64)
        for c in range(rank):
            U_c = U_reshaped[:, :, c]  # (n,m)
            V_c = V_reshaped[:, :, c]  # (m,p)
            W_c = W_reshaped[:, :, c]  # (p,n)
            T += U_c[:, :, None] * V_c[None, :, :] * W_c.T[None, :, :]
        
        # Check T[i,j,k] == 1 for all i,j,k
        if not np.allclose(T, 1.0, atol=1e-9):
            return False
        return True
    
    # Use Strassen's algorithm for 4x4x4 (rank 21) and extend to 4x4x5
    # Strassen's algorithm for 2x2x2 has rank 7
    # For 4x4x4, we can use two levels of Strassen, but the standard Strassen-Winograd
    # for 4x4x4 uses rank 21 (or similar structured decomposition)
    
    # Let's construct a rank-21 decomposition for 4x4x4 first
    # Using the standard Strassen decomposition extended to 4x4
    
    def strassen_4x4x4():
        """Return U, V, W for 4x4x4 matrix multiplication with rank 21."""
        rank = 21
        U = [[0] * rank for _ in range(16)]
        V = [[0] * rank for _ in range(16)]
        W = [[0] * rank for _ in range(16)]
        
        # Strassen's 7 multiplications for 2x2:
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B12-B22)
        # M4 = A22(B21-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # For 4x4x4, we can split into 2x2 blocks and apply Strassen
        # But this gives rank 7*7 = 49 if done naively
        # The rank-21 result comes from a more clever structure
        
        # Actually, let's use a known rank-21 decomposition for 4x4x4
        # This is based on the work of Strassen and later improvements
        
        # For simplicity, let's use the standard Strassen extension:
        # Split 4x4 into four 2x2 blocks
        # A = [[A11, A12], [A21, A22]] where each is 2x2
        # B = [[B11, B12], [B21, B22]] where each is 2x2
        
        # Apply Strassen to the block level:
        # M1 = (A11+A22)(B11+B22) -> 2x2 multiplication, rank 7
        # M2 = (A21+A22)B11 -> 2x2 multiplication, rank 7
        # M3 = A11(B12-B22) -> 2x2 multiplication, rank 7
        # M4 = A22(B21-B11) -> 2x2 multiplication, rank 7
        # M5 = (A11+A12)B22 -> 2x2 multiplication, rank 7
        # M6 = (A21-A11)(B11+B12) -> 2x2 multiplication, rank 7
        # M7 = (A12-A22)(B21+B22) -> 2x2 multiplication, rank 7
        
        # Total rank: 7*7 = 49, which is worse than schoolbook 64
        # But we can share some computations
        
        # Let's use a different approach: the rank-21 decomposition is known
        # to exist for 4x4x4. Let me construct it explicitly.
        
        # Actually, the known best for 4x4x4 is rank 21, achieved by 
        # a specific tensor decomposition. Let me hardcode it.
        
        # For now, let's try a simpler approach: use Strassen for the 2x2 blocks
        # but with shared structure to get rank 21
        
        # This is complex, so let's try a different strategy:
        # Use the fact that for 4x4x5, we can use a rank-21 decomposition for the 
        # first 4 columns and add 4 more terms for the 5th column, giving rank 25.
        # Or use a rank-24 decomposition for 4x4x4 and add 1 for the 5th column.
        
        # Let's try to construct a rank-24 decomposition for 4x4x4
        # using two levels of Strassen with some sharing.
        
        # For now, let's return a placeholder and try random search
        return None
    
    # Try to find a better decomposition using structured approach
    # For 4x4x5, the theoretical minimum rank is at least 4*4*5 / (4+4+5-2) = 80/11 ≈ 7.27
    # by the Coppersmith-Winograd lower bound, but actual known bounds are higher
    
    # Let's try a randomized search for rank 22-30 with integer entries
    np.random.seed(42)
    
    for target_rank in [22, 24, 26, 28, 30, 32, 36, 40, 44, 48]:
        if time.time() > deadline - 10:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 5:
                break
            
            U = np.random.choice([-1, 0, 1], size=(n * m, target_rank))
            V = np.random.choice([-1, 0, 1], size=(m * p, target_rank))
            W = np.random.choice([-1, 0, 1], size=(p * n, target_rank))
            
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            
            if check_decomposition(U_list, V_list, W_list):
                result = {"U": U_list, "V": V_list, "W": W_list}
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_result = result
                break
    
    return best_result
# EVOLVE-BLOCK-END
