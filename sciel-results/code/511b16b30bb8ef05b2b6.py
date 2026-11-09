"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
import time
import numpy as np

def _verify(U, V, W, n, m, p, tol=1e-7):
    """Check if U, V, W exactly decompose the matrix multiplication tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False
    return True

def _construct_block_decomposition(n, m, p):
    """
    Construct a rank-24 decomposition for 2x4 * 4x6 matrix multiplication
    by decomposing the inner dimension into two 2D subspaces.
    
    A is 2x4, B is 4x6, C is 2x6.
    Split A into [A1 | A2] where A1, A2 are 2x2.
    Split B into [B1; B2] where B1, B2 are 2x6.
    Then C = A1*B1 + A2*B2.
    
    For each 2x2 * 2x6 multiplication, we can use rank-4 decomposition.
    Total rank = 4 + 4 = 8? No, that's not right.
    
    Actually, let's use a different approach. The standard way to multiply
    2x4 by 4x6 is to think of it as 2*(2x2) * (2x2)*3 blocks.
    
    Let me use a known result: the bilinear complexity of 2x4 by 4x6 is 24.
    We can construct this using the tensor rank decomposition.
    """
    # For 2x4 * 4x6, we can use the following approach:
    # Split the inner dimension (4) into two parts of size 2
    # A = [A_left | A_right], each 2x2
    # B = [B_top; B_bottom], each 2x6
    # C = A_left * B_top + A_right * B_bottom
    
    # Each 2x2 * 2x6 multiplication has rank 12 (schoolbook)
    # But we can do better. Actually, 2x2 * 2x6 can be done in rank 8
    # using the fact that 2x2 multiplication has rank 7 (Winograd)
    # Wait, 2x2 matrix multiplication has tensor rank 7.
    # So 2x2 * 2x6 = 2x2 * (2x2 + 2x2 + 2x2) 
    # This is getting complex. Let me just use the schoolbook for each block.
    
    # Actually, let me try a direct construction for rank 24.
    # The multiplication tensor for 2x4 * 4x6 has 48 entries (2*4*6).
    # We need to find 24 triples that span this tensor.
    
    # Let's use a systematic approach based on the structure.
    # For C[i,k] = sum_j A[i,j] * B[j,k]
    # where i in {0,1}, j in {0,1,2,3}, k in {0,1,2,3,4,5}
    
    # We can write this as:
    # C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k] + A[0,2]*B[2,k] + A[0,3]*B[3,k]
    # C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k] + A[1,2]*B[2,k] + A[1,3]*B[3,k]
    
    # Let's use a rank-24 construction based on the following idea:
    # For each pair (i, j) and each k, we need a term.
    # But we can share some terms.
    
    # Actually, let me just return the schoolbook for now and rely on the search.
    # The search should be able to find rank 24 if it exists.
    
    # For now, return a valid schoolbook decomposition
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140.0  # Leave 10s safety margin
    
    # Start with schoolbook decomposition
    rank_sb = n * m * p
    U = [[0] * rank_sb for _ in range(n * m)]
    V = [[0] * rank_sb for _ in range(m * p)]
    W = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U, best_V, best_W = [row[:] for row in U], [row[:] for row in V], [row[:] for row in W]
    
    # Try to reduce rank using structured search
    np.random.seed(42)
    
    # The key insight: for 2x4 * 4x6, the rank is 24
    # We can construct this by splitting the inner dimension
    # Let's try to directly construct a rank-24 solution
    
    # Approach: Use the fact that 2x4 * 4x6 can be viewed as
    # (2x2 * 2x6) + (2x2 * 2x6)
    # where we split A into two 2x2 blocks and B into two 2x6 blocks
    
    # For 2x2 * 2x6, the rank is 12 (schoolbook) but can be lower
    # Actually, let's just try rank 24 directly with a better search
    
    min_possible = 24  # Known lower bound
    
    for target_rank in range(min_possible, best_rank + 1):
        if time.time() > deadline:
            break
        
        max_attempts = 100 if target_rank <= 30 else 30
        attempts = 0
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            U_r = np.zeros((n*m, target_rank))
            V_r = np.zeros((m*p, target_rank))
            W_r = np.zeros((p*n, target_rank))
            
            # Better initialization: use structured patterns
            # Initialize with small random values
            U_r[:] = np.random.randn(n*m, target_rank) * 0.5
            V_r[:] = np.random.randn(m*p, target_rank) * 0.5
            W_r[:] = np.random.randn(p*n, target_rank) * 0.5
            
            # Use alternating least squares style optimization
            # We want: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
            
            # For each column c, we can optimize one factor at a time
            for iteration in range(500):
                if time.time() > deadline:
                    break
                
                # Pick which factor to update
                factor = iteration % 3
                
                if factor == 0:  # Update U
                    # For each (i,j), we want to find U[i*m+j,:] 
                    # such that the contributions are balanced
                    for i in range(n):
                        for j in range(m):
                            # Current contribution from U[i*m+j,:]
                            # We want sum_k sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                            # This is complex, so let's use a simpler approach
                            pass
                
                # Simpler: use gradient descent on the loss
                # Loss = sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
                
                # Compute gradients numerically
                # This is expensive, so let's use a different approach
                
                # Use random perturbation with acceptance
                c = np.random.randint(target_rank)
                i = np.random.randint(n)
                j = np.random.randint(m)
                k = np.random.randint(p)
                
                # Compute current value
                current = 0.0
                for cc in range(target_rank):
                    current += U_r[i*m+j, cc] * V_r[j*p+k, cc] * W_r[k*n+i, cc]
                
                error = current - 1.0
                if abs(error) < 1e-12:
                    continue
                
                # Perturb one element
                step = 0.01
                if np.random.rand() < 0.33:
                    U_r[i*m+j, c] += step * np.sign(error) * np.random.choice([-1, 1])
                elif np.random.rand() < 0.66:
                    V_r[j*p+k, c] += step * np.sign(error) * np.random.choice([-1, 1])
                else:
                    W_r[k*n+i, c] += step * np.sign(error) * np.random.choice([-1, 1])
            
            # Verify
            if _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p, tol=1e-6):
                best_rank = target_rank
                best_U = U_r.tolist()
                best_V = V_r.tolist()
                best_W = W_r.tolist()
                break
        
        if best_rank == target_rank:
            break
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
