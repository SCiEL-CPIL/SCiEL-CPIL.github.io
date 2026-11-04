"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140  # safety margin before 150s kill
    n, m, p = DIMENSIONS  # 2, 4, 7

    # Schoolbook baseline: rank = n*m*p = 56
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook

    def verify(U, V, W):
        """Check if U, V, W form a valid decomposition of matrix multiplication tensor."""
        n, m, p = 2, 4, 7
        r = U.shape[1]
        # Build tensor T[i, j, k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        # T should equal 1 for all (i,j,k)
        # Efficient check: for each c, outer product structure
        # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
        # We can check by building the full tensor
        T = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            Uc = U[:, c]  # shape (n*m,)
            Vc = V[:, c]  # shape (m*p,)
            Wc = W[:, c]  # shape (p*n,)
            # Reshape
            Uc_r = Uc.reshape(n, m)
            Vc_r = Vc.reshape(m, p)
            Wc_r = Wc.reshape(p, n)
            # T[i,j,k] += Uc_r[i,j] * Vc_r[j,k] * Wc_r[k,i]
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += Uc_r[i, j] * Vc_r[j, k] * Wc_r[k, i]
        return np.allclose(T, 1.0, atol=1e-10)

    def try_rank_reduction(U, V, W, max_iters=500):
        """Attempt to reduce rank by finding linear dependencies among columns."""
        n, m, p = 2, 4, 7
        r = U.shape[1]
        
        # Build the tensor
        T = np.zeros((n, m, p), dtype=float)
        for c in range(r):
            Uc = U[:, c].reshape(n, m)
            Vc = V[:, c].reshape(m, p)
            Wc = W[:, c].reshape(p, n)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
        
        # Try to find if any column can be expressed as linear combination of others
        # For each column c, check if removing it still gives valid decomposition
        # This is expensive, so try a greedy approach
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                break
            # Randomly try to remove a column
            c_try = np.random.randint(r)
            if r <= 1:
                break
            
            # Try removing column c_try
            mask = [i for i in range(r) if i != c_try]
            U_new = U[:, mask]
            V_new = V[:, mask]
            W_new = W[:, mask]
            
            # Check if still valid
            T_new = np.zeros((n, m, p), dtype=float)
            for c in range(len(mask)):
                Uc = U_new[:, c].reshape(n, m)
                Vc = V_new[:, c].reshape(m, p)
                Wc = W_new[:, c].reshape(p, n)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            T_new[i, j, k] += Uc[i, j] * Vc[j, k] * Wc[k, i]
            
            if np.allclose(T_new, 1.0, atol=1e-10):
                U, V, W = U_new, V_new, W_new
                r = len(mask)
        
        return U, V, W, r

    # Try Winograd-style approach for 1x4x7 multiplication
    # Decompose into two 1x4x7 problems (one for each row of the 2x4 matrix)
    # For 1xm x mxp multiplication, we can use Winograd algorithm
    
    # For 1x4 x 4x7:
    # The tensor has size 1*4*7 = 28
    # Winograd for 1x2x2 convolution has rank 4 (vs schoolbook 4)
    # For 1x4x4, we can tile 1x2x2 blocks
    
    # Let's try a direct approach: use the known optimal rank for small matrix multiplication
    # 2x4 x 4x7: we know rank <= 56 (schoolbook)
    # Strassen gives rank = 7 * ceil(2/2)^2 * ... no, that's for square matrices
    
    # For 2x4 x 4x7, we can view it as:
    # Split 4x7 into two 4x3 and 4x4 blocks? No, that doesn't help directly.
    
    # Better: use the fact that 2x4 = [1x4; 1x4]
    # Each 1x4 x 4x7 multiplication has tensor rank at most 4*7=28
    # But we can do better with Winograd-style methods
    
    # Winograd for 1x4 x 4x4: 
    # Use 1x2x2 Winograd tiles: 4 = 2*2, so we need 2x2 = 4 tiles
    # Each 1x2x2 Winograd has rank 4, so total rank = 4*4 = 16 for 1x4x4
    # For 1x4x7, we can split 7 into 4+3 or use padding
    
    # Actually, let's try a simpler approach: random initialization + rank minimization
    
    # Strategy: Start with schoolbook, then try to find lower rank decompositions
    # by using random linear combinations and checking validity
    
    # First, let's try to construct a rank-49 decomposition using Strassen-like ideas
    # 2x4 x 4x7: Think of it as multiplying [A1 A2] (2x4) by [B1; B2] (4x7)
    # where A1, A2 are 2x2 and B1, B2 are 2x7
    # Result = A1*B1 + A2*B2
    # Each 2x2 x 2x7 can be done with Strassen-like: rank = 7*7 = 49? No.
    # 2x2 x 2x7: tensor size = 2*2*7 = 28
    # Schoolbook rank = 28, but we can do better?
    # Actually for 2x2 x 2xp, the rank is known to be 2p for p>=1? No.
    # 2x2 x 2x2 has rank 7 (Strassen)
    # 2x2 x 2x7: we can tile Strassen: 7/2 is not integer, so pad to 8
    # 2x2 x 2x8: use 4 Strassen blocks, rank = 4*7 = 28
    # So 2x2 x 2x7 has rank <= 28 (with padding to 8, rank 28)
    
    # For 2x4 x 4x7:
    # Split into four 2x2 x 2x2 blocks? No, the dimensions don't align that way.
    
    # Let me try a different approach: use the known result that
    # rank(2x4 x 4x7) <= rank(2x4 x 4x4) + rank(2x4 x 4x3)
    # rank(2x4 x 4x4) <= 2*rank(1x4 x 4x4) = 2*16 = 32 (using Winograd 1x4x4 rank 16)
    # rank(2x4 x 4x3) <= 2*rank(1x4 x 4x3) = 2*12 = 24 (using Winograd 1x4x3)
    # Total: 32 + 24 = 56. Same as schoolbook. Not helpful.
    
    # Let's try: rank(2x4 x 4x7) using the approach of splitting the 7 dimension
    # 7 = 4 + 3
    # 2x4 x 4x4: can we do better than 32?
    # 2x4 x 4x4 = [1x4; 1x4] x 4x4
    # Each 1x4 x 4x4 has rank 16 (Winograd), so total 32.
    # But can we share computations between the two 1x4 rows?
    # If the two 1x4 vectors are arbitrary, probably not much sharing.
    
    # Let's just try to find any rank reduction from 56 using optimization
    
    # Approach: parameterize U, V, W with random entries and solve for constraints
    # T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1 for all i,j,k
    
    # This is a system of 56 equations in r*56 + r*28 + r*14 = 98r unknowns (with r free)
    # For r=49: 98*49 = 4802 unknowns, 56 equations. Very underdetermined.
    # We can set most variables to 0 and solve for the rest.
    
    # Let's try a structured approach:
    # Use the schoolbook decomposition but merge some columns
    
    # Try rank 49: 7 fewer than schoolbook
    # Idea: for each pair of (i,j) or (j,k) or (k,i), see if we can combine
    
    # Actually, let's try a numerical approach:
    # Initialize with schoolbook, then randomly perturb and try to reduce rank
    # by setting some entries to 0 and re-solving
    
    # Simpler: just return schoolbook if we can't find better
    # But let's try one more thing: use the fact that n=2 is small
    
    # For n=2, the tensor T[i,j,k] for i=0,1
    # T[0,j,k] and T[1,j,k] are two 4x7 matrices
    # Each is the "outer product" structure from the decomposition
    
    # Let's try to construct a rank-49 decomposition explicitly
    # by modifying the schoolbook decomposition
    
    # In schoolbook, column (i,j,k) has U[i,j]=1, V[j,k]=1, W[k,i]=1
    # We want to find 49 columns that span the same tensor
    
    # Try: for each (j,k), we have 2 columns (one for i=0, one for i=1)
    # Can we combine them? If we set U[0,j]=1, U[1,j]=1, V[j,k]=1, W[k,0]=a, W[k,1]=b
    # Then T[0,j,k] += 1*1*a = a, T[1,j,k] += 1*1*b = b
    # We need a=1 and b=1, so we can't combine unless we use more complex structures.
    
    # Let's just try random search for a lower rank
    
    np.random.seed(42)
    
    for target_rank in range(49, 56):  # Try ranks 49-55
        if time.time() > deadline:
            break
        
        # Try to find a decomposition of rank target_rank
        # Use random initialization and least squares
        
        # We have 56 constraints: T[i,j,k] = 1
        # Variables: U (8*r), V (28*r), W (14*r) = 50*r variables
        # For r=49: 2450 variables, 56 constraints
        
        # Set up the system: for each (i,j,k), 
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        
        # This is nonlinear (trilinear). Use alternating minimization.
        
        # Initialize
        U_curr = np.random.randn(n * m, target_rank) * 0.1
        V_curr = np.random.randn(m * p, target_rank) * 0.1
        W_curr = np.random.randn(p * n, target_rank) * 0.1
        
        # Scale to satisfy constraints approximately
        # Each term U*V*W should be about 1/target_rank
        scale = (target_rank / 56) ** (1/3)
        U_curr *= scale
        V_curr *= scale
        W_curr *= scale
        
        # Alternating minimization
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each (i,j), minimize ||sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1||
            # This is linear in U[i,j,:] for fixed V, W
            for i in range(n):
                for j in range(m):
                    # We need: for each k, sum_c U[i,j,c] * (V[j,k,c] * W[k,i,c]) = 1
                    # Let A[k, c] = V[j,k,c] * W[k,i,c]
                    # Then sum_c U[i,j,c] * A[k,c] = 1 for all k
                    # This is a linear system: A @ U[i,j,:] = ones(p)
                    A = np.zeros((p, target_rank))
                    for k in range(p):
                        A[k, :] = V_curr[j * p + k, :] * W_curr[k * n + i, :]
                    # Solve A @ x = ones(p)
                    try:
                        x, residuals, rank_a, sv = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U_curr[i * m + j, :] = x
                    except:
                        pass
            
            # Update V: for each (j,k), minimize ||sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1||
            for j in range(m):
                for k in range(p):
                    # We need: for each i, sum_c (U[i,j,c] * W[k,i,c]) * V[j,k,c] = 1
                    # Let B[i, c] = U[i,j,c] * W[k,i,c]
                    # Then sum_c B[i,c] * V[j,k,c] = 1 for all i
                    B = np.zeros((n, target_rank))
                    for i in range(n):
                        B[i, :] = U_curr[i * m + j, :] * W_curr[k * n + i, :]
                    try:
                        x, residuals, rank_b, sv = np.linalg.lstsq(B, np.ones(n), rcond=None)
                        V_curr[j * p + k, :] = x
                    except:
                        pass
            
            # Update W: for each (k,i), minimize ||sum_c U[i,j,c]*V[j,k,c]*W[k,i,c] - 1||
            for k in range(p):
                for i in range(n):
                    # We need: for each j, sum_c (U[i,j,c] * V[j,k,c]) * W[k,i,c] = 1
                    # Let C[j, c] = U[i,j,c] * V[j,k,c]
                    # Then sum_c C[j,c] * W[k,i,c] = 1 for all j
                    C = np.zeros((m, target_rank))
                    for j in range(m):
                        C[j, :] = U_curr[i * m + j, :] * V_curr[j * p + k, :]
                    try:
                        x, residuals, rank_c, sv = np.linalg.lstsq(C, np.ones(m), rcond=None)
                        W_curr[k * n + i, :] = x
                    except:
                        pass
        
        # Check if we found a valid decomposition
        if verify(U_curr, V_curr, W_curr):
            best_U, best_V, best_W = U_curr, V_curr, W_curr
            best_rank = target_rank
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
