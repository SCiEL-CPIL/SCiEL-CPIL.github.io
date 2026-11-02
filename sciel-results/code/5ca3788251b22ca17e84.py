<<<<<<< SEARCH
"""Find rank-minimal tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 145  # safety margin before 150s

    # Schoolbook decomposition: rank = n*m*p = 52
    rank_sb = n * m * p
    nA = n * m  # 4
    nB = m * p  # 26
    nC = p * n  # 26

    U = np.zeros((nA, rank_sb))
    V = np.zeros((nB, rank_sb))
    W = np.zeros((nC, rank_sb))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W

    # Try rank-26 decomposition using numerical optimization
    # Structure: terms indexed by (k, s), k in 0..12, s in {0,1}
    # U[2i+j, (k,s)] = u_{i,j,s}  (independent of k)
    # V[2j+k', (k,s)] = delta_{k',k} * v_{j,s}
    # W[13k''+i', (k,s)] = delta_{k'',k} * w_{i',s}
    #
    # Need: sum_s u_{i,j,s} * v_{j,s} * w_{i',s} = delta_{i,i'} * delta_{j,j'}
    #
    # This requires for each j: M_j * W_mat^T = delta_{j,j'} * I_2
    # where M_j[i,s] = u_{i,j,s} * v_{j,s}
    #
    # For j=0: M_0 * W_mat^T = I_2
    # For j=1: M_1 * W_mat^T = 0
    #
    # With 2 terms, W_mat is 2x2. If invertible, M_1=0, M_0=W_mat^{-T}
    # Then for j=j'=1: sum = 0 != delta_{i,i'}. FAILS.
    #
    # The rank-26 structured approach fails. Try general rank-26 search.
    # Use random initialization and gradient-free optimization on a subset of constraints.
    
    if time.time() < deadline - 10:
        # Try random rank-26 decomposition
        rng = np.random.default_rng(42)
        R = 26
        U_rand = rng.standard_normal((nA, R))
        V_rand = rng.standard_normal((nB, R))
        W_rand = rng.standard_normal((nC, R))
        
        # Build target tensor
        T = np.zeros((nA, nB, nC))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[i*m+j, j*p+k, k*n+i] = 1.0
        
        # Check if random decomposition is close (unlikely)
        # Compute reconstructed tensor
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # This is expensive, so skip and just return schoolbook
        
        pass

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
=======
"""Find rank-minimal tensor decomposition for 2x2 * 2x13 matrix multiplication."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 2, 13]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s

    nA = n * m  # 4
    nB = m * p  # 26
    nC = p * n  # 26

    # Build target tensor
    T = np.zeros((nA, nB, nC))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition: rank = n*m*p = 52
    rank_sb = n * m * p
    U = np.zeros((nA, rank_sb))
    V = np.zeros((nB, rank_sb))
    W = np.zeros((nC, rank_sb))

    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1

    best_rank = rank_sb
    best_U, best_V, best_W = U, V, W

    # Try to find a lower rank decomposition
    # For 2x2 * 2x13, we can use the structure:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # 
    # Key insight: We can decompose this using rank-1 updates for each (i,k) pair
    # but exploit that A is 2x2 (only 4 entries) and B is 2x13 (26 entries)
    #
    # Alternative: Use the fact that we can write
    # C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
    #
    # A structured decomposition:
    # For each k (0..12), we need to compute:
    #   C[0,k] = A[0,0]*B[0,k] + A[0,1]*B[1,k]
    #   C[1,k] = A[1,0]*B[0,k] + A[1,1]*B[1,k]
    #
    # This is a 2x2 matrix times a 2-vector (column k of B)
    #
    # Rank of 2x2 * 2x1 is 4 (schoolbook) or can it be less?
    # For 2x2 * 2x1: C[i] = sum_j A[i,j]*B[j]
    # This is bilinear in A and B. The tensor is in R^4 x R^2 x R^2
    # Rank of this tensor: it's the tensor for 2x2 * 2x1 matrix mult
    # which has rank 4 (since 2x2*2x1 requires at least 4 multiplications)
    #
    # But we have 13 independent k's. If we could share multiplications across k,
    # we might reduce the total rank.
    #
    # However, B[0,k] and B[1,k] are independent for each k, so there's no sharing
    # possible in the straightforward approach.
    #
    # Let's try a different approach: use random rank-r matrices and verify
    # by checking if the tensor equation holds.

    def verify(U_mat, V_mat, W_mat):
        """Check if U, V, W exactly decompose the tensor T."""
        # T_recon[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        # Check against T
        for a in range(nA):
            for b in range(nB):
                # T[a,b,:] should equal sum_r U[a,r]*V[b,r]*W[:,r]
                recon = U_mat[a, :] @ (V_mat[b, :][:, None] * W_mat.T)
                # Actually: sum_r U[a,r]*V[b,r]*W[c,r] for each c
                # = (U[a,:][:,None] * V[b,:][:,None] * W.T).sum(axis=1) -- no
                # = np.einsum('r,r,c->c', U_mat[a,:], V_mat[b,:], W_mat)
                recon = np.einsum('r,r,c->c', U_mat[a, :], V_mat[b, :], W_mat)
                if not np.allclose(recon, T[a, b, :], atol=1e-9):
                    return False
        return True

    # Try various rank values from low to high
    # Theoretical lower bound: max(n*m, m*p, p*n) / something
    # For matrix multiplication, rank >= max(n*m, m*p, p*n) is not true
    # but rank >= max(n, m, p) is a weak lower bound
    # Better: rank >= n*m*p / max(n*m, m*p, p*n) ... no that's not right either
    
    # Let's just try to find any decomposition with rank < 52
    # Start with rank 40 and go down
    
    rng = np.random.default_rng(12345)
    
    # Try a structured approach first:
    # Use the fact that the problem separates in k for the B matrix
    # But A is shared across all k
    
    # Actually, let's think about this more carefully.
    # The tensor T has entries T[a,b,c] where:
    # a = i*m + j (i in 0..1, j in 0..1)
    # b = j*p + k (j in 0..1, k in 0..12)
    # c = k*n + i (k in 0..12, i in 0..1)
    #
    # T[a,b,c] = 1 if and only if the indices are consistent:
    # a = i*m+j, b = j'*p+k, c = k'*n+i'
    # T = 1 if j=j' and k=k' and i=i'
    #
    # So T[i*m+j, j*p+k, k*n+i] = 1
    
    # This is exactly the matrix multiplication tensor for 2x2 * 2x13
    
    # For small n,m, we can try to find optimal decompositions
    # Let's try rank 46, 45, 44, ... and see if we can find valid ones
    
    # Use a simple approach: try random initializations and check
    # This is unlikely to work for exact decompositions, but let's try
    
    # Better: use the structure. Since p=13 is prime and odd,
    # and n=m=2, we can try:
    
    # Decomposition idea:
    # Group the k indices. Since p=13, we can't group evenly.
    
    # Let's just return the schoolbook for now, but try a few random
    # lower-rank attempts quickly.
    
    start_time = time.time()
    for R in range(50, 30, -1):  # Try ranks 50 down to 31
        if time.time() - start_time > 30:
            break
        # Try a few random initializations
        for attempt in range(5):
            U_rand = rng.standard_normal((nA, R))
            V_rand = rng.standard_normal((nB, R))
            W_rand = rng.standard_normal((nC, R))
            
            # Quick check: compute a few tensor entries
            # If any mismatch, skip
            valid = True
            for a in range(min(nA, 4)):
                for b in range(min(nB, 26)):
                    recon = np.einsum('r,r,c->c', U_rand[a, :], V_rand[b, :], W_rand)
                    if not np.allclose(recon, T[a, b, :], atol=1e-6):
                        valid = False
                        break
                if not valid:
                    break
            
            if valid:
                # Full verification
                if verify(U_rand, V_rand, W_rand):
                    best_rank = R
                    best_U, best_V, best_W = U_rand, V_rand, W_rand
                    break
        if best_rank < rank_sb:
            break

    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
>>>>>>> REPLACE
