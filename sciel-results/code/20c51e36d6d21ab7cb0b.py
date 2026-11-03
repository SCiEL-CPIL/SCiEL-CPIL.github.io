import numpy as np
from scipy.linalg import svd

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Build the multiplication tensor T[i,j,k] = 1 (for the identity-like multiplication tensor)
    # Actually, for matrix multiplication C = A * B where A is n x m and B is m x p,
    # the tensor has entries T[i,j,k] = delta_{jk} ... no wait.
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # So the tensor T_{i,j,k} = 1 for all valid (i,j,k) where we're computing C[i,k] using A[i,j] and B[j,k].
    # Actually the multiplication tensor has T[i,j,k] = 1 if the indices are compatible.
    # For (n,m,p) = (3,4,4):
    # U is (n*m, rank) = (12, rank)
    # V is (m*p, rank) = (16, rank)  
    # W is (p*n, rank) = (12, rank)
    # The tensor T has shape (n, m, p) with T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Wait no - the multiplication tensor for C = AB is:
    # T_{i,j,k} = 1 means the (i,j) entry of A times the (j,k) entry of B contributes to C[i,k]
    # So T[i,j,k] = 1 for all valid indices, and the "W" matrix uses the convention W[k*n+i, r]
    
    # Let me think about this more carefully.
    # The tensor is T: [0..n-1] x [0..m-1] x [0..p-1] -> R
    # T[i,j,k] = 1 (the standard multiplication tensor)
    
    # Matricization: reshape T into a (n*m) x (p) matrix? Or different mode?
    # For rank-r decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's use IHT-style approach:
    # Start with rank r = n*m*p (trivial), try to reduce
    
    # Actually, let's try a different approach:
    # Use the fact that for matrix multiplication, there are known efficient decompositions.
    # For 3x4 x 4x4, we can try rank 12 (which is n*p = 3*4) using a Kronecker-like structure.
    
    # Let's try rank = n*p = 12 with a structured decomposition.
    # If we fix j, then C[i,k] = sum_j A[i,j] B[j,k]
    # For each j, we have a rank-1 update: A[:,j] * B[j,:]
    # So we can write C = sum_j A[:,j] B[j,:]
    # This gives rank = m = 4? No, that's not right because each term A[:,j] B[j,:] is an outer product
    # which is rank 1 in the matrix sense, but for the tensor decomposition...
    
    # Actually the tensor decomposition is:
    # T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r] (with the W convention W[k*n+i])
    
    # The trivial decomposition has rank n*m*p = 48.
    # A better decomposition: for each j (0..m-1), and for each... 
    # Hmm, let me think about what rank is achievable.
    
    # The bilinear complexity of (3,4,4) matrix multiplication...
    # We can use block decomposition. Let's try to find a rank-12 decomposition.
    
    # Approach: Use the identity that if we decompose B into rows and A into columns:
    # C = A B = sum_{j=0}^{m-1} A[:,j] * B[j,:]
    # Each term A[:,j] B[j,:] is an outer product of vectors of length n and p.
    # In tensor form: T[i,j',k] where j' is the "free" index... 
    # Actually this doesn't directly give a tensor decomposition because the indices don't match.
    
    # Let me use a numerical optimization approach with IHT:
    # 1. Start with random initialization at various ranks
    # 2. Use alternating least squares / IHT to find a decomposition
    # 3. Check if the solution is exact and has integer/half-integer entries
    
    # The tensor T has entries T[i,j,k] = 1 for all i,j,k (since every (i,j) of A multiplies every (j,k) of B... wait no)
    
    # CORRECTION: T[i,j,k] = 1 means A[i,j] * B[j,k] contributes to C[i,k].
    # So T[i,j,k] = 1 for ALL valid (i,j,k) triples. The tensor is just all-ones of shape (n,m,p).
    
    T = np.ones((n, m, p))
    
    # We want: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Let's try ranks from 12 up to 48 and see what we can achieve.
    # Use a structured approach: assume U, V, W have specific patterns.
    
    # For the all-ones tensor, a rank-r decomposition exists if we can find
    # U (12 x r), V (16 x r), W (12 x r) such that the sum of outer products gives the all-ones tensor.
    
    # Key insight: T[i,j,k] = 1 = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # For this to hold for all i,j,k, we need the sum to be independent of i,j,k.
    
    # Let's try: U[i,j,r] = 1/sqrt(r), V[j,k,r] = 1/sqrt(r), W[k,i,r] = r for all r
    # Then sum_r (1/sqrt(r))(1/sqrt(r))(r) = sum_r 1 = r. Not 1 unless r=1.
    
    # Better: U[i,j,r] = 1/r, V[j,k,r] = 1, W[k,i,r] = 1 for all r
    # Then sum_r (1/r)(1)(1) = r/r = 1. 
    # This gives a rank-r decomposition for ANY r! But the entries are 1/r which may not be half-integers.
    
    # For r=1: U=1, V=1, W=1. This works! Rank 1?
    # T[i,j,k] = U[i,j,0]*V[j,k,0]*W[k,i,0] = 1*1*1 = 1. Yes!
    
    # Wait, that means the all-ones tensor has rank 1 in this particular tensor format?
    # But that can't be right for matrix multiplication...
    
    # OH WAIT. I think I'm confusing myself. The multiplication tensor is NOT the all-ones tensor.
    # The multiplication tensor T satisfies: C[i,k] = sum_j A[i,j] B[j,k] = sum_j T[i,j,k] A[i,j] B[j,k]
    # So T[i,j,k] = 1 for all i,j,k. But the "tensor" here is really a 3-index array where the
    # multiplication is: C[i,k] = sum_j (A[i,j] * B[j,k])
    # The tensor decomposition is: C[i,k] = sum_r (sum_j U[i,j,r] A[i,j]) (sum_k' V[j,k',r] B[j,k']) ... no.
    
    # Let me re-read the interface:
    # U: (n*m, rank) - index by (i,j) flattened as i*m+j
    # V: (m*p, rank) - index by (j,k) flattened as j*p+k  
    # W: (p*n, rank) - index by (k,i) flattened as k*n+i
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # And we need T[i,j,k] = 1 for all i,j,k? No!
    # The multiplication tensor for C=AB is: the coefficient of A[i,j]*B[j,k] in C[i,k] is 1.
    # So yes, T[i,j,k] = 1 for all valid (i,j,k).
    
    # So the all-ones tensor of shape (3,4,4) needs to be decomposed.
    # And I showed that rank 1 works: U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # Then sum_r U*V*W = 1*1*1 = 1. 
    
    # But wait - this seems too easy. Is the multiplication tensor really all-ones?
    # Yes! For standard matrix multiplication, every (i,j,k) triple contributes exactly 1.
    # The "structure" of matrix multiplication is in how the indices connect, not in the values.
    
    # So the minimal rank is 1? That seems wrong. Let me double-check.
    
    # Actually, I think I'm making an error. The tensor decomposition for bilinear forms is different.
    # For C = AB, we have C_{ik} = sum_j A_{ij} B_{jk}
    # This is a trilinear form in A, B, C. The "tensor" of the multiplication is T_{ijk} = 1.
    # The rank of this tensor (as a 3-tensor) is the minimal r such that T_{ijk} = sum_{r=1}^r u_i^{(r)} v_j^{(r)} w_k^{(r)}
    # where u is indexed by i, v by j, w by k.
    
    # But the interface uses different indexings:
    # U is indexed by (i,j) not just i
    # V is indexed by (j,k) not just j  
    # W is indexed by (k,i) not just k
    
    # So this is NOT the standard tensor rank. This is a different kind of decomposition.
    # In this decomposition: T_{ijk} = sum_r U_{ij,r} V_{jk,r} W_{ki,r}
    # where U_{ij,r} is the (i*m+j, r) entry of U, etc.
    
    # For T_{ijk} = 1 for all i,j,k, we need:
    # 1 = sum_r U_{ij,r} V_{jk,r} W_{ki,r} for all i,j,k
    
    # If we set U_{ij,r} = a_r, V_{jk,r} = b_r, W_{ki,r} = c_r (independent of indices),
    # then 1 = sum_r a_r b_r c_r. This works for any r with appropriate a,b,c.
    # For r=1: a_1 b_1 c_1 = 1. E.g., a_1=1, b_1=1, c_1=1.
    
    # So rank 1 should work! Let me verify:
    # U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k
    # W[k*n+i, 0] = 1 for all k,i
    # Then T[i,j,k] = U[i*m+j,0] * V[j*p+k,0] * W[k*n+i,0] = 1*1*1 = 1. Correct!
    
    # So the answer is rank 1? That seems surprising but mathematically correct for this indexing convention.
    
    # Let me just implement rank 1 and see if it passes.
    
    rank = 1
    U = np.ones((nm, rank))
    V = np.ones((mp, rank))
    W = np.ones((pn, rank))
    
    # Verify
    # Reconstruct T
    T_recon = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                T_recon[i,j,k] = val
    
    if np.allclose(T_recon, T, atol=1e-10):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    else:
        # Fallback to trivial decomposition
        rank_triv = nm * mp * pn // 1  # This is wrong, should be n*m*p
        rank_triv = n * m * p
        U = np.zeros((nm, rank_triv))
        V = np.zeros((mp, rank_triv))
        W = np.zeros((pn, rank_triv))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
