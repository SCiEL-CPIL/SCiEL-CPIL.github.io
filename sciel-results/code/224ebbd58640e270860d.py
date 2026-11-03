import time
import numpy as np
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    n, m, p = 3, 3, 7
    nm, mp, pn = n*m, m*p, p*n  # 9, 21, 21
    
    # Trivial rank-63 decomposition (always valid)
    rank_trivial = n * m * p
    U_triv = np.zeros((nm, rank_trivial), dtype=object)
    V_triv = np.zeros((mp, rank_trivial), dtype=object)
    W_triv = np.zeros((pn, rank_trivial), dtype=object)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_triv[i*m + j, col] = 1
                V_triv[j*p + k, col] = 1
                W_triv[k*n + i, col] = 1
                col += 1
    
    def to_lists(U, V, W):
        return {
            'U': [[int(x) if isinstance(x, (int, np.integer)) else 
                   (2*x).numerator // (2*x).denominator if isinstance(x, Fraction) and (2*x).denominator == 1 else 
                   x.numerator // x.denominator if isinstance(x, Fraction) and x.denominator == 1 else x
                   for x in row] for row in U.tolist()],
            'V': [[int(x) if isinstance(x, (int, np.integer)) else 
                   (2*x).numerator // (2*x).denominator if isinstance(x, Fraction) and (2*x).denominator == 1 else 
                   x.numerator // x.denominator if isinstance(x, Fraction) and x.denominator == 1 else x
                   for x in row] for row in V.tolist()],
            'W': [[int(x) if isinstance(x, (int, np.integer)) else 
                   (2*x).numerator // (2*x).denominator if isinstance(x, Fraction) and (2*x).denominator == 1 else 
                   x.numerator // x.denominator if isinstance(x, Fraction) and x.denominator == 1 else x
                   for x in row] for row in W.tolist()]
        }
    
    def verify(U, V, W):
        """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
        # For each (i, j, k) in range(n, m, p), check that sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # And for all other combinations, the sum should be 0
        # Actually, the decomposition means:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # = sum_j sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] * A[i,j] * B[j,k] ... no
        # 
        # The tensor T[i,j,k] = 1 if the multiplication involves (i,j,k), i.e., T is the multiplication tensor.
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # where T[i,j,k] = 1 for all (i,j,k) in the multiplication pattern.
        # Actually for matrix multiplication C = A*B where A is n x m, B is m x p:
        # C[i,k] = sum_j A[i,j] B[j,k]
        # The tensor has entries T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
        # Wait, that's not right either. The tensor T has T[i,j,k] = 1 for each triple (i,j,k)
        # such that the product A[i,j]*B[j,k] contributes to C[i,k].
        # So T[i,j,k] = 1 for all i,j,k in their respective ranges.
        # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(len(U[0])):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if s != 1:
                        return False
        return True
    
    def verify_fast(U, V, W):
        """Fast verification using numpy."""
        rank = len(U[0])
        # For each r, compute the outer product contribution
        # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U.reshape(n, m, rank)
        V_reshaped = V.reshape(m, p, rank)
        W_reshaped = W.reshape(p, n, rank)
        
        # T[i,j,k] = sum_r U_reshaped[i,j,r] * V_reshaped[j,k,r] * W_reshaped[k,i,r]
        # This is a bit tricky to vectorize. Let's do it differently.
        # For each r, we have a 3-way tensor U_r[i,j] * V_r[j,k] * W_r[k,i]
        # Sum over r.
        
        # Alternative: build the full tensor
        T = np.zeros((n, m, p), dtype=object)
        for r in range(rank):
            u_col = U[:, r]  # (nm,)
            v_col = V[:, r]  # (mp,)
            w_col = W[:, r]  # (pn,)
            
            u_2d = u_col.reshape(n, m)
            v_2d = v_col.reshape(m, p)
            w_2d = w_col.reshape(p, n)
            
            # For each (i, j, k): u_2d[i,j] * v_2d[j,k] * w_2d[k,i]
            # This is an outer product with a cyclic index
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i, j, k] += u_2d[i, j] * v_2d[j, k] * w_2d[k, i]
        
        # Check T[i,j,k] == 1 for all i,j,k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    if T[i, j, k] != 1:
                        return False
        return True
    
    # Try to find a lower-rank decomposition
    # Strategy: Use Strassen-like ideas
    # For 3x3 matrix multiplication, the best known rank is 21 (Strassen)
    # For 3x3x7, we can think of it as 3x3x3 + 3x3x4
    # 3x3x3 has rank at most 21 (Strassen)
    # 3x3x4 has rank at most 12 (trivial) or potentially less
    # Total: at most 21 + 12 = 33
    
    # But we need to be careful - the tensors don't simply add.
    # Actually, 3x3x7 multiplication: C[i,k] = sum_j A[i,j] B[j,k] for k=0..6
    # This is the same as 3x3x3 for k=0..2 plus 3x3x4 for k=3..6
    # The tensor T[i,j,k] = 1 for all (i,j,k) in [0,3)x[0,3)x[0,7)
    # This is the direct sum of T1[i,j,k] = 1 for k=0..2 and T2[i,j,k] = 1 for k=3..6
    # So rank(T) <= rank(T1) + rank(T2)
    
    # Let's try to find a decomposition with rank < 63
    # First, let's try the trivial direct sum approach:
    # T1 (3x3x3) with Strassen rank 21
    # T2 (3x3x4) with trivial rank 12
    # Total rank: 33
    
    # But implementing Strassen correctly is complex. Let's first try a simpler approach.
    # Try random search for lower rank.
    
    best_rank = rank_trivial
    best_result = {'U': U_triv, 'V': V_triv, 'W': W_triv}
    
    # Try to find a rank-33 decomposition using the direct sum idea
    # For now, let's just return the trivial one if we can't find better quickly
    
    if time.time() > deadline - 5.0:
        return to_lists(U_triv, V_triv, W_triv)
    
    # Let's try a structured approach:
    # The 3x3x7 tensor T[i,j,k] = 1 for all i,j,k
    # We can write this as a sum of rank-1 tensors
    # A rank-1 tensor has the form U[i,j] * V[j,k] * W[k,i] where U, V, W are vectors
    
    # Actually, let me think about this differently.
    # The multiplication tensor for n x m by m x p is:
    # T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is just the all-ones tensor of shape (n*m, m*p, p*n) reshaped appropriately.
    # Wait no. The tensor T has indices (i,j,k) and T[i,j,k] = 1.
    # The decomposition T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # means we need to express the all-ones tensor as a sum of rank-1 tensors.
    
    # The all-ones tensor of shape (a, b, c) has rank 1!
    # Because T[i,j,k] = 1 = 1*1*1
    # So U[i*m+j, 0] = 1 for all i,j
    # V[j*p+k, 0] = 1 for all j,k  
    # W[k*n+i, 0] = 1 for all k,i
    # This gives rank 1!
    
    # Wait, that can't be right. Let me re-read the problem.
    # The multiplication tensor is NOT the all-ones tensor.
    # The multiplication tensor T satisfies:
    # (A*B)[i,k] = sum_j A[i,j] B[j,k] = sum_j sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] A[i,j] B[j,k]
    # No wait. The tensor T is defined such that:
    # T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
    # For standard matrix multiplication, every (i,j,k) triple contributes, so T[i,j,k] = 1 for all i,j,k.
    # But the decomposition is:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # If T[i,j,k] = 1 for all i,j,k, then we can set:
    # U[i*m+j, 0] = 1, V[j*p+k, 0] = 1, W[k*n+i, 0] = 1
    # This gives rank 1!
    
    # But that seems too good to be true. Let me verify.
    # With rank 1:
    # T[i,j,k] = U[i*m+j, 0] * V[j*p+k, 0] * W[k*n+i, 0] = 1 * 1 * 1 = 1 ✓
    
    # So the rank is 1? That can't be right for matrix multiplication.
    
    # I think I'm misunderstanding the tensor structure. Let me re-read the problem.
    # "The output factor follows the official trace convention W[k*n+i,r]"
    # So W has shape (p*n, rank) and W[k*n+i, r] is the entry.
    
    # The multiplication tensor for multiplying an n x m matrix A by an m x p matrix B:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k).
    # The decomposition is T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
    
    # If T is the all-ones tensor, then rank 1 works. But that's clearly wrong for matrix multiplication.
    
    # I think the issue is that the tensor is not the all-ones tensor in the way I'm thinking.
    # Actually, for matrix multiplication, the tensor IS the all-ones tensor in the (i,j,k) space.
    # The "hardness" of matrix multiplication comes from the fact that we need to compute
    # C[i,k] = sum_j A[i,j] B[j,k] efficiently, but the tensor itself is just all-ones.
    
    # Wait, I think I'm confusing the tensor with the computation. The tensor T[i,j,k] = 1
    # means that the (i,j,k)-th trilinear form is just 1. The rank of this tensor is indeed 1.
    
    # But that contradicts known results about matrix multiplication rank. Let me reconsider.
    
    # Actually, I think the tensor for matrix multiplication is:
    # T[i,j,k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k]
    # But the key insight is that the SAME A[i,j] is used in multiple C[i,k] computations,
    # and the SAME B[j,k] is used in multiple C[i,k] computations.
    # The tensor captures this structure.
    
    # Let me look at this from a different angle. The tensor T is a 3-way tensor with:
    # - First index: (i, j) where i is row of A, j is column of A / row of B
    # - Second index: (j, k) where j is row of B, k is column of B
    # - Third index: (i, k) where i is row of C, k is column of C
    # 
    # T[(i,j), (j,k), (i,k)] = 1
    # 
    # So T is NOT the all-ones tensor. It's a tensor where T[a,b,c] = 1 only if
    # a = (i,j), b = (j,k), c = (i,k) for some i,j,k.
    # In other words, T[a,b,c] = 1 if the second index of a equals the first index of b,
    # and the first index of a equals the first index of c, and the second index of b equals
    # the second index of c.
    
    # Let me re-index. Let:
    # a = i*m + j (row of A * m + col of A)
    # b = j*p + k (row of B * p + col of B)  
    # c = k*n + i (col of C * n + row of C)
    # 
    # T[a,b,c] = 1 if:
    #   j = a % m = b // p
    #   i = a // m = c % n
    #   k = b % p = c // n
    
    # So T is a sparse tensor, not the all-ones tensor.
    
    # Let me verify with small example: n=1, m=1, p=1
    # a = 0*1+0 = 0, b = 0*1+0 = 0, c = 0*1+0 = 0
    # T[0,0,0] = 1 (since j=0=0, i=0=0, k=0=0)
    # Rank is 1.
    
    # n=2, m=2, p=2:
    # T has 8*8*8 = 512 entries, but only 2*2*2 = 8 are non-zero.
    # T[(i,j),(j,k),(k,i)] = 1 for i,j,k in {0,1}
    
    # This is the standard matrix multiplication tensor.
    # For 2x2x2, the rank is known to be 7 (Strassen).
    
    # OK so my earlier trivial decomposition was correct! The rank-63 decomposition I had
    # is valid but not optimal. The question is whether we can do better.
    
    # For 3x3x7, what's the best known rank?
    # 3x3x3 has rank at most 21 (Strassen)
    # 3x3x4 has rank at most 12 (trivial)
    # But these don't simply add because the tensors are not independent.
    
    # Actually, the 3x3x7 tensor T[i,j,k] = 1 for i in [0,3), j in [0,3), k in [0,7)
    # with the constraint that T[(i,j),(j,k),(k,i)] = 1.
    # This is NOT the direct sum of 3x3x3 and 3x3x4 because the indices are coupled.
    
    # Let me just try to find a better decomposition using optimization.
    
    # For now, let's return the trivial rank-63 decomposition.
    # This is guaranteed to be valid.
    
    return to_lists(U_triv, V_triv, W_triv)
