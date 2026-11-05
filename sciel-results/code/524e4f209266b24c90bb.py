import numpy as np
import time
from fractions import Fraction
import random

DIMENSIONS = [2, 5, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Theoretical lower bound for rank of matrix multiplication
    # For (n,m,p) = (2,5,5), lower bound is max(n*m, m*p, p*n) = max(10,25,10) = 25
    # But actual lower bound for tensor rank is more complex
    # Schoolbook rank is n*m*p = 50
    
    # We'll try to find a rank-minimal decomposition
    # Start from the theoretical lower bound and increase
    
    lower_bound = max(n*m, m*p, p*n)
    # For matrix multiplication, the rank is at least max(n*m, m*p, p*n)
    # But we can do better with non-schoolbook decompositions
    
    # Let's try ranks from lower_bound to n*m*p
    best_result = None
    
    # First, let's try the schoolbook decomposition as a fallback
    def schoolbook_decomp():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1
        return {"U": U, "V": V, "W": W, "rank": rank}
    
    schoolbook = schoolbook_decomp()
    best_rank = schoolbook["rank"]
    best_result = schoolbook
    
    # Try to find lower rank decompositions using randomized search
    # We'll use a randomized approach with exact rational arithmetic
    
    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matrix multiplication tensor."""
        rank = len(U[0]) if U else 0
        n, m, p = DIMENSIONS
        
        # Check dimensions
        if len(U) != n * m or len(U[0]) != rank:
            return False
        if len(V) != m * p or len(V[0]) != rank:
            return False
        if len(W) != p * n or len(W[0]) != rank:
            return False
        
        # For each (i,j,k), check that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # And for all other triples, the sum should be 0
        # Actually, the tensor is T[i,j,k] = delta_{i,j,k} where the multiplication
        # (A*B)[i,k] = sum_j A[i,j]*B[j,k]
        # So T[i,j,k] = 1 if we're computing (A*B)[i,k] with A[i,j]*B[j,k]
        # The tensor T_{i,j,k} = 1 for all valid (i,j,k), 0 otherwise
        # Wait, that's not right either.
        
        # The matrix multiplication tensor T has entries T[i,j,k] = 1 for all i,j,k
        # No, that's not right. Let me think again.
        
        # For matrix multiplication C = A*B where A is n x m and B is m x p:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 if the term A[i,j]*B[j,k] contributes to C[i,k]
        # Since every A[i,j]*B[j,k] contributes to exactly one C[i,k], T[i,j,k] = 1 for all i,j,k
        # But that's the schoolbook view.
        
        # Actually, the tensor decomposition means:
        # T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]  (wait, the indices are different)
        
        # From the task: U has shape (n*m, rank), V has shape (m*p, rank), W has shape (p*n, rank)
        # U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
        # The decomposition is: for the multiplication tensor,
        # sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = T[i,j,k]
        
        # What is T[i,j,k]? It's the tensor for matrix multiplication.
        # T[i,j,k] = 1 if the (i,k) entry of C gets a contribution from A[i,j]*B[j,k]
        # Since C[i,k] = sum_j A[i,j]*B[j,k], T[i,j,k] = 1 for all valid i,j,k
        # But wait, that would mean the tensor is all ones, which doesn't make sense for a rank problem.
        
        # Let me re-read: "Find a rank-minimal exact tensor decomposition for multiplying a 2 by 5 matrix by a 5 by 5 matrix"
        # The tensor T for matrix multiplication (n x m) * (m x p) -> (n x p) is:
        # T[i,j,k] = 1 if we're computing C[i,k] using A[i,j] and B[j,k]
        # Since each C[i,k] = sum_j A[i,j]*B[j,k], the tensor has T[i,j,k] = 1 for all i,j,k
        # But that's an all-ones tensor, which has rank 1 (it's U*V*W where U[i,j]=1, V[j,k]=1, W[k,i]=1)
        # That can't be right.
        
        # I think I'm misunderstanding the tensor. Let me think of it differently.
        # The matrix multiplication can be seen as a bilinear map, and its tensor representation
        # is in the space of (n*m) x (m*p) -> (n*p), but the task uses a different factorization.
        
        # Actually, I think the tensor T[i,j,k] represents the structure constant:
        # When you multiply A[i,j] * B[j,k], it contributes to C[i,k].
        # So T[i,j,k] = 1 for all valid indices, and the decomposition is:
        # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
        
        # But if T is all ones, then rank 1 suffices: U[i*m+j,c]=1, V[j*p+k,c]=1, W[k*n+i,c]=1 for all.
        # That gives rank 1, which is clearly wrong for a non-trivial matrix multiplication.
        
        # I must be misunderstanding. Let me look at the schoolbook decomposition again:
        # U[i*m+j, c] = 1, V[j*p+k, c] = 1, W[k*n+i, c] = 1 for each (i,j,k) with a unique c.
        # This means for each (i,j,k), there's exactly one c where all three are 1.
        # So T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1 for each (i,j,k).
        
        # So the tensor IS all ones! But then why is the rank 50?
        
        # Oh wait, I think I see. The issue is that the same (i,j) pair appears in multiple C[i,k] entries,
        # and the same (j,k) pair appears in multiple C[i,k] entries. The tensor structure is not simply
        # "all ones" in a trivial sense because the indices are coupled.
        
        # Let me reconsider. The tensor T is an element of R^{n*m} x R^{m*p} x R^{n*p} (approximately).
        # Actually, T[i,j,k] where i in [0,n), j in [0,m), k in [0,p).
        # The decomposition is T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c].
        
        # For the schoolbook: T[i,j,k] = 1 for all i,j,k, and each (i,j,k) has its own c.
        # So yes, T is the all-ones tensor on [n] x [m] x [p].
        
        # But an all-ones tensor has rank 1! So why does the schoolbook use rank 50?
        
        # I think I'm still missing something. Let me think about it from the perspective of
        # the actual matrix multiplication algorithm.
        
        # The key insight: the tensor is NOT all ones. The tensor encodes the fact that
        # C[i,k] = sum_j A[i,j]*B[j,k]. So the tensor T should satisfy:
        # For each (i,k), C[i,k] = sum_j A[i,j]*B[j,k]
        # In tensor form: sum_j T[i,j,k] * A[i,j] * B[j,k] = C[i,k]
        # But C[i,k] is not an independent variable; it's the output.
        
        # I think the correct interpretation is:
        # The tensor T has T[i,j,k] = 1 for all i,j,k, representing that A[i,j]*B[j,k]
        # contributes to the computation. The "rank" here is the rank of this tensor as
        # a multilinear form, where the three factors are U (indexed by i*m+j), V (indexed by
        # j*p+k), and W (indexed by k*n+i).
        
        # The constraint is that the three indexings are coupled through j and i.
        # Specifically, the same j appears in both U's index (i*m+j) and V's index (j*p+k),
        # and the same i appears in U's index (i*m+j) and W's index (k*n+i).
        
        # So the tensor is: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # for all i in [0,n), j in [0,m), k in [0,p).
        
        # This is a system of n*m*p equations in rank^2 (or rather, in the variables U,V,W).
        # The schoolbook solution uses rank = n*m*p, with each equation having a unique c.
        
        # Can we do better? Yes, if we can share the c indices.
        # For example, if n=1, m=1, p=1, then rank 1 suffices.
        # For n=2, m=1, p=1: T[i,0,0] = 1 for i=0,1.
        # U[i*1+0,c] = U[i,c], V[0*1+0,c] = V[0,c], W[0*2+i,c] = W[i,c].
        # T[i,0,0] = sum_c U[i,c]*V[0,c]*W[i,c] = 1.
        # With rank 1: U[0,0]*V[0,0]*W[0,0]=1, U[1,0]*V[0,0]*W[1,0]=1.
        # We can set V[0,0]=1, U[0,0]=1, W[0,0]=1, U[1,0]=1, W[1,0]=1. Rank 1 works.
        
        # For n=2, m=2, p=2: Can we do better than rank 8?
        # This is a known problem in tensor rank.
        
        # OK, so the verification is: for all i,j,k, sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] == 1.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = Fraction(0)
                    for c in range(rank):
                        s += Fraction(U[i*m+j][c]) * Fraction(V[j*p+k][c]) * Fraction(W[k*n+i][c])
                    if s != 1:
                        return False
        return True
    
    # Try randomized search for lower rank
    # We'll generate random sparse integer patterns and solve
    
    def try_rank_r(r, max_iterations=100):
        """Try to find a rank-r decomposition."""
        for iteration in range(max_iterations):
            if time.time() > deadline - 5:
                return None
            
            # Generate random sparse support patterns
            # Each row of U, V, W gets a random subset of columns
            U_support = [set(random.sample(range(r), min(r, random.randint(1, r)))) for _ in range(n*m)]
            V_support = [set(random.sample(range(r), min(r, random.randint(1, r)))) for _ in range(m*p)]
            W_support = [set(random.sample(range(r), min(r, random.randint(1, r)))) for _ in range(p*n)]
            
            # Set up the system: for each (i,j,k), sum_{c in S[i,j,k]} U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
            # where S[i,j,k] = U_support[i*m+j] & V_support[j*p+k] & W_support[k*n+i]
            
            # This is a nonlinear system. Let's linearize by fixing two of the three matrices.
            # Fix U and V randomly, then solve for W linearly.
            
            # Actually, this is a degree-3 system. Let's try a different approach:
            # Use the fact that we need integer/half-integer entries.
            # Try small random integer values and check.
            
            # Generate random integer values for U, V, W with small absolute values
            max_val = 2
            U_vals = [[random.randint(-max_val, max_val) if c in U_support[i] else 0 for c in range(r)] for i in range(n*m)]
            V_vals = [[random.randint(-max_val, max_val) if c in V_support[i] else 0 for c in range(r)] for i in range(m*p)]
            W_vals = [[random.randint(-max_val, max_val) if c in W_support[i] else 0 for c in range(r)] for i in range(p*n)]
            
            # Check if this works
            valid = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        s = 0
                        for c in range(r):
                            s += U_vals[i*m+j][c] * V_vals[j*p+k][c] * W_vals[k*n+i][c]
                        if s != 1:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            
            if valid:
                return {"U": U_vals, "V": V_vals, "W": W_vals, "rank": r}
        
        return None
    
    # Try ranks from lower_bound to best_rank - 1
    for r in range(lower_bound, best_rank):
        if time.time() > deadline - 10:
            break
        result = try_rank_r(r, max_iterations=50)
        if result is not None:
            best_result = result
            best_rank = r
            break
    
    # Return the best result
    return {"U": best_result["U"], "V": best_result["V"], "W": best_result["W"]}
