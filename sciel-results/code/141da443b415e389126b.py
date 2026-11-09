"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
import numpy as np
from fractions import Fraction

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    total_eq = n * m * p
    
    # Schoolbook rank
    schoolbook_rank = total_eq
    
    # Try to find a lower rank decomposition using structured approach
    # For matrix multiplication, we can use the fact that:
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # The tensor is T[i,j,k] = delta_{i,j} * delta_{j,k} in some sense
    
    # Let's try ranks from a reasonable lower bound up to schoolbook
    # Lower bound: max(n*m, m*p, p*n) = max(12, 28, 21) = 28
    # But we can do better using tensor rank techniques
    
    best_result = None
    
    # Try schoolbook first as fallback
    rank = schoolbook_rank
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to improve using a structured decomposition
    # For small matrices, we can try rank = max(n*m, m*p, p*n) or lower
    # Use a greedy approach: try to match the tensor with rank-1 terms
    
    # Actually, let's use a proper optimization approach
    # The matrix multiplication tensor can be decomposed as:
    # T = sum_c u_c ⊗ v_c ⊗ w_c
    # where (u_c ⊗ v_c ⊗ w_c)[i,j,k] = u_c[i] * v_c[j] * w_c[k]
    
    # For exact decomposition, we need:
    # sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    # where the "1" corresponds to the multiplication tensor
    
    # Let's try a different construction: use the fact that we can factor through
    # intermediate matrices
    
    # Try rank = 28 (the max of the three dimensions)
    for target_rank in range(28, schoolbook_rank + 1):
        success = False
        for attempt in range(20):
            # Initialize with random values
            U = np.random.randn(nm, target_rank) * 0.1
            V = np.random.randn(mp, target_rank) * 0.1
            W = np.random.randn(pn, target_rank) * 0.1
            
            # Alternating minimization
            for iteration in range(100):
                # Update U: for each (i,j), we have:
                # sum_c U[i*m+j, c] * (V * W)[c] = T[i,j]
                # where (V*W)[c] depends on which (j,k) we're considering
                # This is complex, so let's use a simpler approach
                
                # Actually, let's just check if we can solve this as a linear system
                # for each fixed pair of two factors
                
                # Fix V and W, solve for U
                # For each (i,j): sum_c U[i*m+j, c] * V[j*p+?, c] * W[?, i, c] = ...
                # This is still complex
                
                # Let's use a different strategy: since this is a small problem,
                # we can try to find an exact decomposition using rational arithmetic
                pass
            
            # Check if this is a valid decomposition
            # Compute the tensor from U, V, W and compare to the target
            error = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for c in range(target_rank):
                            val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                        # The target is 1 if this corresponds to a multiplication term
                        # Actually, the target tensor for matrix multiplication is:
                        # T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
                        # So the tensor has T[i,j,k] = 1 for all i,j,k (in the appropriate indexing)
                        # Wait, I need to be more careful about the indexing.
                        
                        # The convention is W[k*n+i, r], so W is indexed by (k,i)
                        # U is indexed by (i,j) -> U[i*m+j, r]
                        # V is indexed by (j,k) -> V[j*p+k, r]
                        
                        # The product should give: for each (i,j,k), the coefficient
                        # in the decomposition of the multiplication tensor
                        # The multiplication tensor has a 1 at position (i,j,k) 
                        # corresponding to A[i,j] * B[j,k] contributing to C[i,k]
                        
                        # So the target is 1 for all (i,j,k)
                        if abs(val - 1) > 1e-6:
                            error += abs(val - 1)
            
            if error < 1e-6:
                # Found a valid decomposition
                # Convert to fractions for exactness
                U_frac = [[Fraction(U[i, c]).limit_denominator(1000) for c in range(target_rank)] for i in range(nm)]
                V_frac = [[Fraction(V[i, c]).limit_denominator(1000) for c in range(target_rank)] for i in range(mp)]
                W_frac = [[Fraction(W[i, c]).limit_denominator(1000) for c in range(target_rank)] for i in range(pn)]
                
                best_result = {
                    "U": [[float(f) for f in row] for row in U_frac],
                    "V": [[float(f) for f in row] for row in V_frac],
                    "W": [[float(f) for f in row] for row in W_frac]
                }
                success = True
                break
        
        if success:
            break
    
    return best_result
# EVOLVE-BLOCK-END
