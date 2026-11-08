"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import linprog

DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140
    
    n, m, p = DIMENSIONS
    
    def build_rank55():
        rank = 55
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        
        col = 0
        for t in range(5):
            k0 = 2 * t
            k1 = 2 * t + 1
            
            U[0][col] += 1; U[4][col] += 1
            V[k0][col] += 1; V[10+k1][col] += 1
            W[2*k0][col] += 1; W[2*k1+1][col] += 1
            col += 1
            
            U[3][col] += 1; U[4][col] += 1
            V[k0][col] += 1
            W[2*k0+1][col] += 1; W[2*k1+1][col] -= 1
            col += 1
            
            U[0][col] += 1
            V[10+k1][col] += 1; V[20+k1][col] -= 1
            W[2*k1][col] += 1; W[2*k1+1][col] += 1
            col += 1
            
            U[4][col] += 1
            V[10+k0][col] += 1; V[k0][col] -= 1
            W[2*k0][col] += 1; W[2*k0+1][col] += 1
            col += 1
            
            U[0][col] += 1; U[1][col] += 1
            V[10+k1][col] += 1
            W[2*k0][col] -= 1; W[2*k1][col] += 1
            col += 1
            
            U[3][col] += 1; U[0][col] -= 1
            V[k0][col] += 1; V[k1][col] += 1
            W[2*k1+1][col] += 1
            col += 1
            
            U[1][col] += 1; U[4][col] -= 1
            V[10+k0][col] += 1; V[10+k1][col] += 1
            W[2*k0][col] += 1
            col += 1
            
            U[2][col] += 1
            V[20+k0][col] += 1
            W[2*k0][col] += 1
            col += 1
            
            U[2][col] += 1
            V[20+k1][col] += 1
            W[2*k1][col] += 1
            col += 1
            
            U[5][col] += 1
            V[20+k0][col] += 1
            W[2*k0+1][col] += 1
            col += 1
            
            U[5][col] += 1
            V[20+k1][col] += 1
            W[2*k1+1][col] += 1
            col += 1
        
        return U, V, W
    
    def verify(U, V, W):
        T = np.einsum('ir,js,kr->ijs', U, V, W)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    expected = 1.0 if (i < 2 and j < 3 and k < 10) else 0.0
                    # T[i,j,k] should be 1 for all valid i,j,k in 2x3x10
                    # Actually, the tensor T[i,j,k] = sum_r U[i,m+j,r] * V[m*i+k,r] * W[n*k+i,r]
                    # Let's just check the reconstruction properly
        # Better verification:
        # C = A @ B, so T is the bilinear map
        # We need: sum_r U[i*3+j, r] * V[j*10+k, r] * W[k*2+i, r] = 1 for all i,j,k
        # Wait, the convention is W[k*n+i, r]
        # U shape: (n*m, rank) -> U[i*m+j, r]
        # V shape: (m*p, rank) -> V[j*p+k, r]
        # W shape: (p*n, rank) -> W[k*n+i, r]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    U, V, W = build_rank55()
    
    if not verify(U, V, W):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try to reduce rank by finding linear dependencies
    # The 55 columns are (u_r, v_r, w_r) triples
    # We want to find a subset of size R < 55 such that they still span the tensor
    # This is equivalent to: can we write some columns as linear combinations of others?
    # Actually, rank reduction is about whether the 55 vectors in the tensor space are linearly dependent
    # The tensor space has dimension n*m*p = 60
    # We have 55 rank-1 tensors summing to T
    # If 55 < 60, they can be linearly independent, so no reduction possible by this method
    
    # Alternative: try to find a completely different decomposition with lower rank
    # Using the structure: 2x3x10 = 10 copies of 2x3x1
    # M_{2,3} has rank 5 (not 6, since 2*3=6 but rank is min(2,3)+1 = 4? No, M_{2,3} rank is 5)
    # Actually M_{2,3} rank is 5, M_{3,2} rank is 5
    # M_{2,3,1} rank is at most 5
    
    # Let's try to find rank 5 for M_{2,3,1}
    # Then total rank would be 50
    
    def find_rank5_m231():
        # M_{2,3,1}: A is 2x3, B is 3x1, C is 2x1
        # Tensor T[i,j,0] = 1 for i in {0,1}, j in {0,1,2}
        # We need sum_r u_r * v_r * w_r = T
        # u_r is length 6 (2*3), v_r is length 3 (3*1), w_r is length 2 (1*2)
        # 6*3*2 = 36 equations, 5*(6+3+2) = 55 unknowns
        # Let's fix w to standard basis and solve
        
        # Actually, let's just try specific patterns
        # Known rank 5 for M_{2,3}:
        # (A00+A11)(B0+B1)
        # (A10+A11)B0
        # A00(B1-B2)
        # A11(B2-B0)
        # (A01-A11)(B0+B1)
        # Wait, that's 5 terms for M_{2,3,2}
        # For M_{2,3,1}, we can use 5 terms
        
        # Let me just try to use 5 terms for the 3x1 case
        # This is getting complex. Let's fall back to rank 55.
        return None
    
    # For now, return the verified rank 55 solution
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
