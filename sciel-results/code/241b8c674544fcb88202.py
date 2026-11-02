"""Structural change: search for a rank-27 exact decomposition of 2x4 * 4x4
by exploiting the small first dimension (n=2).

Key structural insight:
  C[0,j] = sum_k A[0,k] * B[k,j]
  C[1,j] = sum_k A[1,k] * B[k,j]

We can group terms by the B-index.  For each column j of C, the two entries
C[0,j] and C[1,j] are linear combinations of the same B[k,j] values.

A rank-r decomposition has:
  U: 8 x r  (indices i*m+k, i in {0,1}, k in {0..3})
  V: 16 x r (indices k*p+j)
  W: 8 x r  (indices j*n+i, j in {0..3}, i in {0,1})

The tensor equation is:
  sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = delta_{i,0} * (if we look at it differently)

Actually the standard decomposition condition is:
  For all (i, k, j):  sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1
  (This is the rank-1 decomposition of the matrix multiplication tensor T[i,k,j] = delta_{i,0}... 
   wait, no. The matrix multiplication tensor for C = A*B where C is n x p, A is n x m, B is m x p:
   T_{i,k,j} = 1 if we index as T[i, k, j] meaning the coefficient of A[i,k]*B[k,j] in C[i,j].
   So T[i,k,j] = 1 for all valid i,k,j (it's just the identity on the product space).
   
   Actually no. The tensor is defined so that:
   C[i,j] = sum_{k} A[i,k] * B[k,j]
   
   The rank-1 decomposition of this tensor means:
   C[i,j] = sum_s U[i,k,s] * V[k,j,s] * W[i,j,s]  -- this doesn't quite work with 3 indices.
   
   Let me re-examine the convention. The task says:
   U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
   Output factor follows W[k*n+i, r] convention.
   
   Wait, W has shape (p*n, rank) = (4*2, rank) = (8, rank).
   The index into W is j*n + i where j is the column of C (0..p-1) and i is the row of C (0..n-1).
   
   The decomposition condition is:
   For the matrix multiplication tensor, we need:
   sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1  for all i in [0,n), k in [0,m), j in [0,p)
   
   This is because each "slice" of the tensor T[i,k,j] = 1 (the coefficient of A[i,k]*B[k,j] in C[i,j] is 1).
   
   So we need to find U (8 x r), V (16 x r), W (8 x r) such that for every (i,k,j):
   sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1

This is a system of 2*4*4 = 32 polynomial equations (each is a sum of r products of 3 variables)
in 8*r + 16*r + 8*r = 32*r unknowns.

For r=27, that's 864 unknowns and 32 equations - very underdetermined.

Strategy: Use a structured ansatz. Since n=2 is small, we can try to exploit the fact that
for fixed (k,j), the two equations (for i=0 and i=1) share the same V[k*p+j, s] factors.

Let's use a different structural approach: fix V to have a specific sparse structure and solve
for U and W, or use numerical optimization.

Given the time budget, I'll implement a numerical search:
1. Start from the rank-32 solution
2. Try to merge pairs of rank-1 terms (reducing rank by 1 each time)
3. Use least-squares / Newton iteration to find the merged coefficients

Actually, a cleaner structural approach: 
- The matrix multiplication tensor M_{n,m,p} has known lower bounds.
- For 2x4*4x4, the rank is at most 32 (trivial) and at least... 
- We can try to find rank 28, 27, etc.

Let me implement a direct numerical search using the fact that the system is highly underdetermined.
I'll parameterize with a structured ansatz that allows for cancellations.
"""
import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 4
    deadline = time.time() + 110  # safety margin
    
    # The tensor T[i,k,j] = 1 for all i in [0,n), k in [0,m), j in [0,p)
    # We need: sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1
    
    # Start with rank-32 solution and try to reduce
    # First, let's try a structured ansatz for lower rank
    
    # Approach: Use the fact that for n=2, we can write
    # C[0,j] = sum_k A[0,k] B[k,j]
    # C[1,j] = sum_k A[1,k] B[k,j]
    # 
    # If we group by j: for each j, we have a 2x4 "column" of C.
    # The contribution from B[:,j] (a 4x1 vector) multiplied by A (2x4) gives C[:,j].
    #
    # A rank-1 term contributes: (u_i * v_k * w_j) to T[i,k,j]
    # where u_i depends on (i,k), v_k depends on (k,j), w_j depends on (j,i)
    
    # Let me try a direct numerical optimization approach
    # For a given rank r, we have 32*r variables and 32 constraints.
    # We can use the method of: fix some variables, solve linear systems for others.
    
    # Structural ansatz: 
    # For r = 28, we can start from Strassen-like decomposition
    # For r < 28, we need a more clever approach.
    
    # Let me try: for each j (column of C), the 2x4 subproblem of computing
    # C[:,j] = A * B[:,j] can be done with rank 4 (trivially) or possibly less.
    # But the B[:,j] values are shared across different j's in a decomposition.
    
    # Alternative structural approach: 
    # Use the "border rank" or "approximate" decomposition and then try to make it exact.
    
    # Let me implement a simple but effective search:
    # 1. Start with rank-32
    # 2. For each pair of terms, try to merge them into one term
    # 3. Use the underdetermined system to find a solution
    
    # Actually, let me try a more direct approach:
    # For small ranks, use random initialization + Newton's method on the 32 equations
    
    def try_rank(r, max_iter=5000, seed=42):
        """Try to find a rank-r decomposition using numerical optimization."""
        if time.time() > deadline:
            return None
        
        rng = np.random.RandomState(seed)
        
        # Initialize with random values
        U = rng.randn(n * m, r)
        V = rng.randn(m * p, r)
        W = rng.randn(p * n, r)
        
        # The 32 equations: for each (i, k, j):
        # sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = 1
        
        # This is a system of 32 polynomial equations in 32*r variables.
        # We can use gradient descent or Newton's method.
        
        # Let's use a simple approach: fix W to be identity-like and solve for U, V
        # Or use alternating minimization.
        
        # Better: use the fact that for fixed V and W, the equations are linear in U,
        # and for fixed U and W, they're linear in V, etc.
        
        # Alternating least squares approach:
        # For fixed V, W: solve for U (linear system)
        # For fixed U, W: solve for V (linear system)  
        # For fixed U, V: solve for W (linear system)
        
        # But the equations are not linear - they're trilinear.
        # However, for fixed two matrices, they're linear in the third.
        
        # Let me define the residual:
        # R[i,k,j] = sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] - 1
        
        # For fixed V, W: R is linear in U. We can write:
        # For each (i,k,j): sum_s U[i*m+k, s] * (V[k*p+j, s] * W[j*n+i, s]) = 1
        # This is a linear system in U. But it's not a standard linear system because
        # each equation involves a different row of U.
        
        # Actually, for fixed (i,k): sum_j sum_s U[i*m+k, s] * V[k*p+j, s] * W[j*n+i, s] = p
        # No, that's not right either. Each (i,k,j) is a separate equation.
        
        # Let me just use gradient descent on the sum of squares of residuals.
        
        best_residual = np.inf
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        
        lr = 0.01
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Compute residual
            R = np.zeros((n, m, p))
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        R[i,k,j] = np.sum(U[i*m+k, :] * V[k*p+j, :] * W[j*n+i, :]) - 1.0
            
            res_sq = np.sum(R**2)
            if res_sq < best_residual:
                best_residual = res_sq
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            if best_residual < 1e-10:
                break
            
            # Compute gradient (simple finite differences would be too slow)
            # Use analytical gradient:
            # dR[i,k,j]/dU[i*m+k, s] = V[k*p+j, s] * W[j*n+i, s]
            # dR[i,k,j]/dV[k*p+j, s] = U[i*m+k, s] * W[j*n+i, s]
            # dR[i,k,j]/dW[j*n+i, s] = U[i*m+k, s] * V[k*p+j, s]
            
            dU = np.zeros_like(U)
            dV = np.zeros_like(V)
            dW = np.zeros_like(W)
            
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        r = R[i,k,j]
                        for s in range(r):
                            dU[i*m+k, s] += 2 * r * V[k*p+j, s] * W[j*n+i, s]
                            dV[k*p+j, s] += 2 * r * U[i*m+k, s] * W[j*n+i, s]
                            dW[j*n+i, s] += 2 * r * U[i*m+k, s] * V[k*p+j, s]
            
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        
        if best_residual < 1e-8:
            return best_U, best_V, best_W
        return None
    
    # Try ranks from 27 upward
    for r in range(27, 33):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iter=3000, seed=r)
        if result is not None:
            U, V, W = result
            # Convert to lists with rounded values (should be close to integers or half-integers)
            U_list = [[float(round(x, 10)) for x in row] for row in U]
            V_list = [[float(round(x, 10)) for x in row] for row in V]
            W_list = [[float(round(x, 10)) for x in row] for row in W]
            
            # Verify
            valid = True
            for i in range(n):
                for k in range(m):
                    for j in range(p):
                        val = sum(U_list[i*m+k][s] * V_list[k*p+j][s] * W_list[j*n+i][s] 
                                  for s in range(r))
                        if abs(val - 1.0) > 1e-6:
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    break
            if valid:
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: return rank-32 solution
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for i in range(n):
        for k in range(m):
            for j in range(p):
                s = i * (m * p) + k * p + j
                U[i * m + k][s] = 1
                V[k * p + j][s] = 1
                W[j * n + i][s] = 1
    
    return {"U": U, "V": V, "W": W}
