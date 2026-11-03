"""
Structural approach: Exact integer MILP for rank-minimal tensor decomposition.

For 2x4x8 matrix multiplication, the tensor T[i,j,k] = 1 for all (i,j,k).
We seek the smallest r such that there exist integer vectors U[i,j], V[j,k], W[k,i]
with sum_t U[i,j,t] * V[j,k,t] * W[k,i,t] = 1 for all (i,j,k).

We formulate this as a MILP:
- Binary variable b[i,j,k,t]: entry (i,j,k) is "handled" by term t.
- Integer variables u[i,j,t], v[j,k,t], w[k,i,t]: factor values.
- Constraints: for each (i,j,k), sum_t b[i,j,k,t] = 1 (each entry assigned to exactly one term).
- For each (i,j,k) assigned to term t: u[i,j,t]*v[j,k,t]*w[k,i,t] = 1.
  Since we need integer solutions and the product must be 1, the factors must be
  (+1,+1,+1) or (+1,-1,-1) etc. We can restrict to values in {-1, 0, 1} initially,
  but actually the product being 1 with integers means each factor is ±1.

Wait, that's too restrictive. The factors can be any integers, and the sum over t
must equal 1. But each term contributes u*v*w to the (i,j,k) entry. If we allow
multiple terms to contribute to the same (i,j,k), then we don't need b variables.

Actually, the correct formulation is:
  sum_t u[i,j,t] * v[j,k,t] * w[k,i,t] = 1  for all (i,j,k)

This is a system of 64 equations in 15*(8+32+16)=720 unknowns (for r=15).
The unknowns are integers (or half-integers). This is a linear system!
But the products u*v*w make it nonlinear.

Hmm, the tensor decomposition is:
  T = sum_t  u_t ⊗ v_t ⊗ w_t
  T[i,j,k] = sum_t u_t[i,j] * v_t[j,k] * w_t[k,i] = 1

This is a system of 64 cubic equations in 720 integer unknowns. Not linear.

Alternative: For a fixed rank r, we can try to find integer solutions by:
1. Choose random integer vectors for U and V (with small entries).
2. For each (i,j,k), compute partial sum S[i,j,k] = sum_t u[i,j,t]*v[j,k,t].
3. We need sum_t S[i,j,k] * w[k,i,t] = 1.
   This is a linear system in w: for each i, sum_k w[k,i,t] * S[i,j,k] = 1 for all j,k.
   
Actually, fixing U and V, the constraint becomes:
  For each (i,j): sum_k v[j,k,t] * u[i,j,t] * w[k,i,t] = 1
  Let A_i[j,k] = u[i,j] (scalar per j, but u[i,j,t] depends on t)
  
Let me re-approach. Fix U and V. Then for each i, we need:
  sum_t u[i,j,t] * v[j,k,t] * w[k,i,t] = 1  for all j,k.
  
This is linear in w! For fixed i, define M_i[j,k,t] = u[i,j,t]*v[j,k,t].
Then sum_t M_i[j,k,t] * w[k,i,t] = 1 for all j,k.
This is a system of m*p = 32 equations in r unknowns w[k,i,t] for fixed i.

So the algorithm is:
1. For a given rank r, randomly initialize U (8 x r) and V (32 x r) with small integers.
2. For each i in {0,1}, solve the linear system for W[:, i, :] (16 x r).
3. Check if the solution is integer (or half-integer).
4. Verify the full decomposition.

This is much more tractable! Let me implement this.
"""
import numpy as np
import time
import random

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    U_dim = n * m  # 8
    V_dim = m * p  # 32
    W_dim = p * n  # 16
    
    deadline = time.time() + 130.0  # Safety margin before 150s kill
    best_rank = n * m * p  # schoolbook fallback
    best_U = np.zeros((U_dim, best_rank))
    best_V = np.zeros((V_dim, best_rank))
    best_W = np.zeros((W_dim, best_rank))
    
    # Schoolbook decomposition as fallback
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                best_U[i*m+j, col] = 1
                best_V[j*p+k, col] = 1
                best_W[k*n+i, col] = 1
                col += 1
    
    def is_close_to_integer(x, tol=1e-6):
        return np.all(np.abs(x - np.round(x)) < tol)
    
    def is_close_to_half_integer(x, tol=1e-6):
        return np.all(np.abs(2*x - np.round(2*x)) < tol)
    
    def verify_decomp(U, V, W, tol=1e-6):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Try ranks from small to large
    for rank in range(8, n*m*p):
        if time.time() > deadline - 5:
            break
        
        found = False
        for attempt in range(5000):
            if time.time() > deadline - 3:
                break
            
            # Random small integer initialization
            scale = random.choice([1, 2, 3])
            U = np.random.randint(-scale, scale+1, size=(U_dim, rank))
            V = np.random.randint(-scale, scale+1, size=(V_dim, rank))
            
            # For each i, solve for W[:, i, :]
            W = np.zeros((W_dim, rank))
            valid = True
            for i in range(n):
                # Build linear system: for all (j,k), sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1
                # This is linear in W for fixed i.
                # Unknowns: W[k*n+i, t] for k in range(p), t in range(rank)
                # Equations: m*p = 32 equations
                A_mat = np.zeros((m*p, rank*p))
                rhs = np.ones(m*p)
                eq_idx = 0
                for j in range(m):
                    for k in range(p):
                        for t in range(rank):
                            A_mat[eq_idx, t*p + k] = U[i*m+j, t] * V[j*p+k, t]
                        eq_idx += 1
                
                # Solve least squares
                try:
                    W_i, residuals, rank_sol, sv = np.linalg.lstsq(A_mat, rhs, rcond=None)
                except:
                    valid = False
                    break
                
                # Check if solution is close to integer
                if not is_close_to_integer(W_i, tol=1e-4):
                    # Try half-integer
                    if is_close_to_half_integer(W_i, tol=1e-4):
                        W_i = np.round(W_i * 2) / 2
                    else:
                        valid = False
                        break
                
                W[k*n+i, :] = W_i.reshape(rank, p).T  # W[k*n+i, t] for each k,t
            
            if not valid:
                continue
            
            # Verify full decomposition
            if verify_decomp(U, V, W):
                found = True
                best_rank = rank
                best_U = U
                best_V = V
                best_W = W
                break
        
        if found:
            break
    
    # Return best found
    U_out = np.round(best_U * 2) / 2  # Ensure half-integer
    V_out = np.round(best_V * 2) / 2
    W_out = np.round(best_W * 2) / 2
    
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
