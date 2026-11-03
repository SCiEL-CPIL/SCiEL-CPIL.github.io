"""
Rank-optimal tensor decomposition for 3x3 * 3x7 matrix multiplication.

Strategy:
The schoolbook decomposition has rank 63. We attempt to find a lower rank
decomposition by exploiting the structure.

The tensor T_{i,j,k} for multiplying A(3x3) * B(3x7) -> C(3x7) is defined such that:
C_{i,k} = sum_{j} A_{i,j} B_{j,k}
In the tensor decomposition formulation:
C_{i,k} = sum_{t} (sum_{j} U_{i,j,t} A_{i,j}) * (sum_{j} V_{j,k,t} B_{j,k}) * W_{k,i,t} ... No, that's not right.

The standard tensor decomposition for matrix multiplication T_{n,m,p} is:
T_{i,j,k} = 1 if the term A_{i,j} B_{j,k} contributes to C_{i,k}.
Wait, the standard definition is:
The multiplication tensor T is of size (n*m) x (m*p) x (p*n).
Let indices be I = i*m+j, J = j*p+k, K = k*n+i.
T_{I, J, K} = 1 if I, J, K are compatible, i.e., the j in I matches the j in J, and the k in J matches the k in K?
No, the indices are:
A has indices (i, j).
B has indices (j, k).
C has indices (i, k).
The term is A_{i,j} B_{j,k} -> C_{i,k}.
So the tensor T is defined on indices (i,j), (j,k), (i,k).
T_{(i,j), (j,k), (i,k)} = 1.
All other entries are 0.
This is a sparse tensor.
The rank of this tensor is the minimum number of terms r such that:
T_{I, J, K} = sum_{t=1}^r U_{I, t} V_{J, t} W_{K, t}
where I=(i,j), J=(j,k), K=(i,k).

For 3x3x3, the rank is 23 (Landsberg).
For 3x3x7, the structure is different.
The tensor is a block diagonal? No.
Let's look at the indices.
I ranges 0..8 (i=0..2, j=0..2).
J ranges 0..20 (j=0..2, k=0..6).
K ranges 0..20 (k=0..6, i=0..2).

T_{I, J, K} = 1 if:
I = i*m + j
J = j*p + k
K = k*n + i
This implies:
j(I) = j(J)
k(J) = k(K)
i(I) = i(K)

So, for a fixed j, we have a sub-tensor involving i, k.
For fixed j, I = i*m + j, J = j*p + k, K = k*n + i.
This looks like the 3x1x7 multiplication tensor? No, it's 3x3x7.
Actually, for fixed j, the variables are i (0..2) and k (0..6).
The indices are:
I depends on i.
J depends on k.
K depends on k, i.
This is essentially the tensor for multiplying a 3x1 vector by a 1x7 vector? No.
It is the tensor for multiplying a 3x3 matrix? No.

Let's re-evaluate the rank.
The tensor is the direct sum of 3 tensors, one for each j?
No, the indices are coupled.
However, we can use the known rank of 3x3x3 (which is 23) to bound or construct a solution for 3x3x7?
Actually, 3x3x7 is much larger.
The schoolbook rank is 3*3*7 = 63.
Is there a better rank?
For 3x3x3, rank is 23.
For 3x3x7, we can view it as 7 copies of 3x3x1? No.
It is the tensor product of 3x3x1 and 1x1x7? No.

Let's use the "block" structure.
The tensor T can be viewed as a 3x3 block matrix where each block is a 7x7? No.
Let's just use the schoolbook decomposition as a baseline and try to optimize.
However, finding a lower rank for 3x3x7 is hard.
Given the time limit, and the fact that 3x3x3 rank 23 is a specific known result,
maybe we can use a similar structure?
But 3x3x7 is not 3x3x3.

Let's stick to the schoolbook decomposition but ensure it is correct.
The prompt asks for rank-minimal. If we can't find a better one, schoolbook is the safe bet.
However, the score is seed/value. If seed is 63, and we return 63, score is 1.0.
If we find a lower rank, score > 1.0.

Let's try to find a decomposition of rank < 63.
One idea: use the rank-23 decomposition for 3x3x3 and extend it?
The 3x3x7 tensor is not a simple extension.

Let's try a numerical optimization approach with a few restarts, aiming for rank 50, 40, etc.
But the tensor is 9x21x21.
The number of constraints is 9*21*21 = 3969? No, only the non-zero entries matter.
There are 3*3*7 = 63 non-zero entries.
So we have 63 equations.
We want to find U (9xr), V (21xr), W (21xr) such that:
sum_t U[i*m+j, t] V[j*p+k, t] W[k*n+i, t] = 1
for all valid (i,j,k), and 0 otherwise.

This is a system of 63 cubic equations in 3*r*(9+21+21) = 126r variables.
For r=63, it's easy (schoolbook).
For r=23? 126*23 = 2898 variables. 63 equations. Underdetermined.
Maybe we can find a solution?

Let's implement a solver that tries to find a solution for a given rank r.
We will use scipy.optimize.least_squares or minimize.
Objective: sum of squares of (T_recon - T_target).
T_target is the 9x21x21 tensor with 1s at the valid positions.

We will try ranks: 63, 60, 55, 50, 45, 40, 35, 30, 25, 23.
If we find a valid solution (error < 1e-6), we check if it can be snapped to integers/half-integers.
If not, we might still return the floating point solution if the evaluator accepts it?
The prompt says "integer/half-integer entries". So we MUST snap.
If snapping fails, we discard that solution.

This is risky. The schoolbook solution is guaranteed to work.
Let's implement the optimization carefully with a time limit.
If no better solution is found, return schoolbook.

To speed up, we can precompute the indices.
Indices (i,j,k) where T=1:
i in 0..2, j in 0..2, k in 0..6.
I = i*3 + j
J = j*7 + k
K = k*3 + i

We will flatten U, V, W into a vector x.
U = x[0 : 9*r].reshape(9, r)
V = x[9*r : 9*r + 21*r].reshape(21, r)
W = x[9*r + 21*r : 9*r + 21*r + 21*r].reshape(21, r)

Compute T_recon[I, J, K] = sum_t U[I, t] V[J, t] W[K, t].
We only need to check the 63 valid (I,J,K) positions.
We can store these positions in arrays I_idx, J_idx, K_idx.
Then T_recon_valid = sum_t U[I_idx, t] * V[J_idx, t] * W[K_idx, t] (element-wise product summed over t).
Actually, for each t, we can compute the vector of products for the 63 positions.
prod_t = U[I_idx, t] * V[J_idx, t] * W[K_idx, t]
T_recon_valid = sum_t prod_t

Error = sum (T_recon_valid - 1)^2.

This is efficient.

Let's try this.
"""

import time
import numpy as np
import random

# Precompute indices for the 3x3x7 tensor
n, m, p = 3, 3, 7
I_idx = []
J_idx = []
K_idx = []
for i in range(n):
    for j in range(m):
        for k in range(p):
            I_idx.append(i * m + j)
            J_idx.append(j * p + k)
            K_idx.append(k * n + i)

I_idx = np.array(I_idx)
J_idx = np.array(J_idx)
K_idx = np.array(K_idx)

def compute_error(params, r):
    """
    Compute the sum of squared errors for the tensor decomposition.
    params: flattened vector of U, V, W.
    r: rank.
    """
    # Unpack
    U = params[0 : 9*r].reshape(9, r)
    V = params[9*r : 9*r + 21*r].reshape(21, r)
    W = params[9*r + 21*r : 9*r + 21*r + 21*r].reshape(21, r)
    
    # Compute T_recon for the 63 valid positions
    # T_recon[s] = sum_t U[I_idx[s], t] * V[J_idx[s], t] * W[K_idx[s], t]
    # We can do this by iterating t
    T_recon = np.zeros(63)
    for t in range(r):
        u_col = U[:, t]
        v_col = V[:, t]
        w_col = W[:, t]
        # Extract values at indices
        u_vals = u_col[I_idx]
        v_vals = v_col[J_idx]
        w_vals = w_col[K_idx]
        T_recon += u_vals * v_vals * w_vals
        
    diff = T_recon - 1.0
    return np.sum(diff * diff)

def construct():
    start_time = time.time()
    time_limit = 140.0  # Safety margin
    
    # Schoolbook solution
    rank_schoolbook = n * m * p
    U_school = np.zeros((9, rank_schoolbook))
    V_school = np.zeros((21, rank_schoolbook))
    W_school = np.zeros((21, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m + j, col] = 1
                V_school[j*p + k, col] = 1
                W_school[k*n + i, col] = 1
                col += 1
                
    best_solution = (U_school, V_school, W_school)
    best_rank = rank_schoolbook
    
    # Try to find lower rank solutions
    # We try ranks in descending order
    candidate_ranks = [60, 55, 50, 45, 40, 35, 30, 25, 23]
    
    for r in candidate_ranks:
        if time.time() - start_time > time_limit - 10:
            break
            
        found_valid = False
        # Try a few random restarts
        for trial in range(5):
            if time.time() - start_time > time_limit - 5:
                break
                
            random.seed(r * 100 + trial)
            np.random.seed(r * 100 + trial)
            
            # Initialize
            # Use small random values
            U0 = np.random.rand(9, r) * 0.1
            V0 = np.random.rand(21, r) * 0.1
            W0 = np.random.rand(21, r) * 0.1
            
            # Bias towards 1?
            U0 += 1.0
            V0 += 1.0
            W0 += 1.0
            
            params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
            
            # Optimize
            try:
                from scipy.optimize import minimize
                
                def objective(params):
                    return compute_error(params, r)
                    
                # Use L-BFGS-B
                res = minimize(objective, params0, method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12})
                
                if res.success or res.fun < 1e-8:
                    params_opt = res.x
                    U_opt = params_opt[0 : 9*r].reshape(9, r)
                    V_opt = params_opt[9*r : 9*r + 21*r].reshape(21, r)
                    W_opt = params_opt[9*r + 21*r : 9*r + 21*r + 21*r].reshape(21, r)
                    
                    # Check error
                    err = compute_error(params_opt, r)
                    if err < 1e-6:
                        # Try to snap to integers/half-integers
                        U_snap = np.round(U_opt * 2) / 2
                        V_snap = np.round(V_opt * 2) / 2
                        W_snap = np.round(W_opt * 2) / 2
                        
                        # Verify snapped solution
                        params_snap = np.concatenate([U_snap.flatten(), V_snap.flatten(), W_snap.flatten()])
                        err_snap = compute_error(params_snap, r)
                        
                        if err_snap < 1e-6:
                            best_rank = r
                            best_solution = (U_snap, V_snap, W_snap)
                            found_valid = True
                            break # Break trial loop
                        
            except Exception:
                pass
        
        if found_valid:
            break # Break rank loop
            
    U_res, V_res, W_res = best_solution
    
    return {"U": U_res.tolist(), "V": V_res.tolist(), "W": W_res.tolist()}

# For testing
if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
