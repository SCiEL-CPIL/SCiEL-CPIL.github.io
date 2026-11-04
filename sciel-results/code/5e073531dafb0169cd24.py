"""
Find a rank-minimal exact tensor decomposition for multiplying a 3x4 matrix by a 4x5 matrix.

Strategy:
1. Start with schoolbook (rank 60) as a valid baseline.
2. Use block decomposition: split m=4 into 2+2, and for each 3x2x5 block,
   find a low-rank decomposition.
3. For 3x2x5: C[i,k] = A[i,0]*B[0,k] + A[i,1]*B[1,k]
   The schoolbook rank is 3*2*5=30, but we can do better by sharing.
4. Key insight: For each (i,k), we need the bilinear form in (A[i,0],A[i,1]) and (B[0,k],B[1,k]).
   This is equivalent to computing 3*5=15 outputs, each from 2x2 inputs.
   The rank of the 2x2 multiplication tensor is 7 (Strassen), but here we have
   a different structure: we're not multiplying 2x2 matrices, we're computing
   15 separate bilinear forms.
5. Alternative: use the fact that the tensor is separable in a certain way.

Let me try a concrete construction:
- Split j into {0,1} and {2,3}
- For the first block (j=0,1): we have a 3x2x5 subproblem
- For the second block (j=2,3): we have another 3x2x5 subproblem
- Total schoolbook: 30 + 30 = 60

For each 3x2x5 subproblem, can we find rank < 30?
The tensor T[i,j,k] = 1 for i in [3], j in [2], k in [5].
We need sum_r U[i*2+j,r] * V[j*5+k,r] * W[k*3+i,r] = 1.

This is 3*2*5 = 30 equations. With rank r, we have (6+10+15)*r = 31r unknowns.
So r >= ceil(30/31) = 1, but the structure makes it harder.

Let me try to construct a rank-15 solution for 3x2x5:
If I set U[i*2+j, r] = 1 for all, V[j*5+k, r] = 1 for all, W[k*3+i, r] = 1 for all,
then sum_r = r, so I need r=1... but that gives 1, not 30 independent equations.

Actually, the issue is that all equations are identical: sum_r = 1.
So I just need one term: U[i*2+j,0]=1, V[j*5+k,0]=1, W[k*3+i,0]=1 for all i,j,k.
But that's rank 1! Wait, that can't be right.

Let me check: if U[i*2+j, 0] = 1 for all i,j, V[j*5+k, 0] = 1 for all j,k, W[k*3+i, 0] = 1 for all k,i,
then for each (i,j,k): sum_r U[i*2+j,r]*V[j*5+k,r]*W[k*3+i,r] = 1*1*1 = 1. Yes!

So the rank of the 3x2x5 "all-ones" tensor is 1? That seems too good to be true.

Wait, I think I'm misunderstanding the tensor. Let me re-read.

The matrix multiplication tensor is NOT all-ones. It's:
T[i,j,k] = 1 if the output C[i,k] depends on A[i,j] and B[j,k], which is always true.

But the tensor decomposition is for the bilinear map (A,B) -> AB, not for a single entry.

I think the correct interpretation is:
The tensor T in R^{nm} ⊗ R^{mp} ⊗ R^{np} is defined by:
T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{(i,j)} ⊗ e_{(j,k)} ⊗ e_{(i,k)}

where e_{(i,j)} is the basis vector for position (i,j) in A, etc.

A rank-r decomposition is:
T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r

where u_r in R^{nm}, v_r in R^{mp}, w_r in R^{np}.

This means: for all (i,j) in A, (j',k) in B, (i',k') in C:
sum_r u_r[i*m+j] * v_r[j'*p+k] * w_r[i'*p+k'] = 1 if i=i', j=j', k=k', else 0.

Wait, that's not right either. Let me think more carefully.

The multiplication tensor T is an element of R^{nm} ⊗ R^{mp} ⊗ R^{np} such that for any A in R^{n×m}, B in R^{m×p}:
vec(AB) = T(A, B) = sum_{i,j,k} A[i,j]*B[j,k] * e_{(i,k)}

As a tensor, T can be written as:
T = sum_{i=0}^{n-1} sum_{j=0}^{m-1} sum_{k=0}^{p-1} e_{(i,j)}^{(A)} ⊗ e_{(j,k)}^{(B)} ⊗ e_{(i,k)}^{(C)}

A rank-r decomposition:
T = sum_{r=1}^{R} u_r ⊗ v_r ⊗ w_r

means:
sum_{i,j,k} A[i,j]*B[j,k]*e_{(i,k)} = sum_{r=1}^{R} (u_r · A) * (v_r · B) * w_r

where u_r · A = sum_{i,j} u_r[i*m+j]*A[i,j], etc.

So the condition is:
For all A, B: sum_{i,j,k} A[i,j]*B[j,k]*e_{(i,k)} = sum_{r=1}^{R} (sum_{i,j} u_r[i*m+j]*A[i,j]) * (sum_{j,k} v_r[j*p+k]*B[j,k]) * w_r

This is equivalent to:
For all (i,j) in A, (j',k) in B, (i',k') in C:
delta_{i,i'} * delta_{j,j'} * delta_{k,k'} = sum_{r=1}^{R} u_r[i*m+j] * v_r[j'*p+k] * w_r[i'*p+k']

Wait, I'm still confusing myself. Let me use the standard formulation.

The multiplication tensor μ_{n,m,p} in R^{nm} ⊗ R^{mp} ⊗ R^{np} is defined by:
μ_{n,m,p} = sum_{i=1}^{n} sum_{j=1}^{m} sum_{k=1}^{p} e_i ⊗ e_j ⊗ e_k ⊗ e_{ik}

No wait, that's 4 indices. Let me use the correct formulation.

Actually, the standard formulation is:
μ_{n,m,p} ∈ R^{n×m} ⊗ R^{m×p} ⊗ R^{n×p}

μ_{n,m,p} = sum_{i,j,k} A_{ij} ⊗ B_{jk} ⊗ C_{ik}

where A_{ij} is the basis element for A[i,j], etc.

A rank-r decomposition:
μ_{n,m,p} = sum_{r=1}^{R} U_r ⊗ V_r ⊗ W_r

where U_r ∈ R^{n×m}, V_r ∈ R^{m×p}, W_r ∈ R^{n×p}.

This means:
sum_{i,j,k} A[i,j]*B[j,k]*C[i,k] = sum_{r=1}^{R} (U_r : A) * (V_r : B) * (W_r : C)

for all A, B, C. But that's not right either, because C is determined by A and B.

OK, I think the correct interpretation for this task is:
The decomposition must satisfy:
For all i in [n], k in [p]:
(AB)[i,k] = sum_{r=1}^{R} (sum_j U[i*m+j, r] * A[i,j]) * (sum_j' V[j'*p+k, r] * B[j',k]) * W[k*n+i, r]

Wait, that doesn't look right either. Let me just use the formulation from the task:
sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i, j, k.

This is the condition that the decomposition is exact. Let me just implement this and verify.

Given the confusion, let me just implement a clean schoolbook and a structured search.
"""

import time
import numpy as np
import random

DIMENSIONS = [3, 4, 5]

def verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that the decomposition is exact."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # 110s to be safe within 120s
    
    # Start with schoolbook
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    
    best_rank = rank
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try structured block decomposition
    # Split m=4 into two blocks of 2: j in {0,1} and j in {2,3}
    # For each block, we have a 3x2x5 subproblem
    
    # Try to find a low-rank decomposition for 3x2x5
    # The schoolbook for 3x2x5 has rank 30
    # Let's try rank 15 for each block (total 30)
    
    # Actually, let me try a different approach: 
    # Use the fact that for each (i,k), the equation is:
    # sum_{j=0}^{m-1} sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # 
    # If I fix i and k, and define for each r:
    # a_r = sum_j U[i*m+j, r] * V[j*p+k, r]
    # then I need sum_r a_r * W[k*n+i, r] = 1
    #
    # This is a system that can be solved if I have enough freedom.
    
    # Let me try a greedy construction:
    # For each (i,j,k), assign a unique rank index, but try to merge some.
    
    # Try random search with more attempts and structured initialization
    np.random.seed(42)
    
    # Try ranks from 59 down to 45
    for r in range(rank - 1, max(rank - 15, 1), -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        attempts = 20
        for attempt in range(attempts):
            if time.time() > deadline - 30:
                break
            
            # Initialize with small random values, biased towards 0 and 1
            U_r = np.zeros((n * m, r), dtype=float)
            V_r = np.zeros((m * p, r), dtype=float)
            W_r = np.zeros((p * n, r), dtype=float)
            
            # Set some entries to 1, others to 0 or -1
            for idx in range(n * m):
                if random.random() < 0.5:
                    U_r[idx, random.randint(0, r-1)] = 1
            for idx in range(m * p):
                if random.random() < 0.5:
                    V_r[idx, random.randint(0, r-1)] = 1
            for idx in range(p * n):
                if random.random() < 0.5:
                    W_r[idx, random.randint(0, r-1)] = 1
            
            # Check validity
            if verify_decomposition(U_r, V_r, W_r, n, m, p):
                best_rank = r
                best_U, best_V, best_W = U_r, V_r, W_r
                break
        
        if best_rank == r:
            break
    
    # Convert to lists with integer/half-integer values
    def to_list(arr):
        return [[float(round(x * 2) / 2) for x in row] for row in arr]
    
    return {
        "U": to_list(best_U),
        "V": to_list(best_V),
        "W": to_list(best_W)
    }

if __name__ == "__main__":
    result = construct()
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    print(f"Rank: {U.shape[1]}")
    print(f"U shape: {U.shape}, V shape: {V.shape}, W shape: {W.shape}")
