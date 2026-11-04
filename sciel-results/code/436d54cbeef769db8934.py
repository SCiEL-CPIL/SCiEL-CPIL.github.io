"""
Rank-minimal exact tensor decomposition for 4x5 * 5x6 matrix multiplication.

Strategy:
  The schoolbook rank is 4*5*6 = 120.
  We try to find a lower-rank decomposition by exploiting the rectangular structure.
  
  Known results:
  - For square n x n x n, the best known upper bounds are below n^3 for n >= 2.
  - For rectangular cases, the rank can be lower than n*m*p.
  
  Approach:
  We use a structured search based on the "border rank" or "slice rank" ideas.
  However, since we need an EXACT decomposition (not border rank), we look for
  decompositions that come from known algorithms or structured patterns.

  One well-known technique is to use the "Laderman" or "Strassen-like" decompositions
  for small cases. For 4x5x6, we can try to construct a decomposition by combining
  smaller known decompositions.

  Actually, let's think about this differently. The tensor for matrix multiplication
  M_{n,m,p} has entries T[i,j,k] = 1 if (i,j,k) corresponds to a valid (i,j,k) triple
  in the multiplication. The rank of this tensor is the minimum number of rank-1 tensors
  needed to express it.

  For small dimensions, we can try to use integer/half-integer entries and search.
  But a full search is infeasible for rank < 120.

  Instead, we use a known result: the rank of the 4x4x4 matrix multiplication tensor is
  at most 20 (Laderman's algorithm gives 20, and it's known that 19 is possible).
  
  For 4x5x6, we can try to construct a decomposition by "padding" or using block structures.

  However, given the time constraints, let's try a different approach:
  We'll use a randomized search with local optimization to try to reduce the rank
  from 120.

  Actually, let me reconsider. The task says "Find a rank-minimal exact tensor decomposition".
  This suggests we should find the true minimum rank, not just a good upper bound.

  For 4x5x6, the minimum rank is not known to be much below 120 in the literature for
  exact (not border) rank. Let's try to construct a decomposition with rank < 120.

  One idea: use the fact that 4x5 = 20, 5x6 = 30, 6x4 = 24. The schoolbook uses 120 terms.
  
  Let's try a structured approach: decompose the 4x5 matrix into blocks and use
  known decompositions for smaller cases.

  Actually, I'll implement a search that tries to find a decomposition with rank < 120
  by using a randomized approach with integer/half-integer constraints.

  Given the complexity, let's start with a reasonable upper bound and try to improve.

  After reflection, I think the best practical approach within the time limit is:
  1. Start with the schoolbook decomposition (rank 120).
  2. Try to reduce the rank by identifying linear dependencies in the tensor.
  3. Use a greedy or local search approach to find a lower-rank decomposition.

  However, this is computationally intensive. Let's try a simpler approach:
  use known decompositions from the literature for small rectangular cases.

  For now, I'll return the schoolbook decomposition, as finding a provably lower rank
  decomposition for 4x5x6 is a research-level problem.

  Wait, the task says "rank-minimal". Let me try to at least find a decomposition
  that is better than schoolbook if possible.

  Let me try a different strategy: use the "substitution" method or "tensor slicing".

  Actually, I'll implement a search that tries to find a decomposition with rank
  between 100 and 120, using a randomized approach.
"""

import time
import random
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Build the multiplication tensor as a 3D array
    # T[i, j, k] = 1 if this term appears in C[i,k] = sum_j A[i,j] * B[j,k]
    # Actually, the tensor is defined such that C[i,k] = sum_{j} A[i,j] B[j,k]
    # The tensor T has T[i,j,k] = 1 for all valid (i,j,k)
    # The rank is the minimum r such that T = sum_{c=1}^{r} u_c v_c w_c
    # where u_c is in R^{n*m}, v_c in R^{m*p}, w_c in R^{p*n}
    
    # Let's construct the tensor explicitly
    # Indexing: U row = i*m + j, V row = j*p + k, W row = k*n + i
    T = np.zeros((n * m, m * p, p * n), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    # Try to find a low-rank decomposition
    # We'll use a randomized approach: initialize U, V, W with random values
    # and try to minimize the reconstruction error
    
    best_rank = schoolbook_rank
    best_U = None
    best_V = None
    best_W = None
    
    # First, set up the schoolbook decomposition as initial best
    U_school = np.zeros((n * m, schoolbook_rank))
    V_school = np.zeros((m * p, schoolbook_rank))
    W_school = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    # Try ranks from 110 down to 80
    for target_rank in range(110, 79, -1):
        if time.time() > deadline - 20:
            break
        
        # Try a few random initializations
        for attempt in range(5):
            if time.time() > deadline - 10:
                break
            
            random.seed(attempt * 1000 + target_rank)
            np.random.seed(attempt * 1000 + target_rank)
            
            # Initialize with random values
            U = np.random.randn(n * m, target_rank)
            V = np.random.randn(m * p, target_rank)
            W = np.random.randn(p * n, target_rank)
            
            # Try to optimize using a simple gradient-free approach
            # We'll use a local search: perturb one entry at a time
            
            # Compute initial error
            def compute_error(U, V, W):
                # Reconstruct tensor and compare
                recon = np.zeros((n * m, m * p, p * n))
                for c in range(target_rank):
                    recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
                return np.linalg.norm(T - recon)
            
            current_error = compute_error(U, V, W)
            
            # Local search
            max_iters = 50
            for it in range(max_iters):
                if time.time() > deadline - 5:
                    break
                
                # Pick a random matrix and entry to perturb
                mat_choice = random.choice(['U', 'V', 'W'])
                if mat_choice == 'U':
                    mat = U
                    rows = n * m
                elif mat_choice == 'V':
                    mat = V
                    rows = m * p
                else:
                    mat = W
                    rows = p * n
                
                row = random.randint(0, rows - 1)
                col = random.randint(0, target_rank - 1)
                old_val = mat[row, col]
                
                # Try a small perturbation
                delta = random.choice([-0.5, -0.25, -0.1, 0.1, 0.25, 0.5])
                mat[row, col] = old_val + delta
                
                new_error = compute_error(U, V, W)
                if new_error < current_error:
                    current_error = new_error
                else:
                    mat[row, col] = old_val
            
            # Check if we found a valid decomposition
            if current_error < 1e-6:
                # Round to nearest half-integer
                U = np.round(U * 2) / 2
                V = np.round(V * 2) / 2
                W = np.round(W * 2) / 2
                
                # Verify after rounding
                recon = np.zeros((n * m, m * p, p * n))
                for c in range(target_rank):
                    recon += np.outer(np.outer(U[:, c], V[:, c]), W[:, c])
                if np.linalg.norm(T - recon) < 1e-6:
                    best_rank = target_rank
                    best_U = U
                    best_V = V
                    best_W = W
                    break
        
        if best_rank < schoolbook_rank:
            break
    
    # Convert to lists
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
