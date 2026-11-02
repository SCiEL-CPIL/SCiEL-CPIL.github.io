import time
import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix
    by a 2x10 matrix. The output factor follows the official trace convention W[k*n+i,r].
    Returns {'U', 'V', 'W'} with shapes (n*m, rank), (m*p, rank), (p*n, rank).
    """
    t0 = time.time()
    DEADLINE = 140  # seconds, safety margin before 150s kill

    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # A is n x m = 2x2
    # B is m x p = 2x10
    # C is n x p = 2x10

    # The bilinear map is:
    # A (4 entries) x B (20 entries) -> C (20 entries)
    # C_{i,k} = sum_j A_{i,j} * B_{j,k}
    # For each output C_{i,k}, it depends on A_{i,0}, A_{i,1}, B_{0,k}, B_{1,k}

    # Schoolbook: rank = n * m * p = 40 (one term per (i,j,k))
    # We can do better using Strassen on the 2x2 blocks.
    # B has shape 2x10 = 5 blocks of 2x2
    # Each 2x2x2 multiplication can be done in rank 7 (Strassen)
    # So total rank = 5 * 7 = 35

    # But can we do better? The 5 blocks share the same A matrix.
    # This is a structured tensor: T = sum_{b=0}^{4} A \otimes B_b \otimes C_b
    # where A is 2x2x2 (the multiplication tensor), B_b is the b-th 2x2 block of B,
    # and C_b is the b-th 2x2 block of C.
    
    # Actually, let me think more carefully. The tensor for (2x2) * (2x10) -> (2x10) is:
    # T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'}
    # where (i,j) indexes A, (j',k) indexes B, (i',k) indexes C.
    
    # This tensor has a block structure. For each pair of output columns (k, k+1)
    # forming a 2x2 block, the sub-tensor is the 2x2x2 matrix multiplication tensor.
    # Since A is shared across all blocks, we might be able to exploit this.
    
    # Let's set up the problem as finding a rank-r decomposition.
    # U: (n*m) x r = 4 x r
    # V: (m*p) x r = 20 x r  
    # W: (p*n) x r = 20 x r
    # such that for all (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] = W[k*n+i, r] ... no wait.
    
    # The correct decomposition: T_{alpha, beta, gamma} = sum_r U_{alpha,r} V_{beta,r} W_{gamma,r}
    # where alpha = (i,j) for A, beta = (j,k) for B, gamma = (k,i) for C (with convention W[k*n+i])
    # Actually the convention says W[k*n+i, r], so gamma = k*n + i.
    
    # The multiplication tensor: T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    # Wait, that's not right either. Let me re-derive.
    
    # C_{i,k} = sum_j A_{i,j} B_{j,k}
    # So the tensor T has components T_{(i,j),(j',k),(i',k')} = 1 if i=i' and j=j' and k=k', else 0.
    # No, that's not right. The tensor for matrix multiplication M_{n,m,p} has:
    # T_{(i,j),(j',k),(i',k')} = delta_{i,i'} * delta_{j,j'} * delta_{k,k'}
    # This is the standard definition.
    
    # For our case: n=2, m=2, p=10
    # alpha = i*m + j, i in {0,1}, j in {0,1} -> alpha in {0,1,2,3}
    # beta = j*p + k, j in {0,1}, k in {0..9} -> beta in {0..19}
    # gamma = k*n + i, k in {0..9}, i in {0,1} -> gamma in {0..19}
    
    # T[alpha, beta, gamma] = 1 if (i,j)=(i',j') and (j',k)=(j',k') and (k',i)=(k,i')
    # i.e., T[(i,j),(j',k),(k',i')] = delta_{i,i'} delta_{j,j'} delta_{k,k'}
    
    # Let's build the schoolbook decomposition (rank 40) and then try to reduce.
    
    # First, let's construct the Strassen-based rank-35 solution as a starting point.
    rank35 = _strassen_rank35(n, m, p)
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 30:
        return rank35
    
    # Now try to reduce the rank further.
    # The key insight: the 5 blocks of the 2x10 matrix share the same A.
    # This means the tensor is a "sum" of 5 copies of the 2x2x2 multiplication tensor,
    # but they share the A factor.
    
    # Let's formulate this as a linear algebra problem.
    # We want to find U (4 x r), V (20 x r), W (20 x r) such that:
    # For all (i,j,k): sum_{r=0}^{r-1} U[i*m+j, r] * V[j*p+k, r] = 1 if this is a valid term, else 0
    # Wait, no. The tensor T is what we're decomposing.
    
    # T[alpha, beta, gamma] = sum_r U[alpha, r] * V[beta, r] * W[gamma, r]
    
    # Let's list the non-zero entries of T:
    # T[(i,j),(j,k),(k,i)] = 1 for all i,j,k (with appropriate index mapping)
    # In our indexing:
    # alpha = i*m + j = 2*i + j
    # beta = j*p + k = 10*j + k
    # gamma = k*n + i = 2*k + i
    
    # So T has 1 at positions (2*i+j, 10*j+k, 2*k+i) for i in {0,1}, j in {0,1}, k in {0..9}
    # Total: 2*2*10 = 40 non-zero entries.
    
    # The Strassen rank-35 solution works because within each 2x2 block of k,
    # we use 7 multiplications instead of 4, but the A factors are shared.
    
    # Can we go below 35? Let's try a rank-reduction search.
    # We'll use the fact that the tensor has a special structure to set up a system of equations.
    
    # For a given rank r, we need to find U, V, W such that:
    # sum_{rho=0}^{r-1} U[2*i+j, rho] * V[10*j+k, rho] * W[2*k+i, rho] = 1 for all i,j,k
    # and = 0 for all other (alpha, beta, gamma) combinations.
    
    # This is a nonlinear system. Let's try to solve it iteratively.
    
    # Strategy: Start with the rank-35 solution and try to merge factors.
    # Or use a different approach: exploit the block structure directly.
    
    # Let's think about it differently. The tensor T can be written as:
    # T = sum_{k=0}^{9} sum_{i,j} e_{2*i+j} \otimes e_{10*j+k} \otimes e_{2*k+i}
    
    # Group by k: for each k, we have a 4x1 tensor (over i,j) \otimes e_{10*j+k} \otimes e_{2*k+i}
    # Actually for each k, the terms are:
    # (i=0,j=0): e_0 \otimes e_k \otimes e_{2k}
    # (i=0,j=1): e_1 \otimes e_{10+k} \otimes e_{2k}
    # (i=1,j=0): e_2 \otimes e_k \otimes e_{2k+1}
    # (i=1,j=1): e_3 \otimes e_{10+k} \otimes e_{2k+1}
    
    # For a pair of consecutive k values (k=2b, k=2b+1), we get 8 terms that form a 2x2x2 multiplication tensor.
    # Strassen reduces each such group from rank 4 to rank 7... wait, that's worse!
    
    # Oh wait, I made an error. In the standard Strassen algorithm for (2x2)*(2x2)->(2x2),
    # the rank is 7 instead of 8 (schoolbook). But here each "block" has 4 terms, and Strassen uses 7.
    # So Strassen makes it WORSE in this context?
    
    # No wait. Let me re-examine. In the current program, the rank is 35 = 5*7.
    # Schoolbook would be 40 = 5*8. So Strassen gives 35 < 40. That's correct.
    # Each 2x2x2 block has 4 non-zero tensor entries (one for each (i,j) pair given fixed k-pair),
    # and Strassen decomposes the 2x2x2 multiplication into 7 rank-1 tensors.
    # But wait, the 2x2x2 multiplication tensor has 8 non-zero entries (one for each (i,j,k) triple).
    # Hmm, I'm getting confused.
    
    # Let me just verify: the current program claims rank 35 and score 1.1429.
    # The theoretical lower bound for 2x2x2 matrix multiplication over a field is 7 (Strassen).
    # For 5 independent 2x2x2 blocks sharing the same A, the minimum rank is at least 7*5=35?
    # Not necessarily, because the A factors are shared.
    
    # Actually, the tensor here is NOT a direct sum of 5 independent 2x2x2 tensors.
    # It's more like: T = sum_b (A \otimes B_b \otimes C_b) where B_b and C_b vary with b.
    # The A factor is the SAME for all b.
    
    # This is a different tensor structure. Let me think about whether rank < 35 is possible.
    
    # Let's try a computational approach: set up the equations and try to find a rank-34 solution.
    
    def _try_rank_reduction(rank, U_init, V_init, W_init, max_iters=50, timeout=20):
        """Try to verify and potentially improve a rank-r decomposition."""
        U = np.array(U_init, dtype=float)
        V = np.array(V_init, dtype=float)
        W = np.array(W_init, dtype=float)
        
        # Verify the decomposition
        def _verify(U, V, W):
            # Check: for all (i,j,k), sum_r U[2i+j,r] V[10j+k,r] W[2k+i,r] should equal 1
            # And for all other (alpha,beta,gamma), it should be 0.
            errors = 0
            for i in range(2):
                for j in range(2):
                    for k in range(10):
                        alpha = 2*i + j
                        beta = 10*j + k
                        gamma = 2*k + i
                        val = np.dot(U[alpha, :], V[beta, :] * W[gamma, :])
                        if abs(val - 1.0) > 1e-9:
                            errors += 1
            # Check some zero entries
            for _ in range(100):
                i = np.random.randint(0, 2)
                j = np.random.randint(0, 2)
                k = np.random.randint(0, 10)
                alpha = 2*i + j
                beta = 10*j + k
                gamma = 2*k + i
                # This is a non-zero entry, so skip
                # Pick a random (alpha,beta,gamma) that should be zero
                while True:
                    a2 = np.random.randint(0, 4)
                    b2 = np.random.randint(0, 20)
                    g2 = np.random.randint(0, 20)
                    # Check if this is a non-zero entry
                    i2 = a2 // 2
                    j2 = a2 % 2
                    j2b = b2 // 10
                    k2 = b2 % 10
                    i2g = g2 % 2
                    k2g = g2 // 2
                    if not (i2 == i2g and j2 == j2b and k2 == k2g):
                        break
                val = np.dot(U[a2, :], V[b2, :] * W[g2, :])
                if abs(val) > 1e-9:
                    errors += 1
            return errors
        
        initial_errors = _verify(U, V, W)
        if initial_errors == 0:
            return U, V, W, True
        
        return U, V, W, False
    
    # For now, return the Strassen rank-35 solution.
    # A more sophisticated approach would be to set up the full nonlinear system
    # and use Newton's method or gradient descent to find a rank-34 solution if one exists.
    
    # Let's attempt a simple rank-reduction: try to see if any two of the 35 factors
    # can be merged.
    
    U35 = np.array(rank35['U'], dtype=float)
    V35 = np.array(rank35['V'], dtype=float)
    W35 = np.array(rank35['W'], dtype=float)
    
    elapsed = time.time() - t0
    if elapsed > DEADLINE - 20:
        return rank35
    
    # Try to find a rank-34 solution by setting up the equations.
    # This is a hard problem in general, but let's try a heuristic approach.
    
    # Actually, let's just return the rank-35 solution for now, as it's known to be valid.
    # The score of 1.1429 suggests this is the current best.
    
    return rank35

def _strassen_rank35(n, m, p):
    """Construct the rank-35 Strassen-based decomposition."""
    rank = 35
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for A (2x2)
    A_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): 1},
        {(0,0): 1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (0,0): -1},
        {(0,1): 1, (1,1): -1}
    ]
    
    # Strassen coefficients for B (2x2)
    B_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(0,0): 1},
        {(0,1): 1, (1,1): -1},
        {(1,0): 1, (0,0): -1},
        {(1,1): 1},
        {(0,0): 1, (0,1): 1},
        {(1,0): 1, (1,1): 1}
    ]
    
    # Strassen coefficients for C (2x2)
    C_coeffs = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    for b in range(p // 2):
        for s in range(7):
            r = b * 7 + s
            
            for (i, j), val in A_coeffs[s].items():
                U[i * m + j][r] = val
                
            for (j, k_bar), val in B_coeffs[s].items():
                k = b * 2 + k_bar
                V[j * p + k][r] = val
                
            for (i, k_bar), val in C_coeffs[s].items():
                k = b * 2 + k_bar
                W[k * n + i][r] = val
                
    return {"U": U, "V": V, "W": W}
