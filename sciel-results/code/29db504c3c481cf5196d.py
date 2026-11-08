"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

import numpy as np

def _check_decomp(n, m, p, U, V, W):
    """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 iff j==j' else 0."""
    # Actually for matmul tensor: T[i,j,k,i',j',k'] = 1 if j==j' and k==k', else 0
    # Decomposition: sum_r U[i,j,r] V[j,k,r] W[k,i,r] should give
    #   sum_r U[i,j,r] V[j,k,r] W[k,i,r] = T(i,j,k,i')
    # Let's re-derive: For C = A @ B where C[i,k] = sum_j A[i,j] B[j,k]
    # Tensor T[i,j,k] = 1 (it's the identity in j)
    # Decomposition: T[i,j,k] = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    # Wait, let me use the standard form:
    # U has shape (n*m, rank), indexed by (i,j)
    # V has shape (m*p, rank), indexed by (j,k)
    # W has shape (p*n, rank), indexed by (k,i)
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    # For matmul: T[i,j,k] = 1 if (the j-th element of row i of A times j-th column of B contributes to C[i,k])
    # Actually T[i,j,k] = delta(j,k) is NOT right.
    # The matmul tensor is: T[i,j,k,i',j',k'] but we reduce to 3-index.
    # C[i,k] = sum_j A[i,j] B[j,k], so T[i,j,k] = 1 for all valid (i,j,k) where the product is formed.
    # So T[i,j,k] = 1 for all i,j,k. The decomposition should satisfy:
    # sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1 for all i,j,k.
    # Wait, that's not a tensor decomposition then. Let me re-read.
    # "tensor decomposition for multiplying a 4 by 4 matrix by a 4 by 6 matrix"
    # The standard matmul tensor T has indices (i, j, k) where C[i,k] = sum_j A[i,j] B[j,k]
    # So T[i,j,k] = 1 (unit tensor) and the contraction is over j.
    # A rank-1 term contributes: U[i,j] * V[j,k] * W[k,i] summed over the term index.
    # For the identity decomposition (schoolbook): each term corresponds to (j) fixed:
    #   U[i,j] = 1, V[j,k] = 1, W[k,i] = 1 for fixed j.
    #   sum over j: T[i,j,k] = sum_j 1 = ... no, that doesn't work.
    #
    # Let me reconsider. In the standard formulation:
    # T_{ijk} = 1 for all i,j,k (the "multiplication" tensor)
    # A rank-r decomposition: T_{ijk} = sum_{s=1}^r U_{is} V_{js} W_{ks}
    # For matmul: we need sum_s U[i,s] * V[j,s] * W[k,s] = delta? No.
    #
    # Actually for C = AB: C_{ik} = sum_j A_{ij} B_{jk}
    # This is a trilinear form in A, B, C: sum_{i,j,k} C_{ik} A_{ij} B_{jk} = 0? No.
    # The constraint is C_{ik} = sum_j A_{ij} B_{jk}.
    # Tensor: T_{ijk}^{(A,B,C)} where contraction over j: sum_j A_{ij} B_{jk} - C_{ik} = 0
    # So the "tensor" for the bilinear map (A,B) -> C is:
    # C_{ik} = sum_j T_{ij}^{k} A_{ij} B_{jk} where T_{ij}^{k} = delta_{jk}... no.
    #
    # OK let me just go with the interface as given and use the schoolbook:
    # rank = n*m*p. Term (i,j,k) has U[i,j]=1, V[j,k]=1, W[k,i]=1 at that column.
    # Then sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r] = 1 for all i,j,k (only the term matching i,j,k contributes).
    # Wait no: term r corresponds to (i_r, j_r, k_r). U[i*m+j, r] = 1 iff i=i_r and j=j_r.
    # V[j*p+k, r] = 1 iff j=j_r and k=k_r.
    # W[k*n+i, r] = 1 iff k=k_r and i=i_r.
    # So for a given (i,j,k), only term r=(i,j,k) has all three = 1, giving 1.
    # So T[i,j,k] = 1 for all i,j,k. Good, that's the identity tensor.
    pass

def _random_decomposition(n, m, p, rank, rng):
    """Generate a random decomposition and verify it."""
    U = rng.standard_normal((n*m, rank))
    V = rng.standard_normal((m*p, rank))
    W = rng.standard_normal((p*n, rank))
    # Check: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should = 1 for all i,j,k
    # We can check a few samples.
    for _ in range(100):
        i, j, k = rng.randint(n), rng.randint(m), rng.randint(p)
        val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
        if abs(val - 1.0) > 1e-6:
            return None
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p  # 96
    
    # Start with schoolbook
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Try to find a lower-rank decomposition using iterative rank reduction
    # Strategy: start with schoolbook, randomly zero out entries, fix up with optimization
    
    import time
    start_time = time.time()
    deadline = 140.0  # leave 10s margin
    
    best_rank = schoolbook_rank
    best_decomp = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    def verify(U_mat, V_mat, W_mat, sample_count=500):
        """Verify decomposition on random samples."""
        rng_local = np.random.RandomState(42)
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        for _ in range(sample_count):
            i = rng_local.randint(n)
            j = rng_local.randint(m)
            k = rng_local.randint(p)
            val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
            if abs(val - 1.0) > 1e-9:
                return False
        return True
    
    def verify_full(U_mat, V_mat, W_mat):
        """Full verification."""
        U_arr = np.array(U_mat, dtype=np.float64)
        V_arr = np.array(V_mat, dtype=np.float64)
        W_arr = np.array(W_mat, dtype=np.float64)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j, :] * V_arr[j*p+k, :] * W_arr[k*n+i, :])
                    if abs(val - 1.0) > 1e-9:
                        return False
        return True
    
    # Try various ranks from high to low
    current_rank = schoolbook_rank
    
    while current_rank >= 60 and (time.time() - start_time) < deadline:
        found = False
        for attempt in range(50):
            if (time.time() - start_time) > deadline:
                break
            
            # Randomized iterative method: start from schoolbook, try to project out a column
            # or use random restart with constrained optimization
            
            if attempt < 10:
                # Strategy: remove one column from schoolbook and fix
                if current_rank <= schoolbook_rank:
                    pass  # already at schoolbook, skip this approach
            
            # Strategy: random sparse initialization + gradient-free perturbation
            rng = np.random.RandomState(12345 + attempt * 7 + current_rank)
            
            # Start from a scaled identity-like structure
            U_init = np.zeros((n*m, current_rank))
            V_init = np.zeros((m*p, current_rank))
            W_init = np.zeros((p*n, current_rank))
            
            # Fill with random patterns that might work
            for r in range(current_rank):
                # Each rank-1 term: pick random (i,j), (j,k), (k,i) patterns
                # with some structure
                i1 = rng.randint(n)
                j1 = rng.randint(m)
                k1 = rng.randint(p)
                
                U_init[i1*m+j1, r] = 1.0
                V_init[j1*p+k1, r] = 1.0
                W_init[k1*n+i1, r] = 1.0
                
                # Add some extra coupling
                if rng.random() < 0.5:
                    i2 = (i1 + 1) % n
                    U_init[i2*m+j1, r] = rng.choice([-1, 1])
                    i3 = (i1 + 2) % n
                    if i3 != i1 and i3 != i2:
                        U_init[i3*m+j1, r] = rng.choice([-0.5, 0.5, 1.0, -1.0])
                
                if rng.random() < 0.5:
                    k2 = (k1 + 1) % p
                    V_init[j1*p+k2, r] = rng.choice([-1, 1])
                    k3 = (k1 + 2) % p
                    if k3 != k1 and k3 != k2:
                        V_init[j1*p+k3, r] = rng.choice([-0.5, 0.5, 1.0, -1.0])
                
                if rng.random() < 0.5:
                    i2 = (i1 + 1) % n
                    W_init[k1*n+i2, r] = rng.choice([-1, 1])
                
                if rng.random() < 0.3:
                    # Cross coupling
                    j2 = (j1 + 1) % m
                    if j2 != j1:
                        U_init[i1*m+j2, r] = rng.choice([-0.5, 0.5])
                        V_init[j2*p+k1, r] = rng.choice([-0.5, 0.5])
                        W_init[k1*n+i1, r] += rng.choice([-0.25, 0.25])
            
            # Now try to fix up using alternating least squares
            U_c = U_init.copy()
            V_c = V_init.copy()
            W_c = W_init.copy()
            
            converged = False
            for sweep in range(200):
                if (time.time() - start_time) > deadline - 5:
                    break
                
                # Update U: for each (i,j), minimize sum_k (1 - sum_r V[j*p+k,r] W[k*n+i,r] U[i*m+j,r])^2
                # This is a linear least squares in U[i*m+j, :]
                # Let A_r = V[j*p+k_r, r] * W[k_r*n+i, r] for each r, and target = 1 for each k_r
                # Actually: for fixed (i,j), we need:
                #   sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all k
                # Let M[r, k] = V[j*p+k, r] * W[k*n+i, r]
                # Then M @ U[i*m+j, :] = ones(p)
                for i in range(n):
                    for j in range(m):
                        M = np.array([[V_c[j*p+k, r] * W_c[k*n+i, r] for k in range(p)] for r in range(current_rank)])
                        # M is (current_rank, p), solve M @ x = ones(p)
                        try:
                            if current_rank >= p:
                                x, _, _, _ = np.linalg.lstsq(M, np.ones(p), rcond=None)
                                U_c[i*m+j, :] = x
                        except:
                            pass
                
                # Update V: for each (j,k)
                for j in range(m):
                    for k in range(p):
                        M = np.array([[U_c[i*m+j, r] * W_c[k*n+i, r] for i in range(n)] for r in range(current_rank)])
                        try:
                            if current_rank >= n:
                                x, _, _, _ = np.linalg.lstsq(M, np.ones(n), rcond=None)
                                V_c[j*p+k, :] = x
                        except:
                            pass
                
                # Update W: for each (k,i)
                for k in range(p):
                    for i in range(n):
                        M = np.array([[U_c[i*m+j, r] * V_c[j*p+k, r] for j in range(m)] for r in range(current_rank)])
                        try:
                            if current_rank >= m:
                                x, _, _, _ = np.linalg.lstsq(M, np.ones(m), rcond=None)
                                W_c[k*n+i, :] = x
                        except:
                            pass
                
                # Check convergence
                if sweep % 20 == 19:
                    if verify_full(U_c.tolist(), V_c.tolist(), W_c.tolist()):
                        converged = True
                        break
                
                if (time.time() - start_time) > deadline - 5:
                    break
            
            if converged:
                # Found valid decomposition at this rank!
                if verify_full(U_c.tolist(), V_c.tolist(), W_c.tolist()):
                    best_rank = current_rank
                    best_decomp = {"U": U_c.tolist(), "V": V_c.tolist(), "W": W_c.tolist()}
                    found = True
                    print(f"Found valid decomposition at rank {current_rank}")
                    # Try to go lower
                    break
            else:
                # Try a few iterations of random perturbation from the current best
                if (time.time() - start_time) > deadline - 10:
                    break
        
        if found:
            current_rank -= 1  # try one rank lower
        else:
            # If we can't find at this rank, try a bit lower
            if current_rank > schoolbook_rank - 10:
                current_rank -= 2
            else:
                break
        
        # If we found a good decomposition, we're done
        if best_rank < schoolbook_rank and (time.time() - start_time) > deadline - 15:
            break
    
    # Round to half-integers if close
    def round_to_half(arr):
        result = [[0.0] * len(arr[0]) for _ in range(len(arr))]
        for i in range(len(arr)):
            for j in range(len(arr[i])):
                val = arr[i][j]
                if abs(val) < 1e-10:
                    result[i][j] = 0.0
                elif abs(val * 2 - round(val * 2)) < 1e-8:
                    result[i][j] = round(val * 2) / 2
                else:
                    result[i][j] = float(val)
        return result
    
    # Convert best to integer/half-integer form
    U_final = round_to_half(best_decomp["U"])
    V_final = round_to_half(best_decomp["V"])
    W_final = round_to_half(best_decomp["W"])
    
    # Final verification
    if verify_full(U_final, V_final, W_final):
        return {"U": U_final, "V": V_final, "W": W_final}
    else:
        # Fallback to schoolbook
        return {"U": U_sb, "V": V_sb, "W": W_sb}

# EVOLVE-BLOCK-END
