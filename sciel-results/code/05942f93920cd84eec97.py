"""Rewritten program for matrix multiplication 3x3x7 tensor decomposition."""
import time
import numpy as np

DIMENSIONS = [3, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 140.0  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    nm = n * m  # 9
    mp = m * p  # 21
    pn = p * n  # 21
    rank_school = n * m * p  # 63

    # Build schoolbook decomposition
    U_school = np.zeros((nm, rank_school))
    V_school = np.zeros((mp, rank_school))
    W_school = np.zeros((pn, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1

    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_school

    # Try to find a better decomposition using structured approaches
    # Strategy: Use the tensor structure to find low-rank decompositions
    # The multiplication tensor T[i,j,k] = 1 if we need output[i,k] += A[i,j]*B[j,k]
    
    # Approach: Try rank values from low to high, using random initialization + optimization
    # We'll use a simple iterative approach: initialize with random vectors, then project
    
    def verify_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition."""
        # The tensor is: sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # For each (i,j,k), the coefficient should be 1 if i,j,k are valid indices
        # Actually, the matrix multiplication tensor has entries:
        # T[i,j,k] = 1 for all valid (i,j,k) where i in [0,n), j in [0,m), k in [0,p)
        # And the decomposition is: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        
        # This is a 3-way tensor, so we can't easily verify without building the full tensor
        # Instead, we can check by computing C = A*B for specific A, B and comparing
        # with the decomposition-based computation
        
        # For efficiency, let's just check a few random instances
        np.random.seed(42)
        for _ in range(3):
            A = np.random.randn(n, m)
            B = np.random.randn(m, p)
            C_expected = A @ B
            
            # Compute C via decomposition: C[i,k] = sum_r (sum_j U[i*m+j,r]*A[i,j]) * (sum_k' V[j*p+k',r]*B[j,k']) * W[k*n+i,r]
            # Actually: C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * ... 
            # Wait, the standard form is:
            # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k])
            # No, that's not right either. Let me think again.
            
            # The tensor decomposition for matrix multiplication:
            # C[i,k] = sum_{j} A[i,j] * B[j,k]
            # = sum_r (sum_j U[i*m+j, r] * A[i,j]) * (sum_j V[j*p+k, r] * B[j,k]) * W[k*n+i, r]
            # No, that's not the standard form.
            
            # Standard tensor form: T_{i,j,k} = sum_r U_{i,j}^{(r)} * V_{j,k}^{(r)} * W_{k,i}^{(r)}
            # where T_{i,j,k} = 1 for the multiplication tensor
            # And C[i,k] = sum_j T_{i,j,k} * A[i,j] * B[j,k] = sum_j A[i,j] * B[j,k]
            
            # So the decomposition gives: for each r, we have a term
            # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
            # and summing over r should give T[i,j,k] = 1 for all valid (i,j,k)
            
            # Let's verify by checking that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] = 1
            # for all i in [0,n), j in [0,m), k in [0,p)
            pass  # We'll do a direct check below
        
        # Direct check: build the tensor and verify
        # T[i,j,k] should equal sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # for all i, j, k
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    # Wait, that's not right. The tensor entry is:
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    # This is an element-wise product followed by sum
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True

    # Try a few structured decompositions
    # For 3x3x3, Strassen's algorithm gives rank 7
    # We can embed this into 3x3x7 by using it for the first 3 columns of B
    
    # Strassen's algorithm for 3x3x3:
    # The 7 rank-1 tensors are defined by specific U, V, W matrices
    # Let's define them explicitly
    
    # Strassen's 7 terms for multiplying 3x3 by 3x3:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B12-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1 - M4 + M5 - M7
    # C12 = M3 + M7
    # C21 = M2 + M6
    # C22 = M1 - M3 + M2 - M6
    
    # For our 3x3x3 subproblem (i in 0..2, j in 0..2, k in 0..2):
    # We need U[i*m+j, r], V[j*p+k, r], W[k*n+i, r]
    
    # Let me define the Strassen decomposition properly
    # The 7 rank-1 components:
    
    # Component 1: (A11+A22)(B11+B22) contributes to C11 and C22
    # U: A11+A22 means U[0*3+0]=1, U[1*3+1]=1 (i=0,j=0 and i=1,j=1)
    # V: B11+B22 means V[0*3+0]=1, V[1*3+1]=1 (j=0,k=0 and j=1,k=1)
    # W: contributes to C11 (i=0,k=0) and C22 (i=1,k=1)
    # W[0*3+0]=1, W[1*3+1]=1
    
    # Actually, let me use a more systematic approach.
    # For the 3x3x3 block, I'll use known Strassen coefficients.
    
    # Let's try a different approach: use random search with verification
    # for small ranks, and fall back to schoolbook if nothing works.
    
    # Try ranks from 20 to 62 (below schoolbook)
    # Use random initialization with values from {-1, 0, 1} and {-0.5, 0, 0.5}
    
    def try_random_decomposition(rank, max_iters=500, time_limit=10.0):
        """Try to find a decomposition of given rank using random search."""
        nonlocal best_U, best_V, best_W, best_rank
        start = time.time()
        
        for attempt in range(max_iters):
            if time.time() - start > time_limit:
                break
            
            # Initialize with random values from {-1, -0.5, 0, 0.5, 1}
            vals = np.array([-1, -0.5, 0, 0.5, 1])
            U = np.random.choice(vals, size=(nm, rank))
            V = np.random.choice(vals, size=(mp, rank))
            W = np.random.choice(vals, size=(pn, rank))
            
            # Simple iterative refinement: for each rank-1 component, try to optimize
            # This is a heuristic, not guaranteed to work
            
            # Check if this random initialization happens to work (unlikely)
            if verify_decomposition(U, V, W):
                best_U = U.copy()
                best_V = V.copy()
                best_W = W.copy()
                best_rank = rank
                return True
        
        return False

    # Try a few ranks
    for rank in [21, 24, 28, 30, 35, 40, 45, 50, 55, 60]:
        if time.time() > deadline - 30:
            break
        if rank >= best_rank:
            break
        if try_random_decomposition(rank, max_iters=100, time_limit=5.0):
            break

    # Convert to lists for return
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()

    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
