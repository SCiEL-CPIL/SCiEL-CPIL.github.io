"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Shapes: U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    nm = n * m  # 16
    mp = m * p  # 32
    pn = p * n  # 32
    
    # The tensor T[i*m+j, j*p+k, k*n+i] should equal 1 for all valid (i,j,k)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Build the list of (i,j,k) triples and their corresponding indices
    triples = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                triples.append((u_idx, v_idx, w_idx))
    
    num_triples = len(triples)  # 4*4*8 = 128
    
    # Try to find minimal rank decomposition
    # Start with schoolbook and try to reduce
    schoolbook_rank = n * m * p  # 128
    
    def verify(U, V, W, rank):
        """Check if decomposition is exact."""
        for (u_idx, v_idx, w_idx) in triples:
            val = sum(U[u_idx][r] * V[v_idx][r] * W[w_idx][r] for r in range(rank))
            if abs(val - 1.0) > 1e-6:
                return False
        return True
    
    def verify_numpy(U, V, W, rank):
        """Vectorized verification."""
        U_arr = np.array(U)
        V_arr = np.array(V)
        W_arr = np.array(W)
        # For each triple, compute sum_r U[u,r]*V[v,r]*W[w,r]
        for (u_idx, v_idx, w_idx) in triples:
            val = np.dot(U_arr[u_idx], V_arr[v_idx] * W_arr[w_idx])
            if abs(val - 1.0) > 1e-6:
                return False
        return True
    
    # Schoolbook decomposition as fallback
    def schoolbook():
        rank = schoolbook_rank
        U = [[0] * rank for _ in range(nm)]
        V = [[0] * rank for _ in range(mp)]
        W = [[0] * rank for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    start_time = time.time()
    deadline = start_time + 130  # safety margin
    
    best_result = None
    
    # Try ranks from a reasonable minimum up to schoolbook
    # Theoretical lower bound: at least max(n*m, m*p, p*n) = max(16, 32, 32) = 32
    # But actually the rank must be at least the number of nonzero entries in any "slice"
    # Let's try a range of ranks
    
    # First, try to see if we can do better than schoolbook with a search
    # Use a randomized approach: for a given rank, try to find U,V,W
    
    def try_rank(rank, num_iters=500, seed=42):
        """Try to find a rank-r decomposition using iterative refinement."""
        if rank < max(nm, mp, pn):
            return None
        
        rng = random.Random(seed)
        np_rng = np.random.RandomState(seed)
        
        # Initialize with small random values
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        
        for _ in range(num_iters):
            # Initialize randomly
            U = np_rng.randn(nm, rank) * 0.1
            V = np_rng.randn(mp, rank) * 0.1
            W = np_rng.randn(pn, rank) * 0.1
            
            # Iterative refinement: fix two, solve for third
            converged = False
            for it in range(200):
                # Solve for U: for each u_idx, sum_r U[u,r]*V[v,r]*W[w,r] = 1
                # This is a linear system in U[u,:] for each u_idx
                # But it's coupled across different (v,w) pairs...
                # Actually, for a fixed u_idx, the constraint is:
                # sum over all (j,k) where i*m+j=u_idx: U[u,r] * V[j*p+k,r] * W[k*n+i,r] = 1
                # This is not a simple linear system in U[u,:] because of the product V*W
                
                # Alternative: use gradient descent or alternating minimization
                # Let's use a simple approach: for each rank element r, we have
                # U[:,r], V[:,r], W[:,r] as vectors. The constraint is bilinear/trilinear.
                
                # Let's try: fix V and W, solve for U via least squares
                # For each u_idx, we need: sum_r U[u,r] * (V[v,r]*W[w,r]) = 1 for all (v,w) with u_idx=i*m+j
                # This is underdetermined in general
                
                # Simpler: just do gradient descent on the loss
                pass
            
            break  # This approach is too complex, let's try something else
        
        return None
    
    # Alternative approach: use the fact that we can express the problem as
    # finding matrices U, V, W such that the trilinear form matches.
    # 
    # Let's try a different strategy: use known results or structured decompositions.
    #
    # For 4x4x8 matrix multiplication, one approach is to note that 8 = 4+4,
    # so we can split B into two 4x4 blocks: B = [B1, B2]
    # Then C = [A*B1, A*B2]
    # Each A*B1 and A*B2 is 4x4x4 matrix multiplication.
    # If we can do 4x4x4 in rank r, then 4x4x8 might be doable in rank 2r.
    #
    # Known: 4x4x4 matrix multiplication can be done in rank 49 (Strassen-like)
    # But that's not exact minimal. The schoolbook for 4x4x4 is 64.
    #
    # Actually, let me just try a numerical search more carefully.
    
    # Better approach: set up the problem as a system and use numpy to solve
    # For a given rank r, we have r^3 * (nm + mp + pn) variables... no.
    # We have nm*r + mp*r + pn*r variables and nm*mp*pn... no, we have num_triples equations.
    
    # Actually, the number of equations is num_triples = 128, and the number of
    # unknowns is (nm + mp + pn) * r = (16 + 32 + 32) * r = 80r.
    # For r >= 2, we have more unknowns than equations, so solutions may exist.
    
    # Let's try a direct approach: for small r, set up and solve.
    
    def solve_for_rank(r, max_time=10):
        """Try to find a rank-r decomposition by solving the system."""
        t0 = time.time()
        
        # Variables: U (16 x r), V (32 x r), W (32 x r)
        # Total variables: 16r + 32r + 32r = 80r
        # Equations: 128 (one for each (i,j,k) triple)
        
        # This is a nonlinear system (trilinear). Let's use a different approach.
        
        # Strategy: fix W, then the system becomes bilinear in U and V.
        # For each r, define a_r = W[:, r] (32-vector)
        # Then for each (i,j,k): sum_r U[i*m+j, r] * V[j*p+k, r] * a_r[k*n+i] = 1
        
        # Let's try: initialize W randomly, then for each "slice" try to solve.
        
        # Actually, let me try a simpler heuristic: 
        # Start with schoolbook and try to merge columns.
        
        # Schoolbook has 128 columns. Can we find linear dependencies?
        # Column c corresponds to (i,j,k). The column is:
        # U_col: 1 at position i*m+j, 0 elsewhere
        # V_col: 1 at position j*p+k, 0 elsewhere  
        # W_col: 1 at position k*n+i, 0 elsewhere
        
        # Two columns c1=(i1,j1,k1) and c2=(i2,j2,k2) are "mergeable" if
        # we can find alpha, beta such that the sum still works... but that's
        # not straightforward because the trilinear form needs to remain 1.
        
        # Let me just return the schoolbook for now and add a simple search.
        
        return None
    
    # For now, let's just return the schoolbook decomposition.
    # But let's try one thing: can we use the structure to reduce rank?
    
    # Idea: group by i. For fixed i, we have a 4x4 * 4x8 = 4x8 multiplication.
    # That's 4*4*8 = 128 operations per i, but they share the A[i,:] row.
    # Not obviously helpful.
    
    # Let me try a computational search with limited time.
    # I'll try ranks from 100 down to 32, and for each, do a few random initializations
    # with gradient-based optimization.
    
    def gradient_search(rank, num_restarts=10, iters_per_restart=500, lr=0.01):
        """Try to find rank-r decomposition via gradient descent."""
        np_rng = np.random.RandomState(rank * 7 + 13)
        
        best_loss = float('inf')
        best_U, best_V, best_W = None, None, None
        
        for restart in range(num_restarts):
            if time.time() - start_time > deadline:
                break
                
            U = np_rng.randn(nm, rank) * 0.5
            V = np_rng.randn(mp, rank) * 0.5
            W = np_rng.randn(pn, rank) * 0.5
            
            for it in range(iters_per_restart):
                if time.time() - start_time > deadline:
                    break
                    
                # Compute loss and gradients
                # Loss = sum over triples of (sum_r U[u,r]*V[v,r]*W[w,r] - 1)^2
                # This is expensive to compute directly. Let's sample.
                
                # Compute all trilinear products
                # For each r, compute outer products
                loss = 0.0
                dU = np.zeros_like(U)
                dV = np.zeros_like(V)
                dW = np.zeros_like(W)
                
                for (u_idx, v_idx, w_idx) in triples:
                    # val = sum_r U[u,r] * V[v,r] * W[w,r]
                    val = np.dot(U[u_idx], V[v_idx] * W[w_idx])
                    err = val - 1.0
                    loss += err * err
                    # dU[u,r] += 2*err * V[v,r] * W[w,r]
                    # dV[v,r] += 2*err * U[u,r] * W[w,r]
                    # dW[w,r] += 2*err * U[u,r] * V[v,r]
                    dU[u_idx] += 2 * err * (V[v_idx] * W[w_idx])
                    dV[v_idx] += 2 * err * (U[u_idx] * W[w_idx])
                    dW[w_idx] += 2 * err * (U[u_idx] * V[v_idx])
                
                if loss < best_loss:
                    best_loss = loss
                    best_U = U.copy()
                    best_V = V.copy()
                    best_W = W.copy()
                
                if loss < 1e-8:
                    break
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            if best_loss < 1e-6:
                break
        
        if best_loss < 1e-6:
            # Convert to lists
            U_list = [[float(x) for x in row] for row in best_U]
            V_list = [[float(x) for x in row] for row in best_V]
            W_list = [[float(x) for x in row] for row in best_W]
            return U_list, V_list, W_list, best_loss
        return None
    
    # Try a few ranks
    for r in [128, 120, 112, 104, 96, 88, 80, 72, 64, 56, 48, 40, 32]:
        if time.time() - start_time > deadline:
            break
        
        result = gradient_search(r, num_restarts=5, iters_per_restart=200, lr=0.005)
        if result is not None:
            U_list, V_list, W_list, loss = result
            if verify_numpy(U_list, V_list, W_list, r):
                return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback to schoolbook
    return schoolbook()
# EVOLVE-BLOCK-END
