import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    target_rank = 50 # Start with schoolbook as fallback
    
    # Precompute the target tensor indices
    # T[u, v, w] = 1 if (u, v, w) is a valid multiplication term
    # u = i*m + j, v = j*p + k, w = k*n + i
    # Valid if u corresponds to (i, j) and v corresponds to (j, k)
    # and w corresponds to (k, i).
    
    # We will attempt to find a decomposition of rank r < 50.
    # Search range: 49 down to 40.
    
    # Deadline guard
    start_time = time.time()
    deadline = 110.0 # Safety margin
    
    # Generate random seeds for reproducibility in this run, but distinct from seed
    # We use random, but fixed seed for deterministic behavior within the run
    random.seed(42)
    np.random.seed(42)

    def verify(U, V, W):
        # Check if U, V, W exactly decompose the tensor
        # Compute the product tensor
        # prod[u, v, w] = sum_t U[u, t] * V[v, t] * W[w, t]
        # We can compute this efficiently using broadcasting
        # U: (nm, r), V: (mp, r), W: (pn, r)
        # Tensor product is expensive for large r, but r is small.
        
        # Iterate over valid triples to check == 1
        # Iterate over invalid triples to check == 0?
        # Checking all 2500 triples is fast.
        
        r = U.shape[1]
        # Compute sum for all triples
        # This is a 3D tensor of size (nm, mp, pn)
        # Sum over t of U[:,t] V[:,t] W[:,t]
        
        # Using einsum or loops
        # Loops are safer for clarity and memory if r is small
        res = np.zeros((nm, mp, pn), dtype=int)
        for t in range(r):
            # Outer product of U[:,t], V[:,t], W[:,t]
            # u_idx, v_idx, w_idx
            res += np.outer(np.outer(U[:, t], V[:, t]), W[:, t])
            
        # Check valid indices
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    if res[u, v, w] != 1:
                        return False
        
        # Check invalid indices are 0
        # We can just check if any other element is non-zero
        # Create a mask of valid indices
        valid_mask = np.zeros((nm, mp, pn), dtype=bool)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u = i * m + j
                    v = j * p + k
                    w = k * n + i
                    valid_mask[u, v, w] = True
        
        # Check if any invalid element is non-zero
        # res is int, so check res[~valid_mask]
        if np.any(res[~valid_mask] != 0):
            return False
            
        return True

    def solve_for_W(U, V):
        # Solve linear system for W given U, V
        # Eq: sum_t U[u,t] V[v,t] W[w,t] = T[u,v,w]
        # Variables: W[w,t] for w in 0..pn-1, t in 0..r-1
        # Number of variables: pn * r
        # Number of equations: nm * mp * pn = 2500
        
        r = U.shape[1]
        num_vars = pn * r
        num_eqs = nm * mp * pn
        
        # We only need to enforce the equations for valid indices (50)
        # and a subset of invalid indices?
        # If we only enforce valid indices, we might have non-zero values elsewhere.
        # We should enforce all equations.
        
        # Construct A matrix and b vector
        # A is sparse.
        # Instead of full matrix, use least squares solver on the full system?
        # 2500 eqs, ~500 vars.
        
        # Let's try to build the system only for the non-zero targets?
        # No, we need zeros elsewhere too.
        
        # Optimization:
        # The equation for (u,v,w) is:
        # sum_t (U[u,t] * V[v,t]) * W[w,t] = T[u,v,w]
        # Note that the coefficient of W[w,t] depends on u,v,w.
        # Specifically, for a fixed w, the variable is W[w,t].
        # The coefficient for W[w,t] in equation (u,v,w) is U[u,t] * V[v,t].
        
        # We can group equations by w.
        # For each w, we have variables W[w, 0], ..., W[w, r-1].
        # But the equations mix different w's? No, each equation (u,v,w) only involves variables W[w, t].
        # Wait, the sum is over t. The variable W[w,t] is specific to w and t.
        # So for a fixed w, the equations that involve W[w,t] are those with that specific w.
        # How many equations have a specific w?
        # w = k*n + i.
        # u = i*m + j. v = j*p + k.
        # Given w (which fixes i,k), u is determined by j (m choices), v is determined by j (m choices).
        # So for a fixed w, there are m equations (one for each j).
        # Total equations = (n*p) * m = n*p*m = 50?
        # Wait, nm*mp*pn = 2500.
        # My grouping logic:
        # Equation index is (u,v,w).
        # Variable index is (w,t).
        # In equation (u,v,w), the variables are W[w, 0..r-1].
        # The coefficients are U[u,t] * V[v,t].
        # So for a fixed w, we have a system of equations for the vector W[w, :].
        # The number of equations for a fixed w is the number of pairs (u,v) that are compatible with w.
        # u = i*m + j. v = j*p + k. w = k*n + i.
        # i, k are fixed by w.
        # j can vary from 0 to m-1.
        # So for each j, we have one equation.
        # So for each w, there are m equations.
        # Total equations = pn * m = 10 * 5 = 50.
        
        # BUT, this only covers the "valid" structural equations?
        # No, the tensor T is zero for invalid (u,v,w).
        # Are there invalid (u,v,w) that share the same w?
        # Yes.
        # u = i*m + j, v = j'*p + k.
        # If j != j', then T[u,v,w] = 0.
        # So for a fixed w (fixed i,k), we have m*m pairs of (j, j').
        # Valid ones: j=j'. (m equations, target 1)
        # Invalid ones: j!=j'. (m^2-m equations, target 0)
        # So for each w, we have m^2 equations for the r variables W[w, :].
        
        # Total equations = pn * m^2 = 10 * 25 = 250.
        # This matches a subset of the 2500.
        # What about the other 2250 equations?
        # They involve (u,v,w) where the indices don't match the "chain" i->j->k->i?
        # No, u,v,w are just indices.
        # u encodes (i1, j1). v encodes (j2, k1). w encodes (k2, i2).
        # The tensor T[u,v,w] is 1 iff i1=i2, j1=j2, k1=k2.
        # So T is 1 only when the indices form a consistent chain.
        # If they don't, T=0.
        
        # My grouping by w assumes w encodes (k2, i2).
        # The equation is sum_t U[u,t]V[v,t]W[w,t] = T[u,v,w].
        # If I fix w, I fix i2, k2.
        # The variables are W[w,t].
        # The coefficients depend on u (i1,j1) and v (j2,k1).
        # So for a fixed w, the variables are the same for all u,v.
        # So yes, we can group by w.
        # For each w, we have a linear system A_w W[w,:] = b_w.
        # A_w has size (m^2 * ... ?) No.
        # The number of equations for fixed w is the number of (u,v) pairs.
        # u can be any of nm. v can be any of mp.
        # So there are nm*mp equations for each w?
        # No, the equation is specific to the triple (u,v,w).
        # If I fix w, I still have freedom in u and v.
        # So yes, for each w, there are nm*mp equations involving the variables W[w,:].
        # Wait, if I fix w, do all u,v combinations produce an equation involving W[w,:]?
        # Yes, the term W[w,t] appears in the sum for ANY u,v, as long as we look at the triple (u,v,w).
        # So for a fixed w, we have a system with nm*mp = 10*25 = 250 equations.
        # And r variables.
        # Total equations = pn * (nm*mp) = 10 * 250 = 2500.
        
        # So we can solve 10 independent linear systems, each of size (250 x r).
        # This is very efficient!
        
        A_list = []
        b_list = []
        
        for w in range(pn):
            A_w = np.zeros((nm * mp, r), dtype=float)
            b_w = np.zeros(nm * mp, dtype=float)
            
            idx = 0
            for u in range(nm):
                for v in range(mp):
                    # Determine T[u,v,w]
                    # Decode u, v, w
                    i1, j1 = divmod(u, m)
                    j2, k1 = divmod(v, p)
                    k2, i2 = divmod(w, n)
                    
                    if i1 == i2 and j1 == j2 and k1 == k2:
                        target = 1.0
                    else:
                        target = 0.0
                    
                    b_w[idx] = target
                    
                    for t in range(r):
                        A_w[idx, t] = U[u, t] * V[v, t]
                    idx += 1
            
            A_list.append(A_w)
            b_list.append(b_w)
        
        # Solve each system using Least Squares
        W_sol = np.zeros((pn, r), dtype=float)
        for w in range(pn):
            # Solve A_w W[w,:] = b_w
            # Use np.linalg.lstsq
            sol, residuals, rank, sv = np.linalg.lstsq(A_list[w], b_list[w], rcond=None)
            W_sol[w, :] = sol
        
        return W_sol

    def round_to_half_integer(W):
        # Multiply by 2, round, divide by 2
        W2 = np.round(W * 2)
        W_r = W2 / 2.0
        return W_r

    # Search loop
    for r in range(49, 39, -1):
        if time.time() > deadline:
            break
            
        # Try several random initializations for U, V
        # We can also use structured initializations
        for trial in range(100):
            if time.time() > deadline:
                break
                
            # Initialize U, V
            # Use small integers to keep numbers manageable
            U = np.random.randint(-1, 2, size=(nm, r))
            V = np.random.randint(-1, 2, size=(mp, r))
            
            # Solve for W
            W_float = solve_for_W(U, V)
            
            # Round W
            W_r = round_to_half_integer(W_float)
            
            # Verify
            if verify(U, V, W_r):
                return {
                    "U": U.tolist(),
                    "V": V.tolist(),
                    "W": W_r.tolist()
                }
                
            # Also try rounding U and V? No, U,V were integers.
            # Maybe try different random seeds or distributions
            
            # If we have time, try a second pass with slightly different U,V
            # Or try structured U,V

    # Fallback: Schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
