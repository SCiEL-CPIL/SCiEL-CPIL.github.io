"""Matrix multiplication 2x4x7 - rank-50 decomposition using Strassen blocks with rank reduction."""
DIMENSIONS = [2, 4, 7]

import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110  # safety margin before 120s limit
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # 8 x rank
    V = [[0] * rank for _ in range(m * p)]  # 28 x rank
    W = [[0] * rank for _ in range(p * n)]  # 14 x rank
    
    col = 0
    
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1: (A[i0,j0]+A[i0+1,j0+1]) * (B[j0,k0]+B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M2: (A[i0+1,j0]+A[i0+1,j0+1]) * B[j0,k0]
        U[(i0+1)*m + j0][col] += 1
        U[(i0+1)*m + (j0+1)][col] += 1
        V[j0*p + k0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] -= 1
        col += 1
        
        # M3: A[i0,j0] * (B[j0,k0+1]-B[j0+1,k0+1])
        U[i0*m + j0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M4: A[i0+1,j0+1] * (B[j0+1,k0]-B[j0,k0])
        U[(i0+1)*m + (j0+1)][col] += 1
        V[(j0+1)*p + k0][col] += 1
        V[j0*p + k0][col] -= 1
        W[k0*n + i0][col] += 1
        W[k0*n + (i0+1)][col] += 1
        col += 1
        
        # M5: (A[i0,j0]+A[i0,j0+1]) * B[j0+1,k0+1]
        U[i0*m + j0][col] += 1
        U[i0*m + (j0+1)][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] -= 1
        W[(k0+1)*n + i0][col] += 1
        col += 1
        
        # M6: (A[i0+1,j0]-A[i0,j0]) * (B[j0,k0]+B[j0,k0+1])
        U[(i0+1)*m + j0][col] += 1
        U[i0*m + j0][col] -= 1
        V[j0*p + k0][col] += 1
        V[j0*p + (k0+1)][col] += 1
        W[(k0+1)*n + (i0+1)][col] += 1
        col += 1
        
        # M7: (A[i0,j0+1]-A[i0+1,j0+1]) * (B[j0+1,k0]+B[j0+1,k0+1])
        U[i0*m + (j0+1)][col] += 1
        U[(i0+1)*m + (j0+1)][col] -= 1
        V[(j0+1)*p + k0][col] += 1
        V[(j0+1)*p + (k0+1)][col] += 1
        W[k0*n + i0][col] += 1
        col += 1

    # 2x4x4: 4 Strassen blocks for k=0..3, j=0..3
    add_strassen(0, 0, 0)  # i=0,1; j=0,1; k=0,1
    add_strassen(0, 2, 0)  # i=0,1; j=2,3; k=0,1
    add_strassen(0, 0, 2)  # i=0,1; j=0,1; k=2,3
    add_strassen(0, 2, 2)  # i=0,1; j=2,3; k=2,3
    
    # 2x4x2: 2 Strassen blocks for k=4..5, j=0..3
    add_strassen(0, 0, 4)  # i=0,1; j=0,1; k=4,5
    add_strassen(0, 2, 4)  # i=0,1; j=2,3; k=4,5
    # 2x4x1: 8 direct terms for k=6, j=0..3, i=0..1
    for j in range(4):
        for i in range(2):
            U[i*m + j][col] = 1
            V[j*p + 6][col] = 1
            W[6*n + i][col] = 1
            col += 1
    
    # Verify decomposition correctness
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-6:
        # Should not happen; return as-is
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction: check if any term contributes zero everywhere
    for r in range(rank):
        if time.time() > deadline:
            break
        contrib = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    contrib += abs(U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r])
        if contrib < 1e-10:
            # This term contributes nothing; remove it
            U = [row[:r] + row[r+1:] for row in U]
            V = [row[:r] + row[r+1:] for row in V]
            W = [row[:r] + row[r+1:] for row in W]
            rank -= 1
            break
    
    return {"U": U, "V": V, "W": W}
    # Verify decomposition is correct
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    # Check: for each (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == delta
    # We only need to check that it equals 1 when i,j,k are "connected" by the multiplication
    # Actually, the tensor for 2x4x7 mult: T[i,j,k] = 1 if the (i,j) entry of A times (j,k) entry of B
    # contributes to (i,k) entry of C. So T[i,j,k] = 1 for all valid (i,j,k) since it's the identity
    # mapping. Wait no - the multiplication tensor has T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k]
    # So T[i,j,k] = 1 for all i in [0,2), j in [0,4), k in [0,7) - that's the full tensor.
    
    # Verify
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for r in range(rank):
                    val += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                expected = 1.0
                err = abs(val - expected)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-6:
        # Decomposition is wrong, return as-is (shouldn't happen)
        return {"U": U, "V": V, "W": W}
    
    # Attempt rank reduction: check if any term is linearly dependent
    # A term r is redundant if u_r ⊗ v_r ⊗ w_r = sum_{s≠r} c_s u_s ⊗ v_s ⊗ w_s
    # Practical check: for each r, see if removing it still gives a valid rank-(rank-1) decomposition
    # by checking if the tensor can be represented without term r.
    
    # Simpler check: see if any column of (U,V,W) is all zeros
    for r in range(rank):
        if all(U[i][r] == 0 for i in range(n*m)) and \
           all(V[i][r] == 0 for i in range(m*p)) and \
           all(W[i][r] == 0 for i in range(p*n)):
            # Remove this term
            U = [row[:r] + row[r+1:] for row in U]
            V = [row[:r] + row[r+1:] for row in V]
            W = [row[:r] + row[r+1:] for row in W]
            rank -= 1
            break
    
    # Try to find linear dependencies among the rank-1 terms
    # For each pair (r1, r2), check if u_r1 = a*u_r2 and v_r1 = b*v_r2 and w_r1 = c*w_r2
    # with a*b*c = 1 (then they're the same term up to scaling, redundant)
    # Or if u_r1 = a*u_r2, v_r1 = b*v_r2, w_r1 = c*w_r2 with a*b*c != 1 (then one is a multiple)
    
    # More useful: check if the 50 vectors (u_r, v_r, w_r) have any linear dependency
    # that preserves the tensor. Use a greedy approach:
    # For each r, check if T - u_r⊗v_r⊗w_r can be written as sum of 49 rank-1 terms
    # from the remaining set. This is equivalent to checking if u_r⊗v_r⊗w_r is in the
    # span of {u_s⊗v_s⊗w_s : s ≠ r}.
    
    # Practical test: fix U and V, then W is determined by the tensor equation.
    # If we remove term r, we need to check if the system is still consistent with rank-1 W.
    
    # Alternative: try to find a linear combination of terms that sums to zero.
    # If sum_r c_r u_r⊗v_r⊗w_r = 0 with not all c_r = 0, then we can reduce rank.
    
    # Check using the "slice" method: for each k, the matrix T[:,:,k] = sum_r W[k*n+i, r] * u_r * v_r^T
    # Wait, that's not quite right. Let me think more carefully.
    
    # T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # For fixed k: T[:,:,k] = sum_r W[k,i,r] * (U[:,r] outer V[:,r])... no, that's not right either.
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For fixed k and i: T[i,:,k] = sum_r W[k*n+i, r] * U[i*m+0:r... ] * V[0*p+k:r...]
    # This is getting complicated. Let me just try a simple heuristic:
    # Check if any two terms are "similar" (same support pattern up to scaling).
    
    if time.time() < deadline - 5:
        # Try to find a rank reduction by checking linear dependencies
        # Form the matrix M where M[r1, r2] = <u_r1, u_r2> * <v_r1, v_r2> * <w_r1, w_r2>
        # If M is singular, there might be a dependency.
        
        U_f = np.array(U, dtype=float)
        V_f = np.array(V, dtype=float)
        W_f = np.array(W, dtype=float)
        
        # Check if the 50 rank-1 tensors are linearly independent
        # Flatten each term into a vector of size (n*m)*(m*p)*(p*n) = 8*28*14 = 3136
        # That's too large. Instead, use a random projection.
        
        # Random projection: pick random vectors a (size n*m), b (size m*p), c (size p*n)
        # Project each term: proj_r = a^T u_r * b^T v_r * c^T w_r
        # If the 50 projected values have a linear dependency, the original terms might too.
        
        rng = np.random.RandomState(42)
        found_reduction = False
        
        for trial in range(50):
            a = rng.randn(n*m)
            b = rng.randn(m*p)
            c = rng.randn(p*n)
            
            # Projected values for each term
            proj = np.zeros(rank)
            for r in range(rank):
                proj[r] = np.dot(a, U_f[:, r]) * np.dot(b, V_f[:, r]) * np.dot(c, W_f[:, r])
            
            # Check if proj is linearly dependent (should be full rank if terms are independent)
            # But this only checks 1-dimensional projection, not the full tensor.
            # This won't work well. Let me try a different approach.
            break
        
        # Try a more direct approach: check if removing each term and re-solving gives a valid decomposition
        # For each r, remove term r and check if the remaining 49 terms still decompose the tensor.
        # This is expensive but let's try a few.
        
        for r_check in range(min(rank, 10)):  # Only check first 10 for time
            if time.time() > deadline - 10:
                break
            
            # Remove term r_check
            U_rem = np.delete(U_f, r_check, axis=1)
            V_rem = np.delete(V_f, r_check, axis=1)
            W_rem = np.delete(W_f, r_check, axis=1)
            
            # Check if sum_{s≠r} U_rem[i,j,s] * V_rem[j,k,s] * W_rem[k,i,s] == 1 for all (i,j,k)
            # This is equivalent to checking if the removed term contributed 0 everywhere,
            # which means the original decomposition was not minimal.
            
            # Quick check: compute the residual
            residual_max = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0.0
                        for s in range(rank - 1):
                            val += U_rem[i*m+j, s] * V_rem[j*p+k, s] * W_rem[k*n+i, s]
                        err = abs(val - 1.0)
                        if err > residual_max:
                            residual_max = err
            
            if residual_max < 1e-6:
                # Term r_check is redundant!
                U = [row[:r_check] + row[r_check+1:] for row in U]
                V = [row[:r_check] + row[r_check+1:] for row in V]
                W = [row[:r_check] + row[r_check+1:] for row in W]
                rank -= 1
                found_reduction = True
                break
    
    return {"U": U, "V": V, "W": W}
