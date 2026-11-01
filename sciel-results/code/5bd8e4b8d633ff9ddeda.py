"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 3, 4
    deadline = time.time() + 140  # 10s safety margin before 150s kill
    
    # Build the tensor: T[(i,j,k)] = 1 for all valid (i,j,k)
    # Decomposition: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    # Start with schoolbook rank-24 decomposition
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        r = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][r] = 1
                    V[j * p + k][r] = 1
                    W[k * n + i][r] = 1
                    r += 1
        return U, V, W, rank
    
    U, V, W, rank = schoolbook()
    
    # Precompute the tensor entries as list of (i,j,k) triples
    triples = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    def verify(U, V, W):
        """Check if decomposition is exact."""
        for i, j, k in triples:
            s = 0
            for r in range(len(U[0])):
                s += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
            if s != 1:
                return False
        return True
    
    def try_merge(U, V, W, r1, r2):
        """Try to merge terms r1 and r2 into a single term.
        Returns (success, new_U, new_V, new_W, new_rank) or None."""
        rank = len(U[0])
        if not (0 <= r1 < rank and 0 <= r2 < rank and r1 != r2):
            return None
        
        # Collect the rows involved
        # U rows: i*m+j for all i,j
        # V rows: j*p+k for all j,k
        # W rows: k*n+i for all k,i
        
        u1 = [U[i * m + j][r1] for i in range(n) for j in range(m)]
        u2 = [U[i * m + j][r2] for i in range(n) for j in range(m)]
        v1 = [V[j * p + k][r1] for j in range(m) for k in range(p)]
        v2 = [V[j * p + k][r2] for j in range(m) for k in range(p)]
        w1 = [W[k * n + i][r1] for k in range(p) for i in range(n)]
        w2 = [W[k * n + i][r2] for k in range(p) for i in range(n)]
        
        # After merge, contribution from these two terms becomes:
        # For each (i,j,k): U_new[i*m+j] * V_new[j*p+k] * W_new[k*n+i]
        # should equal u1[i,j]*v1[j,k]*w1[k,i] + u2[i,j]*v2[j,k]*w2[k,i]
        #
        # Try ansatz: U_new = a*u1 + b*u2, V_new = c*v1 + d*v2, W_new = e*w1 + f*w2
        # where a,c,e and b,d,f are half-integers.
        #
        # This is complex. Instead, try simpler: U_new = u1+u2, V_new = v1+v2, W_new = w1+w2
        # This gives (u1+u2)(v1+v2)(w1+w2) which has extra cross terms.
        #
        # Better approach: try to find a rank-1 tensor that equals the sum.
        # The sum tensor is S[i,j,k] = u1[i,j]*v1[j,k]*w1[k,i] + u2[i,j]*v2[j,k]*w2[k,i]
        # We need S = U_new ⊗ V_new ⊗ W_new (outer product)
        
        # Build S as a 3D array
        S = np.zeros((n, m, p))
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    S[i, j, k] = (u1[i * m + j] * v1[j * p + k] * w1[k * n + i] +
                                   u2[i * m + j] * v2[j * p + k] * w2[k * n + i])
        
        # Check if S has rank 1 (as a 3-tensor, i.e., is an outer product)
        # A 3-tensor is an outer product iff all its 2D slices are rank-1
        # and consistent.
        
        # Flatten and check: if S[i,j,k] = a_i * b_j * c_k, then
        # S[i,j,k]*S[i',j',k'] = S[i,j',k]*S[i',j,k'] etc.
        
        # Simple check: try to find a_i, b_j, c_k
        # Pick a non-zero entry to normalize
        nz = np.argwhere(S != 0)
        if len(nz) == 0:
            return None
        
        # Try each non-zero entry as anchor
        for anchor in nz[:5]:  # Limit to 5 anchors for speed
            ai, aj, ak = anchor
            if S[ai, aj, ak] == 0:
                continue
            
            # Assume S[ai,aj,ak] = 1 by scaling (we'll adjust later)
            # S[i,j,k] / S[ai,aj,ak] = (a_i/a_ai) * (b_j/b_aj) * (c_k/c_ak)
            # Let x_i = a_i/a_ai, y_j = b_j/b_aj, z_k = c_k/c_ak
            # Then S[i,j,k]/S[ai,aj,ak] = x_i * y_j * z_k
            # With x_ai = y_aj = z_ak = 1
            
            x = np.zeros(n)
            y = np.zeros(m)
            z = np.zeros(p)
            x[ai] = 1
            y[aj] = 1
            z[ak] = 1
            
            valid = True
            for i in range(n):
                if i == ai:
                    continue
                # S[i,aj,ak] = x_i * y_aj * z_ak = x_i
                if S[ai, aj, ak] == 0 or S[i, aj, ak] == 0:
                    if S[i, aj, ak] != 0 or S[ai, aj, ak] != 0:
                        valid = False
                        break
                    continue
                x[i] = S[i, aj, ak] / S[ai, aj, ak]
            
            if not valid:
                continue
            
            for j in range(m):
                if j == aj:
                    continue
                if S[ai, aj, ak] == 0 or S[ai, j, ak] == 0:
                    if S[ai, j, ak] != 0 or S[ai, aj, ak] != 0:
                        valid = False
                        break
                    continue
                y[j] = S[ai, j, ak] / S[ai, aj, ak]
            
            if not valid:
                continue
            
            for k in range(p):
                if k == ak:
                    continue
                if S[ai, aj, ak] == 0 or S[ai, aj, k] == 0:
                    if S[ai, aj, k] != 0 or S[ai, aj, ak] != 0:
                        valid = False
                        break
                    continue
                z[k] = S[ai, aj, k] / S[ai, aj, ak]
            
            if not valid:
                continue
            
            # Verify reconstruction
            recon_ok = True
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        expected = S[i, j, k]
                        actual = x[i] * y[j] * z[k] * S[ai, aj, ak]
                        if abs(expected - actual) > 1e-10:
                            recon_ok = False
                            break
                    if not recon_ok:
                        break
                if not recon_ok:
                    break
            
            if not recon_ok:
                continue
            
            # Now we have S = (S[ai,aj,ak]) * x ⊗ y ⊗ z
            # But we need S = U_new ⊗ V_new ⊗ W_new exactly with half-integers
            # U_new[i*m+j] = a * x[i] for some scale... wait, this doesn't decompose cleanly
            # because U_new depends on both i and j, but x only depends on i.
            
            # This approach only works if S is a "separable" tensor.
            # For general merges, this won't work. Let me try a different approach.
            break
        
        # Alternative: brute-force search for half-integer coefficients
        # U_new[i*m+j] = a*u1[i*m+j] + b*u2[i*m+j] for half-integers a,b
        # Similarly for V_new, W_new
        # This is too many variables.
        
        # Practical approach: try specific linear combinations
        # Case 1: U_new = u1 + u2, V_new = v1 + v2, W_new = w1 + w2
        # Case 2: U_new = u1 + u2, V_new = v1 + v2, W_new = w1 - w2
        # etc.
        
        for a in [-1, -0.5, 0.5, 1]:
            for b in [-1, -0.5, 0.5, 1]:
                u_new = [a * u1[idx] + b * u2[idx] for idx in range(n * m)]
                # Check if u_new has all half-integers
                if any(abs(x * 2 - round(x * 2)) > 1e-10 for x in u_new):
                    continue
                
                for c in [-1, -0.5, 0.5, 1]:
                    for d in [-1, -0.5, 0.5, 1]:
                        v_new = [c * v1[idx] + d * v2[idx] for idx in range(m * p)]
                        if any(abs(x * 2 - round(x * 2)) > 1e-10 for x in v_new):
                            continue
                        
                        for e in [-1, -0.5, 0.5, 1]:
                            for f in [-1, -0.5, 0.5, 1]:
                                w_new = [e * w1[idx] + f * w2[idx] for idx in range(p * n)]
                                if any(abs(x * 2 - round(x * 2)) > 1e-10 for x in w_new):
                                    continue
                                
                                # Check if this combination reproduces the sum
                                ok = True
                                for i, j, k in triples:
                                    expected = (u1[i * m + j] * v1[j * p + k] * w1[k * n + i] +
                                                 u2[i * m + j] * v2[j * p + k] * w2[k * n + i])
                                    actual = u_new[i * m + j] * v_new[j * p + k] * w_new[k * n + i]
                                    if abs(expected - actual) > 1e-10:
                                        ok = False
                                        break
                                
                                if ok:
                                    # Create new matrices with merged terms
                                    new_U = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in U]
                                    new_V = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in V]
                                    new_W = [row[:r1] + row[r1+1:r2] + row[r2+1:] for row in W]
                                    
                                    # Insert new term at position r1
                                    new_U = [new_U[r] for r in range(n * m)]
                                    for idx in range(n * m):
                                        new_U[idx] = new_U[idx][:r1] + [u_new[idx]] + new_U[idx][r1:]
                                    
                                    new_V = [new_V[r] for r in range(m * p)]
                                    for idx in range(m * p):
                                        new_V[idx] = new_V[idx][:r1] + [v_new[idx]] + new_V[idx][r1:]
                                    
                                    new_W = [new_W[r] for r in range(p * n)]
                                    for idx in range(p * n):
                                        new_W[idx] = new_W[idx][:r1] + [w_new[idx]] + new_W[idx][r1:]
                                    
                                    # Remove term r2 (which is now at position r2-1)
                                    new_U = [row[:r2-1] + row[r2:] for row in new_U]
                                    new_V = [row[:r2-1] + row[r2:] for row in new_V]
                                    new_W = [row[:r2-1] + row[r2:] for row in new_W]
                                    
                                    return (True, new_U, new_V, new_W, rank - 1)
        
        return None
    
    # Greedy merge loop
    max_iter = 200
    for iteration in range(max_iter):
        if time.time() > deadline:
            break
        if rank <= 1:
            break
        
        merged = False
        # Try all pairs
        for r1 in range(rank):
            if time.time() > deadline:
                break
            for r2 in range(r1 + 1, rank):
                result = try_merge(U, V, W, r1, r2)
                if result is not None and result[0]:
                    U, V, W, rank = result[1], result[2], result[3], result[4]
                    merged = True
                    break
            if merged:
                break
        
        if not merged:
            # Try random perturbation: add noise to some entries and try again
            # This is a simplified local search
            if iteration % 10 == 0 and rank > 12:
                # Add a random term and try to remove a different one
                pass  # Simplified for now
    
    # Final verification
    if not verify(U, V, W):
        # Fallback to schoolbook
        U, V, W, rank = schoolbook()
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
