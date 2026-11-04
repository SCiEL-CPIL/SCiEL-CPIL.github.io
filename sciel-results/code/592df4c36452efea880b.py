import numpy as np
import time
from itertools import product

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110.0

    # Index mapping
    # U[i*m + j, r] for i in range(n), j in range(m)
    # V[j*p + k, r] for j in range(m), k in range(p)  
    # W[k*n + i, r] for k in range(p), i in range(n)
    
    def verify(U, V, W):
        """Verify that sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r] == 1 for all valid (i,j,k)"""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False
        # Also check that invalid entries are 0
        # T[a,b,c] should be 0 when the indices don't form a valid (i,j,k)
        # a = i*m+j, b = j'*p+k, c = k'*n+i'
        # Valid iff j == j' and k == k' and i == i'
        for a in range(n*m):
            for b in range(m*p):
                for c in range(p*n):
                    i = a // m
                    j = a % m
                    j2 = b // p
                    k = b % p
                    k2 = c // n
                    i2 = c % n
                    if j == j2 and k == k2 and i == i2:
                        continue  # Should be 1, already checked
                    # Should be 0
                    s = 0.0
                    for r in range(rank):
                        s += U[a, r] * V[b, r] * W[c, r]
                    if abs(s) > 1e-9:
                        return False
        return True

    def snap_to_half_int(U, V, W):
        """Snap values to nearest half-integer"""
        U_s = np.round(U * 2) / 2.0
        V_s = np.round(V * 2) / 2.0
        W_s = np.round(W * 2) / 2.0
        return U_s, V_s, W_s

    # Start with the known rank-17 solution
    rank17 = 17
    U17 = np.zeros((n*m, rank17))
    V17 = np.zeros((m*p, rank17))
    W17 = np.zeros((p*n, rank17))
    
    # Strassen 2x2x2 for i in {0,1}, j in {0,1}, k in {0,1}
    # Col 0: (A00+A11)(B00+B11) -> contributes to C00, C01, C10, C11
    U17[0,0]=1; U17[4,0]=1  # A00, A11
    V17[0,0]=1; V17[4,0]=1  # B00, B11
    W17[0,0]=1; W17[3,0]=1  # C00, C10
    
    # Col 1: (A10+A11)B00 -> C10, C11
    U17[3,1]=1; U17[4,1]=1
    V17[0,1]=1
    W17[1,1]=1; W17[3,1]=-1
    
    # Col 2: A00(B01-B11) -> C01, C11
    U17[0,2]=1
    V17[1,2]=1; V17[4,2]=-1
    W17[2,2]=1; W17[3,2]=1
    
    # Col 3: A11(B10-B00) -> C10, C11
    U17[4,3]=1
    V17[3,3]=1; V17[0,3]=-1
    W17[0,3]=1; W17[1,3]=1
    
    # Col 4: (A00+A01)B11 -> C01, C11
    U17[0,4]=1; U17[1,4]=1
    V17[4,4]=1
    W17[0,4]=-1; W17[2,4]=1
    
    # Col 5: (A10-A00)(B00+B01) -> C10, C11
    U17[3,5]=1; U17[0,5]=-1
    V17[0,5]=1; V17[1,5]=1
    W17[3,5]=1
    
    # Col 6: (A01-A11)(B10+B11) -> C01, C11
    U17[1,6]=1; U17[4,6]=-1
    V17[3,6]=1; V17[4,6]=1
    W17[0,6]=1
    
    # 2x1x1 for j in {0,1}, k=2: C02 = A00*B02 + A01*B12, C12 = A10*B02 + A11*B12
    # Col 7: A00*B02
    U17[0,7]=1
    V17[2,7]=1
    W17[4,7]=1
    
    # Col 8: A01*B12
    U17[1,8]=1
    V17[5,8]=1
    W17[4,8]=1
    
    # Col 9: A10*B02
    U17[3,9]=1
    V17[2,9]=1
    W17[5,9]=1
    
    # Col 10: A11*B12
    U17[4,10]=1
    V17[5,10]=1
    W17[5,10]=1
    
    # 2x1x3 for j=2: C0k = A02*B2k, C1k = A12*B2k for k in {0,1,2}
    # Col 11: A02*B20
    U17[2,11]=1
    V17[6,11]=1
    W17[0,11]=1
    
    # Col 12: A02*B21
    U17[2,12]=1
    V17[7,12]=1
    W17[2,12]=1
    
    # Col 13: A02*B22
    U17[2,13]=1
    V17[8,13]=1
    W17[4,13]=1
    
    # Col 14: A12*B20
    U17[5,14]=1
    V17[6,14]=1
    W17[1,14]=1
    
    # Col 15: A12*B21
    U17[5,15]=1
    V17[7,15]=1
    W17[3,15]=1
    
    # Col 16: A12*B22
    U17[5,16]=1
    V17[8,16]=1
    W17[5,16]=1
    
    # Verify rank 17
    if not verify(U17, V17, W17):
        raise ValueError("Rank 17 solution verification failed")
    
    best_U, best_V, best_W = U17.copy(), V17.copy(), W17.copy()
    best_rank = 17

    # Try to find rank 16 by systematically trying to merge columns
    # Strategy: try to express one of the simple product columns as a combination
    # of Strassen columns plus one new column
    
    # The key insight: for the 2x1x1 block, we have 4 terms:
    # C02 = A00*B02 + A01*B12
    # C12 = A10*B02 + A11*B12
    # 
    # Can we do this in 3 terms? Let's try:
    # Term A: (A00+A10)*(B02) -> gives A00*B02 + A10*B02
    # Term B: (A01+A11)*(B12) -> gives A01*B12 + A11*B12  
    # Term C: correction term to fix cross terms
    #
    # But we need C02 to only have A00*B02 + A01*B12, not A10*B02
    # And C12 to only have A10*B02 + A11*B12, not A01*B12
    #
    # Alternative: use the Strassen structure more cleverly
    # What if we modify one Strassen column to also contribute to the k=2 block?
    
    # Try: modify col 0 to include B02 and B12 contributions
    # Original col 0: (A00+A11)(B00+B11) -> C00, C10
    # Modified: (A00+A11)(B00+B11+alpha*B02+beta*B12)
    # This would add (A00+A11)*alpha*B02 to C00 and C10, which is wrong
    # Unless we also modify W to cancel these in C00 and C10
    
    # This is getting complex. Let me try a different approach:
    # Use randomized search with the constraint that entries are in {-1, 0, 1, 0.5, -0.5}
    
    # Actually, let me try a more targeted approach:
    # The rank 16 solution, if it exists, must have a specific structure.
    # Let me try to solve the system of equations directly for rank 16.
    
    # For rank 16, we have 16 columns. The constraints are:
    # For each (i,j,k) in the 18 valid positions: sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] = 1
    # For each invalid (a,b,c): sum_r U[a,r]*V[b,r]*W[c,r] = 0
    # Total: 18 + (6*9*6 - 18) = 18 + 288 = 306 equations
    # Unknowns: 16*(6+9+6) = 16*21 = 336
    # So we have more unknowns than equations, suggesting solutions may exist.
    
    # Let me try a local search starting from the rank-17 solution with one column removed
    
    def try_remove_column(U, V, W, col_to_remove, deadline):
        """Try to find a rank-16 solution by removing one column and adjusting others"""
        rank = U.shape[1]
        if rank != 17:
            return None
        
        # Create rank-16 matrices by removing the specified column
        U16 = np.delete(U, col_to_remove, axis=1)
        V16 = np.delete(V, col_to_remove, axis=1)
        W16 = np.delete(W, col_to_remove, axis=1)
        
        # Now try to adjust the remaining columns to compensate
        # Use gradient descent with half-integer snapping
        
        def compute_residual(Ur, Vr, Wr):
            """Compute the residual tensor"""
            r = Ur.shape[1]
            residual = np.zeros((n*m, m*p, p*n))
            for a in range(n*m):
                for b in range(m*p):
                    for c in range(p*n):
                        s = 0.0
                        for rr in range(r):
                            s += Ur[a, rr] * Vr[b, rr] * Wr[c, rr]
                        # Determine target
                        i = a // m
                        j = a % m
                        j2 = b // p
                        k = b % p
                        k2 = c // n
                        i2 = c % n
                        if j == j2 and k == k2 and i == i2:
                            target = 1.0
                        else:
                            target = 0.0
                        residual[a, b, c] = s - target
            return residual
        
        def compute_loss(Ur, Vr, Wr):
            res = compute_residual(Ur, Vr, Wr)
            return np.sum(res**2)
        
        # Gradient descent
        lr = 0.01
        for step in range(2000):
            if time.time() > deadline:
                break
            
            loss = compute_loss(U16, V16, W16)
            if loss < 1e-10:
                U_s, V_s, W_s = snap_to_half_int(U16, V16, W16)
                if verify(U_s, V_s, W_s):
                    return U_s, V_s, W_s
            
            # Compute gradients numerically
            eps = 1e-5
            dU = np.zeros_like(U16)
            dV = np.zeros_like(V16)
            dW = np.zeros_like(W16)
            
            # Only compute a subset of gradients for speed
            # Sample random entries
            np.random.seed(step)
            for _ in range(50):
                idx_u = (np.random.randint(n*m), np.random.randint(16))
                idx_v = (np.random.randint(m*p), np.random.randint(16))
                idx_w = (np.random.randint(p*n), np.random.randint(16))
                
                # Gradient w.r.t. U
                U16[idx_u] += eps
                loss_plus = compute_loss(U16, V16, W16)
                U16[idx_u] -= 2*eps
                loss_minus = compute_loss(U16, V16, W16)
                U16[idx_u] += eps
                dU[idx_u] = (loss_plus - loss_minus) / (2*eps)
                
                # Gradient w.r.t. V
                V16[idx_v] += eps
                loss_plus = compute_loss(U16, V16, W16)
                V16[idx_v] -= 2*eps
                loss_minus = compute_loss(U16, V16, W16)
                V16[idx_v] += eps
                dV[idx_v] = (loss_plus - loss_minus) / (2*eps)
                
                # Gradient w.r.t. W
                W16[idx_w] += eps
                loss_plus = compute_loss(U16, V16, W16)
                W16[idx_w] -= 2*eps
                loss_minus = compute_loss(U16, V16, W16)
                W16[idx_w] += eps
                dW[idx_w] = (loss_plus - loss_minus) / (2*eps)
            
            # Update
            U16 -= lr * dU
            V16 -= lr * dV
            W16 -= lr * dW
            
            # Snap to half-integers periodically
            if step % 50 == 0:
                U_s, V_s, W_s = snap_to_half_int(U16, V16, W16)
                if verify(U_s, V_s, W_s):
                    return U_s, V_s, W_s
        
        return None
    
    # Try removing each of the 17 columns
    for col in range(17):
        if time.time() > deadline - 10:
            break
        
        result = try_remove_column(U17, V17, W17, col, deadline - 5)
        if result is not None:
            U_s, V_s, W_s = result
            best_U, best_V, best_W = U_s, V_s, W_s
            best_rank = 16
            break
    
    # If we didn't find rank 16, return rank 17
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
