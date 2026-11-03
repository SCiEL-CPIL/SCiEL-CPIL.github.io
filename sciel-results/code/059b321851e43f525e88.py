import numpy as np
import time

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Build the target tensor as a list of (a,b,c) indices that should be 1
    target_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                target_indices.append((a,b,c))
    
    # Try to find a low-rank decomposition using ALS
    # Start with schoolbook rank and try to reduce
    
    best_U, best_V, best_W = None, None, None
    best_rank = n*m*p  # schoolbook
    
    # Schoolbook solution
    rank0 = n*m*p
    U0 = np.zeros((nm, rank0))
    V0 = np.zeros((mp, rank0))
    W0 = np.zeros((pn, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i*m + j
                b = j*p + k
                c = k*n + i
                U0[a, col] = 1
                V0[b, col] = 1
                W0[c, col] = 1
                col += 1
    
    # Verify schoolbook
    def check_decomp(U, V, W, tol=1e-6):
        """Check if U,V,W exactly decompose the tensor."""
        # Check all target indices give 1
        for (a,b,c) in target_indices:
            val = np.dot(U[a,:], V[b,:] * W[c,:])
            if abs(val - 1.0) > tol:
                return False
        # Check some random non-target indices give 0
        # Actually need to check ALL non-target indices
        # This is expensive for large tensors, so we'll be careful
        all_indices = set(target_indices)
        for a in range(nm):
            for b in range(mp):
                for c in range(pn):
                    if (a,b,c) not in all_indices:
                        val = np.dot(U[a,:], V[b,:] * W[c,:])
                        if abs(val) > tol:
                            return False
        return True
    
    # The full check is 8*20*10 = 1600 checks, each O(rank)
    # For rank 40, that's 64000 ops - fine
    
    best_rank = rank0
    best_U, best_V, best_W = U0, V0, W0
    
    # Try to reduce rank by trying ranks from lower values
    # Use ALS optimization
    deadline = time.time() + 140  # safety margin
    
    # Try various ranks
    for target_rank in range(15, rank0+1):  # try from 15 up to schoolbook
        if time.time() > deadline:
            break
        
        # Initialize randomly
        np.random.seed(42)
        U = np.random.randn(nm, target_rank)
        V = np.random.randn(mp, target_rank)
        W = np.random.randn(pn, target_rank)
        
        # ALS iterations
        for iteration in range(200):
            if time.time() > deadline:
                break
            
            # Update U: for each row a, U[a,:] should satisfy
            # U[a,:] dot (V[b,:] * W[c,:]) = T[a,b,c] for all b,c
            # This is a linear system: U[a,:] @ (V*W)[b,c,:]^T = T[a,b,c]
            # Where (V*W)[b,c,:] = V[b,:] * W[c,:] (element-wise)
            
            # Build the design matrix for U
            # For row a, we need: for each (b,c), U[a,:] dot (V[b,:]*W[c,:]) = T[a,b,c]
            # This is: U[a,:] @ M_a = t_a where M_a is a matrix with rows (V[b,:]*W[c,:])
            
            # Actually, let's do it more efficiently
            # Fix V and W, update U:
            # For each a, minimize sum_{b,c} (U[a,:] dot (V[b,:]*W[c,:]) - T[a,b,c])^2
            # This is a least squares problem
            
            # Build the matrix: rows are V[b,:]*W[c,:] for all b,c
            # That's mp*pn rows, each of length target_rank
            # mp*pn = 20*10 = 200 rows
            # This is fine
            
            VW = np.zeros((mp*pn, target_rank))
            for b in range(mp):
                for c in range(pn):
                    VW[b*pn+c, :] = V[b,:] * W[c,:]
            
            # For each a, solve U[a,:] @ VW^T = t_a
            # Where t_a[b*pn+c] = T[a,b,c]
            for a in range(nm):
                t_a = np.zeros(mp*pn)
                for (aa,bb,cc) in target_indices:
                    if aa == a:
                        t_a[bb*pn+cc] = 1.0
                # Solve U[a,:] @ VW^T = t_a
                # This is: VW @ U[a,:]^T = t_a
                U[a,:], _, _, _ = np.linalg.lstsq(VW, t_a, rcond=None)
            
            # Similarly update V and W
            UW = np.zeros((nm*pn, target_rank))
            for a in range(nm):
                for c in range(pn):
                    UW[a*pn+c, :] = U[a,:] * W[c,:]
            
            for b in range(mp):
                t_b = np.zeros(nm*pn)
                for (aa,bb,cc) in target_indices:
                    if bb == b:
                        t_b[aa*pn+cc] = 1.0
                V[b,:], _, _, _ = np.linalg.lstsq(UW, t_b, rcond=None)
            
            UV = np.zeros((nm*mp, target_rank))
            for a in range(nm):
                for b in range(mp):
                    UV[a*mp+b, :] = U[a,:] * V[b,:]
            
            for c in range(pn):
                t_c = np.zeros(nm*mp)
                for (aa,bb,cc) in target_indices:
                    if cc == c:
                        t_c[aa*mp+bb] = 1.0
                W[c,:], _, _, _ = np.linalg.lstsq(UV, t_c, rcond=None)
        
        # Check if we found a valid decomposition
        if check_decomp(U, V, W, tol=1e-4):
            best_rank = target_rank
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists with rounding to half-integers
    def round_to_half(x):
        return round(2*x)/2
    
    U_list = [[round_to_half(best_U[i,j]) for j in range(best_rank)] for i in range(nm)]
    V_list = [[round_to_half(best_V[i,j]) for j in range(best_rank)] for i in range(mp)]
    W_list = [[round_to_half(best_W[i,j]) for j in range(best_rank)] for i in range(pn)]
    
    # Verify the rounded solution
    # Re-check with exact arithmetic using the rounded values
    def check_exact(Ul, Vl, Wl, target_idx):
        for (a,b,c) in target_idx:
            s = 0
            for t in range(len(Ul[0])):
                s += Ul[a][t] * Vl[b][t] * Wl[c][t]
            if abs(s - 1.0) > 1e-9:
                return False
        # Check zeros
        target_set = set(target_idx)
        for a in range(len(Ul)):
            for b in range(len(Vl)):
                for c in range(len(Wl)):
                    if (a,b,c) not in target_set:
                        s = 0
                        for t in range(len(Ul[0])):
                            s += Ul[a][t] * Vl[b][t] * Wl[c][t]
                        if abs(s) > 1e-9:
                            return False
        return True
    
    if not check_exact(U_list, V_list, W_list, target_indices):
        # Fall back to schoolbook
        U_list = [[0]*rank0 for _ in range(nm)]
        V_list = [[0]*rank0 for _ in range(mp)]
        W_list = [[0]*rank0 for _ in range(pn)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_list[i*m+j][col] = 1
                    V_list[j*p+k][col] = 1
                    W_list[k*n+i][col] = 1
                    col += 1
        best_rank = rank0
    
    return {"U": U_list, "V": V_list, "W": W_list}
