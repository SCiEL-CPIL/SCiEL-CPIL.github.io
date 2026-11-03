"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def solve(a, b):
    n = len(a)
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    # Ensure int64 for exact arithmetic
    a = a.astype(np.int64)
    b = b.astype(np.int64)
    
    def calc_cost(p):
        b_perm = b[:, p][:, p]
        return int(np.sum(a * b_perm))
    
    # Precompute for fast vectorized delta calculation
    # dot_row[i] = a[i, :] @ B_p (shape n) -- sum over k: a[i,k]*B_p[k,:]
    # dot_col[i] = B_p @ a[:, i] (shape n) -- sum over k: B_p[:,k]*a[k,i]
    # After swap of i,j (locations u=p[i], v=p[j]):
    #   B_p changes: rows u,v swap content; cols i,j swap content
    #   dot_row and dot_col can be updated incrementally in O(n)
    
    def compute_dot_products(p):
        """Compute dot_row[i] = a[i,:] @ B_p and dot_col[i] = B_p @ a[:,i]"""
        B_p = b[:, p][:, p]
        # dot_row[i] = a[i,:] @ B_p = sum_k a[i,k] * B_p[k,:]  -> shape (n,)
        dot_row = a @ B_p  # (n,n) @ (n,n) -> (n,n), dot_row[i] is row i
        # dot_col[i] = B_p @ a[:,i] = sum_k B_p[:,k] * a[k,i] -> shape (n,)
        # dot_col[i][j] = sum_k B_p[j,k] * a[k,i] = (B_p @ a[:,i])[j]
        # So dot_col as a matrix: dot_col_mat[j,i] = sum_k B_p[j,k]*a[k,i] = (B_p @ a)[j,i]
        dot_col = B_p @ a  # (n,n) @ (n,n) -> (n,n), dot_col[:,i] is col i
        return B_p, dot_row, dot_col
    
    def get_delta_vectorized(i, j, p, B_p, dot_row, dot_col):
        """Compute delta for swapping facilities i and j (at locations u, v)"""
        u = p[i]
        v = p[j]
        
        # delta = (a[i]-a[j]) @ (B_p[v,:]-B_p[u,:]) + (B_p[:,v]-B_p[:,u]) @ (a[:,i]-a[:,j])
        #       + corner corrections
        # 
        # Using dot products:
        # (a[i]-a[j]) @ B_p[v,:] = dot_row[i,v] - dot_row[j,v]
        # (a[i]-a[j]) @ B_p[u,:] = dot_row[i,u] - dot_row[j,u]
        # B_p[:,v] @ (a[:,i]-a[:,j]) = dot_col[v,i] - dot_col[v,j]  (using dot_col as B_p@a)
        # B_p[:,u] @ (a[:,i]-a[:,j]) = dot_col[u,i] - dot_col[u,j]
        
        main = (dot_row[i, v] - dot_row[j, v]) - (dot_row[i, u] - dot_row[j, u]) + \
               (dot_col[v, i] - dot_col[v, j]) - (dot_col[u, i] - dot_col[u, j])
        
        # Corner corrections:
        # The main term double-counts some terms. Let's derive carefully.
        # Cost = sum_{k,l} a[k,l] * B_p[k,l]
        # After swap i<->j: B_p_new differs from B_p only in rows u,v and cols i,j
        # Changed entries: (u,i), (u,j), (v,i), (v,j)
        # 
        # Old contribution: a[i,i]*b[u,u] + a[i,j]*b[u,v] + a[j,i]*b[v,u] + a[j,j]*b[v,v]
        # New contribution: a[j,j]*b[v,v] + a[j,i]*b[v,u] + a[i,j]*b[u,v] + a[i,i]*b[u,u]
        # Wait, that's the same! Let me reconsider.
        # 
        # Actually B_p[k,l] = b[p[k], p[l]]
        # Old: B_p[u,i]=b[u,u], B_p[u,j]=b[u,v], B_p[v,i]=b[v,u], B_p[v,j]=b[v,v]
        # New: B_p[u,i]=b[v,v], B_p[u,j]=b[v,u], B_p[v,i]=b[u,v], B_p[v,j]=b[u,u]
        # 
        # Delta from corners = a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) + 
        #                       a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
        
        b_uu = b[u, u]
        b_vv = b[v, v]
        b_uv = b[u, v]
        b_vu = b[v, u]
        
        corner = (a[i, i] * (b_vv - b_uu) + a[i, j] * (b_vu - b_uv) + 
                  a[j, i] * (b_uv - b_vu) + a[j, j] * (b_uu - b_vv))
        
        # The main term from dot products includes all k,l terms where the B_p change affects
        # the dot product. But the dot product formula (a[i]-a[j])@(B_p[v]-B_p[u]) + (B_p[:,v]-B_p[:,u])@(a[:,i]-a[:,j])
        # counts the corner terms (k in {i,j}, l in {i,j}) twice in a specific way.
        # Let's verify: the full delta is sum_{k,l} (a[k,l]) * (B_p_new[k,l] - B_p[k,l])
        # B_p_new[k,l] - B_p[k,l] is nonzero only for k in {u,v} and l in {i,j}
        # So delta = a[i,i]*(B_new[u,i]-B[u,i]) + a[i,j]*(B_new[u,j]-B[u,j]) + 
        #            a[j,i]*(B_new[v,i]-B[v,i]) + a[j,j]*(B_new[v,j]-B[v,j])
        #          = a[i,i]*(b[v,v]-b[u,u]) + a[i,j]*(b[v,u]-b[u,v]) +
        #            a[j,i]*(b[u,v]-b[v,u]) + a[j,j]*(b[u,u]-b[v,v])
        # 
        # This is EXACTLY the corner term! So the main term from dot products should be zero?
        # No wait, I made an error. Let me redo.
        # 
        # B_p[k,l] = b[p[k], p[l]]. When we swap p[i] and p[j]:
        # - For k not in {i,j} and l not in {i,j}: B_p unchanged
        # - For k in {i,j}, l not in {i,j}: p[k] changes, so B_p[k,l] = b[p_new[k], p[l]] changes
        # - For k not in {i,j}, l in {i,j}: p[l] changes, so B_p[k,l] = b[p[k], p_new[l]] changes  
        # - For k in {i,j}, l in {i,j}: both change
        # 
        # So the delta involves ALL k,l where k or l is in {i,j}, not just the 4 corners!
        # 
        # delta = sum_{k} a[k,i]*(B_new[k,i]-B[k,i]) + sum_{k} a[k,j]*(B_new[k,j]-B[k,j])
        #       + sum_{l} a[i,l]*(B_new[i,l]-B[i,l]) + sum_{l} a[j,l]*(B_new[j,l]-B[j,l])
        #       - (double counted corners)
        # 
        # B_new[k,i] - B[k,i] = b[p_new[k], p[i]] - b[p[k], p[i]]
        #   = b[p[k], p[j]] - b[p[k], p[i]] for k not in {i,j}  (since p_new[k]=p[k])
        #   = b[p[j], p[j]] - b[p[i], p[i]] for k=i  (p_new[i]=p[j]=v, p[i]=u)
        #   = b[p[i], p[i]] - b[p[j], p[j]] for k=j  (p_new[j]=p[i]=u, p[j]=v)
        # 
        # So sum_k a[k,i]*(B_new[k,i]-B[k,i]) = sum_{k not in {i,j}} a[k,i]*(b[p[k],v]-b[p[k],u]) + a[i,i]*(b[v,v]-b[u,u]) + a[j,i]*(b[u,u]-b[v,v])
        #   = (B_p[:,i] @ (a[:,i] with i,j zeroed)) ... this is getting complex.
        # 
        # Let me just use the direct formula:
        # delta = sum_{k,l} a[k,l] * delta_B[k,l]
        # where delta_B[k,l] = B_new[k,l] - B[k,l]
        # 
        # delta_B is nonzero when k in {i,j} or l in {i,j}
        # 
        # Let's compute it as:
        # delta = sum_l a[i,l]*(B_new[i,l]-B[i,l]) + sum_l a[j,l]*(B_new[j,l]-B[j,l])
        #       + sum_k a[k,i]*(B_new[k,i]-B[k,i]) + sum_k a[k,j]*(B_new[k,j]-B[k,j])
        #       - sum_{l in {i,j}} a[i,l]*(B_new[i,l]-B[i,l]) - sum_{l in {i,j}} a[j,l]*(B_new[j,l]-B[j,l])
        #       (to avoid double counting the corners)
        # 
        # Actually simpler: just compute the 4 corner terms directly and the "cross" terms.
        
        # Let me use a cleaner approach:
        # delta = sum_{k not in {i,j}} a[k,i]*(b[p[k],v]-b[p[k],u]) + sum_{k not in {i,j}} a[k,j]*(b[p[k],u]-b[p[k],v])
        #       + sum_{l not in {i,j}} a[i,l]*(b[v,p[l]]-b[u,p[l]]) + sum_{l not in {i,j}} a[j,l]*(b[u,p[l]]-b[v,p[l]])
        #       + corner_terms
        # 
        # The first two sums: sum_k a[k,i]*(B_p[k,v]-B_p[k,u]) - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[u,u]-b[v,v])
        #   = dot_col[v,i] - dot_col[u,i] - a[i,i]*(b[v,v]-b[u,u]) - a[j,i]*(b[u,u]-b[v,v])
        # Wait, dot_col[v,i] = sum_k B_p[v,k]*a[k,i]... no, I defined dot_col = B_p @ a
        # dot_col[k,i] = sum_l B_p[k,l]*a[l,i]
        # That's not what I need. I need sum_k a[k,i]*B_p[k,l] which is (a @ B_p)[i,l] = dot_row[i,l]
        
        # Let me redefine:
        # R = a @ B_p  so R[i,l] = sum_k a[i,k]*B_p[k,l]
        # C = B_p @ a  so C[k,i] = sum_l B_p[k,l]*a[l,i]
        # 
        # sum_k a[k,i]*B_p[k,v] = ? This is not directly R or C.
        # R[k,v] = sum_l a[k,l]*B_p[l,v]. Not quite.
        # 
        # Actually: sum_k a[k,i]*B_p[k,v] = (a @ B_p)[i,v]... no.
        # (a @ B_p)[i,v] = sum_k a[i,k]*B_p[k,v]
        # I need sum_k a[k,i]*B_p[k,v] which is (a.T @ B_p)[i,v] if a were... 
        # Or equivalently, if a is symmetric, (a @ B_p)[i,v].
        # 
        # For general a, let me precompute:
        # R1[i,l] = sum_k a[i,k]*B_p[k,l] = (a @ B_p)[i,l]
        # R2[k,l] = sum_i a[i,k]*B_p[i,l]... this is (a.T @ B_p)[k,l]
        
        # This is getting too complex for a clean vectorized formula. Let me use a different strategy:
        # Precompute for each location pair (u,v) the vector difference, and use matrix-vector products.
        
        # Actually, the simplest fast approach for n=150:
        # Precompute:
        #   row_diff[i,j] = a[i,:] - a[j,:]  (n x n x n) -- too much memory
        # Instead, compute delta on the fly but vectorize over one index.
        
        # For a given swap (i,j) with u=p[i], v=p[j]:
        # delta = sum_{k,l} a[k,l] * (B_new[k,l] - B[k,l])
        # B_new[k,l] - B[k,l] is:
        #   for k not in {i,j}, l not in {i,j}: 0
        #   for k not in {i,j}, l=i: b[p[k],u] - b[p[k],v]  (since p_new[i]=v, p[i]=u... wait)
        # 
        # I'm confusing myself. Let me just use the O(n) formula per pair with numpy.
        
        # Simple O(n) delta computation:
        # delta = (a[i,:]-a[j,:]) @ (B_p[v,:]-B_p[u,:]) + (a[:,i]-a[:,j]) @ (B_p[:,v]-B_p[:,u])
        #         - (a[i,:]-a[j,:])[i]*(B_p[v,:]-B_p[u,:])[i] - (a[i,:]-a[j,:])[j]*(B_p[v,:]-B_p[u,:])[j]
        #         - (a[:,i]-a[:,j])[i]*(B_p[:,v]-B_p[:,u])[i] - (a[:,i]-a[:,j])[j]*(B_p[:,v]-B_p[:,u])[j]
        #         + corner_correction
        # 
        # This is what get_delta_fast was doing. Let me just optimize that.
        
        da_row = a[i] - a[j]
        da_col = a[:, i] - a[:, j]
        db_row = B_p[v] - B_p[u]
        db_col = B_p[:, v] - B_p[:, u]
        
        main = np.dot(da_row, db_row) + np.dot(da_col, db_col)
        
        # Subtract the k=i and k=j terms from main that shouldn't be there
        # In main, the term for k=i in first dot: da_row[i]*db_row[i] = (a[i,i]-a[j,i])*(b[v,u]-b[u,u])
        # In main, the term for k=j in first dot: da_row[j]*db_row[j] = (a[i,j]-a[j,j])*(b[v,v]-b[u,v])
        # In main, the term for l=i in second dot: da_col[i]*db_col[i] = (a[i,i]-a[j,i])*(b[u,v]-b[u,u])
        # In main, the term for l=j in second dot: da_col[j]*db_col[j] = (a[i,j]-a[j,j])*(b[v,v]-b[v,u])
        
        ki = (a[i, i] - a[j, i]) * (b[v, u] - b[u, u])
        kj = (a[i, j] - a[j, j]) * (b[v, v] - b[u, v])
        li = (a[i, i] - a[j, i]) * (b[u, v] - b[u, u])
        lj = (a[i, j] - a[j, j]) * (b[v, v] - b[v, u])
        
        # Direct corner delta:
        corner = (a[i, i] * (b[v, v] - b[u, u]) + a[i, j] * (b[v, u] - b[u, v]) + 
                  a[j, i] * (b[u, v] - b[v, u]) + a[j, j] * (b[u, u] - b[v, v]))
        
        delta = main - ki - kj - li - lj + corner
        return int(delta)
    
    def update_dot_products(p, B_p, dot_row, dot_col, i, j):
        """Incrementally update dot_row and dot_col after swapping p[i] and p[j]"""
        u = p[i]
        v = p[j]
        
        # After swap: p[i]=v, p[j]=u
        # B_p changes in rows u,v and cols i,j
        # 
        # dot_row[k,l] = sum_m a[k,m]*B_p[m,l]
        # dot_col[k,l] = sum_m B_p[k,m]*a[m,l]
        # 
        # For dot_row:
        #   dot_row[k,i] = sum_m a[k,m]*B_p[m,i]
        #   B_p[m,i] changes only for m in {u,v}
        #   So dot_row[k,i] changes by: a[k,u]*(B_new[u,i]-B[u,i]) + a[k,v]*(B_new[v,i]-B[v,i])
        #   B_new[u,i] = b[u, p_new[i]] = b[u,v], B[u,i] = b[u,u]
        #   B_new[v,i] = b[v, p_new[i]] = b[v,v], B[v,i] = b[v,u]
        #   So delta_dot_row[k,i] = a[k,u]*(b[u,v]-b[u,u]) + a[k,v]*(b[v,v]-b[v,u])
        #   Similarly delta_dot_row[k,j] = a[k,u]*(b[u,u]-b[u,v]) + a[k,v]*(b[v,u]-b[v,v])
        #   For other l, dot_row[k,l] unchanged
        # 
        # For dot_col:
        #   dot_col[k,l] = sum_m B_p[k,m]*a[m,l]
        #   B_p[k,m] changes only for k in {u,v}
        #   So dot_col[u,l] changes by: B_new[u,m]-B[u,m] for m in {i,j}
        #   B_new[u,i]=b[u,v], B[u,i]=b[u,u]; B_new[u,j]=b[u,u], B[u,j]=b[u,v]
        #   delta_dot_col[u,l] = (b[u,v]-b[u,u])*a[i,l] + (b[u,u]-b[u,v])*a[j,l]
        #   delta_dot_col[v,l] = (b[v,v]-b[v,u])*a[i,l] + (b[v,u]-b[v,v])*a[j,l]
        
        # Update dot_row columns i and j
        # delta_dot_row[:,i] = a[:,u]*(b[u,v]-b[u,u]) + a[:,v]*(b[v,v]-b[v,u])
        dot_row[:, i] += a[:, u] * (b[u, v] - b[u, u]) + a[:, v] * (b[v, v] - b[v, u])
        # delta_dot_row[:,j] = a[:,u]*(b[u,u]-b[u,v]) + a[:,v]*(b[v,u]-b[v,v])
        dot_row[:, j] += a[:, u] * (b[u, u] - b[u, v]) + a[:, v] * (b[v, u] - b[v, v])
        
        # Update dot_col rows u and v
        # delta_dot_col[u,:] = (b[u,v]-b[u,u])*a[i,:] + (b[u,u]-b[u,v])*a[j,:]
        dot_col[u, :] += (b[u, v] - b[u, u]) * a[i, :] + (b[u, u] - b[u, v]) * a[j, :]
        # delta_dot_col[v,:] = (b[v,v]-b[v,u])*a[i,:] + (b[v,u]-b[v,v])*a[j,:]
        dot_col[v, :] += (b[v, v] - b[v, u]) * a[i, :] + (b[v, u] - b[v, v]) * a[j, :]
        
        # Update B_p
        # B_p[:,i] = b[:,v], B_p[:,j] = b[:,u]
        # B_p[u,:] = b[u,p], B_p[v,:] = b[v,p]
        B_p[:, i] = b[:, v]
        B_p[:, j] = b[:, u]
        B_p[u, :] = b[u, p]
        B_p[v, :] = b[v, p]
        # Fix corners
        B_p[u, i] = b[u, v]
        B_p[u, j] = b[u, u]
        B_p[v, i] = b[v, v]
        B_p[v, j] = b[v, u]
    
    # Initialization Strategies
    
    def init_rank():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_rank_reversed():
        a_row_sums = a.sum(axis=1)
        b_row_sums = b.sum(axis=1)
        heavy_order = np.argsort(-a_row_sums, kind="stable")
        central_order = np.argsort(b_row_sums, kind="stable")[::-1]
        p = np.empty(n, dtype=np.int64)
        p[heavy_order] = central_order
        return p

    def init_spectral():
        """Spectral initialization using eigendecomposition of B"""
        try:
            eigvals, eigvecs = np.linalg.eigh(b.astype(np.float64))
            # Use eigenvector corresponding to smallest eigenvalue
            idx = np.argmin(np.abs(eigvals))
            loc_scores = eigvecs[:, idx]
            loc_order = np.argsort(loc_scores, kind="stable")
            
            a_row_sums = a.sum(axis=1)
            fac_order = np.argsort(a_row_sums, kind="stable")
            
            p = np.empty(n, dtype=np.int64)
            p[fac_order] = loc_order
            return p
        except:
            return np.random.permutation(n).astype(np.int64)

    def init_admm():
        """ADMM-style initialization: solve relaxed QAP via alternating minimization"""
        try:
            # Minimize f(X) = tr(A X B X^T) subject to X being a permutation matrix
            # Relax to Birkhoff polytope (doubly stochastic matrices)
            # Use ADMM: X^{k+1} = argmin tr(A X B X^T) + (rho/2)||X - Z^k + U^k||^2
            # Z^{k+1} = projection of X^{k+1} + U^k onto Birkhoff polytope
            # U^{k+1} = U^k + X^{k+1} - Z^{k+1}
            
            rho = 1.0
            max_iter = 50
            tol = 1e-6
            
            X = np.eye(n, dtype=np.float64) / n  # Initialize as uniform doubly stochastic
            Z = X.copy()
            U = np.zeros((n, n), dtype=np.float64)
            
            B_float = b.astype(np.float64)
            A_float = a.astype(np.float64)
            
            for it in range(max_iter):
                # Update X: minimize tr(A X B X^T) + (rho/2)||X - Z + U||^2
                # Gradient of tr(A X B X^T) w.r.t. X = A^T X B^T + A X B (if A,B symmetric: 2AXB)
                # For simplicity, use gradient descent step
                grad = A_float.T @ X @ B_float.T + A_float @ X @ B_float
                X_new = X - 0.01 * grad + (rho * (Z - U))
                
                # Project X_new onto Birkhoff polytope (approximate with row/col normalization)
                # Simple approximation: normalize rows then columns iteratively
                X_proj = X_new.copy()
                for _ in range(10):
                    row_sums = X_proj.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    X_proj = X_proj / row_sums
                    col_sums = X_proj.sum(axis=0, keepdims=True)
                    col_sums[col_sums == 0] = 1
                    X_proj = X_proj / col_sums
                
                # Update Z: project X_new + U onto Birkhoff polytope
                Z_new = X_new + U
                for _ in range(10):
                    row_sums = Z_new.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    Z_new = Z_new / row_sums
                    col_sums = Z_new.sum(axis=0, keepdims=True)
                    col_sums[col_sums == 0] = 1
                    Z_new = Z_new / col_sums
                
                # Update U
                U = U + X_proj - Z_new
                
                if np.linalg.norm(X_proj - X) < tol:
                    X = X_proj
                    break
                X = X_proj
            
            # Extract permutation from X using Hungarian algorithm or greedy
            # Use scipy if available, otherwise greedy
            try:
                from scipy.optimize import linear_sum_assignment
                row_idx, col_idx = linear_sum_assignment(-X)  # Maximize X[i,j]
                p = np.empty(n, dtype=np.int64)
                p[col_idx] = row_idx  # p[j] = i means facility i at location j
                # But we want p[i] = location of facility i
                # linear_sum_assignment gives assignment of rows to cols
                # row_idx[k] = i, col_idx[k] = j means facility i assigned to location j
                # So p[i] = j
                p2 = np.empty(n, dtype=np.int64)
                p2[row_idx] = col_idx
                return p2
            except ImportError:
                # Greedy: for each facility, pick best location
                p = np.full(n, -1, dtype=np.int64)
                used = set()
                for i in range(n):
                    best_j = -1
                    best_val = -np.inf
                    for j in range(n):
                        if j not in used:
                            if X[i, j] > best_val:
                                best_val = X[i, j]
                                best_j = j
                    p[i] = best_j
                    used.add(best_j)
                return p
        except:
            return np.random.permutation(n).astype(np.int64)

    def init_random():
        return np.random.permutation(n).astype(np.int64)

    # Fast Local Search with vectorized delta
    def local_search(p, current_cost, max_time=None):
        if max_time is None:
            max_time = deadline - 5.0
        
        best_p = p.copy()
        best_cost = current_cost
        B_p, dot_row, dot_col = compute_dot_products(p)
        
        tabu_tenure = 10
        tabu_list = {}
        iteration = 0
        no_improve = 0
        max_no_improve = 20
        
        while time.time() < max_time:
            best_delta = np.inf
            best_i = -1
            best_j = -1
            
            # Vectorized search for best improving move
            for i in range(n):
                for j in range(i + 1, n):
                    move_key = (i, j)
                    tabu_until = tabu_list.get(move_key, -1)
                    if tabu_until > iteration:
                        # Check aspiration
                        delta = get_delta_vectorized(i, j, p, B_p, dot_row, dot_col)
                        if current_cost + delta < best_cost:
                            pass  # aspiration, allow
                        else:
                            continue
                    
                    delta = get_delta_vectorized(i, j, p, B_p, dot_row, dot_col)
                    if delta < best_delta:
                        best_delta = delta
                        best_i = i
                        best_j = j
            
            if best_i == -1:
                # No improving move; pick random
                while True:
                    best_i = random.randint(0, n - 1)
                    best_j = random.randint(0, n - 1)
                    if best_i != best_j and tabu_list.get((min(best_i,best_j), max(best_i,best_j)), -1) <= iteration:
                        break
                best_delta = get_delta_vectorized(best_i, best_j, p, B_p, dot_row, dot_col)
            
            # Perform swap
            u_old = p[best_i]
            v_old = p[best_j]
            p[best_i], p[best_j] = v_old, u_old
            current_cost += best_delta
            
            # Update precomputed structures
            update_dot_products(p, B_p, dot_row, dot_col, best_i, best_j)
            
            # Update tabu
            tabu_list[(min(best_i, best_j), max(best_i, best_j))] = iteration + tabu_tenure
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_p = p.copy()
                no_improve = 0
            else:
                no_improve += 1
            
            iteration += 1
            
            if no_improve > max_no_improve:
                # Perturb
                num_perturb = random.randint(3, 5)
                for _ in range(num_perturb):
                    i = random.randint(0, n - 1)
                    j = random.randint(0, n - 1)
                    if i != j:
                        p[i], p[j] = p[j], p[i]
                current_cost = calc_cost(p)
                B_p, dot_row, dot_col = compute_dot_products(p)
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_p = p.copy()
                no_improve = 0
                tabu_list.clear()
        
        return best_p, best_cost

    # Multi-start
    best_overall_p = None
    best_overall_cost = np.inf
    
    inits = [
        init_rank(),
        init_rank_reversed(),
        init_spectral(),
        init_random()
    ]
    
    if time.time() < deadline - 20.0:
        inits.append(init_admm())
    
    for init_p in inits:
        if time.time() > deadline - 10.0:
            break
        
        init_cost = calc_cost(init_p)
        if init_cost < best_overall_cost:
            best_overall_cost = init_cost
            best_overall_p = init_p.copy()
        
        ls_p, ls_cost = local_search(init_p, init_cost)
        
        if ls_cost < best_overall_cost:
            best_overall_cost = ls_cost
            best_overall_p = ls_p.copy()
    
    if best_overall_p is not None and time.time() < deadline - 5.0:
        final_p, final_cost = local_search(best_overall_p.copy(), best_overall_cost, max_time=deadline - 2.0)
        if final_cost < best_overall_cost:
            best_overall_cost = final_cost
            best_overall_p = final_p
    
    return best_overall_p.tolist()

# EVOLVE-BLOCK-END
