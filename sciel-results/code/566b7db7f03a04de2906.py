def _als_optimize(U, V, W, T, max_iters=100, deadline=None):
    """Optimize U,V,W using Alternating Least Squares."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    n, m, p = T.shape
    r = U.shape[1]
    
    best_loss = float('inf')
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
            
        # Solve for W: T[a,b,:] = sum_r U[a,r] V[b,r] W[:,r]
        # For each r, we have a linear system for W[:,r]?
        # No, it's a joint system.
        # T[a,b,c] = sum_r U[a,r] V[b,r] W[c,r]
        # Let's reshape T to (n*m, m*p, p*n)
        # For fixed r, the term is U[:,r] V[:,r] W[:,r]
        # This is not linear in W if U and V are fixed?
        # T = sum_r (U[:,r] V[:,r]) \otimes W[:,r] ?
        # Let M_r = U[:,r] V[:,r] (outer product).
        # T[a,b,c] = sum_r M_r[a,b] W[c,r].
        # This is a linear system in W.
        # We can solve for W by least squares.
        # Construct a matrix A of size (n*m*m*p, p*n*r) ?
        # No, it's easier to loop over c?
        
        # Let's use the fact that T is sparse.
        # Actually, we can just use np.linalg.lstsq on a flattened system.
        # But that's expensive.
        
        # Alternative: 
        # For each c, T[:,:,c] = sum_r W[c,r] (U[:,r] V[:,r])
        # This is a linear combination of rank-1 matrices U[:,r] V[:,r].
        # We can solve for W[:,c] ? No, W is (p*n, r).
        # Let's flatten U and V.
        
        # Let's stick to a simpler gradient method but faster.
        # Or just use ALS with a simplified update.
        
        # Update W:
        # Minimize || T - sum_r U[:,r] V[:,r] W[:,r] ||^2
        # This is equivalent to solving:
        # For each (a,b), T[a,b,c] = sum_r (U[a,r] V[b,r]) W[c,r]
        # Let's form a matrix B of size (n*m, m*p, p*n) -> no.
        
        # Let's just use a few steps of gradient descent with a larger learning rate.
        # Or use L-BFGS-B?
        
        pass
