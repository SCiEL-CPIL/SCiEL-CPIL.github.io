<<<<<<< SEARCH
    # Try ranks from low to high
    best_rank = n * m * p
    best_result = None
    
    U0, V0, W0 = schoolbook()
    
    for rank in range(28, best_rank + 1):
        if time.time() > deadline:
            break
        
        # Try multiple random restarts
        found = False
        max_restarts = 50 if rank <= 35 else 20
        max_iter = 5000 if rank <= 35 else 2000
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            # Random initialization with small values
            U = np.random.randn(nm, rank) * 0.5
            V = np.random.randn(mp, rank) * 0.5
            W = np.random.randn(pn, rank) * 0.5
            
            # Gradient descent on loss = sum (T_comp - T)^2
            lr = 0.01
            prev_loss = np.inf
            
            for it in range(max_iter):
                if time.time() > deadline:
                    break
                
                # Compute T_comp and loss
                T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
                loss = np.sum((T_comp - T) ** 2)
                
                if loss < 1e-12:
                    break
                if loss > prev_loss * 1.1:
                    break
                prev_loss = loss
                
                # Gradients
                # dL/dU[u,r] = 2 * sum_{v,w} (T_comp[u,v,w] - T[u,v,w]) * V[v,r] * W[w,r]
                # dL/dV[v,r] = 2 * sum_{u,w} (T_comp[u,v,w] - T[u,v,w]) * U[u,r] * W[w,r]
                # dL/dW[w,r] = 2 * sum_{u,v} (T_comp[u,v,w] - T[u,v,w]) * U[u,r] * V[v,r]
                
                diff = T_comp - T  # shape (nm, mp, pn)
                
                dU = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
                dV = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
                dW = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            # Check if we converged
            T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
            loss = np.sum((T_comp - T) ** 2)
            
            if loss < 1e-8:
                # Try rounding
                U2 = np.round(U * 2) / 2.0
                V2 = np.round(V * 2) / 2.0
                W2 = np.round(W * 2) / 2.0
                
                if verify(U2, V2, W2):
                    found = True
                    break
        
        if found:
            best_rank = rank
            best_result = (U2, V2, W2)
            break
=======
    # Try ranks from low to high
    best_rank = n * m * p
    best_result = None
    
    U0, V0, W0 = schoolbook()
    
    # Limit search to a small range below schoolbook to save time
    # Schoolbook is 48. Try 47, 46, 45.
    search_ranks = [47, 46, 45]
    
    for rank in search_ranks:
        if time.time() > deadline - 20:  # Reserve 20s for final checks/return
            break
        
        # Try a few random restarts with limited iterations
        found = False
        max_restarts = 10
        max_iter = 200
        
        for restart in range(max_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U = np.random.randn(nm, rank) * 0.1
            V = np.random.randn(mp, rank) * 0.1
            W = np.random.randn(pn, rank) * 0.1
            
            lr = 0.05
            prev_loss = np.inf
            
            for it in range(max_iter):
                if time.time() > deadline - 20:
                    break
                
                T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
                loss = np.sum((T_comp - T) ** 2)
                
                if loss < 1e-10:
                    break
                if loss > prev_loss * 1.05:
                    break
                prev_loss = loss
                
                diff = T_comp - T
                dU = 2.0 * np.einsum('uvw,vr,wr->ur', diff, V, W)
                dV = 2.0 * np.einsum('uvw,ur,wr->vr', diff, U, W)
                dW = 2.0 * np.einsum('uvw,ur,vr->wr', diff, U, V)
                
                U -= lr * dU
                V -= lr * dV
                W -= lr * dW
            
            T_comp = np.einsum('ur,vr,wr->uvw', U, V, W)
            loss = np.sum((T_comp - T) ** 2)
            
            if loss < 1e-8:
                U2 = np.round(U * 2) / 2.0
                V2 = np.round(V * 2) / 2.0
                W2 = np.round(W * 2) / 2.0
                
                if verify(U2, V2, W2):
                    found = True
                    break
        
        if found:
            best_rank = rank
            best_result = (U2, V2, W2)
            break
>>>>>>> REPLACE
