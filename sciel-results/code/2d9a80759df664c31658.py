<<<<<<< SEARCH
        for step in range(1000):
            # Compute current tensor
            C = np.einsum('ir, jr, kr -> ijk', U, V, W)
            error = T - C
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                return U, V, W
=======
        for step in range(1000):
            if _time.time() > _deadline:
                return None
            # Compute current tensor
            C = np.einsum('ir, jr, kr -> ijk', U, V, W)
            error = T - C
            loss = np.sum(error**2)
            
            if loss < 1e-10:
                return U, V, W
>>>>>>> REPLACE
