<<<<<<< SEARCH
        # Run L-BFGS-B optimization
        x, y, r = optimize_lbfgsb(x, y, r, maxiter=500)
        
        # Repair
        x, y, r = repair_config(x, y, r)
        
        # Local Search
        x, y, r = local_search(x, y, r, max_iter=300)
        
        # Final Repair
        x, y, r = repair_config(x, y, r)
        
        # Maximize Radii
        x, y, r = maximize_radii(x, y, r)
        
        # Check Validity
        if validity_penalty(x, y, r) == 0:
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
        
        if time.time() > deadline - 15:
            break
=======
        # Run L-BFGS-B optimization
        x, y, r = optimize_lbfgsb(x, y, r, maxiter=300)
        
        # Local Search
        x, y, r = local_search(x, y, r, max_iter=200)
        
        # Maximize Radii (this function ensures validity)
        x, y, r = maximize_radii(x, y, r)
        
        # Final Check
        if is_valid(x, y, r):
            score = np.sum(r)
            if score > best_score:
                best_score = score
                best_config = np.column_stack((x, y, r))
        
        if time.time() > deadline - 10:
            break
>>>>>>> REPLACE

