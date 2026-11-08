    # Phase 3: Iterated Local Search (ILS)
    if time.time() < deadline - 5 and best_in_set is not None:
        current_best = best_in_set.copy()
        current_size = current_best.sum()
        
        # Heuristic for greedy fill:
        # We want to fill the holes created by perturbation.
        # We will use a simple greedy: repeatedly pick a vertex with 0 conflicts 
        # among remaining candidates.
        
        while time.time() < deadline - 5:
            # 1. Perturb: Remove k random vertices from current_best
            k_remove = np.random.randint(1, 6) # Remove 1 to 5
            if current_size < k_remove:
                k_remove = current_size
            
            # Select indices to remove
            in_indices = np.where(current_best)[0]
            if len(in_indices) == 0: break
            
            remove_indices = np.random.choice(in_indices, k_remove, replace=False)
            
            perturbed = current_best.copy()
            perturbed[remove_indices] = False
            
            # 2. Re-optimize: Greedy fill
            # We need to add vertices back.
            # Strategy: Iterate through all non-in_set vertices.
            # To be efficient, we can sort by "desirability".
            # Desirability: vertices with fewer conflicts with the *perturbed* set are better.
            
            # Calculate conflicts with perturbed set for all vertices
            # conflict[:, perturbed] gives an (n, n) bool matrix? No, (n, k) where k is size of perturbed.
            # We want to know for each vertex v, if it conflicts with ANY in perturbed.
            # has_conflict = conflict[:, perturbed].any(axis=1)
            
            has_conflict = conflict[:, perturbed].any(axis=1)
            candidates = np.where(~perturbed & ~has_conflict)[0]
            
            # Greedy add
            # We can just add all candidates that don't conflict with each other?
            # No, they might conflict with each other.
            # We need to select an independent set from candidates.
            # Since candidates are vertices that don't conflict with the *core* (perturbed set),
            # they only need to be independent among themselves.
            
            # This is a smaller independent set problem on the subgraph induced by candidates.
            # However, candidates can be large.
            # Simple Greedy:
            # Shuffle candidates, add if no conflict with already added in this pass.
            
            np.random.shuffle(candidates)
            new_in_set = perturbed.copy()
            added_count = 0
            for v in candidates:
                if time.time() > deadline - 5: break
                # Check conflict with new_in_set
                if not conflict[v, new_in_set].any():
                    new_in_set[v] = True
                    added_count += 1
            
            new_size = new_in_set.sum()
            if new_size > current_size:
                current_size = new_size
                current_best = new_in_set
                # Optional: restart perturbation loop with smaller k? Or keep going.
            else:
                # If no improvement, we might get stuck.
                # We can accept the perturbed state if it's same size to escape local opt?
                # No, we want to maximize size.
                # If we don't improve, we can try a different perturbation.
                # To avoid getting stuck in a loop of no improvement, we can just continue.
                pass

        best_in_set = current_best
