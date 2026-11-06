    def solve_instance():
        rng = random.Random(42)
        
        # Initialization strategies
        inits = []
        inits.append(np.array([rng.random() < 0.5 for _ in range(n)]))
        
        lit_counts = np.zeros(n, dtype=np.int32)
        for c in clauses:
            for lit in c:
                v = abs(lit) - 1
                if lit > 0:
                    lit_counts[v] += 1
                else:
                    lit_counts[v] -= 1
        polarity_init = np.array([lit_counts[i] > 0 for i in range(n)])
        inits.append(polarity_init)
        inits.append(~polarity_init)
        
        best_global_unsat = m + 1
        best_global_assign = None
        
        # Try each init with multiple restarts if needed?
        # Or just run long enough for each.
        
        for init_assign in inits:
            if time.time() - start_time > timeout - 10:
                break
            
            # Run with restarts
            current_assign = init_assign.copy()
            for restart in range(5): # 5 restarts per init
                if time.time() - start_time > timeout - 5:
                    break
                
                result, final_unsat = run_annealed_walksat(current_assign, rng)
                
                if final_unsat == 0:
                    return result.tolist()
                
                if final_unsat < best_global_unsat:
                    best_global_unsat = final_unsat
                    best_global_assign = result
                    
                # Perturb for next restart
                if restart < 4:
                    # Flip some random variables
                    num_flips = max(1, n // 10)
                    flip_indices = rng.sample(range(n), num_flips)
                    for idx in flip_indices:
                        current_assign[idx] = not current_assign[idx]
        
        if best_global_assign is not None:
            return best_global_assign.tolist()
            
        return [rng.random() < 0.5 for _ in range(n)]
