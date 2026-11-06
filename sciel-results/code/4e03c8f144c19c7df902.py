    def construct():
        # ... setup ...
        best = initial_network
        best, _ = _greedy_remove_pass(best, deadline)
        
        # Multi-Start SA
        start_time = time.time()
        max_time = deadline - 5 # safety margin
        num_starts = 10 # adjust based on time
        
        for start in range(num_starts):
            if time.time() > max_time: break
            
            # Perturb best
            current = best[:]
            # Randomly remove 5-10 comparators to create a "broken" state?
            # No, we need valid states for SA to work well.
            # Better: Perturb by swapping/adding, then greedy repair?
            # Or just start from best and run SA.
            
            # To escape local optima, we can start from a slightly different state.
            # Let's try: Start from best, but apply a few random valid moves?
            # Or: Start from a random valid network? Too hard.
            
            # Let's stick to: Start from best, run SA.
            # To ensure diversity, we can change the random seed or the move distribution.
            
            T = 2.0
            T_min = 0.01
            cooling = 0.99
            rng = random.Random(start * 1000 + 1)
            
            steps = 0
            max_steps = 5000 # per start
            
            while steps < max_steps and time.time() < max_time:
                # Generate neighbor
                # ... (same as before) ...
                
                if _verify(neighbor):
                    # Intensify
                    neighbor, _ = _greedy_remove_pass(neighbor, max_time)
                    
                    if len(neighbor) < len(current):
                        current = neighbor
                        if len(current) < len(best):
                            best = current[:]
                            # Optional: Reset T? Or just continue.
                    elif rng.random() < math.exp(-(len(neighbor) - len(current)) / T):
                        current = neighbor
                else:
                    # Rejection
                    pass
                    
                T *= cooling
                steps += 1
    