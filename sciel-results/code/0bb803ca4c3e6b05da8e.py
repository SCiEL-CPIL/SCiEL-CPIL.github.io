<<<<<<< SEARCH
    # 3. Local Search (Hill Climbing)
    def local_search(A, score, time_limit):
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try all swaps, adds, removes
            # Swaps are most likely to improve.
            
            # Collect all possible moves
            # To speed up, we can just iterate.
            
            # 1. Try Adds
            for x in range(1, N + 1):
                if not (A & (1 << x)):
                    if is_valid_add(A, x):
                        g = gain_add(A, x)
                        if g > 0:
                            A |= (1 << x)
                            score += g
                            improved = True
                            # Restart loop to recalculate? 
                            # Or just continue. First improvement strategy.
                            # Let's restart to ensure we pick the best next move.
                            break
            if improved:
                continue
                
            # 2. Try Removes
            # Removing usually decreases score, so only useful if it enables a better swap?
            # In pure hill climbing, we only accept if score increases.
            # So we skip pure removes unless we are doing SA.
            # For ILS, we rely on swaps.
            
            # 3. Try Swaps
            # Iterate x in A, y not in A
            # This is O(N^2). 256^2 = 65536. 
            # Each swap check is O(degree). Degree is small.
            # This might be slow in Python if done naively.
            # Optimization: Only check promising swaps?
            # Or just do it. 65k * 50 ops = 3M ops per local search step.
            # If we do 100 local search steps, 300M ops. Might be too slow.
            
            # Better: Randomized Hill Climbing?
            # Pick random x in A, random y not in A.
            # If valid and score increases, accept.
            # Repeat until no improvement in K random tries.
            
            no_improve_count = 0
            max_random_tries = 1000
            while no_improve_count < max_random_tries and time.time() < time_limit:
                # Pick random x in A
                # We need a list of active elements
                active = [i for i in range(1, N+1) if A & (1<<i)]
                inactive = [i for i in range(1, N+1) if not (A & (1<<i))]
                
                if not active or not inactive:
                    break
                    
                x = random.choice(active)
                y = random.choice(inactive)
                
                if is_valid_swap(A, x, y):
                    d = delta_swap(A, x, y)
                    if d > 0:
                        # Apply swap
                        A ^= (1 << x)
                        A ^= (1 << y)
                        score += d
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
            # If we exhausted random tries without improvement, try a full scan?
            # Or just stop local search.
            # Full scan is expensive. Let's try a limited full scan if time permits.
            if not improved:
                # Try a few deterministic swaps?
                pass
                
        return A, score
=======
    # 3. Optimized Local Search
    def local_search(A, score, time_limit):
        # Maintain active and inactive lists
        active = [i for i in range(1, N+1) if A & (1<<i)]
        inactive = [i for i in range(1, N+1) if not (A & (1<<i))]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            
            # 1. Try Adds (First Improvement)
            # Iterate through inactive list
            # To be faster, we can shuffle inactive occasionally or just iterate
            for i, y in enumerate(inactive):
                if is_valid_add(A, y):
                    g = gain_add(A, y)
                    if g > 0:
                        A |= (1 << y)
                        score += g
                        # Update lists
                        active.append(y)
                        inactive.pop(i)
                        improved = True
                        break # Restart add phase
            
            if improved:
                continue
            
            # 2. Try Swaps (Randomized Hill Climbing with List Maintenance)
            no_improve_count = 0
            max_random_tries = 2000 # Increased tries as list maintenance is cheap
            
            while no_improve_count < max_random_tries and time.time() < time_limit:
                if not active or not inactive:
                    break
                    
                x_idx = random.randint(0, len(active) - 1)
                y_idx = random.randint(0, len(inactive) - 1)
                x = active[x_idx]
                y = inactive[y_idx]
                
                if is_valid_swap(A, x, y):
                    d = delta_swap(A, x, y)
                    if d > 0:
                        # Apply swap
                        A ^= (1 << x)
                        A ^= (1 << y)
                        score += d
                        # Update lists
                        active[x_idx] = y
                        inactive[y_idx] = x
                        no_improve_count = 0
                        improved = True
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
                    
        return A, score, active, inactive
>>>>>>> REPLACE

