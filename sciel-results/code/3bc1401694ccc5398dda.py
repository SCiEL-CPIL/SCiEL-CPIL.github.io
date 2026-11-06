<<<<<<< SEARCH
    # Light local search: improve the found model via hill-climbing in spare time.
    def improve(a, b, budget):
        nonlocal best_a, best_b, best_len
        rng = random.Random(12345)
        cur_len = best_len
        while time.time() < budget:
            improved = False
            for _ in range(40):
                if time.time() > budget:
                    break
                r = rng.random()
                if r < 0.5:
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    nb = list(best_b); nb[i], nb[j] = nb[j], nb[i]
                    L = eval_ab(best_a, nb)
                    if L > cur_len:
                        best_b, cur_len, improved = nb, L, True
                elif r < 0.9:
                    i, j = rng.randrange(N), rng.randrange(N)
                    if i == j:
                        continue
                    na = list(best_a); na[i], na[j] = na[j], na[i]
                    L = eval_ab(na, best_b)
                    if L > cur_len:
                        best_a, cur_len, improved = na, L, True
                else:
                    na, nb = list(best_b), list(best_a)  # swap letters
                    L = eval_ab(na, nb)
                    if L > cur_len:
                        best_a, best_b, cur_len, improved = na, nb, L, True
            if not improved:
                break
        best_len = cur_len

    improve(best_a, best_b, deadline)
=======
    # Simulated Annealing to escape local optima
    def improve(a, b, budget):
        nonlocal best_a, best_b, best_len
        rng = random.Random(12345)
        cur_a, cur_b = list(best_a), list(best_b)
        cur_len = best_len
        
        # Cooling schedule
        start_temp = 100.0
        end_temp = 0.1
        steps = 5000 # We will run as many as fit in time, but scale temp
        
        # Estimate max steps based on time
        # Each eval is ~microseconds, so we can do many.
        # Let's just run until budget is close.
        
        t = start_temp
        # We don't know exact step count, so we use a time-based cooling or fixed steps with time check.
        # Let's use a fixed number of iterations that should fit, or just loop while time remains.
        
        # Better: Loop while time < budget, with a cooling factor per step.
        # If we do ~1000 steps/sec, and we have ~50s, that's 50000 steps.
        # Let's define total_steps dynamically or just use a decay per step.
        
        # Simple approach: Run for a fixed number of iterations if time allows, or until time runs out.
        # Let's estimate: 1 eval ~ 50us. 10000 evals = 0.5s.
        # We have ~50s. So ~100,000 evals possible.
        
        # We will perform iterations.
        # To ensure we explore, we accept worse moves with probability exp(-delta/T).
        
        # Initial T
        T = 50.0
        decay = 0.9999 # Slow decay
        
        while time.time() < budget - 2.0: # Leave 2s safety margin
            # Generate a neighbor
            r = rng.random()
            na, nb = cur_a[:], cur_b[:]
            
            if r < 0.7:
                # Swap two destinations in one letter
                letter = rng.random() < 0.5
                i, j = rng.randrange(N), rng.randrange(N)
                if i == j: continue
                if letter:
                    na[i], na[j] = na[j], na[i]
                else:
                    nb[i], nb[j] = nb[j], nb[i]
            else:
                # Change one destination
                letter = rng.random() < 0.5
                i = rng.randrange(N)
                new_dest = rng.randrange(N)
                if letter:
                    na[i] = new_dest
                else:
                    nb[i] = new_dest
                    
            L = eval_ab(na, nb)
            if L == -1:
                continue # Not synchronizing, ignore or penalize heavily. 
                         # Actually, if it's not synchronizing, it's a dead end for this path.
                         # We should probably reject it.
            
            delta = L - cur_len
            if delta > 0:
                # Always accept improvement
                cur_a, cur_b, cur_len = na, nb, L
                if L > best_len:
                    best_a, best_b, best_len = list(na), list(nb), L
            else:
                # Accept worse with probability
                # If delta is large negative, prob is small.
                # If T is small, prob is small.
                if T > 0:
                    prob = math.exp(delta / T)
                    if rng.random() < prob:
                        cur_a, cur_b, cur_len = na, nb, L
                        if L > best_len:
                            best_a, best_b, best_len = list(na), list(nb), L
            
            T *= decay
            if T < end_temp:
                T = end_temp

    improve(best_a, best_b, deadline)
>>>>>>> REPLACE

I need to import `math` for `exp`.

