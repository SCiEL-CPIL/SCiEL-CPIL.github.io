def construct():
            import time
            import random
            from math import gcd

            start_time = time.time()
            deadline = start_time + 50  # Safety margin
            n = 128
            all_elements = list(range(1, n + 1))

            # Precompute GCDs and Conflicts
            gcd_table = [[0]*(n+1) for _ in range(n+1)]
            for i in range(1, n+1):
                for j in range(1, n+1):
                    gcd_table[i][j] = gcd(i, j)

            # conflict[x] = list of (y, z) such that (x,y,z) is bad
            conflict = [[] for _ in range(n+1)]
            # We need to find all bad triples.
            # Triple (a,b,c) is bad if gcd(a,b)==gcd(a,c)==gcd(b,c)
            # To speed up, iterate a < b < c.
            for a in range(1, n+1):
                for b in range(a+1, n+1):
                    g_ab = gcd_table[a][b]
                    for c in range(b+1, n+1):
                        if gcd_table[a][c] == g_ab and gcd_table[b][c] == g_ab:
                            conflict[a].append((b, c))
                            conflict[b].append((a, c))
                            conflict[c].append((a, b))
            
            # Sort conflicts by frequency? Or just keep as is.
            # For faster checking, we might want to order conflicts such that "likely" conflicts are checked first?
            # Or just randomize.
            
            best_size = 0
            best_set = []
            rng = random.Random(42)

            # Helper to check if x can be added given in_set
            def can_add(x, in_set):
                for y, z in conflict[x]:
                    if in_set[y] and in_set[z]:
                        return False
                return True

            # Greedy construction
            def greedy_fill(in_set, S_list, missing_list, order):
                # order is a list of elements to try
                # We iterate through order, if not in S, try to add
                # This is O(N * AvgConflict)
                for x in order:
                    if not in_set[x]:
                        if can_add(x, in_set):
                            in_set[x] = True
                            S_list.append(x)
                            # Remove from missing if present? 
                            # Maintaining missing_list is expensive. 
                            # Instead, just check `not in_set[x]` in the loop.
            
            # Local Search
            def local_search(in_set, S_list, max_iters=200):
                # S_list is mutable
                improved = True
                iters = 0
                while improved and iters < max_iters and time.time() < deadline:
                    improved = False
                    iters += 1
                    # 1. Try to add any missing element
                    # Create list of missing elements
                    missing = [x for x in range(1, n+1) if not in_set[x]]
                    rng.shuffle(missing)
                    added_any = False
                    for x in missing:
                        if can_add(x, in_set):
                            in_set[x] = True
                            S_list.append(x)
                            added_any = True
                            improved = True
                    if added_any:
                        continue # Restart adding loop
                    
                    # 2. Try 1-for-1 swaps (neutral size, might open up space for additions)
                    # Or just 1-for-2 kicks?
                    # Let's try swaps first.
                    S_list_copy = list(S_list)
                    rng.shuffle(S_list_copy)
                    swap_found = False
                    for x_out in S_list_copy:
                        if time.time() > deadline: return
                        # Try to remove x_out and add some x_in
                        # To be a "swap", we remove x_out, then try to add x_in.
                        # But if we just remove x_out, we might be able to add multiple?
                        # Standard move: Remove x_out. Then try to add x_in.
                        # If we can add x_in, we do it. Size stays same.
                        # Then we check if we can add MORE.
                        in_set[x_out] = False
                        S_list.remove(x_out)
                        
                        # Try to add any missing element
                        success = False
                        for x_in in missing: # 'missing' from before, but x_out is now missing
                            if not in_set[x_in]:
                                if can_add(x_in, in_set):
                                    in_set[x_in] = True
                                    S_list.append(x_in)
                                    success = True
                                    break
                        if success:
                            improved = True
                            # After a swap, we might be able to add more.
                            # Let's loop back to step 1
                            break
                        else:
                            # Restore
                            in_set[x_out] = True
                            S_list.append(x_out)
                    
                    if improved:
                        continue

                    # 3. Kick: Remove k random elements, then greedily fill
                    # This is a stochastic restart.
                    if len(S_list) > 5:
                        k = min(len(S_list), rng.randint(2, 5))
                        to_remove = rng.sample(S_list, k)
                        for x in to_remove:
                            in_set[x] = False
                            S_list.remove(x)
                        
                        # Greedy fill
                        # Use a random order for the fill
                        order = list(range(1, n+1))
                        rng.shuffle(order)
                        greedy_fill(in_set, S_list, None, order)
                        
                        if len(S_list) > 0: # It will be, unless we removed everything and couldn't add back
                            improved = True # Force loop to restart check

            # Main Loop
            # Try multiple initializations
            num_restarts = 20
            for r in range(num_restarts):
                if time.time() > deadline - 5:
                    break
                
                in_set = [False] * (n+1)
                S_list = []
                
                # Random initial order
                order = list(range(1, n+1))
                rng.shuffle(order)
                
                greedy_fill(in_set, S_list, None, order)
                
                # Local search
                local_search(in_set, S_list, max_iters=50)
                
                if len(S_list) > best_size:
                    best_size = len(S_list)
                    best_set = list(S_list)
                
                # Early exit if we found a very large set?
                # The maximum possible is 128.
                # If best_size is close to 128, we can stop.
                # But we don't know the max.
            
            return best_set
