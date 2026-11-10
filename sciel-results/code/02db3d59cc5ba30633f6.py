<<<<<<< SEARCH
        # Local search loop
        while time.time() < deadline:
            # Get ranked missing edges
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            # Shuffle for diversity
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                
                if is_c6_free(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = [set(s) for s in adj_set]
            
            if added_in_pass == 0:
                # Stuck: perturbation
                if ranked_missing:
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = get_blocking_c6(target_u, target_v)
                    if block_c6:
                        # Remove 1-3 random edges from the blocking C6
                        num_removals = random.randint(1, 3)
                        for _ in range(num_removals):
                            if edges:
                                e_to_remove = random.choice(block_c6)
                                remove_edge(e_to_remove[0], e_to_remove[1])
                    elif edges:
                        e = random.choice(edges)
                        remove_edge(e[0], e[1])
=======
        # Local search loop
        consecutive_failures = 0
        while time.time() < deadline:
            # Get ranked missing edges
            ranked_missing = compute_priority()
            if not ranked_missing:
                break
            
            # Shuffle for diversity
            random.shuffle(ranked_missing)
            
            added_in_pass = 0
            for u, v in ranked_missing:
                if time.time() > deadline:
                    break
                if (adj_mask[u] >> v) & 1:
                    continue
                
                if is_c6_free(u, v):
                    add_edge(u, v)
                    added_in_pass += 1
                    consecutive_failures = 0
                    if len(edges) > local_best:
                        local_best = len(edges)
                        local_best_adj = [set(s) for s in adj_set]
            
            if added_in_pass == 0:
                consecutive_failures += 1
                # Stuck: perturbation
                if ranked_missing:
                    target_u, target_v = random.choice(ranked_missing)
                    block_c6 = get_blocking_c6(target_u, target_v)
                    if block_c6:
                        # Remove 1-4 random edges from the blocking C6
                        num_removals = random.randint(1, 4)
                        for _ in range(num_removals):
                            if edges:
                                e_to_remove = random.choice(block_c6)
                                remove_edge(e_to_remove[0], e_to_remove[1])
                    elif edges:
                        # Remove 1-2 random edges
                        num_removals = random.randint(1, 2)
                        for _ in range(num_removals):
                            if edges:
                                e = random.choice(edges)
                                remove_edge(e[0], e[1])
                
                # If stuck many times, do a bigger perturbation
                if consecutive_failures >= 3:
                    num_removals = random.randint(3, 6)
                    for _ in range(num_removals):
                        if edges:
                            e = random.choice(edges)
                            remove_edge(e[0], e[1])
                    consecutive_failures = 0
>>>>>>> REPLACE

