<<<<<<< SEARCH
    def local_search(mask, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 2
        improved = True
        iters = 0
        max_iters = 100
        
        while improved and time.time() < time_limit and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        improved = True
                        break
            if improved:
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z
            found_swap = False
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            for x in s_list:
                if time.time() > time_limit: break
                mask_no_x = mask & ~(1 << (x - 1))
                added_count = 0
                new_mask = mask_no_x
                for y in range(1, n + 1):
                    if y == x: continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                            if added_count == 2:
                                break
                if added_count >= 2:
                    if popcount(new_mask) > popcount(mask):
                        mask = new_mask
                        improved = True
                        found_swap = True
                        break
            if found_swap:
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            if time.time() < time_limit - 1:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                tried = 0
                for i in range(min(len(s_list), 15)):
                    if time.time() > time_limit - 1 or tried > 30: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+8)):
                        if tried > 30: break
                        y = s_list[j]
                        tried += 1
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y: continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                                    if added_count == 3:
                                        break
                        if added_count >= 3:
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                break
                    if improved:
                        break
                if improved:
                    continue

            # 4. Random perturbation if stuck
            if not improved and iters > 3:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                if len(s_list) > 4:
                    k = random.randint(1, min(3, len(s_list) - 1))
                    remove_idx = random.sample(range(len(s_list)), k)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) > popcount(mask):
                        mask = perturb_mask
                        improved = True
                        continue

            if not improved:
                break
                
        return mask
=======
    def local_search(mask, time_limit=None):
        if time_limit is None:
            time_limit = deadline - 2
        improved = True
        iters = 0
        max_iters = 200
        
        while improved and time.time() < time_limit and iters < max_iters:
            improved = False
            iters += 1
            
            # 1. Try to add any missing element (try all, not just first)
            added_any = False
            for x in range(1, n + 1):
                if not (mask & (1 << (x - 1))):
                    if can_add(x, mask):
                        mask |= (1 << (x - 1))
                        added_any = True
            if added_any:
                improved = True
                continue

            # 2. Try 1-for-2 swap: remove x, add y, z (try all candidates)
            found_swap = False
            s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
            best_gain = 0
            best_new_mask = mask
            for x in s_list:
                if time.time() > time_limit: break
                mask_no_x = mask & ~(1 << (x - 1))
                new_mask = mask_no_x
                added_count = 0
                for y in range(1, n + 1):
                    if y == x: continue
                    if not (new_mask & (1 << (y - 1))):
                        if can_add(y, new_mask):
                            new_mask |= (1 << (y - 1))
                            added_count += 1
                if added_count >= 2:
                    gain = popcount(new_mask) - popcount(mask)
                    if gain > best_gain:
                        best_gain = gain
                        best_new_mask = new_mask
            if best_gain > 0:
                mask = best_new_mask
                improved = True
                continue

            # 3. Try 2-for-3 swap: remove x, y, add z, w, v
            if time.time() < time_limit - 1:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                tried = 0
                found_23 = False
                for i in range(min(len(s_list), 20)):
                    if time.time() > time_limit - 1 or tried > 50: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+12)):
                        if tried > 50: break
                        y = s_list[j]
                        tried += 1
                        mask_no_xy = mask & ~(1 << (x - 1)) & ~(1 << (y - 1))
                        added_count = 0
                        new_mask = mask_no_xy
                        for z in range(1, n + 1):
                            if z == x or z == y: continue
                            if not (new_mask & (1 << (z - 1))):
                                if can_add(z, new_mask):
                                    new_mask |= (1 << (z - 1))
                                    added_count += 1
                        if added_count >= 3:
                            if popcount(new_mask) > popcount(mask):
                                mask = new_mask
                                improved = True
                                found_23 = True
                                break
                    if found_23:
                        break
                if found_23:
                    continue

            # 4. Try 3-for-4 swap
            if time.time() < time_limit - 2 and not improved:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                random.shuffle(s_list)
                tried = 0
                found_34 = False
                for i in range(min(len(s_list), 10)):
                    if time.time() > time_limit - 2 or tried > 20: break
                    x = s_list[i]
                    for j in range(i+1, min(len(s_list), i+6)):
                        if tried > 20: break
                        y = s_list[j]
                        for k in range(j+1, min(len(s_list), j+4)):
                            if tried > 20: break
                            z = s_list[k]
                            tried += 1
                            mask_no_xyz = mask & ~(1 << (x - 1)) & ~(1 << (y - 1)) & ~(1 << (z - 1))
                            added_count = 0
                            new_mask = mask_no_xyz
                            for w in range(1, n + 1):
                                if w == x or w == y or w == z: continue
                                if not (new_mask & (1 << (w - 1))):
                                    if can_add(w, new_mask):
                                        new_mask |= (1 << (w - 1))
                                        added_count += 1
                            if added_count >= 4:
                                if popcount(new_mask) > popcount(mask):
                                    mask = new_mask
                                    improved = True
                                    found_34 = True
                                    break
                        if found_34:
                            break
                    if found_34:
                        break
                if found_34:
                    continue

            # 5. Simulated annealing-style random perturbation
            if not improved and iters > 5:
                s_list = [i for i in range(1, n + 1) if (mask & (1 << (i - 1)))]
                if len(s_list) > 4:
                    # Try removing 1-3 random elements and greedily refilling
                    k = random.randint(1, min(3, len(s_list) - 1))
                    remove_idx = random.sample(range(len(s_list)), k)
                    perturb_mask = mask
                    for idx in remove_idx:
                        perturb_mask &= ~(1 << (s_list[idx] - 1))
                    # Greedy refill
                    for x in range(1, n + 1):
                        if not (perturb_mask & (1 << (x - 1))):
                            if can_add(x, perturb_mask):
                                perturb_mask |= (1 << (x - 1))
                    if popcount(perturb_mask) >= popcount(mask):
                        # Accept with probability based on "temperature" that decreases
                        temp = max(0.1, 1.0 - iters / 50.0)
                        if popcount(perturb_mask) > popcount(mask) or random.random() < temp:
                            mask = perturb_mask
                            improved = True
                            continue

            if not improved:
                break
                
        return mask
>>>>>>> REPLACE

