<<<<<<< SEARCH
    def try_add_fast(s, v):
        return len(conflicts[v] & s) == 0
        
    def local_search_improve(s):
        s = set(s)
        improved = True
        while improved:
            improved = False
            
            # 1. Try to add any single node
            for v in all_candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved = True
            
            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            removal_improved = False
            for u in list(s):
                s.remove(u)
                
                addable = []
                for v in all_candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)
                
                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    removal_improved = True
                    improved = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflicts[u] & s)
                    deg_v = len(conflicts[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        removal_improved = True
                        improved = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)
                    
        return s
=======
    # Precompute bitmasks for fast addability checks
    n_cand = len(all_candidates)
    cand_index = {a: i for i, a in enumerate(all_candidates)}
    conflict_masks = [0] * n_cand
    for i, a in enumerate(all_candidates):
        mask = 0
        for b in conflicts[a]:
            mask |= (1 << cand_index[b])
        conflict_masks[i] = mask

    def try_add_fast(s, v):
        return (conflict_masks[cand_index[v]] & s) == 0

    def local_search_improve(s):
        s = set(s)
        improved = True
        while improved:
            improved = False

            # 1. Try to add any single node
            for v in all_candidates:
                if v not in s and try_add_fast(s, v):
                    s.add(v)
                    improved = True

            # 2. Try to remove one node to add >= 1 node (prefer >= 2 for size gain)
            for u in list(s):
                s.remove(u)

                addable = []
                for v in all_candidates:
                    if v not in s and v != u:
                        if try_add_fast(s, v):
                            addable.append(v)

                if len(addable) >= 2:
                    for v in addable:
                        s.add(v)
                    improved = True
                    break
                elif len(addable) == 1:
                    # Swap: remove u, add v. Size same.
                    # Accept if new node has lower degree (potential for future growth)
                    deg_u = len(conflicts[u] & s)
                    deg_v = len(conflicts[addable[0]] & s)
                    if deg_v < deg_u:
                        s.add(addable[0])
                        improved = True
                        break
                    else:
                        s.add(u)
                else:
                    s.add(u)

            # 3. Try 2-swap: remove two nodes, add two or more
            if not improved:
                s_list = list(s)
                two_swap_found = False
                for i in range(len(s_list)):
                    if time.time() - start_time > TIME_LIMIT:
                        break
                    for j in range(i + 1, len(s_list)):
                        u1, u2 = s_list[i], s_list[j]
                        s.remove(u1)
                        s.remove(u2)

                        addable = []
                        for v in all_candidates:
                            if v not in s and v != u1 and v != u2:
                                if try_add_fast(s, v):
                                    addable.append(v)

                        if len(addable) >= 2:
                            for v in addable:
                                s.add(v)
                            two_swap_found = True
                            improved = True
                            break
                        else:
                            s.add(u1)
                            s.add(u2)

                    if two_swap_found:
                        break

        return s
>>>>>>> REPLACE

