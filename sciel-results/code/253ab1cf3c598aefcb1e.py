    def greedy_independent_set(randomize=False, seed=None):
        if seed is not None:
            random.seed(seed)
        
        # We maintain a list of available nodes
        available = list(candidates)
        result_set = set()
        
        while available:
            if not randomize:
                # Pick node with minimum degree in the original graph? 
                # Or minimum conflicts with current result_set?
                # Minimum conflicts with current result_set is the standard "Greediest" for IS.
                min_conf = float('inf')
                best_v = None
                for v in available:
                    conf_count = 0
                    for u in result_set:
                        if u in conflicts[v]:
                            conf_count += 1
                            if conf_count >= min_conf:
                                break
                    if conf_count < min_conf:
                        min_conf = conf_count
                        best_v = v
                # If best_v is None, something went wrong, but it shouldn't be
                if best_v is None:
                    break
                v = best_v
            else:
                # Randomized: Pick node with min conflicts, break ties randomly
                min_conf = float('inf')
                candidates_for_pick = []
                for v in available:
                    conf_count = 0
                    for u in result_set:
                        if u in conflicts[v]:
                            conf_count += 1
                            if conf_count >= min_conf:
                                break
                    if conf_count < min_conf:
                        min_conf = conf_count
                        candidates_for_pick = [v]
                    elif conf_count == min_conf:
                        candidates_for_pick.append(v)
                v = random.choice(candidates_for_pick)
                
            result_set.add(v)
            # Remove v from available
            available.remove(v)
            # Optional: We could remove neighbors of v from available if we were doing "Maximum Independent Set" via deletion,
            # but here we are just adding to a set. The "conflict check" against result_set ensures validity.
            # However, if we don't remove neighbors, we might pick a neighbor later? No, because it would conflict with v (which is in result_set).
            # So it's safe.
            
        return list(result_set)
