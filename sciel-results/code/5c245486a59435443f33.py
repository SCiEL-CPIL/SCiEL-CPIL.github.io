        import time
        
        def construct():
            initial = [1, 2, 5, 11]
            
            def get_sums(s):
                sums = set()
                for i in range(len(s)):
                    for j in range(i, len(s)):
                        sums.add(s[i] + s[j])
                return sums
            
            init_sums = get_sums(initial)
            
            # We will try to find a set of size 20, then 19, etc.
            # To avoid hanging on impossible targets, we limit search nodes.
            
            def search(target, node_limit=500000):
                curr_set = initial[:]
                curr_sums = set(init_sums)
                nodes = 0
                start_time = time.time()
                
                def backtrack(start_val):
                    nonlocal nodes
                    nodes += 1
                    if nodes > node_limit or time.time() - start_time > 10:
                        return False
                    
                    if len(curr_set) == target:
                        return True
                    
                    # Pruning
                    if len(curr_set) + (284 - start_val + 1) < target:
                        return False
                    
                    for x in range(start_val, 285):
                        # Check validity
                        if 2*x in curr_sums:
                            continue
                            
                        new_sums = {2*x}
                        valid = True
                        for y in curr_set:
                            s = x + y
                            if s in curr_sums:
                                valid = False
                                break
                            new_sums.add(s)
                        
                        if valid:
                            curr_set.append(x)
                            curr_sums.update(new_sums)
                            
                            if backtrack(x + 1):
                                return True
                            
                            curr_set.pop()
                            curr_sums.difference_update(new_sums)
                    
                    return False
                
                if backtrack(12):
                    return curr_set
                return None

            for target in range(20, 3, -1):
                res = search(target)
                if res:
                    return res
            
            return initial
        