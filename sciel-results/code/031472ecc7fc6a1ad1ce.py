import time

def construct():
    start_time = time.time()
    deadline = start_time + 55.0
    best_seq = []
    
    def run_search(order_func):
        nonlocal best_seq
        seq = []
        
        def dfs_inner(last_val, seen_mask, suffix_sums):
            nonlocal best_seq
            if time.time() > deadline:
                return
            
            # Pruning: Can we beat the best?
            # Max elements we can add is (256 - last_val)
            if len(seq) + (256 - last_val) <= len(best_seq):
                return
            
            if len(seq) > len(best_seq):
                best_seq = seq.copy()
                
            candidates = list(range(last_val + 1, 257))
            if order_func == 'high_to_low':
                candidates.reverse()
            
            for x in candidates:
                if time.time() > deadline:
                    return
                
                # Build new suffix sums
                new_sums = [x]
                for ss in suffix_sums:
                    new_sums.append(x + ss)
                    
                # Check collision and build mask
                new_mask = 0
                valid = True
                for val in new_sums:
                    if (seen_mask >> val) & 1:
                        valid = False
                        break
                    new_mask |= (1 << val)
                    
                if not valid:
                    continue
                
                # Recurse
                seq.append(x)
                dfs_inner(x, seen_mask | new_mask, new_sums)
                seq.pop()

        dfs_inner(0, 0, [])

    run_search('low_to_high')
    
    remaining_time = deadline - time.time()
    if remaining_time > 2.0:
        run_search('high_to_low')
        
    return best_seq
