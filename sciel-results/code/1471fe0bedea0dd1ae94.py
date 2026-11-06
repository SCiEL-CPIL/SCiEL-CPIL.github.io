import random
import time

def construct():
    # LCM of 1 to 22
    L_global = 232792560
    V_global = [L_global // a for a in range(1, 23)]
    
    def find_collision(indices):
        """
        indices: list of global indices (0-21)
        Returns (is_dissociated, mask1, mask2)
        mask1, mask2 are bitmasks relative to the order in indices
        """
        vals = [V_global[i] for i in indices]
        n = len(vals)
        if n == 0:
            return True, 0, 0
        
        # sum -> mask
        sum_to_mask = {0: 0}
        
        for i in range(n):
            v = vals[i]
            bit = 1 << i
            new_entries = {}
            for s, m in sum_to_mask.items():
                ns = s + v
                nm = m | bit
                # Check collision with existing sums
                if ns in sum_to_mask:
                    return False, sum_to_mask[ns], nm
                # Check collision with new sums generated in this step
                if ns in new_entries:
                    return False, new_entries[ns], nm
                new_entries[ns] = nm
            for ns, nm in new_entries.items():
                sum_to_mask[ns] = nm
                
        return True, 0, 0

    best_set = None
    start_time = time.time()
    time_limit = 50.0  # Leave 10s margin
    
    # Strategies for removing an element from the collision set
    # We try several deterministic and random strategies to maximize the chance of finding a large set
    # The "diff" passed to the function is a list of global indices involved in the collision
    
    def strategy_min(diff):
        return min(diff)
        
    def strategy_max(diff):
        return max(diff)
        
    def strategy_random(diff):
        return random.choice(diff)
        
    # We also try a strategy that removes the element with the smallest value (largest reciprocal)
    # and one that removes the largest value (smallest reciprocal)
    
    strategies = [
        strategy_min,
        strategy_max,
        strategy_random
    ]
    
    # Run multiple rounds with different strategies
    # Since the search space is small, we can run this many times
    for _ in range(100): # Max 100 restarts, but likely breaks early
        if time.time() - start_time > time_limit:
            break
            
        # Shuffle strategies or just iterate
        random.shuffle(strategies)
        
        for strat in strategies:
            if time.time() - start_time > time_limit:
                break
                
            S = set(range(22))
            
            while S:
                S_list = sorted(S)
                is_diss, mask1, mask2 = find_collision(S_list)
                
                if is_diss:
                    curr_set = [i + 1 for i in S_list]
                    if best_set is None or len(curr_set) > len(best_set):
                        best_set = curr_set
                    break
                
                # Get involved global indices
                diff_mask = mask1 ^ mask2
                involved = []
                for i in range(len(S_list)):
                    if (diff_mask >> i) & 1:
                        involved.append(S_list[i])
                
                if not involved:
                    # Should not happen
                    break
                    
                to_remove = strat(involved)
                S.remove(to_remove)
                
                if time.time() - start_time > time_limit:
                    break

    if best_set is None:
        return [1]
        
    return best_set
