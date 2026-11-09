import random
import time

def construct():
    N = 128
    # Precompute all 4-APs
    aps = []
    for d in range(1, (N - 3) // 3 + 1):
        for start in range(N - 3 * d):
            aps.append((start, start + d, start + 2 * d, start + 3 * d))
    
    num_aps = len(aps)
    
    # For each index, store the list of AP indices it belongs to
    index_to_aps = [[] for _ in range(N)]
    for idx, ap in enumerate(aps):
        for pos in ap:
            index_to_aps[pos].append(idx)

    def is_valid_assignment(colors, i, c):
        """
        Check if assigning color c to index i is consistent with all 
        already assigned neighbors in any 4-AP containing i.
        """
        for ap_idx in index_to_aps[i]:
            ap = aps[ap_idx]
            # Get the other 3 indices
            others = [pos for pos in ap if pos != i]
            # If any of the others is unassigned, we can't check yet (it's not a violation yet)
            # We only check if ALL others are assigned.
            if all(colors[pos] != -1 for pos in others):
                # Check if the set of colors in this AP has < 3 distinct values
                # Colors at others + c
                c_set = {c}
                for pos in others:
                    c_set.add(colors[pos])
                if len(c_set) < 3:
                    return False
        return True

    def try_construct(K, max_restarts=200):
        for _ in range(max_restarts):
            colors = [-1] * N
            order = list(range(N))
            random.shuffle(order)
            
            success = True
            for i in order:
                valid_colors = []
                for c in range(K):
                    if is_valid_assignment(colors, i, c):
                        valid_colors.append(c)
                
                if not valid_colors:
                    success = False
                    break
                
                # Random choice among valid colors to explore different paths
                colors[i] = random.choice(valid_colors)
            
            if success:
                # Final verification (paranoid check)
                for ap in aps:
                    if len({colors[ap[0]], colors[ap[1]], colors[ap[2]], colors[ap[3]]}) < 3:
                        success = False
                        break
                if success:
                    return colors
        return None

    start_time = time.time()
    time_limit = 55.0
    
    # Try K from 3 upwards
    for K in range(3, 15):
        if time.time() - start_time > time_limit:
            break
            
        # Increase restarts for smaller K if time permits? 
        # Or just fixed restarts.
        # For K=3, it might be hard. For K=4, easier.
        # Let's give more tries to smaller K if we have time.
        # But simple fixed restarts is safer for time management.
        
        res = try_construct(K, max_restarts=100)
        if res is not None:
            return res

    # Fallback: If we didn't find a solution with K < 15, 
    # we must return a valid solution. 
    # A trivial valid solution is colors[i] = i (uses 128 colors).
    # But we can do better: use a pattern that guarantees validity.
    # E.g., colors[i] = i % 5 ? No, 4-APs might collide.
    # Just use distinct colors for a safe fallback, or a known good pattern.
    # Actually, if K=10 failed, K=11, 12... should be very easy.
    # If we hit time limit, we return a valid coloring.
    # Let's try one more quick local search with K=10 if we have time?
    # No, just return a valid one.
    
    # A very safe valid coloring:
    # colors[i] = i
    # This uses 128 colors.
    # Can we do better quickly?
    # If we have time, run try_construct(15).
    if time.time() - start_time < time_limit:
         res = try_construct(15, max_restarts=50)
         if res is not None:
             return res

    # Absolute fallback: Identity
    return list(range(N))
