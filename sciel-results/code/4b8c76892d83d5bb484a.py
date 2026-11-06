    remaining_budget = 110.0
    for n in range(34, 14, -1):
        if time.time() > start_time + 110.0:
            break
            
        # Allocate some time for this n
        # Higher n is harder, give it more time?
        # Or just give it a fixed chunk, e.g., 5 seconds per n?
        # 20 * 5 = 100s. Fits.
        
        local_deadline = time.time() + 5.0
        if local_deadline > start_time + 110.0:
            local_deadline = start_time + 110.0
            
        success = False
        while time.time() < local_deadline:
            colors, cost = local_search(n, local_deadline)
            if cost == 0:
                success = True
                best_n = n
                best_colors = colors
                break
        
        if success:
            break
