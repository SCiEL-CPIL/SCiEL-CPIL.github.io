    restarts = 0
    max_restarts = 10
    
    while time.time() - start_time < timeout:
        # ... removal logic ...
        
        if not removed:
            # ... plateau logic ...
            
            if not plateau_moved:
                restarts += 1
                if restarts > max_restarts:
                    break
                # Reset current_network to best_network?
                # Or try a different randomization?
                # Let's just break for now to be safe.
                break
            else:
                restarts = 0
