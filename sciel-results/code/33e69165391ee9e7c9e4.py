def dispatch(ready_operations, time):
        if not ready_operations:
            return 0 # or handle error
        
        best_idx = 0
        best_machine = None
        min_score = float('inf')
        
        for i, op in enumerate(ready_operations):
            options = op['machine_options']
            if not options:
                continue
            
            # Find the best machine for this operation (shortest duration)
            # This represents the "least loaded" assumption if all are free,
            # or simply the most efficient machine.
            best_m = min(options.keys(), key=lambda m: options[m])
            dur = options[best_m]
            
            # Heuristic: Shortest Processing Time (SPT)
            # We want to minimize the duration to clear the bottleneck quickly.
            # Tie-breaker: If durations are equal, prefer job with more remaining work?
            # Or just stable sort?
            
            # Let's use a composite score: Primary is duration, Secondary is remaining work (higher is better, so we subtract or invert logic)
            # To keep it simple and robust: Minimize duration.
            if dur < min_score:
                min_score = dur
                best_idx = i
                best_machine = best_m
            elif dur == min_score:
                # Tie-breaker: Prefer job with more remaining work to maintain urgency
                if op['remaining_work'] > ready_operations[best_idx]['remaining_work']:
                    best_idx = i
                    best_machine = best_m
        
        # Return index and optionally the best machine
        # The interface allows (index, machine). Returning the specific machine
        # that minimizes duration is a valid optimization.
        return (best_idx, best_machine)
