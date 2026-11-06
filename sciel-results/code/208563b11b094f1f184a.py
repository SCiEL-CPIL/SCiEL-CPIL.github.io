def dispatch(ready_operations, time):
        if not ready_operations:
            return 0

        best_idx = -1
        best_machine = None
        best_min_dur = float('inf')
        best_remaining_ops = float('inf')
        best_remaining_work = -1.0

        for i, op in enumerate(ready_operations):
            machine_options = op["machine_options"]
            if not machine_options:
                continue
                
            # Find min duration machine
            min_dur = float('inf')
            min_mach = None
            for m, d in machine_options.items():
                if d < min_dur:
                    min_dur = d
                    min_mach = m
            
            # Compare
            if (min_dur < best_min_dur or
                (min_dur == best_min_dur and op["remaining_ops"] < best_remaining_ops) or
                (min_dur == best_min_dur and op["remaining_ops"] == best_remaining_ops and op["remaining_work"] > best_remaining_work)):
                
                best_min_dur = min_dur
                best_idx = i
                best_machine = min_mach
                best_remaining_ops = op["remaining_ops"]
                best_remaining_work = op["remaining_work"]
        
        if best_idx == -1:
            return 0
            
        return (best_idx, best_machine)
