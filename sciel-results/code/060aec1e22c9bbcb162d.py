def dispatch(ready_operations, time):
    if not ready_operations:
        return 0, 0

    best_idx = 0
    best_machine = None
    best_min_dur = float('inf')
    best_work = -1

    for i, op in enumerate(ready_operations):
        if op["machine_options"]:
            min_dur, best_mach = min(op["machine_options"].items(), key=lambda x: x[1])
        else:
            min_dur = 0
            best_mach = 0

        # Primary: Shortest Duration
        # Secondary: Most Work Remaining
        if min_dur < best_min_dur:
            best_min_dur = min_dur
            best_work = op["remaining_work"]
            best_idx = i
            best_machine = best_mach
        elif min_dur == best_min_dur:
            if op["remaining_work"] > best_work:
                best_work = op["remaining_work"]
                best_idx = i
                best_machine = best_mach

    return best_idx, best_machine
