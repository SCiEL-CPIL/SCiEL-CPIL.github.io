def validate_and_fix(schedule, vertices, parents, inputs, outputs, budget):
    # Simulate the schedule and check validity
    fast = set()
    slow = set(inputs)
    computed = set()
    valid = True
    for op, v in schedule:
        if op == 'load':
            if v not in slow:
                valid = False
                break
            if len(fast) >= budget:
                valid = False
                break
            fast.add(v); slow.remove(v)
        elif op == 'store':
            if v not in fast:
                valid = False
                break
            fast.remove(v); slow.add(v)
        elif op == 'compute':
            for p in parents[v]:
                if p not in fast:
                    valid = False
                    break
            if not valid:
                break
            if len(fast) >= budget:
                valid = False
                break
            fast.add(v); computed.add(v)
        elif op == 'delete':
            if v not in fast:
                valid = False
                break
            fast.remove(v)
        else:
            valid = False
            break
    # Check all outputs are blue (in slow) at the end
    if valid:
        for o in outputs:
            if o not in slow:
                valid = False
                break
    if valid and len(schedule) <= 20000:
        return schedule
    # Fallback: return a simple valid schedule (might be suboptimal)
    return fallback_schedule(vertices, parents, inputs, outputs, budget)
