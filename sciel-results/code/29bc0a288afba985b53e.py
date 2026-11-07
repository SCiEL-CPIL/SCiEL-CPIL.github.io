class TimeExceeded(Exception):
    pass

def wrapped_func(x):
    nonlocal calls, best_val, best_x
    if time.time() - t0 > 320:
        raise TimeExceeded()
    val = func(x)
    calls += 1
    if val < best_val:
        best_val = val
        best_x = x.copy()
    return val
