"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    start_time = time.time()
    deadline = start_time + 80.0  # 10s safety margin before 90s limit
    
    values, sums = [], set()
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    return values
# EVOLVE-BLOCK-END
