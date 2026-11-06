import time
import random

def construct():
    N = 1000
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Greedy Initialization
    elements = []
    is_in = [False] * (N + 1)
    counts = [0] * (2 * N + 1)
    violations = 0
    
    for x in range(1, N + 1):
        temp_viol = 0
        for y in elements:
            if counts[x + y] >= 2:
                temp_viol += 1
        if counts[2 * x] >= 2:
            temp_viol += 1
        if temp_viol == 0:
            is_in[x] = True
            elements.append(x)
            for y in elements[:-1]:
                counts[x + y] += 1
            counts[2 * x] += 1
    
    best_set = list(elements)
    best_size = len(elements)
    current_size = len(elements)
    current_violations = violations
    
    # Recalculate violations accurately
    current_violations = 0
    for s in range(2, 2 * N + 1):
        if counts[s] > 2:
            current_violations += counts[s] - 2

    # SA Parameters
    T = 5.0
    T_min = 0.1
    cooling_rate = 0.9995
    PENALTY = 50
    
    def do_add(x):
        nonlocal current_size, current_violations
        is_in[x] = True
        elements.append(x)
        current_size += 1
        d_v = 0
        for y in elements[:-1]:
            s = x + y
            if counts[s] >= 2:
                d_v += 1
            counts[s] += 1
        s = 2 * x
        if counts[s] >= 2:
            d_v += 1
        counts[s] += 1
        current_violations += d_v

    def do_remove(x):
        nonlocal current_size, current_violations
        is_in[x] = False
        elements.remove(x)
        current_size -= 1
        d_v = 0
        for y in elements:
            s = x + y
            if counts[s] >= 3:
                d_v -= 1
            counts[s] -= 1
        s = 2 * x
        if counts[s] >= 3:
            d_v -= 1
        counts[s] -= 1
        current_violations += d_v

    # SA Loop
    while time.time() < deadline:
        if random.random() < 0.5:
            x = random.randint(1, N)
            if not is_in[x]:
                # Calculate d_viol for acceptance
                d_v = 0
                for y in elements:
                    if counts[x + y] >= 2:
                        d_v += 1
                if counts[2 * x] >= 2:
                    d_v += 1
                dE = -1 + PENALTY * d_v
                if dE < 0 or (T > T_min and random.random() < pow(2.71828, -dE / T)):
                    do_add(x)
                    if current_violations == 0 and current_size > best_size:
                        best_size = current_size
                        best_set = list(elements)
        else:
            if elements:
                x = random.choice(elements)
                d_v = 0
                for y in elements:
                    if y != x and counts[x + y] >= 3:
                        d_v -= 1
                if counts[2 * x] >= 3:
                    d_v -= 1
                dE = 1 + PENALTY * d_v
                if dE < 0 or (T > T_min and random.random() < pow(2.71828, -dE / T)):
                    do_remove(x)
                    if current_violations == 0 and current_size > best_size:
                        best_size = current_size
                        best_set = list(elements)
        
        T *= cooling_rate
        if T < T_min:
            T = T_min

    return best_set
