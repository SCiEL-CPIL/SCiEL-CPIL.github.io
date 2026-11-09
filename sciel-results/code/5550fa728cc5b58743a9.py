import random
import time
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    limit = 85.0
    
    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    L = reduce(lcm, range(1, 31))
    
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i
        
    seed_val = A[29] - A[30]
    best_val = abs(seed_val)
    best_c = [0] * 30
    best_c[28] = 1
    best_c[29] = -1
    
    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i+1]
        return s
        
    def try_update(new_c, new_sum):
        nonlocal best_val, best_c
        if new_sum != 0:
            new_abs = abs(new_sum)
            if new_abs < best_val:
                best_val = new_abs
                best_c = new_c[:]
                
    def local_search(c):
        current_sum = get_sum(c)
        
        if current_sum != 0:
            try_update(c, current_sum)
        
        steps = 0
        while time.time() - start_time < limit:
            if current_sum == 0:
                # If we hit 0, we must escape.
                # Try all single moves that make it non-zero.
                # If any single move makes it non-zero and small, take it.
                # Otherwise, try double moves.
                improved = False
                # Single moves
                for i in range(30):
                    old_val = c[i]
                    for new_val in [-1, 0, 1]:
                        if new_val == old_val:
                            continue
                        delta = (new_val - old_val) * A[i+1]
                        new_sum = current_sum + delta
                        if new_sum != 0:
                            # Check if this is better than current best or just escape?
                            # We just need to escape 0 to continue searching.
                            # But if it's better than best_val, great.
                            c[i] = new_val
                            current_sum = new_sum
                            try_update(c, new_sum)
                            improved = True
                            break
                    if improved:
                        break
                if not improved:
                    # Try double moves
                    for i in range(30):
                        for j in range(i+1, 30):
                            old_i = c[i]
                            old_j = c[j]
                            for new_i in [-1, 0, 1]:
                                for new_j in [-1, 0, 1]:
                                    if (new_i == old_i and new_j == old_j):
                                        continue
                                    delta = (new_i - old_i) * A[i+1] + (new_j - old_j) * A[j+1]
                                    new_sum = current_sum + delta
                                    if new_sum != 0:
                                        c[i] = new_i
                                        c[j] = new_j
                                        current_sum = new_sum
                                        try_update(c, new_sum)
                                        improved = True
                                        break
                                if improved:
                                    break
                            if improved:
                                break
                        if improved:
                            break
                if not improved:
                    # Stuck, restart
                    break
            else:
                # Normal hill climbing
                idx = random.randint(0, 29)
                old_val = c[idx]
                improved = False
                for new_val in [-1, 0, 1]:
                    if new_val == old_val:
                        continue
                    delta = (new_val - old_val) * A[idx+1]
                    new_sum = current_sum + delta
                    if new_sum != 0 and abs(new_sum) < abs(current_sum):
                        c[idx] = new_val
                        current_sum = new_sum
                        improved = True
                        try_update(c, new_sum)
                        break
                
                if not improved:
                    # Try double flip to escape local min
                    i = random.randint(0, 29)
                    j = random.randint(0, 29)
                    if i == j:
                        continue
                    old_i = c[i]
                    old_j = c[j]
                    double_improved = False
                    for new_i in [-1, 0, 1]:
                        for new_j in [-1, 0, 1]:
                            if (new_i == old_i and new_j == old_j):
                                continue
                            delta = (new_i - old_i) * A[i+1] + (new_j - old_j) * A[j+1]
                            new_sum = current_sum + delta
                            if new_sum != 0 and abs(new_sum) < abs(current_sum):
                                c[i] = new_i
                                c[j] = new_j
                                current_sum = new_sum
                                double_improved = True
                                try_update(c, new_sum)
                                break
                        if double_improved:
                            break
                    
                    if not double_improved:
                        # Random jump
                        if random.random() < 0.05:
                            c[idx] = random.choice([-1, 0, 1])
                            current_sum = get_sum(c)
                            if current_sum != 0:
                                try_update(c, current_sum)
                        else:
                            break

    # Run restarts
    while time.time() - start_time < limit:
        c = [random.choice([-1, 0, 1]) for _ in range(30)]
        local_search(c)
        if best_val == 1:
            break

    return best_c
