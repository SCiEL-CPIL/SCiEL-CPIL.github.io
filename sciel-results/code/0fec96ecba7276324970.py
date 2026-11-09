import time
import random
from math import gcd
from functools import reduce

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit

    def lcm(a, b):
        return abs(a * b) // gcd(a, b)
    
    # LCM of 1..30
    L = reduce(lcm, range(1, 31))
    
    # Precompute weights
    A = [0] * 31
    for i in range(1, 31):
        A[i] = L // i

    def get_sum(c):
        s = 0
        for i in range(30):
            if c[i] != 0:
                s += c[i] * A[i+1]
        return s

    def simulate_annealing(c_initial, start_temp, end_temp, steps):
        c = c_initial[:]
        current_sum = get_sum(c)
        
        # If initial sum is 0, it's invalid. We must move away.
        if current_sum == 0:
             # Find a valid move
             for i in range(30):
                 for v in [-1, 0, 1]:
                     if v == c[i]: continue
                     delta = (v - c[i]) * A[i+1]
                     new_sum = current_sum + delta
                     if new_sum != 0:
                         c[i] = v
                         current_sum = new_sum
                         break
                 if current_sum != 0: break
        
        best_c = c[:]
        best_val = abs(current_sum) if current_sum != 0 else float('inf')
        
        if current_sum != 0:
            best_c = c[:]
            best_val = abs(current_sum)
        else:
            # If we couldn't find a non-zero sum immediately (unlikely but possible if stuck),
            # we force a small change.
            # Actually, random init is almost never 0.
            pass

        for step in range(steps):
            if time.time() - start_time > time_limit:
                break
            
            # Calculate current temperature
            # Linear cooling is simple, exponential is standard.
            # T = T_start * (T_end/T_start)^(step/steps)
            # Or simply T = T_start * (1 - step/steps) + T_end
            # Let's use exponential: T = T_start * exp(-step/tau)
            # But we have fixed steps.
            progress = step / steps
            T = start_temp * (end_temp / start_temp) ** progress
            
            # Choose a move
            if random.random() < 0.5:
                # Single flip
                i = random.randint(0, 29)
                old_val = c[i]
                # Pick a new value different from old
                options = [v for v in [-1, 0, 1] if v != old_val]
                new_val = random.choice(options)
                delta = (new_val - old_val) * A[i+1]
                new_sum = current_sum + delta
                
                c[i] = new_val
            else:
                # Double flip
                i = random.randint(0, 29)
                j = random.randint(0, 29)
                if i == j:
                    continue
                
                old_i = c[i]
                old_j = c[j]
                
                opts_i = [v for v in [-1, 0, 1] if v != old_i]
                opts_j = [v for v in [-1, 0, 1] if v != old_j]
                
                new_i = random.choice(opts_i)
                new_j = random.choice(opts_j)
                
                delta = (new_i - old_i) * A[i+1] + (new_j - old_j) * A[j+1]
                new_sum = current_sum + delta
                
                c[i] = new_i
                c[j] = new_j

            # Evaluate
            if new_sum == 0:
                # Invalid state. We must revert or accept with penalty?
                # The problem requires non-zero sum.
                # If we land on 0, we treat it as a very bad state.
                # To keep the search moving, we revert to previous state immediately?
                # Or we can accept it but with high energy?
                # Simplest: Revert.
                if random.random() < 0.5:
                    # Revert
                    if 'i' in locals() and 'j' not in locals():
                        c[i] = old_val
                    else:
                        c[i] = old_i
                        c[j] = old_j
                    current_sum = new_sum - delta
                else:
                    # Keep 0? No, invalid. Revert.
                    if 'i' in locals() and 'j' not in locals():
                        c[i] = old_val
                    else:
                        c[i] = old_i
                        c[j] = old_j
                    current_sum = new_sum - delta
                continue

            # Valid move
            current_abs = abs(new_sum)
            
            if current_abs < best_val:
                best_val = current_abs
                best_c = c[:]
                current_sum = new_sum
                # Accept improvement always
            else:
                # Accept worse move with probability
                # Energy difference is in integer sum space
                # We need to map this to a probability.
                # The "energy" is |sum|.
                # Let's use a scaling factor for temperature.
                # The max possible sum is sum(A[i]) ~ 30 * L.
                # L is ~ 2e12.
                # So differences can be huge.
                # We need T to be comparable to typical |sum| differences.
                # Typical |sum| is around L/sqrt(30) if random?
                # Let's set T_start to be roughly L/100.
                
                delta_E = current_abs - abs(current_sum)
                if delta_E > 0:
                    # Worse
                    # p = exp(-delta_E / T)
                    # To avoid overflow/underflow, if delta_E/T > 50, p is ~0
                    if T < 1e-6:
                        p = 0
                    else:
                        ratio = delta_E / T
                        if ratio > 50:
                            p = 0
                        else:
                            p = 2.718281828 ** (-ratio)
                    
                    if random.random() < p:
                        current_sum = new_sum
                    else:
                        # Revert
                        if 'i' in locals() and 'j' not in locals():
                            c[i] = old_val
                        else:
                            c[i] = old_i
                            c[j] = old_j
                else:
                    # Improvement (handled above, but if logic missed it)
                    current_sum = new_sum

        return best_c, best_val

    best_global_c = [0] * 30
    best_global_c[28] = 1
    best_global_c[29] = -1
    best_global_val = abs(get_sum(best_global_c))
    
    # LCM is ~ 2.3e12
    # We want to explore small sums.
    # Start temp should be high enough to explore widely.
    # Let's try T_start = L / 1000
    T_start = L / 1000.0
    T_end = 1.0
    
    # Number of steps per run
    # We have ~55 seconds.
    # Python loop is slow. ~1e6 simple ops/sec?
    # Let's aim for 2e5 steps per run, maybe 20-50 runs.
    steps_per_run = 100000
    
    while time.time() - start_time < time_limit:
        # Random initialization
        c_init = [random.choice([-1, 0, 1]) for _ in range(30)]
        
        # Run SA
        c_res, val_res = simulate_annealing(c_init, T_start, T_end, steps_per_run)
        
        if val_res < best_global_val:
            best_global_val = val_res
            best_global_c = c_res
            
        # If we found 1, it's likely optimal (or close), break early
        if best_global_val == 1:
            break

    return best_global_c

