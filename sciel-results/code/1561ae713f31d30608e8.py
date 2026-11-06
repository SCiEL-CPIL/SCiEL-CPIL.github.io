import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 80.0
    
    n = 30
    denom = [float(i + 1) for i in range(n)]
    
    def float_sum(c):
        return sum(c[i] / denom[i] for i in range(n))
    
    def exact_sum(c):
        from fractions import Fraction
        return sum(Fraction(c[i], i + 1) for i in range(n))
    
    # Seed solution
    best_c = [0] * 28 + [1, -1]
    best_val = abs(float_sum(best_c))
    
    # Simulated annealing
    best_float_val = best_val
    best_c_float = best_c[:]
    
    while time.time() < deadline - 5:
        # Random initialization
        c = [random.choice([-1, 0, 1]) for _ in range(n)]
        if all(x == 0 for x in c):
            c[0] = 1
        
        s = float_sum(c)
        if s == 0.0:
            continue
        val = abs(s)
        
        # Simulated annealing parameters
        T = 0.01
        T_min = 1e-15
        cooling = 0.99999
        c_cur = c[:]
        s_cur = s
        val_cur = val
        
        for step in range(100000):
            if time.time() > deadline - 5:
                break
            
            idx = random.randint(0, n - 1)
            old_v = c_cur[idx]
            new_choices = [v for v in (-1, 0, 1) if v != old_v]
            new_v = random.choice(new_choices)
            
            delta = (new_v - old_v) / denom[idx]
            s_new = s_cur + delta
            
            if s_new == 0.0:
                continue
            
            val_new = abs(s_new)
            
            if val_new < val_cur:
                c_cur[idx] = new_v
                s_cur = s_new
                val_cur = val_new
                if val_new < best_float_val:
                    best_float_val = val_new
                    best_c_float = c_cur[:]
            else:
                if T > T_min:
                    diff = val_new - val_cur
                    if diff < T * 100:
                        if random.random() < math.exp(-diff / T):
                            c_cur[idx] = new_v
                            s_cur = s_new
                            val_cur = val_new
                            if val_new < best_float_val:
                                best_float_val = val_new
                                best_c_float = c_cur[:]
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # Local search polish (single changes)
        if time.time() < deadline - 5:
            c_polish = best_c_float[:]
            s_polish = float_sum(c_polish)
            improved = True
            while improved and time.time() < deadline - 5:
                improved = False
                for i in range(n):
                    if time.time() > deadline - 5:
                        break
                    for new_v in (-1, 0, 1):
                        if new_v == c_polish[i]:
                            continue
                        old_v = c_polish[i]
                        delta = (new_v - old_v) / denom[i]
                        s_new = s_polish + delta
                        if s_new == 0.0:
                            continue
                        val_new = abs(s_new)
                        if val_new < abs(s_polish):
                            c_polish[i] = new_v
                            s_polish = s_new
                            improved = True
                            if abs(s_polish) < best_float_val:
                                best_float_val = abs(s_polish)
                                best_c_float = c_polish[:]
            
            # Pairwise local search
            if time.time() < deadline - 5:
                c_polish = best_c_float[:]
                s_polish = float_sum(c_polish)
                improved = True
                while improved and time.time() < deadline - 5:
                    improved = False
                    for i in range(n):
                        if time.time() > deadline - 5:
                            break
                        for j in range(i + 1, n):
                            if time.time() > deadline - 5:
                                break
                            for new_vi in (-1, 0, 1):
                                if new_vi == c_polish[i]:
                                    continue
                                for new_vj in (-1, 0, 1):
                                    if new_vj == c_polish[j]:
                                        continue
                                    old_vi = c_polish[i]
                                    old_vj = c_polish[j]
                                    delta = (new_vi - old_vi) / denom[i] + (new_vj - old_vj) / denom[j]
                                    s_new = s_polish + delta
                                    if s_new == 0.0:
                                        continue
                                    val_new = abs(s_new)
                                    if val_new < abs(s_polish):
                                        c_polish[i] = new_vi
                                        c_polish[j] = new_vj
                                        s_polish = s_new
                                        improved = True
                                        if abs(s_polish) < best_float_val:
                                            best_float_val = abs(s_polish)
                                            best_c_float = c_polish[:]
    
    # Final verification
    if best_c_float is not None:
        exact_s = exact_sum(best_c_float)
        if exact_s != 0:
            best_c = best_c_float
    
    # Ensure valid
    final_s = exact_sum(best_c)
    if final_s == 0:
        best_c = [0] * 28 + [1, -1]
    
    return best_c
