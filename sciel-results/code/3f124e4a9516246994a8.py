import random
import math
import time

def construct():
    N = 256
    
    # Precompute all 4-term APs
    aps = []
    index_to_aps = [[] for _ in range(N)]
    
    ap_id = 0
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            i0 = a - 1
            i1 = a + d - 1
            i2 = a + 2 * d - 1
            i3 = a + 3 * d - 1
            
            aps.append((i0, i1, i2, i3))
            index_to_aps[i0].append(ap_id)
            index_to_aps[i1].append(ap_id)
            index_to_aps[i2].append(ap_id)
            index_to_aps[i3].append(ap_id)
            ap_id += 1
            
    total_aps = len(aps)
    
    def initial_zero_counts(colors):
        zc = [0] * total_aps
        for k in range(total_aps):
            ap = aps[k]
            # Count zeros
            count = 0
            if colors[ap[0]] == 0: count += 1
            if colors[ap[1]] == 0: count += 1
            if colors[ap[2]] == 0: count += 1
            if colors[ap[3]] == 0: count += 1
            zc[k] = count
        return zc

    def calc_mono_count(zc):
        count = 0
        for z in zc:
            if z == 0 or z == 4:
                count += 1
        return count

    best_colors = None
    best_count = float('inf')
    
    # Simulated Annealing parameters
    # We run multiple restarts to be safe within time limit
    start_time = time.time()
    time_limit = 110.0  # Leave 10s buffer
    
    # Number of restarts is dynamic based on time, but let's fix a few
    # Each restart runs for a certain number of steps or time slice
    num_restarts = 10 
    steps_per_restart = 50000
    
    for restart in range(num_restarts):
        if time.time() - start_time > time_limit:
            break
            
        random.seed(1000 + restart)
        
        colors = [random.randint(0, 1) for _ in range(N)]
        zc = initial_zero_counts(colors)
        current_energy = calc_mono_count(zc)
        
        if current_energy < best_count:
            best_count = current_energy
            best_colors = colors[:]
            
        T = 1.0
        T_min = 0.01
        cooling_rate = 0.999 # Adjust based on steps
        
        for step in range(steps_per_restart):
            if time.time() - start_time > time_limit:
                break
                
            # Pick random index
            idx = random.randint(0, N - 1)
            
            # Calculate delta
            # If colors[idx] is 0, flipping to 1 decreases zero count by 1
            # If colors[idx] is 1, flipping to 0 increases zero count by 1
            c = colors[idx]
            delta = 0
            affected = index_to_aps[idx]
            
            for k in affected:
                z = zc[k]
                if c == 0:
                    # Flip 0 -> 1. z becomes z-1
                    new_z = z - 1
                    if z == 4 and new_z != 0 and new_z != 4:
                        delta -= 1
                    elif z == 1 and new_z == 0:
                        delta += 1
                    # Note: if z==4, new_z=3 (not mono). delta -1.
                    # if z==1, new_z=0 (mono). delta +1.
                    # if z==0, new_z=-1 (impossible, z>=1 if c==0? No, z is count of zeros. If c==0, z>=1. If c==1, z can be 0).
                    # Wait, if c==0, z is at least 1.
                    # If z==4 (all 0), flip one 0 to 1 -> 3 zeros, 1 one. Not mono. Was mono. Delta -1.
                    # If z==1 (one 0, three 1s), flip 0 to 1 -> 0 zeros, 4 ones. Mono. Was not mono. Delta +1.
                    # If z==2 or 3, no change in mono status.
                else:
                    # Flip 1 -> 0. z becomes z+1
                    new_z = z + 1
                    if z == 0 and new_z != 0 and new_z != 4:
                        delta -= 1
                    elif z == 3 and new_z == 4:
                        delta += 1
                    # If z==0 (all 1), flip one 1 to 0 -> 1 zero, 3 ones. Not mono. Was mono. Delta -1.
                    # If z==3 (three 0, one 1), flip 1 to 0 -> 4 zeros, 0 ones. Mono. Was not mono. Delta +1.
            
            # Acceptance
            if delta < 0:
                accept = True
            else:
                if T > T_min:
                    # Avoid overflow/underflow in exp
                    # If delta is large, exp(-delta/T) is very small
                    try:
                        prob = math.exp(-delta / T)
                    except OverflowError:
                        prob = 0.0
                    accept = random.random() < prob
                else:
                    accept = False
            
            if accept:
                # Apply move
                colors[idx] = 1 - c
                for k in affected:
                    if c == 0:
                        zc[k] -= 1
                    else:
                        zc[k] += 1
                current_energy += delta
                
                if current_energy < best_count:
                    best_count = current_energy
                    best_colors = colors[:]
            
            T *= cooling_rate
            
    return best_colors
