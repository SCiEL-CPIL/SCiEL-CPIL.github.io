import time
import random
import math

N = 256
DEADLINE = 85.0

def _precompute():
    """
    Precompute data structures for fast incremental updates.
    Returns:
    ap3_partners: list of lists of tuples (j, k) for each i in 1..N
    ap4_partners: list of lists of tuples (j, k, l) for each i in 1..N
    """
    ap3_partners = [[] for _ in range(N + 1)]
    ap4_partners = [[] for _ in range(N + 1)]
    
    # 3-APs: (a, a+d, a+2d)
    for d in range(1, N // 2 + 1):
        for a in range(1, N - 2 * d + 1):
            b = a + d
            c = a + 2 * d
            # Update partners for a, b, c
            ap3_partners[a].append((b, c))
            ap3_partners[b].append((a, c))
            ap3_partners[c].append((a, b))
            
    # 4-APs: (a, a+d, a+2d, a+3d)
    for d in range(1, N // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # Update partners for a, b, c, e
            ap4_partners[a].append((b, c, e))
            ap4_partners[b].append((a, c, e))
            ap4_partners[c].append((a, b, e))
            ap4_partners[e].append((a, b, c))
            
    return ap3_partners, ap4_partners

def _count_3aps_slow(in_set, ap3_partners):
    """Slow count for verification or initial setup."""
    total = 0
    for i in range(1, N + 1):
        if in_set[i]:
            for j, k in ap3_partners[i]:
                if in_set[j] and in_set[k]:
                    total += 1
    return total // 3

def construct():
    start_time = time.time()
    ap3_partners, ap4_partners = _precompute()
    
    best_score = -1
    best_set = [False] * (N + 1)
    
    num_restarts = 8
    time_per_restart = (DEADLINE - 8.0) / num_restarts
    
    for restart in range(num_restarts):
        # Check global deadline
        if time.time() - start_time > DEADLINE - 6.0:
            break
            
        # Initialize with greedy random
        in_set = [False] * (N + 1)
        candidates = list(range(1, N + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            valid = True
            for j, k, l in ap4_partners[x]:
                if in_set[j] and in_set[k] and in_set[l]:
                    valid = False
                    break
            if valid:
                in_set[x] = True
        
        # Maintain present/absent lists incrementally
        present = [i for i in range(1, N + 1) if in_set[i]]
        absent = [i for i in range(1, N + 1) if not in_set[i]]
        
        # Calculate initial score
        score = _count_3aps_slow(in_set, ap3_partners)
        
        # Simulated Annealing parameters
        T = 15.0
        T_min = 0.05
        cooling = 0.999
        steps = 0
        
        # We will run until time slice is up
        restart_deadline = start_time + time_per_restart
        while time.time() < restart_deadline:
            if time.time() - start_time > DEADLINE - 2.0:
                break
            
            # Choose move type
            # 60% Swap, 20% Add, 20% Remove
            r = random.random()
            
            if r < 0.6:
                # Swap: remove x, add y
                if not present or not absent:
                    T *= cooling
                    steps += 1
                    continue
                
                x = random.choice(present)
                y = random.choice(absent)
                
                # Check 4-AP constraint for adding y to S \ {x}
                valid = True
                for j, k, l in ap4_partners[y]:
                    if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
                        valid = False
                        break
                
                if not valid:
                    T *= cooling
                    steps += 1
                    continue
                
                # Calculate Delta
                delta_rem = 0
                for j, k in ap3_partners[x]:
                    if in_set[j] and in_set[k]:
                        delta_rem += 1
                        
                delta_add = 0
                for j, k in ap3_partners[y]:
                    if (in_set[j] and j != x) and (in_set[k] and k != x):
                        delta_add += 1
                        
                delta = delta_add - delta_rem
                
                # Metropolis criterion
                if delta > 0 or random.random() < math.exp(delta / T):
                    # Accept
                    in_set[x] = False
                    in_set[y] = True
                    score += delta
                    # Update present/absent lists
                    present.remove(x)
                    present.append(y)
                    absent.remove(y)
                    absent.append(x)
                    if score > best_score:
                        best_score = score
                        best_set = in_set[:]
                        
            elif r < 0.8:
                # Add y
                if not absent:
                    T *= cooling
                    steps += 1
                    continue
                y = random.choice(absent)
                
                # Check 4-AP
                valid = True
                for j, k, l in ap4_partners[y]:
                    if in_set[j] and in_set[k] and in_set[l]:
                        valid = False
                        break
                if not valid:
                    T *= cooling
                    steps += 1
                    continue
                    
                # Delta
                delta = 0
                for j, k in ap3_partners[y]:
                    if in_set[j] and in_set[k]:
                        delta += 1
                        
                if delta > 0 or random.random() < math.exp(delta / T):
                    in_set[y] = True
                    score += delta
                    present.append(y)
                    absent.remove(y)
                    if score > best_score:
                        best_score = score
                        best_set = in_set[:]
                        
            else:
                # Remove x
                if not present:
                    T *= cooling
                    steps += 1
                    continue
                x = random.choice(present)
                
                # Delta
                delta = 0
                for j, k in ap3_partners[x]:
                    if in_set[j] and in_set[k]:
                        delta += 1
                        
                delta = -delta # Removing decreases score by this amount
                
                if delta > 0 or random.random() < math.exp(delta / T):
                    in_set[x] = False
                    score += delta
                    present.remove(x)
                    absent.append(x)
                    if score > best_score:
                        best_score = score
                        best_set = in_set[:]
                        
            T *= cooling
            if T < T_min:
                T = T_min
            steps += 1
            
    # Final local search polish on best_set
    if time.time() - start_time < DEADLINE - 1.0 and best_score > 0:
        in_set = best_set[:]
        score = best_score
        present = [i for i in range(1, N + 1) if in_set[i]]
        absent = [i for i in range(1, N + 1) if not in_set[i]]
        
        # Hill climbing: only accept improvements
        polish_steps = 0
        polish_deadline = min(start_time + DEADLINE - 0.5, time.time() + 5.0)
        while time.time() < polish_deadline and polish_steps < 20000:
            # Try all single swaps
            improved = False
            for x in present[:]:
                for y in absent[:]:
                    # Check 4-AP constraint for adding y to S \ {x}
                    valid = True
                    for j, k, l in ap4_partners[y]:
                        if (in_set[j] and j != x) and (in_set[k] and k != x) and (in_set[l] and l != x):
                            valid = False
                            break
                    if not valid:
                        continue
                    
                    # Calculate Delta
                    delta_rem = 0
                    for j, k in ap3_partners[x]:
                        if in_set[j] and in_set[k]:
                            delta_rem += 1
                    delta_add = 0
                    for j, k in ap3_partners[y]:
                        if (in_set[j] and j != x) and (in_set[k] and k != x):
                            delta_add += 1
                    delta = delta_add - delta_rem
                    
                    if delta > 0:
                        # Accept improvement
                        in_set[x] = False
                        in_set[y] = True
                        score += delta
                        present.remove(x)
                        present.append(y)
                        absent.remove(y)
                        absent.append(x)
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = in_set[:]
                        break
                if improved:
                    break
            
            # Also try single adds
            if not improved:
                for y in absent[:]:
                    valid = True
                    for j, k, l in ap4_partners[y]:
                        if in_set[j] and in_set[k] and in_set[l]:
                            valid = False
                            break
                    if not valid:
                        continue
                    delta = 0
                    for j, k in ap3_partners[y]:
                        if in_set[j] and in_set[k]:
                            delta += 1
                    if delta > 0:
                        in_set[y] = True
                        score += delta
                        present.append(y)
                        absent.remove(y)
                        improved = True
                        if score > best_score:
                            best_score = score
                            best_set = in_set[:]
                        break
            
            if not improved:
                break
            polish_steps += 1
    
    # Convert best_set to list
    result = [i for i in range(1, N + 1) if best_set[i]]
    return result
