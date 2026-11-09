import random
import time
import math

def construct():
    n = 256
    # Precompute 3-APs and 4-APs for each element
    # three_aps[x] = list of (y, z) such that {x,y,z} is a 3-AP
    # four_aps[x] = list of (y, z, w) such that {x,y,z,w} is a 4-AP
    
    three_aps = [[] for _ in range(n + 1)]
    four_aps = [[] for _ in range(n + 1)]
    
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 2 + 1):
            b = a + d
            c = a + 2 * d
            three_aps[a].append((b, c))
            three_aps[b].append((a, c))
            three_aps[c].append((a, b))
            
    for a in range(1, n + 1):
        for d in range(1, (n - a) // 3 + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            four_aps[a].append((b, c, e))
            four_aps[b].append((a, c, e))
            four_aps[c].append((a, b, e))
            four_aps[e].append((a, b, c))

    # Helper to count 3-APs in a set represented by a boolean list
    def count_aps(in_A):
        c3 = 0
        for x in range(1, n + 1):
            if in_A[x]:
                for y, z in three_aps[x]:
                    if in_A[y] and in_A[z]:
                        c3 += 1
        return c3 // 3

    # Initial Solution: Greedy
    in_A = [False] * (n + 1)
    A_set = set()
    
    # Greedy initialization to get a good starting point
    for _ in range(n):
        best_x = -1
        best_gain = -1
        for x in range(1, n + 1):
            if in_A[x]:
                continue
            valid = True
            for y, z, w in four_aps[x]:
                if in_A[y] and in_A[z] and in_A[w]:
                    valid = False
                    break
            if not valid:
                continue
            gain = 0
            for y, z in three_aps[x]:
                if in_A[y] and in_A[z]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x == -1:
            break
        in_A[best_x] = True
        A_set.add(best_x)

    best_score = count_aps(in_A)
    best_A = [i for i in range(1, n + 1) if in_A[i]]
    
    start_time = time.time()
    timeout = 55.0
    
    # Simulated Annealing Parameters
    T = 10.0 # Initial temperature
    cooling_rate = 0.999 # Slow cooling
    min_T = 0.01
    
    current_score = best_score
    current_in_A = in_A[:]
    current_A_set = A_set.copy()
    
    while time.time() - start_time < timeout:
        if T < min_T:
            break
            
        # Pick a random element to flip
        x = random.randint(1, n)
        
        if current_in_A[x]:
            # Try to remove x
            # Removing x is always valid regarding 4-APs
            # Calculate delta score
            delta = 0
            for y, z in three_aps[x]:
                if current_in_A[y] and current_in_A[z]:
                    delta -= 1 # We lose this 3-AP
            
            # Acceptance probability
            if delta > 0:
                accept = True
            else:
                # Energy change is -delta (we minimize energy, maximize score)
                # Delta_E = -delta
                # P = exp(-Delta_E / T) = exp(delta / T)
                # Since delta <= 0, exp(delta/T) < 1
                accept = random.random() < math.exp(delta / T)
            
            if accept:
                current_in_A[x] = False
                current_A_set.remove(x)
                current_score += delta
                if current_score > best_score:
                    best_score = current_score
                    best_A = [i for i in range(1, n + 1) if current_in_A[i]]
        else:
            # Try to add x
            # Check 4-AP validity
            valid = True
            for y, z, w in four_aps[x]:
                if current_in_A[y] and current_in_A[z] and current_in_A[w]:
                    valid = False
                    break
            
            if valid:
                # Calculate delta score
                delta = 0
                for y, z in three_aps[x]:
                    if current_in_A[y] and current_in_A[z]:
                        delta += 1 # We gain this 3-AP
                
                # Acceptance probability
                if delta > 0:
                    accept = True
                else:
                    accept = random.random() < math.exp(delta / T)
                
                if accept:
                    current_in_A[x] = True
                    current_A_set.add(x)
                    current_score += delta
                    if current_score > best_score:
                        best_score = current_score
                        best_A = [i for i in range(1, n + 1) if current_in_A[i]]
        
        # Cool down
        T *= cooling_rate
        # If T gets too low, reheat slightly to escape deep local minima?
        # Or just let it run.
        
    return best_A
