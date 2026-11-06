import time
import random
import math
import bisect

def _score(S_list, S_bits):
    """
    S_list must be sorted.
    S_bits is a bitset integer where bit i is 1 if i is in S.
    Returns count of distinct d > 0 such that a, a+d, a+2d in S.
    """
    diffs = set()
    n = len(S_list)
    for i in range(n):
        a = S_list[i]
        for j in range(i + 1, n):
            b = S_list[j]
            d = b - a
            c = b + d
            if c <= 512:
                if (S_bits >> c) & 1:
                    diffs.add(d)
    return len(diffs)

def _score_fast(S_bits):
    """
    Fast score using only the bitset.
    For each d from 1 to 255, check if there exists a such that
    a, a+d, a+2d are all in S.
    """
    count = 0
    for d in range(1, 256):
        # Check all possible a: a, a+d, a+2d <= 512
        # a ranges from 1 to 512 - 2d
        found = False
        for a in range(1, 513 - 2*d):
            if ((S_bits >> a) & 1) and ((S_bits >> (a+d)) & 1) and ((S_bits >> (a+2*d)) & 1):
                found = True
                break
        if found:
            count += 1
    return count

def _bits_to_list(S_bits):
    """Convert bitset to sorted list of integers."""
    result = []
    for i in range(1, 513):
        if (S_bits >> i) & 1:
            result.append(i)
    return result

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    N = 32
    MAXV = 512
    
    best_score = -1
    best_list = []
    
    # Precompute: for each d, the list of valid starting positions a
    # a, a+d, a+2d all in [1, 512]
    # a ranges from 1 to 512 - 2*d
    # This is used in _score_fast implicitly
    
    # Strategy 1: Try a structured initial set (e.g., powers or geometric)
    # Strategy 2: Random initialization with local search
    
    # Run multiple restarts if time permits
    num_restarts = 0
    while time.time() < deadline:
        # Choose initialization strategy based on restart number
        if num_restarts % 3 == 0:
            # Random initialization
            current_list = random.sample(range(1, MAXV + 1), N)
        elif num_restarts % 3 == 1:
            # Geometric-like initialization: pick numbers with varied spacing
            # Start with a base and multiply by random factors
            current_list = []
            base = random.randint(1, 10)
            while len(current_list) < N:
                val = base
                while val <= MAXV and val not in current_list:
                    current_list.append(val)
                    val = int(val * random.uniform(1.1, 1.5))
            if len(current_list) < N:
                # Fill remaining randomly
                remaining = [x for x in range(1, MAXV+1) if x not in current_list]
                current_list.extend(random.sample(remaining, N - len(current_list)))
        else:
            # Clustered initialization: pick clusters at different scales
            current_list = []
            while len(current_list) < N:
                cluster_center = random.randint(1, MAXV)
                cluster_size = random.randint(3, 8)
                for _ in range(cluster_size):
                    if len(current_list) >= N:
                        break
                    val = cluster_center + random.randint(-10, 10)
                    val = max(1, min(MAXV, val))
                    if val not in current_list:
                        current_list.append(val)
        
        current_list.sort()
        current_bits = 0
        for x in current_list:
            current_bits |= (1 << x)
        
        current_score = _score(current_list, current_bits)
        
        if current_score > best_score:
            best_score = current_score
            best_list = current_list.copy()
        
        # Simulated Annealing with adaptive temperature
        T = 3.0
        T_min = 0.01
        cooling = 0.998
        
        # Local time limit for this restart
        local_deadline = start_time + min(15.0, deadline - start_time)
        
        accepted = 0
        total = 0
        
        while time.time() < local_deadline:
            # Pick move type: 70% swap, 20% remove+add, 10% two-swap
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap: remove x, add y
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                # Pick y to add
                while True:
                    y = random.randint(1, MAXV)
                    if not ((current_bits >> y) & 1):
                        break
                
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                new_list = current_list[:x_idx] + current_list[x_idx+1:]
                ins_idx = bisect.bisect_left(new_list, y)
                new_list.insert(ins_idx, y)
                
            elif move_type < 0.9:
                # Remove one, add one (same as swap but different name for clarity)
                x_idx = random.randint(0, N - 1)
                x = current_list[x_idx]
                
                while True:
                    y = random.randint(1, MAXV)
                    if not ((current_bits >> y) & 1):
                        break
                
                new_bits = current_bits ^ (1 << x) ^ (1 << y)
                new_list = current_list[:x_idx] + current_list[x_idx+1:]
                ins_idx = bisect.bisect_left(new_list, y)
                new_list.insert(ins_idx, y)
                
            else:
                # Two-swap: remove x1, add y1, remove x2, add y2
                x1_idx = random.randint(0, N - 1)
                x2_idx = random.randint(0, N - 1)
                if x1_idx == x2_idx:
                    x2_idx = (x2_idx + 1) % N
                x1 = current_list[x1_idx]
                x2 = current_list[x2_idx]
                
                while True:
                    y1 = random.randint(1, MAXV)
                    if not ((current_bits >> y1) & 1):
                        break
                
                temp_bits = current_bits ^ (1 << x1)
                while True:
                    y2 = random.randint(1, MAXV)
                    if not ((temp_bits >> y2) & 1) and y2 != y1:
                        break
                
                new_bits = current_bits ^ (1 << x1) ^ (1 << x2) ^ (1 << y1) ^ (1 << y2)
                
                # Build new list
                new_list = []
                for i in range(N):
                    if i != x1_idx and i != x2_idx:
                        new_list.append(current_list[i])
                new_list.append(y1)
                new_list.append(y2)
                new_list.sort()
            
            new_score = _score(new_list, new_bits)
            
            delta = new_score - current_score
            total += 1
            
            if delta > 0 or (T > 0 and random.random() < math.exp(delta / T)):
                current_list = new_list
                current_bits = new_bits
                current_score = new_score
                accepted += 1
                
                if current_score > best_score:
                    best_score = current_score
                    best_list = current_list.copy()
            
            # Adaptive cooling: if acceptance rate is high, cool faster
            if total % 100 == 0 and total > 0:
                accept_rate = accepted / total
                if accept_rate > 0.5:
                    cooling = 0.99
                elif accept_rate < 0.1:
                    cooling = 0.999
                else:
                    cooling = 0.998
                accepted = 0
                total = 0
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        num_restarts += 1
    
    return best_list
