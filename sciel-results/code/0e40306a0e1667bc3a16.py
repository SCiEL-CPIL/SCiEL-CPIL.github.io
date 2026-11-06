import time
import random
import math

def construct():
    N = 256
    start_time = time.time()
    time_limit = 58.0  # Safety margin for 60s limit

    # 1. Precomputation
    # Find all 4-APs and 3-APs
    # 4-APs: (a, b, c, d) such that b-a = c-b = d-c = k > 0
    four_aps = []
    for a in range(1, N - 2):
        for d in range(1, (N - a) // 3 + 1):
            b = a + d
            c = b + d
            e = c + d
            if e <= N:
                four_aps.append((a, b, c, e))
    
    # Map each element to the other 3 elements in 4-APs it belongs to
    # four_ap_neighbors[x] = list of tuples (y, z, w) such that {x, y, z, w} is a 4-AP
    four_ap_neighbors = {i: [] for i in range(1, N + 1)}
    for ap in four_aps:
        for i in range(4):
            others = tuple(ap[j] for j in range(4) if j != i)
            four_ap_neighbors[ap[i]].append(others)

    # 3-APs: (a, b, c) such that b-a = c-b = k > 0
    # We need to calculate the gain in 3-APs when adding x.
    # Gain(x) = count of (a,c) in A such that a+c=2x (x is middle)
    #          + count of (b,c) in A such that 2b=x+c (x is start)
    #          + count of (a,b) in A such that 2b=a+x (x is end)
    
    mid_partners = {i: [] for i in range(1, N + 1)}
    start_partners = {i: [] for i in range(1, N + 1)}
    end_partners = {i: [] for i in range(1, N + 1)}
    
    for a in range(1, N + 1):
        for c in range(a + 2, N + 1, 2): # a+c must be even
            s = a + c
            x = s // 2
            if 1 <= x <= N and x > a:
                mid_partners[x].append((a, c))
                
    for x in range(1, N + 1):
        for b in range(x + 1, N + 1):
            c = 2 * b - x
            if c > b and c <= N:
                start_partners[x].append((b, c))
                
    for x in range(1, N + 1):
        for b in range(1, x):
            a = 2 * b - x
            if a > 0 and a < b:
                end_partners[x].append((a, b))

    def count_3aps(A_set):
        count = 0
        for x in A_set:
            for a, c in mid_partners[x]:
                if a in A_set and c in A_set:
                    count += 1
            for b, c in start_partners[x]:
                if b in A_set and c in A_set:
                    count += 1
            for a, b in end_partners[x]:
                if a in A_set and b in A_set:
                    count += 1
        return count

    def is_valid_addition(x, A_set):
        for others in four_ap_neighbors[x]:
            if all(o in A_set for o in others):
                return False
        return True

    def calc_gain_add(x, A_set):
        gain = 0
        for a, c in mid_partners[x]:
            if a in A_set and c in A_set:
                gain += 1
        for b, c in start_partners[x]:
            if b in A_set and c in A_set:
                gain += 1
        for a, b in end_partners[x]:
            if a in A_set and b in A_set:
                gain += 1
        return gain

    def calc_loss_remove(x, A_set):
        loss = 0
        for a, c in mid_partners[x]:
            if a in A_set and c in A_set:
                loss += 1
        for b, c in start_partners[x]:
            if b in A_set and c in A_set:
                loss += 1
        for a, b in end_partners[x]:
            if a in A_set and b in A_set:
                loss += 1
        return loss

    # Initial Solution: Greedy
    A = set()
    for x in range(1, N + 1):
        if is_valid_addition(x, A):
            A.add(x)
            
    current_score = count_3aps(A)
    best_A = set(A)
    best_score = current_score
    
    # Parallel Tempering Setup
    num_replicas = 16
    # Temperatures: geometric progression
    T_min = 0.5
    T_max = 10.0
    temperatures = [T_min * (T_max / T_min) ** (i / (num_replicas - 1)) for i in range(num_replicas)]
    
    replicas = []
    for i in range(num_replicas):
        # Start with slightly perturbed greedy or random
        if i == 0:
            r_A = set(A)
        elif random.random() < 0.5:
            r_A = set(A)
            # Perturb: remove some random elements
            for _ in range(10):
                if r_A:
                    r_A.remove(random.choice(list(r_A)))
        else:
            # Random initial set
            r_A = set()
            for x in range(1, N + 1):
                if random.random() < 0.5:
                    if is_valid_addition(x, r_A):
                        r_A.add(x)
        
        r_score = count_3aps(r_A)
        replicas.append({
            'state': r_A,
            'score': r_score,
            'T': temperatures[i]
        })

    # Assign temperatures
    for i, r in enumerate(replicas):
        r['T'] = temperatures[i]

    # Initial check
    for r in replicas:
        if r['score'] > best_score:
            best_score = r['score']
            best_A = set(r['state'])

    # Parallel Tempering Loop
    while time.time() - start_time < time_limit:
        # Perform one move in each replica
        for i in range(num_replicas):
            if time.time() - start_time > time_limit:
                break
            r = replicas[i]
            A_set = r['state']
            score = r['score']
            T = r['T']
            
            # Pick a random move
            move_type = random.choice(['add', 'remove', 'swap'])
            
            new_A = set(A_set)
            gain = 0
            
            if move_type == 'add':
                candidates = [x for x in range(1, N + 1) if x not in A_set]
                if not candidates: continue
                x = random.choice(candidates)
                
                if is_valid_addition(x, new_A):
                    gain = calc_gain_add(x, new_A)
                    new_A.add(x)
                    new_score = score + gain
                    if new_score > best_score:
                        best_score = new_score
                        best_A = set(new_A)
                    
                    if gain >= 0:
                        A_set = new_A
                        score = new_score
                    else:
                        if random.random() < math.exp(gain / T):
                            A_set = new_A
                            score = new_score
                            
            elif move_type == 'remove':
                if not A_set: continue
                x = random.choice(list(A_set))
                new_A.remove(x)
                
                loss = calc_loss_remove(x, new_A)
                gain = -loss
                new_score = score + gain
                if new_score > best_score:
                    best_score = new_score
                    best_A = set(new_A)
                    
                if gain >= 0:
                    A_set = new_A
                    score = new_score
                else:
                    if random.random() < math.exp(gain / T):
                        A_set = new_A
                        score = new_score
                        
            elif move_type == 'swap':
                if not A_set: continue
                x = random.choice(list(A_set))
                candidates = [y for y in range(1, N + 1) if y not in A_set]
                if not candidates: continue
                y = random.choice(candidates)
                
                new_A.remove(x)
                
                if is_valid_addition(y, new_A):
                    loss_x = calc_loss_remove(x, new_A)
                    gain_y = calc_gain_add(y, new_A)
                    net_gain = gain_y - loss_x
                    new_A.add(y)
                    new_score = score + net_gain
                    if new_score > best_score:
                        best_score = new_score
                        best_A = set(new_A)
                    
                    if net_gain >= 0:
                        A_set = new_A
                        score = new_score
                    else:
                        if random.random() < math.exp(net_gain / T):
                            A_set = new_A
                            score = new_score

            r['state'] = A_set
            r['score'] = score

        # Swap replicas with adjacent temperatures
        for i in range(num_replicas - 1):
            if time.time() - start_time > time_limit:
                break
            if random.random() < 0.5: # Swap probability
                r1 = replicas[i]
                r2 = replicas[i+1]
                T1 = r1['T']
                T2 = r2['T']
                s1 = r1['score']
                s2 = r2['score']
                
                # Accept swap if (s2 - s1)/T2 > (s1 - s2)/T1
                # i.e. s2/T2 - s1/T2 > s1/T1 - s2/T1
                # s2(1/T2 + 1/T1) > s1(1/T2 + 1/T1)
                # This is always true if s2 > s1? No.
                # Log prob: exp((s2/T2 - s1/T2) - (s1/T1 - s2/T1))
                # = exp(s2(1/T2+1/T1) - s1(1/T2+1/T1))
                # = exp((s2-s1)(1/T2+1/T1))
                
                delta = (s2 - s1) * (1.0/T2 + 1.0/T1)
                if delta > 0 or random.random() < math.exp(min(0, delta)):
                    # Swap
                    replicas[i]['state'], replicas[i+1]['state'] = replicas[i+1]['state'], replicas[i]['state']
                    replicas[i]['score'], replicas[i+1]['score'] = replicas[i+1]['score'], replicas[i]['score']

    return sorted(best_A)

# EVOLVE-BLOCK-END
