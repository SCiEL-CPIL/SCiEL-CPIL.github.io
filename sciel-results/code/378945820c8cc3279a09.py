import random

def construct():
    # 1. Generate initial solution using greedy
    values, sums = [], set()
    for x in range(1, 257):
        tails, total = [x], x
        for a in reversed(values):
            total += a
            tails.append(total)
        if not sums.intersection(tails):
            values.append(x)
            sums.update(tails)
    
    best_values = values[:]
    best_len = len(values)
    
    # 2. Simulated Annealing
    # State: list of values
    current_values = values[:]
    
    def get_sums(lst):
        s = set()
        for i in range(len(lst)):
            total = 0
            for j in range(i, len(lst)):
                total += lst[j]
                s.add(total)
        return s

    def cost(lst):
        # Cost is primarily based on length (we want to maximize, so -len)
        # Plus penalty for invalid sums
        if not lst:
            return 0
        sums = get_sums(lst)
        # Number of sums should be n*(n+1)/2
        expected = len(lst) * (len(lst) + 1) // 2
        penalty = expected - len(sums)
        return -len(lst) * 100 + penalty

    current_cost = cost(current_values)
    
    T = 100.0
    min_T = 1.0
    
    for i in range(50000):
        if T < min_T:
            break
            
        # Generate neighbor
        neighbor = current_values[:]
        r = random.random()
        
        if r < 0.3:
            # Add a random element
            candidates = [x for x in range(1, 257) if x not in neighbor]
            if candidates:
                x = random.choice(candidates)
                neighbor.append(x)
                neighbor.sort()
        elif r < 0.6:
            # Remove a random element
            if neighbor:
                idx = random.randint(0, len(neighbor)-1)
                neighbor.pop(idx)
        else:
            # Swap
            if neighbor:
                idx = random.randint(0, len(neighbor)-1)
                old_val = neighbor[idx]
                candidates = [x for x in range(1, 257) if x not in neighbor]
                if candidates:
                    new_val = random.choice(candidates)
                    neighbor[idx] = new_val
                    neighbor.sort()

        neighbor_cost = cost(neighbor)
        
        # Acceptance probability
        if neighbor_cost < current_cost:
            current_values = neighbor
            current_cost = neighbor_cost
            if len(current_values) > best_len:
                best_len = len(current_values)
                best_values = current_values[:]
        else:
            delta = neighbor_cost - current_cost
            if delta > 0:
                prob = 2.71828 ** (-delta / T)
                if random.random() < prob:
                    current_values = neighbor
                    current_cost = neighbor_cost
        
        T *= 0.999
        
    return best_values
