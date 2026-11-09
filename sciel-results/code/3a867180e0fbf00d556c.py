import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 85

    def greedy_smallest():
        values = []
        sums = set()
        for x in range(1, 257):
            t = 0
            ok = True
            for a in reversed(values):
                t += a
                if t in sums:
                    ok = False
                    break
            if not ok:
                continue
            if x in sums:
                continue
            new_sums = [x]
            t = x
            for a in reversed(values):
                t += a
                new_sums.append(t)
            values.append(x)
            sums.update(new_sums)
        return values

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        sset = set()
        for i in range(n):
            acc = 0
            for j in range(i, n):
                acc += seq[j]
                if acc in sset:
                    return False
                sset.add(acc)
        return True

    # Initialize with greedy
    current_seq = greedy_smallest()
    current_set = set(current_seq)
    current_len = len(current_seq)
    
    best_seq = current_seq[:]
    best_len = current_len

    T = 50.0
    T_min = 0.5
    cooling_rate = 0.999
    
    # Precompute list of all values for faster sampling
    all_vals = list(range(1, 257))
    
    while time.time() < DEADLINE:
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Generate neighbor
        r = random.random()
        if r < 0.5:
            # Add
            if len(current_set) == 256:
                continue
            # Pick a random value not in set
            # Efficient way: pick random from 1..256, if in set, retry (few retries expected)
            x = random.randint(1, 256)
            if x in current_set:
                # Try a few times
                for _ in range(10):
                    x = random.randint(1, 256)
                    if x not in current_set:
                        break
                else:
                    continue
            new_set = set(current_set)
            new_set.add(x)
        elif r < 0.8:
            # Remove
            if len(current_set) == 0:
                continue
            x = random.choice(list(current_set))
            new_set = set(current_set)
            new_set.remove(x)
        else:
            # Swap
            if len(current_set) == 0:
                continue
            x_remove = random.choice(list(current_set))
            x_add = random.randint(1, 256)
            if x_add in current_set:
                for _ in range(10):
                    x_add = random.randint(1, 256)
                    if x_add not in current_set:
                        break
                else:
                    continue
            new_set = set(current_set)
            new_set.remove(x_remove)
            new_set.add(x_add)
            
        new_seq = sorted(new_set)
        
        # Check validity
        if not is_valid(new_seq):
            continue
            
        new_len = len(new_seq)
        
        if new_len > current_len:
            current_set = new_set
            current_seq = new_seq
            current_len = new_len
            
            if new_len > best_len:
                best_len = new_len
                best_seq = new_seq[:]
        else:
            # Accept worse or equal with probability
            delta = current_len - new_len
            # delta is >= 0
            # prob = exp(-delta / T)
            if delta == 0:
                # Equal length, accept to explore
                if random.random() < 0.5:
                    current_set = new_set
                    current_seq = new_seq
            else:
                # Worse length
                # Use exp
                import math
                prob = math.exp(-delta / T)
                if random.random() < prob:
                    current_set = new_set
                    current_seq = new_seq
                    current_len = new_len

    return best_seq
