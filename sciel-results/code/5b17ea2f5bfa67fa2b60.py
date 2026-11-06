import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s safety margin
    
    def get_all_block_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def is_valid(seq):
        if len(seq) == 0:
            return True
        if not all(1 <= x <= 256 for x in seq):
            return False
        if not all(seq[i] < seq[i+1] for i in range(len(seq)-1)):
            return False
        sums = get_all_block_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def greedy_build():
        values = []
        used_sums = set()
        for x in range(1, 257):
            # Check if x can be appended
            new_sums = set()
            # Sums ending at x
            current_sum = x
            new_sums.add(current_sum)
            # Sums extending previous suffixes
            # Actually, we need to check all suffix sums of the new sequence
            # Suffix sums of new_seq = [x] + [x + a_n, x + a_n + a_{n-1}, ...]
            # Wait, the sequence is increasing, so we append to the end.
            # New suffix sums are:
            # 1. x
            # 2. x + last_element
            # 3. x + last_element + second_last_element
            # ...
            # Let's compute them efficiently
            suffix_sums = [x]
            total = x
            for a in reversed(values):
                total += a
                suffix_sums.append(total)
            
            if used_sums.isdisjoint(suffix_sums):
                values.append(x)
                used_sums.update(suffix_sums)
        
        return values

    # Start with a good initial solution
    best_seq = greedy_build()
    best_len = len(best_seq)
    
    # Simulated Annealing
    current_seq = list(best_seq)
    current_sums = get_all_block_sums(current_seq)
    current_len = len(current_seq)
    
    T = 100.0
    cooling_rate = 0.995
    
    while time.time() < deadline and T > 0.1:
        # Generate a neighbor
        neighbor = list(current_seq)
        move_type = random.choice(['swap', 'insert', 'delete', 'adjust'])
        
        if move_type == 'swap' and len(neighbor) > 1:
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif move_type == 'insert' and len(neighbor) < 256:
            # Try to insert a new element
            # Find a position and a value
            pos = random.randint(0, len(neighbor))
            # Value must be between neighbor[pos-1] and neighbor[pos]
            if pos == 0:
                low, high = 1, neighbor[0] - 1
            elif pos == len(neighbor):
                low, high = neighbor[-1] + 1, 256
            else:
                low, high = neighbor[pos-1] + 1, neighbor[pos] - 1
            
            if low <= high:
                val = random.randint(low, high)
                neighbor.insert(pos, val)
        elif move_type == 'delete' and len(neighbor) > 1:
            pos = random.randint(0, len(neighbor) - 1)
            neighbor.pop(pos)
        elif move_type == 'adjust' and len(neighbor) > 1:
            # Adjust one element
            pos = random.randint(0, len(neighbor) - 1)
            if pos == 0:
                low, high = 1, neighbor[1] - 1
            elif pos == len(neighbor) - 1:
                low, high = neighbor[pos-1] + 1, 256
            else:
                low, high = neighbor[pos-1] + 1, neighbor[pos+1] - 1
            
            if low < high:
                neighbor[pos] = random.randint(low, high)
        
        # Check validity
        if not is_valid(neighbor):
            continue
        
        new_len = len(neighbor)
        
        # Acceptance criterion
        if new_len > current_len:
            current_seq = neighbor
            current_sums = get_all_block_sums(neighbor)
            current_len = new_len
            
            if current_len > best_len:
                best_seq = list(neighbor)
                best_len = current_len
        else:
            # Accept worse solutions with probability exp(-delta/T)
            delta = current_len - new_len
            if delta > 0:
                prob = pow(2.718281828, -delta / T)
                if random.random() < prob:
                    current_seq = neighbor
                    current_sums = get_all_block_sums(neighbor)
                    current_len = new_len
        
        T *= cooling_rate
    
    return best_seq
