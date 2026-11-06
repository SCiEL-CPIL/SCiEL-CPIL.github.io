import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0  # Safety margin before 90s kill
    
    def elapsed():
        return time.time() - start_time
    
    def within_deadline():
        return elapsed() < DEADLINE
    
    best_seq = []
    best_len = 0
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
    
    def check_validity(seq):
        if not seq:
            return True
        n = len(seq)
        if len(seq) != len(set(seq)):
            return False
        if seq != sorted(seq):
            return False
        if seq[0] < 1 or seq[-1] > 256:
            return False
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def greedy_build():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = []
            total = x
            tails.append(total)
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values
    
    def randomized_greedy(bias='uniform', num_restarts=200, candidates_per_step=50):
        best_local = []
        for r in range(num_restarts):
            if not within_deadline():
                break
            values = []
            sums = set()
            last = 0
            while True:
                if not within_deadline():
                    break
                found = False
                for _ in range(candidates_per_step):
                    if not within_deadline():
                        break
                    min_val = last + 1
                    max_val = 256
                    if min_val > max_val:
                        break
                    if bias == 'uniform':
                        x = random.randint(min_val, max_val)
                    elif bias == 'larger':
                        x = random.randint((min_val + max_val) // 2, max_val)
                    elif bias == 'smaller':
                        x = random.randint(min_val, (min_val + max_val) // 2)
                    else:
                        x = random.randint(min_val, max_val)
                    
                    # Check tails
                    tails = [x]
                    total = x
                    for a in reversed(values):
                        total += a
                        tails.append(total)
                    
                    if not sums.intersection(tails):
                        values.append(x)
                        sums.update(tails)
                        last = x
                        found = True
                        break
                if not found:
                    break
            if len(values) > len(best_local):
                best_local = values
        return best_local
    
    # Phase 1: Deterministic greedy
    seq = greedy_build()
    update_best(seq)
    
    # Phase 2: Randomized greedy with various biases
    for bias in ['uniform', 'larger', 'smaller']:
        if not within_deadline():
            break
        seq = randomized_greedy(bias=bias, num_restarts=300, candidates_per_step=80)
        update_best(seq)
    
    # Phase 3: Local search - try to improve by replacing elements
    def local_search(seq, max_rounds=50):
        seq = list(seq)
        n = len(seq)
        if n == 0:
            return seq
        
        def is_valid(s):
            return check_validity(s)
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and within_deadline():
            improved = False
            rounds += 1
            # Try replacing each element with a different value
            for i in range(n):
                if not within_deadline():
                    break
                old_val = seq[i]
                # Try nearby values
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -8, 8]:
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 256:
                        continue
                    # Check strict increase
                    if i > 0 and new_val <= seq[i-1]:
                        continue
                    if i < n-1 and new_val >= seq[i+1]:
                        continue
                    # Check uniqueness
                    if new_val in seq[:i] or new_val in seq[i+1:]:
                        continue
                    
                    candidate = seq[:i] + [new_val] + seq[i+1:]
                    if is_valid(candidate):
                        # Accept if valid
                        seq = candidate
                        improved = True
                        break
                if improved:
                    break
        return seq
    
    # Phase 4: Try to extend best sequence by inserting elements
    def try_extend(seq):
        seq = list(seq)
        if not seq:
            return seq
        n = len(seq)
        
        # Try inserting at each position
        for pos in range(n + 1):
            if not within_deadline():
                break
            # Determine valid range for insertion at pos
            min_val = (seq[pos-1] + 1) if pos > 0 else 1
            max_val = (seq[pos] - 1) if pos < n else 256
            if min_val > max_val:
                continue
            
            # Try some values in range
            step = max(1, (max_val - min_val + 1) // 20)
            for x in range(min_val, max_val + 1, step):
                if not within_deadline():
                    break
                candidate = seq[:pos] + [x] + seq[pos:]
                if check_validity(candidate):
                    return candidate
        return seq
    
    # Apply local search and extension
    if within_deadline():
        improved = local_search(best_seq, max_rounds=30)
        update_best(improved)
    
    if within_deadline():
        extended = try_extend(best_seq)
        update_best(extended)
    
    # Phase 5: More randomized greedy with more restarts if time permits
    if within_deadline():
        for bias in ['uniform', 'larger']:
            if not within_deadline():
                break
            seq = randomized_greedy(bias=bias, num_restarts=200, candidates_per_step=100)
            update_best(seq)
    
    # Phase 6: Final local search
    if within_deadline():
        improved = local_search(best_seq, max_rounds=50)
        update_best(improved)
    
    if within_deadline():
        extended = try_extend(best_seq)
        update_best(extended)
    
    return best_seq
