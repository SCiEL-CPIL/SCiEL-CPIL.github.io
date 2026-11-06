import random
import time

def construct():
    start_time = time.time()
    DEADLINE = 85.0

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

    # Backtracking search with pruning
    def backtrack_search(max_depth=20, time_limit=5.0):
        """
        Depth-first search with backtracking.
        We build the sequence left to right, maintaining:
        - values: the sequence so far
        - sums: set of all block sums so far
        - last: last value chosen
        
        At each step, try candidate values in some order.
        If we hit a dead end, backtrack.
        """
        best_local = []
        best_local_len = 0
        search_start = time.time()
        nodes_visited = 0
        
        def search(values, sums, last, depth):
            nonlocal best_local, best_local_len, nodes_visited
            
            if time.time() - search_start > time_limit:
                return False  # timeout signal
            
            nodes_visited += 1
            
            if len(values) > best_local_len:
                best_local = list(values)
                best_local_len = len(best_local)
                if len(best_local) >= max_depth:
                    return True  # found max depth
            
            if not within_deadline():
                return False
            
            min_val = last + 1
            max_val = 256
            if min_val > max_val:
                return False
            
            # Generate candidates: try a mix of small, medium, large values
            candidates = []
            # Small values (more likely to fit)
            for x in range(min_val, min(min_val + 10, max_val + 1)):
                candidates.append(x)
            # Medium
            mid = (min_val + max_val) // 2
            for x in range(max(min_val, mid - 5), min(mid + 5, max_val + 1)):
                if x not in candidates:
                    candidates.append(x)
            # Large
            for x in range(max(mid, max_val - 10), max_val + 1):
                if x not in candidates:
                    candidates.append(x)
            
            # Shuffle to add diversity
            random.shuffle(candidates)
            
            for x in candidates:
                if x < min_val or x > max_val:
                    continue
                
                # Check if adding x creates any duplicate block sums
                tails = [x]
                total = x
                for a in reversed(values):
                    total += a
                    tails.append(total)
                
                # Check against existing sums
                conflict = False
                for t in tails:
                    if t in sums:
                        conflict = True
                        break
                
                if conflict:
                    continue
                
                # Also check that new tails don't conflict with each other
                if len(set(tails)) != len(tails):
                    continue
                
                # Try this candidate
                values.append(x)
                sums_copy = sums.copy()
                sums_copy.update(tails)
                
                result = search(values, sums_copy, x, depth + 1)
                
                values.pop()
                
                if result:
                    return True
                
                if time.time() - search_start > time_limit:
                    return False
            
            return False
        
        search([], set(), 0, 0)
        return best_local

    def local_search(seq, max_rounds=50):
        seq = list(seq)
        n = len(seq)
        if n == 0:
            return seq
        
        improved = True
        rounds = 0
        while improved and rounds < max_rounds and within_deadline():
            improved = False
            rounds += 1
            for i in range(n):
                if not within_deadline():
                    break
                old_val = seq[i]
                for delta in [-3, -2, -1, 1, 2, 3, -5, 5, -8, 8]:
                    new_val = old_val + delta
                    if new_val < 1 or new_val > 256:
                        continue
                    if i > 0 and new_val <= seq[i-1]:
                        continue
                    if i < n-1 and new_val >= seq[i+1]:
                        continue
                    if new_val in seq[:i] or new_val in seq[i+1:]:
                        continue
                    
                    candidate = seq[:i] + [new_val] + seq[i+1:]
                    if check_validity(candidate):
                        seq = candidate
                        improved = True
                        break
                if improved:
                    break
        return seq

    def try_extend(seq):
        seq = list(seq)
        if not seq:
            return seq
        n = len(seq)
        
        for pos in range(n + 1):
            if not within_deadline():
                break
            min_val = (seq[pos-1] + 1) if pos > 0 else 1
            max_val = (seq[pos] - 1) if pos < n else 256
            if min_val > max_val:
                continue
            
            step = max(1, (max_val - min_val + 1) // 30)
            for x in range(min_val, max_val + 1, step):
                if not within_deadline():
                    break
                candidate = seq[:pos] + [x] + seq[pos:]
                if check_validity(candidate):
                    return candidate
        return seq

    # Phase 1: Deterministic greedy
    seq = greedy_build()
    update_best(seq)

    # Phase 2: Backtracking search (the key new component)
    # Run multiple backtracking searches with different time limits
    bt_times = [3.0, 5.0, 8.0, 12.0]
    for t in bt_times:
        if not within_deadline():
            break
        if elapsed() + t > DEADLINE:
            t = max(1.0, DEADLINE - elapsed() - 2.0)
        seq = backtrack_search(max_depth=25, time_limit=t)
        if len(seq) > best_len:
            update_best(seq)

    # Phase 3: Randomized greedy with various biases
    for bias in ['uniform', 'larger', 'smaller']:
        if not within_deadline():
            break
        seq = randomized_greedy(bias=bias, num_restarts=200, candidates_per_step=80)
        update_best(seq)

    # Phase 4: Local search on best
    if within_deadline():
        improved = local_search(best_seq, max_rounds=30)
        update_best(improved)

    # Phase 5: Try to extend
    if within_deadline():
        extended = try_extend(best_seq)
        update_best(extended)

    # Phase 6: More backtracking with longer time if budget allows
    if within_deadline() and elapsed() < DEADLINE - 15:
        remaining = DEADLINE - elapsed() - 3
        if remaining > 5:
            seq = backtrack_search(max_depth=25, time_limit=min(remaining, 15.0))
            update_best(seq)

    # Phase 7: Final local search and extension
    if within_deadline():
        improved = local_search(best_seq, max_rounds=50)
        update_best(improved)

    if within_deadline():
        extended = try_extend(best_seq)
        update_best(extended)

    # Phase 8: One more round of randomized greedy if time permits
    if within_deadline() and elapsed() < DEADLINE - 5:
        for bias in ['uniform', 'larger']:
            if not within_deadline():
                break
            seq = randomized_greedy(bias=bias, num_restarts=150, candidates_per_step=100)
            update_best(seq)

    return best_seq
