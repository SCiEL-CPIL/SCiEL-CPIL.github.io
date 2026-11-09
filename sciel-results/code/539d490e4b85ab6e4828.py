import time
import random

def construct():
    start = time.time()
    DEADLINE = start + 85  # Safety margin before 90s limit

    def compute_block_sums(s):
        """Return set of all consecutive block sums using prefix sums."""
        sums = set()
        n = len(s)
        if n == 0:
            return sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i + 1] = prefix[i] + s[i]
        for i in range(n):
            for j in range(i + 1, n + 1):
                sums.add(prefix[j] - prefix[i])
        return sums

    def is_valid(s):
        """Check if sequence is valid: strictly increasing, values in [1,256], distinct block sums."""
        n = len(s)
        if n == 0:
            return True
        for i in range(n - 1):
            if s[i] >= s[i + 1]:
                return False
        for v in s:
            if v < 1 or v > 256:
                return False
        sums = compute_block_sums(s)
        expected = n * (n + 1) // 2
        return len(sums) == expected

    def can_extend(s, x, pos):
        """Check if inserting x at pos keeps sequence valid."""
        # Check increasing order
        if pos > 0 and x <= s[pos - 1]:
            return False
        if pos < len(s) and x >= s[pos]:
            return False
        # Check all new block sums are distinct
        n = len(s)
        # Compute all existing block sums using prefix sums
        existing_sums = set()
        if n > 0:
            prefix = [0] * (n + 1)
            for i in range(n):
                prefix[i + 1] = prefix[i] + s[i]
            for i in range(n):
                for j in range(i + 1, n + 1):
                    existing_sums.add(prefix[j] - prefix[i])
        
        # New block sums involving x
        new_sums = set()
        # Blocks starting at pos and going right
        acc = x
        new_sums.add(acc)
        for j in range(pos, n):
            acc += s[j]
            new_sums.add(acc)
        # Blocks ending at pos from the left (not starting at pos)
        acc = x
        for i in range(pos - 1, -1, -1):
            acc += s[i]
            new_sums.add(acc)
        
        # Check no conflicts with existing sums
        if new_sums & existing_sums:
            return False
        
        # Check internal consistency: all new sums must be distinct
        # Since we're building a set, len check is implicit
        return True

    def greedy_build(direction='smallest'):
        """Build a sequence greedily with prefix-sum optimization."""
        values = []
        if direction == 'smallest':
            candidates = range(1, 257)
        else:
            candidates = range(256, 0, -1)
        
        for x in candidates:
            if time.time() > DEADLINE - 2:
                break
            # Try appending at end (pos = len(values))
            n = len(values)
            if n == 0 or x > values[-1]:
                # Compute existing sums using prefix sums
                sums = set()
                if n > 0:
                    prefix = [0] * (n + 1)
                    for i in range(n):
                        prefix[i + 1] = prefix[i] + values[i]
                    for i in range(n):
                        for j in range(i + 1, n + 1):
                            sums.add(prefix[j] - prefix[i])
                # New sums when appending x
                new_sums = set()
                acc = x
                new_sums.add(acc)
                for a in reversed(values):
                    acc += a
                    new_sums.add(acc)
                if not (new_sums & sums):
                    values.append(x)
        
        return sorted(values)

    def backtracking_search(max_depth=12, time_limit=80):
        """Use backtracking with constraint propagation to find long sequences."""
        best = []
        
        def dfs(current, depth, max_d):
            nonlocal best
            if time.time() > time_limit:
                return
            if len(current) > len(best):
                best = list(current)
            if depth >= max_d:
                return
            
            # Try to extend by adding elements
            last = current[-1] if current else 0
            for x in range(last + 1, 257):
                if time.time() > time_limit:
                    return
                if can_extend(current, x, len(current)):
                    current.append(x)
                    dfs(current, depth + 1, max_d)
                    current.pop()
        
        dfs([], 0, max_depth)
        return best

    def local_search(start_seq, max_iterations=20000):
        """Local search: try to add elements at any position, or replace elements."""
        current = list(start_seq)
        best = list(current)
        best_len = len(current)
        
        for iteration in range(max_iterations):
            if time.time() > DEADLINE - 5:
                break
            
            improved = False
            
            # Try inserting a new element at each position
            n = len(current)
            for pos in range(n + 1):
                if time.time() > DEADLINE - 5:
                    break
                # Determine valid range for x at this position
                if pos > 0:
                    lo = current[pos - 1] + 1
                else:
                    lo = 1
                if pos < n:
                    hi = current[pos] - 1
                else:
                    hi = 256
                
                for x in range(lo, hi + 1):
                    if can_extend(current, x, pos):
                        new_seq = current[:pos] + [x] + current[pos:]
                        if len(new_seq) > len(best):
                            best = new_seq
                            best_len = len(new_seq)
                        current = new_seq
                        improved = True
                        break
                if improved:
                    break
            
            # If no improvement, try replacing an element
            if not improved and len(current) > 1:
                for pos in range(len(current)):
                    if time.time() > DEADLINE - 5:
                        break
                    old_val = current[pos]
                    # Try removing and reinserting a different value
                    temp = current[:pos] + current[pos+1:]
                    # Find a value that fits at pos
                    if pos > 0:
                        lo = temp[pos - 1] + 1
                    else:
                        lo = 1
                    if pos < len(temp):
                        hi = temp[pos] - 1
                    else:
                        hi = 256
                    
                    for x in range(lo, hi + 1):
                        if x == old_val:
                            continue
                        if can_extend(temp, x, pos):
                            new_seq = temp[:pos] + [x] + temp[pos:]
                            # Check if this gives more room for future extensions
                            # For now, just accept if valid
                            current = new_seq
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best

    def simulated_annealing(start_seq, iterations=10000, start_temp=10.0, end_temp=0.01):
        """Simulated annealing to improve sequence length."""
        current = list(start_seq)
        best = list(current)
        best_len = len(current)
        
        def get_score(s):
            """Score: negative of number of conflicts + bonus for length."""
            if not s:
                return 0
            # Check increasing
            for i in range(len(s) - 1):
                if s[i] >= s[i + 1]:
                    return -1000
            # Check range
            for v in s:
                if v < 1 or v > 256:
                    return -1000
            # Count distinct block sums
            sums = compute_block_sums(s)
            expected = len(s) * (len(s) + 1) // 2
            conflicts = expected - len(sums)
            return len(s) * 10 - conflicts * 100
        
        current_score = get_score(current)
        
        for i in range(iterations):
            if time.time() > DEADLINE - 5:
                break
            temp = start_temp * (end_temp / start_temp) ** (i / iterations)
            
            # Generate neighbor
            neighbor = list(current)
            move_type = random.choice(['insert', 'delete', 'swap'])
            
            if move_type == 'insert' and len(neighbor) < 15:
                x = random.randint(1, 256)
                pos = random.randint(0, len(neighbor))
                if pos == 0 or (pos > 0 and x > neighbor[pos-1] and (pos == len(neighbor) or x < neighbor[pos])):
                    neighbor.insert(pos, x)
            elif move_type == 'delete' and len(neighbor) > 1:
                pos = random.randint(0, len(neighbor) - 1)
                neighbor.pop(pos)
            elif move_type == 'swap' and len(neighbor) >= 2:
                i1, i2 = random.sample(range(len(neighbor)), 2)
                neighbor[i1], neighbor[i2] = neighbor[i2], neighbor[i1]
                neighbor.sort()
            
            neighbor_score = get_score(neighbor)
            
            if neighbor_score >= current_score:
                current = neighbor
                current_score = neighbor_score
                if len(current) > best_len:
                    best = list(current)
                    best_len = len(current)
            else:
                delta = neighbor_score - current_score
                if random.random() < pow(2.718281828, delta / max(temp, 0.001)):
                    current = neighbor
                    current_score = neighbor_score
        
        return best

    # Strategy 1: Backtracking
    best = backtracking_search(max_depth=10, time_limit=25)
    
    # Strategy 2: Local search from backtracking result
    ls_result = local_search(best, max_iterations=5000)
    if len(ls_result) > len(best):
        best = ls_result
    
    # Strategy 3: Greedy builds
    for direction in ['smallest', 'largest']:
        if time.time() > DEADLINE - 15:
            break
        g = greedy_build(direction)
        if len(g) > len(best):
            best = g
        # Local search from greedy result
        if time.time() > DEADLINE - 10:
            break
        ls_g = local_search(g, max_iterations=3000)
        if len(ls_g) > len(best):
            best = ls_g
    
    # Strategy 4: Simulated annealing from multiple starts
    starts = [best]
    
    # Add random starts
    for _ in range(3):
        if time.time() > DEADLINE - 10:
            break
        rand_seq = []
        for _ in range(random.randint(5, 10)):
            x = random.randint(1, 256)
            if rand_seq and x <= rand_seq[-1]:
                continue
            if can_extend(rand_seq, x, len(rand_seq)):
                rand_seq.append(x)
        starts.append(rand_seq)
    
    for start_seq in starts:
        if time.time() > DEADLINE - 5:
            break
        result = simulated_annealing(start_seq, iterations=3000)
        if len(result) > len(best):
            best = result
    
    # Final local search polish
    if time.time() < DEADLINE - 5:
        final_ls = local_search(best, max_iterations=5000)
        if len(final_ls) > len(best):
            best = final_ls
    
    # Final verification
    if not is_valid(best):
        n = len(best)
        for i in range(n):
            if is_valid(best[:i + 1]):
                best = best[:i + 1]
            else:
                break
        else:
            best = best[:i]
    
    return best
