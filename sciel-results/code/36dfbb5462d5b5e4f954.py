import time
import random
import heapq

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin

    def is_valid(seq):
        """Check if all non-empty consecutive block sums are distinct."""
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def greedy_increasing():
        values = []
        sums = set()
        for x in range(1, 257):
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        return values

    def greedy_decreasing():
        values = []
        sums = set()
        for x in range(256, 0, -1):
            prefixes = [x]
            total = x
            for a in values:
                total += a
                prefixes.append(total)
            if not sums.intersection(prefixes):
                values.append(x)
                values.sort()
                sums.update(prefixes)
        return values

    # 1. Baselines from greedy
    best_seq = greedy_increasing()
    best_len = len(best_seq)

    dec_seq = greedy_decreasing()
    if len(dec_seq) > best_len:
        best_seq = dec_seq
        best_len = len(best_seq)

    # 2. Randomized greedy with multiple orders (cheap, fast)
    for trial in range(200):
        if time.time() > deadline:
            break
        order = list(range(1, 257))
        random.shuffle(order)
        values = []
        sums = set()
        for x in order:
            if values and x <= values[-1]:
                continue
            tails = [x]
            total = x
            for a in reversed(values):
                total += a
                tails.append(total)
            if not sums.intersection(tails):
                values.append(x)
                sums.update(tails)
        if len(values) > best_len:
            best_len = len(values)
            best_seq = list(values)

    # 3. A*-inspired search in state space (current_max_value, frozenset_of_suffix_sums)
    # We model the problem as a pathfinding problem where each state is
    # (current_max_value, frozenset_of_suffix_sums).
    # Heuristic: estimate maximum remaining length based on density of occupied sums.

    def heuristic(state):
        """Estimate maximum remaining length based on density of occupied sums."""
        max_val, suffix_sums = state
        # Higher max_val means we've used larger numbers, leaving less room
        # More suffix_sums means more collisions, reducing potential length
        # Simple heuristic: (256 - max_val) * 0.1 - len(suffix_sums) * 0.05
        return (256 - max_val) * 0.1 - len(suffix_sums) * 0.05

    def a_star_search(initial_seq, max_iterations=10000):
        """A* search in state space of (current_max_value, frozenset_of_suffix_sums)."""
        # State: (current_max_value, frozenset_of_suffix_sums)
        # We track the actual sequence to reconstruct the path.
        
        # Initial state
        if not initial_seq:
            return []
        
        initial_max = max(initial_seq)
        # Compute suffix sums
        suffix_sums = set()
        current_sum = 0
        for a in reversed(initial_seq):
            current_sum += a
            suffix_sums.add(current_sum)
        
        initial_state = (initial_max, frozenset(suffix_sums))
        
        # Priority queue: (heuristic + g, g, state, sequence)
        # g is the number of elements in the sequence
        start_g = len(initial_seq)
        start_h = heuristic(initial_state)
        
        # Use a heap for A*
        # (f, g, max_val, suffix_sums_frozen, sequence)
        heap = [(start_g + start_h, -start_g, initial_max, frozenset(suffix_sums), list(initial_seq))]
        
        visited = set()
        visited.add((initial_max, frozenset(suffix_sums)))
        
        best_found = list(initial_seq)
        best_len_found = len(initial_seq)
        
        iterations = 0
        while heap and iterations < max_iterations:
            if time.time() > deadline - 2:
                break
                
            f, neg_g, max_val, suffix_sums_frozen, seq = heapq.heappop(heap)
            iterations += 1
            
            # Reconstruct suffix_sums from frozen set
            suffix_sums = set(suffix_sums_frozen)
            
            # Try adding each possible next value
            # Next value must be > max_val
            for x in range(max_val + 1, 257):
                # Compute new suffix sums
                new_suffix_sums = set(suffix_sums)
                new_suffix_sums.add(x)
                # Add x + each existing suffix sum
                for s in suffix_sums:
                    new_suffix_sums.add(x + s)
                
                # Check for collisions with existing sums
                # We need to check against ALL sums, not just suffix sums
                # But for efficiency, we'll do a quick check first
                # If there's a collision in suffix sums, it's definitely invalid
                # This is a necessary but not sufficient condition
                if len(new_suffix_sums) != len(suffix_sums) + len(suffix_sums) + 1:
                    # Collision detected in suffix sums
                    continue
                
                # Full validation
                new_seq = seq + [x]
                if not is_valid(new_seq):
                    continue
                
                new_max = x
                new_state = (new_max, frozenset(new_suffix_sums))
                
                if new_state in visited:
                    continue
                visited.add(new_state)
                
                new_g = len(new_seq)
                new_h = heuristic(new_state)
                
                if new_g > best_len_found:
                    best_len_found = new_g
                    best_found = list(new_seq)
                
                heapq.heappush(heap, (new_g + new_h, -new_g, new_max, frozenset(new_suffix_sums), new_seq))
        
        return best_found

    # Run A* search from the best greedy solution
    if time.time() < deadline - 5:
        a_star_result = a_star_search(best_seq, max_iterations=5000)
        if len(a_star_result) > best_len:
            best_len = len(a_star_result)
            best_seq = list(a_star_result)

    # 4. Local search with position-aware insertion
    current_seq = list(best_seq)
    all_candidates = list(range(1, 257))

    # Helper: try inserting x at every valid position, return best (longest valid) new seq
    def try_insert_all_positions(seq, x):
        """Try inserting x at each position; return (new_seq, new_sums) for the first valid one, or None."""
        L = len(seq)
        # Precompute prefix sums of seq
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        existing_sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                existing_sums.add(P[j] - P[i])

        # Positions 0..L (insert before index pos, i.e., between pos-1 and pos)
        for pos in range(L + 1):
            # x must fit strictly: if pos>0 then seq[pos-1] < x; if pos<L then x < seq[pos]
            if pos > 0 and seq[pos - 1] >= x:
                continue
            if pos < L and x >= seq[pos]:
                continue
            # New sums involving x: for left extension l (0..pos) and right extension r (0..L-pos)
            # not both zero
            new_sums = set()
            base_left = P[pos]
            for l in range(0, pos + 1):
                left_part = base_left - P[pos - l]
                for r in range(0, L - pos + 1):
                    if l == 0 and r == 0:
                        continue
                    right_part = P[pos + r] - P[pos]
                    new_sums.add(left_part + x + right_part)
            # Check conflict with existing sums
            if existing_sums.intersection(new_sums):
                continue
            # Build new sequence
            new_seq = seq[:pos] + [x] + seq[pos:]
            new_sums = existing_sums | new_sums
            return new_seq, new_sums
        return None

    while time.time() < deadline:
        r = random.random()
        if r < 0.55:  # Add with position-aware insertion
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            if not available:
                continue
            # Bias towards values that are "near" the sequence (more likely to fit)
            # Pick randomly but also try a few candidates
            tried = 0
            max_tries = 5
            while tried < max_tries and available:
                x = random.choice(available)
                available.remove(x)
                tried += 1
                result = try_insert_all_positions(current_seq, x)
                if result is not None:
                    new_seq, new_sums = result
                    current_seq = new_seq
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
                    break
        elif r < 0.75:  # Remove
            if current_seq:
                x = random.choice(current_seq)
                current_seq.remove(x)
        else:  # Swap
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    # Remove x_rem
                    temp_seq = [x for x in current_seq if x != x_rem]
                    # Try inserting x_add at all positions
                    result = try_insert_all_positions(temp_seq, x_add)
                    if result is not None:
                        new_seq, new_sums = result
                        current_seq = new_seq
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)

    return sorted(best_seq)
