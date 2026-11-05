import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 55
    
    def compute_sums(seq):
        """Compute all consecutive block sums for a sorted sequence."""
        L = len(seq)
        if L == 0:
            return set()
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                sums.add(P[j] - P[i])
        return sums
    
    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        L = len(seq)
        if L == 0:
            return True
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
    
    def incremental_check(seq, x, existing_sums):
        """
        Check if x can be added to sorted seq without creating duplicate sums.
        Returns True if valid, False otherwise.
        Also returns the set of new sums created by adding x.
        """
        k = bisect.bisect_left(seq, x)
        L = len(seq)
        
        new_sums = set()
        
        # x alone
        if x in existing_sums:
            return False, None
        new_sums.add(x)
        
        # Left extensions: blocks ending at x
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            s = left_sum + x
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Right extensions: blocks starting at x
        right_sum = 0
        for j in range(k, L):
            right_sum += seq[j]
            s = x + right_sum
            if s in existing_sums or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # Both sides: blocks containing x with elements on both sides
        left_sum = 0
        for i in range(k - 1, -1, -1):
            left_sum += seq[i]
            right_sum = 0
            for j in range(k, L):
                right_sum += seq[j]
                s = left_sum + x + right_sum
                if s in existing_sums or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums
    
    def greedy_build(strategy, existing_sums=None):
        """
        Build a sequence greedily using the given strategy.
        strategy: 'increasing', 'decreasing', 'random', or a list of values to try in order.
        """
        seq = []
        sums = set() if existing_sums is None else existing_sums.copy()
        
        if strategy == 'increasing':
            candidates = list(range(1, 257))
        elif strategy == 'decreasing':
            candidates = list(range(256, 0, -1))
        elif strategy == 'random':
            candidates = list(range(1, 257))
            random.shuffle(candidates)
        else:
            candidates = strategy
        
        for x in candidates:
            if x in seq:
                continue
            valid, new_sums = incremental_check(seq, x, sums)
            if valid:
                k = bisect.bisect_left(seq, x)
                seq = seq[:k] + [x] + seq[k:]
                sums.update(new_sums)
        
        return seq, sums
    
    # Try multiple greedy strategies
    best_seq = []
    best_sums = set()
    
    strategies = [
        'increasing',
        'decreasing',
    ]
    
    # Add some randomized strategies
    for _ in range(3):
        rand_strat = list(range(1, 257))
        random.shuffle(rand_strat)
        strategies.append(rand_strat)
    
    for strat in strategies:
        seq, sums = greedy_build(strat)
        if len(seq) > len(best_seq):
            best_seq = seq
            best_sums = sums
    
    # Backtracking DFS to explore non-greedy branches
    def backtracking_dfs(seq, sums, remaining, depth, max_depth):
        """Backtracking DFS to try to extend the sequence."""
        if depth >= max_depth or time.time() > deadline - 1:
            return seq, sums
        
        best_seq = list(seq)
        best_sums = set(sums)
        
        # Try each remaining value
        for x in sorted(remaining, key=lambda v: abs(v - (seq[-1] if seq else 128))):
            valid, new_sums = incremental_check(seq, x, sums)
            if valid:
                k = bisect.bisect_left(seq, x)
                new_seq = seq[:k] + [x] + seq[k:]
                new_sums_set = sums | new_sums
                remaining.remove(x)
                
                deeper_seq, deeper_sums = backtracking_dfs(new_seq, new_sums_set, remaining, depth + 1, max_depth)
                
                remaining.add(x)
                
                if len(deeper_seq) > len(best_seq):
                    best_seq = deeper_seq
                    best_sums = deeper_sums
        
        return best_seq, best_sums
    
    current_seq = list(best_seq)
    current_sums = set(best_sums)
    best_len = len(current_seq)
    best_seq_final = list(current_seq)
    
    all_values = set(range(1, 257))
    current_set = set(current_seq)
    remaining = all_values - current_set
    
    # Try backtracking DFS with limited depth
    if time.time() < deadline - 5 and len(remaining) > 0:
        dfs_seq, dfs_sums = backtracking_dfs(current_seq, current_sums, set(remaining), 0, 2)
        if len(dfs_seq) > best_len:
            best_len = len(dfs_seq)
            best_seq_final = list(dfs_seq)
            current_seq = dfs_seq
            current_sums = dfs_sums
            current_set = set(current_seq)
            remaining = all_values - current_set
    
    # Local search with informed moves
    stuck_count = 0
    max_stuck = 50
    
    while time.time() < deadline - 1:
        r = random.random()
        
        if r < 0.7:
            # Try to add a random available value
            if remaining:
                # Heuristic: prefer values that are spaced out
                available = list(remaining)
                x = random.choice(available)
                valid, new_sums = incremental_check(current_seq, x, current_sums)
                if valid:
                    k = bisect.bisect_left(current_seq, x)
                    new_seq = current_seq[:k] + [x] + current_seq[k:]
                    new_sums_set = current_sums | new_sums
                    current_seq = new_seq
                    current_sums = new_sums_set
                    remaining.remove(x)
                    stuck_count = 0
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq_final = list(current_seq)
                else:
                    stuck_count += 1
                    
        elif r < 0.9:
            # Try to remove a random element and try to add a different one
            if current_seq and len(current_seq) > 1:
                x_rem = random.choice(current_seq)
                current_seq.remove(x_rem)
                remaining.add(x_rem)
                current_sums = compute_sums(current_seq)
                stuck_count = 0
                
                # Try to add a different element
                if remaining:
                    available = list(remaining)
                    for _ in range(min(3, len(available))):
                        x_add = random.choice(available)
                        valid, new_sums = incremental_check(current_seq, x_add, current_sums)
                        if valid:
                            k = bisect.bisect_left(current_seq, x_add)
                            new_seq = current_seq[:k] + [x_add] + current_seq[k:]
                            new_sums_set = current_sums | new_sums
                            current_seq = new_seq
                            current_sums = new_sums_set
                            remaining.remove(x_add)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq_final = list(current_seq)
                            break
        else:
            # Occasionally try a fresh greedy restart
            rand_strat = list(range(1, 257))
            random.shuffle(rand_strat)
            seq, sums = greedy_build(rand_strat)
            if len(seq) > len(current_seq):
                current_seq = seq
                current_sums = sums
                current_set = set(current_seq)
                remaining = all_values - current_set
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq_final = list(current_seq)
            stuck_count = 0
        
        # If stuck for too long, restart with a new random greedy
        if stuck_count > max_stuck:
            rand_strat = list(range(1, 257))
            random.shuffle(rand_strat)
            seq, sums = greedy_build(rand_strat)
            if len(seq) > len(current_seq):
                current_seq = seq
                current_sums = sums
                current_set = set(current_seq)
                remaining = all_values - current_set
                if len(current_seq) > best_len:
                    best_len = len(current_seq)
                    best_seq_final = list(current_seq)
            stuck_count = 0
    
    return sorted(best_seq_final)
