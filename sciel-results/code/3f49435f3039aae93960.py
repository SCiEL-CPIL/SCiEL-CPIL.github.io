import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def is_valid(seq):
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

    def get_all_sums(seq):
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                sums.add(s)
        return sums

    def check_add_valid(seq, x, sums_set):
        """Check if adding x to the end of seq keeps all consecutive sums distinct."""
        L = len(seq)
        # New consecutive blocks ending at x:
        # x, x + seq[L-1], x + seq[L-1] + seq[L-2], ..., x + sum(seq)
        running = x
        for i in range(L - 1, -1, -1):
            if running in sums_set:
                return False, None
            running += seq[i]
        # Also need to check that running (which is x + sum(seq)) is not in sums_set
        if running in sums_set:
            return False, None
        # Build new sums
        new_sums = set(sums_set)
        running = x
        for i in range(L - 1, -1, -1):
            new_sums.add(running)
            running += seq[i]
        new_sums.add(running)
        return True, new_sums

    def check_insert_valid(seq, idx, x, sums_set):
        """Check if inserting x at position idx in seq keeps all consecutive sums distinct."""
        new_seq = seq[:idx] + [x] + seq[idx:]
        return is_valid(new_seq), get_all_sums(new_seq)

    # Greedy initialization with multiple strategies
    def greedy_forward(start_val, end_val, step):
        values = []
        sums_set = set()
        for x in range(start_val, end_val, step):
            valid, new_sums = check_add_valid(values, x, sums_set)
            if valid:
                values.append(x)
                sums_set = new_sums
        return values, sums_set

    def greedy_backward(start_val, end_val, step):
        values = []
        sums_set = set()
        for x in range(start_val, end_val, step):
            valid, new_sums = check_add_valid(values, x, sums_set)
            if valid:
                values.append(x)
                sums_set = new_sums
        return values, sums_set

    # Try multiple greedy strategies
    best_greedy_seq = []
    best_greedy_sums = set()
    
    # Forward from 1
    seq1, sums1 = greedy_forward(1, 257, 1)
    if len(seq1) > len(best_greedy_seq):
        best_greedy_seq = seq1
        best_greedy_sums = sums1
    
    # Forward from 2
    seq2, sums2 = greedy_forward(2, 257, 1)
    if len(seq2) > len(best_greedy_seq):
        best_greedy_seq = seq2
        best_greedy_sums = sums2
    
    # Backward from 256
    seq3, sums3 = greedy_backward(256, 0, -1)
    if len(seq3) > len(best_greedy_seq):
        best_greedy_seq = seq3
        best_greedy_sums = sums3
    
    # Backward from 255
    seq4, sums4 = greedy_backward(255, 0, -1)
    if len(seq4) > len(best_greedy_seq):
        best_greedy_seq = seq4
        best_greedy_sums = sums4

    # DFS backtracking with better pruning
    def dfs_backtrack(current_seq, sums_set, max_depth, time_limit, depth=0):
        if time.time() > time_limit:
            return current_seq
        
        L = len(current_seq)
        if L >= max_depth:
            return current_seq
        
        best_result = list(current_seq)
        last_val = current_seq[-1] if current_seq else 0
        
        # Try values in increasing order, but skip those that are likely to fail
        # Use a heuristic: try smaller increments first
        tried = set()
        for x in range(last_val + 1, min(last_val + 30, 257)):
            if x in tried:
                continue
            tried.add(x)
            
            valid, new_sums = check_add_valid(current_seq, x, sums_set)
            if not valid:
                continue
            
            new_seq = current_seq + [x]
            result = dfs_backtrack(new_seq, new_sums, max_depth, time_limit, depth + 1)
            if len(result) > len(best_result):
                best_result = result
            
            if len(best_result) >= max_depth:
                break
        
        return best_result

    # Try DFS from various starting points
    best_dfs_seq = []
    for start in range(1, 15):
        if time.time() > deadline:
            break
        initial_sums = {start}
        result = dfs_backtrack([start], initial_sums, 25, deadline)
        if len(result) > len(best_dfs_seq):
            best_dfs_seq = result

    # Also try DFS from larger starts
    for start in range(100, 130):
        if time.time() > deadline:
            break
        initial_sums = {start}
        result = dfs_backtrack([start], initial_sums, 25, deadline)
        if len(result) > len(best_dfs_seq):
            best_dfs_seq = result

    best_seq = best_greedy_seq if len(best_greedy_seq) >= len(best_dfs_seq) else best_dfs_seq
    best_sums = best_greedy_sums if len(best_greedy_seq) >= len(best_dfs_seq) else get_all_sums(best_dfs_seq)

    # Local search with targeted moves
    current_seq = list(best_seq)
    current_sums = set(best_sums)
    best_len = len(current_seq)
    best_seq_final = list(current_seq)
    
    all_candidates = set(range(1, 257))
    
    def try_add(x):
        nonlocal current_seq, current_sums, best_len, best_seq_final
        valid, new_sums = check_add_valid(current_seq, x, current_sums)
        if valid:
            current_seq.append(x)
            current_sums = new_sums
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq_final = list(current_seq)
            return True
        return False

    def try_swap(x_remove, x_add):
        nonlocal current_seq, current_sums, best_len, best_seq_final
        new_seq = [x for x in current_seq if x != x_remove]
        new_seq.append(x_add)
        new_seq.sort()
        if is_valid(new_seq):
            current_seq = new_seq
            current_sums = get_all_sums(new_seq)
            if len(current_seq) > best_len:
                best_len = len(current_seq)
                best_seq_final = list(current_seq)
            return True
        return False

    def try_remove(x_remove):
        nonlocal current_seq, current_sums
        current_seq.remove(x_remove)
        current_sums = get_all_sums(current_seq)

    # Iterated local search
    for iteration in range(50):
        if time.time() > deadline:
            break
        
        # Phase 1: Try to add elements
        current_set = set(current_seq)
        available = [x for x in all_candidates if x not in current_set]
        
        # Sort available by how "compatible" they might be
        # Try smaller values first, then larger
        random.shuffle(available)
        
        added = 0
        for x in available:
            if time.time() > deadline:
                break
            if try_add(x):
                added += 1
        
        # Phase 2: Try swaps to potentially enable more additions
        if current_seq and available:
            for _ in range(10):
                if time.time() > deadline:
                    break
                x_rem = random.choice(current_seq)
                x_add = random.choice(available)
                if x_add not in current_set:
                    if try_swap(x_rem, x_add):
                        available.remove(x_add)
                        available.append(x_rem)
                        current_set = set(current_seq)
        
        # Phase 3: Try removals to escape local minima
        if len(current_seq) > 1 and random.random() < 0.3:
            x_rem = random.choice(current_seq)
            try_remove(x_rem)
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
        
        # Restart if stuck
        if len(current_seq) < best_len - 2 and random.random() < 0.5:
            current_seq = list(best_seq_final)
            current_sums = get_all_sums(current_seq)
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]

    return sorted(best_seq_final)
