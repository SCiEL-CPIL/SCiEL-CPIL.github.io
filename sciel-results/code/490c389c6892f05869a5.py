import time
import random

def construct():
    start_time = time.time()
    TIME_LIMIT = 80.0
    MAX_VAL = 256

    def is_valid(seq):
        """Check if all consecutive block sums are distinct."""
        n = len(seq)
        if n == 0:
            return True
        sums = set()
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_block_sums(seq):
        """Get all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def can_extend(seq, new_val, sums_set):
        """Check if adding new_val to prefix keeps all block sums distinct."""
        n = len(seq)
        new_sums = []
        s = new_val
        new_sums.append(s)
        for i in range(n - 1, -1, -1):
            s += seq[i]
            new_sums.append(s)
        
        for ns in new_sums:
            if ns in sums_set:
                return False
        if len(new_sums) != len(set(new_sums)):
            return False
        return True

    def extend_forward(seq, sums_set, start_val=1):
        """Greedily extend sequence from start_val upward."""
        seq = list(seq)
        sums_set = set(sums_set)
        for x in range(start_val, MAX_VAL + 1):
            if can_extend(seq, x, sums_set):
                n = len(seq)
                s = x
                sums_set.add(s)
                for i in range(n - 1, -1, -1):
                    s += seq[i]
                    sums_set.add(s)
                seq.append(x)
        return seq, sums_set

    def extend_backward(seq, sums_set, end_val=MAX_VAL):
        """Greedily extend sequence by prepending smaller values."""
        seq = list(seq)
        sums_set = set(sums_set)
        for x in range(end_val, 0, -1):
            if x < seq[0] and can_extend([x] + seq, x, sums_set):
                n = len(seq)
                s = x
                sums_set.add(s)
                for i in range(n - 1, -1, -1):
                    s += seq[i]
                    sums_set.add(s)
                seq = [x] + seq
        return seq, sums_set

    def bidirectional_greedy(mid_val):
        """Build sequence around a midpoint."""
        seq = [mid_val]
        sums_set = {mid_val}
        # Extend forward
        for x in range(mid_val + 1, MAX_VAL + 1):
            if can_extend(seq, x, sums_set):
                n = len(seq)
                s = x
                sums_set.add(s)
                for i in range(n - 1, -1, -1):
                    s += seq[i]
                    sums_set.add(s)
                seq.append(x)
        # Extend backward
        for x in range(mid_val - 1, 0, -1):
            n = len(seq)
            new_seq = [x] + seq
            s = x
            new_sums = {s}
            for i in range(n - 1, -1, -1):
                s += seq[i]
                new_sums.add(s)
            if all(ns not in sums_set for ns in new_sums) and len(new_sums) == len(seq) + 1:
                sums_set.update(new_sums)
                seq = new_seq
        return seq, sums_set

    def local_search(seq, sums_set, time_budget):
        """Try to improve the sequence by local modifications."""
        seq = list(seq)
        sums_set = set(sums_set)
        best_seq = list(seq)
        best_sums = set(sums_set)
        best_len = len(seq)
        
        while time.time() - start_time < time_budget:
            improved = False
            
            # Try adding elements at the end
            if seq:
                for x in range(seq[-1] + 1, MAX_VAL + 1):
                    if can_extend(seq, x, sums_set):
                        n = len(seq)
                        s = x
                        sums_set.add(s)
                        for i in range(n - 1, -1, -1):
                            s += seq[i]
                            sums_set.add(s)
                        seq.append(x)
                        improved = True
                        break
            else:
                for x in range(1, MAX_VAL + 1):
                    if can_extend(seq, x, sums_set):
                        sums_set.add(x)
                        seq.append(x)
                        improved = True
                        break
            
            if not improved:
                # Try removing last element and trying alternatives
                if len(seq) > 1:
                    removed = seq.pop()
                    sums_set = get_block_sums(seq)
                    
                    for x in range(removed + 1, MAX_VAL + 1):
                        if can_extend(seq, x, sums_set):
                            n = len(seq)
                            s = x
                            sums_set.add(s)
                            for i in range(n - 1, -1, -1):
                                s += seq[i]
                                sums_set.add(s)
                            seq.append(x)
                            improved = True
                            break
                    if not improved:
                        seq.append(removed)
                        sums_set = get_block_sums(seq)
                elif len(seq) == 1:
                    # Try replacing the single element
                    old = seq[0]
                    for x in range(1, old):
                        if can_extend([x], x, set()):
                            seq = [x]
                            sums_set = {x}
                            improved = True
                            break
                    if not improved:
                        seq = [old]
                        sums_set = {old}
            
            if len(seq) > best_len:
                best_seq = list(seq)
                best_sums = set(sums_set)
                best_len = len(seq)
            
            if not improved:
                break
        
        return best_seq, best_sums

    def hill_climb(seq, sums_set, time_budget):
        """More aggressive local search with multiple moves."""
        seq = list(seq)
        sums_set = set(sums_set)
        best_seq = list(seq)
        best_sums = set(sums_set)
        best_len = len(seq)
        
        while time.time() - start_time < time_budget:
            improved = False
            
            # Try various local moves
            # 1. Add at end
            if seq:
                for x in range(seq[-1] + 1, MAX_VAL + 1):
                    if can_extend(seq, x, sums_set):
                        n = len(seq)
                        s = x
                        sums_set.add(s)
                        for i in range(n - 1, -1, -1):
                            s += seq[i]
                            sums_set.add(s)
                        seq.append(x)
                        improved = True
                        break
            
            # 2. Prepend
            if not improved and seq:
                for x in range(1, seq[0]):
                    n = len(seq)
                    new_seq = [x] + seq
                    s = x
                    new_sums = {s}
                    for i in range(n - 1, -1, -1):
                        s += seq[i]
                        new_sums.add(s)
                    if all(ns not in sums_set for ns in new_sums) and len(new_sums) == n + 1:
                        sums_set.update(new_sums)
                        seq = new_seq
                        improved = True
                        break
            
            # 3. Replace last element with larger
            if not improved and len(seq) > 1:
                last = seq.pop()
                sums_set = get_block_sums(seq)
                for x in range(last + 1, MAX_VAL + 1):
                    if can_extend(seq, x, sums_set):
                        n = len(seq)
                        s = x
                        sums_set.add(s)
                        for i in range(n - 1, -1, -1):
                            s += seq[i]
                            sums_set.add(s)
                        seq.append(x)
                        improved = True
                        break
                if not improved:
                    seq.append(last)
                    sums_set = get_block_sums(seq)
            
            # 4. Replace first element with smaller
            if not improved and len(seq) > 1:
                first = seq.pop(0)
                sums_set = get_block_sums(seq)
                for x in range(1, first):
                    new_seq = [x] + seq
                    n = len(seq)
                    s = x
                    new_sums = {s}
                    for i in range(n - 1, -1, -1):
                        s += seq[i]
                        new_sums.add(s)
                    if all(ns not in sums_set for ns in new_sums) and len(new_sums) == n + 1:
                        sums_set.update(new_sums)
                        seq = new_seq
                        improved = True
                        break
                if not improved:
                    seq = [first] + seq
                    sums_set = get_block_sums(seq)
            
            if len(seq) > best_len:
                best_seq = list(seq)
                best_sums = set(sums_set)
                best_len = len(seq)
            
            if not improved:
                break
        
        return best_seq, best_sums

    best = []
    best_sums = set()

    # Strategy 1: Forward greedy from various starts
    for start in range(1, 30):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        seq, sums_set = extend_forward([], set(), start)
        if len(seq) > len(best):
            best = list(seq)
            best_sums = set(sums_set)

    # Strategy 2: Bidirectional greedy
    for mid in range(20, 80):
        if time.time() - start_time > TIME_LIMIT - 15:
            break
        seq, sums_set = bidirectional_greedy(mid)
        if len(seq) > len(best):
            best = list(seq)
            best_sums = set(sums_set)

    # Strategy 3: Local search on best
    best, best_sums = local_search(best, best_sums, TIME_LIMIT - 20)

    # Strategy 4: Randomized restarts with local search
    random.seed(123)
    for trial in range(100):
        if time.time() - start_time > TIME_LIMIT - 10:
            break
        
        # Random starting point with random gaps
        start = random.randint(1, 50)
        seq, sums_set = extend_forward([], set(), start)
        seq, sums_set = local_search(seq, sums_set, min(TIME_LIMIT - 10, time.time() - start_time + 0.5))
        
        if len(seq) > len(best):
            best = list(seq)
            best_sums = set(sums_set)

    # Strategy 5: Hill climbing on multiple initial solutions
    for start in [1, 5, 10, 20, 30, 40]:
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        seq, sums_set = extend_forward([], set(), start)
        seq, sums_set = hill_climb(seq, sums_set, min(TIME_LIMIT - 5, time.time() - start_time + 0.5))
        
        if len(seq) > len(best):
            best = list(seq)
            best_sums = set(sums_set)

    # Strategy 6: Backward greedy
    for end in range(MAX_VAL, MAX_VAL - 30, -1):
        if time.time() - start_time > TIME_LIMIT - 5:
            break
        seq, sums_set = extend_backward([end], {end}, end - 1)
        if len(seq) > len(best):
            best = list(seq)
            best_sums = set(sums_set)

    # Final: try to extend best further
    best, best_sums = local_search(best, best_sums, TIME_LIMIT - 2)

    # Final verification
    if is_valid(best):
        return best
    else:
        # Fallback: return a known valid sequence
        return [1, 2, 4, 8, 16, 32, 64, 128]
