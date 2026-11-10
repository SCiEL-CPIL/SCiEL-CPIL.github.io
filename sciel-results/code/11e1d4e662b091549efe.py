import time
import random

def construct():
    start_time = time.time()
    time_limit = 85.0  # Evaluator kills at 90s, leave 5s margin

    def get_all_sums(seq):
        """Compute all consecutive block sums for a sequence."""
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums

    def check_append_conflict(seq, x, sums_set):
        """Check if appending x creates duplicate consecutive-block sums.
        Only need to check new suffix sums ending at x."""
        total = x
        if total in sums_set:
            return True
        for a in reversed(seq):
            total += a
            if total in sums_set:
                return True
        return False

    def get_suffix_sums(seq, x):
        """Get new suffix sums when appending x to seq."""
        sums = set()
        total = x
        sums.add(total)
        for a in reversed(seq):
            total += a
            sums.add(total)
        return sums

    # 1. Greedy baseline with lookahead
    values = []
    sums_set = set()
    for x in range(1, 257):
        if not check_append_conflict(values, x, sums_set):
            values.append(x)
            new_s = get_suffix_sums(values[:-1], x)
            sums_set.update(new_s)

    best_seq = values[:]
    best_len = len(best_seq)
    best_sums = sums_set.copy()

    # 2. Try multiple greedy strategies with different orderings
    strategies = [
        list(range(1, 257)),
        list(range(256, 0, -1)),
        list(range(128, 257)) + list(range(1, 129)),
        list(range(1, 257, 2)) + list(range(2, 257, 2)),
        list(range(128, 257, 2)) + list(range(129, 257, 2)),
    ]
    
    for strategy in strategies:
        if time.time() > time_limit - 10:
            break
        vals = []
        s_set = set()
        for x in strategy:
            if x <= (vals[-1] if vals else 0):
                continue
            if not check_append_conflict(vals, x, s_set):
                vals.append(x)
                new_s = get_suffix_sums(vals[:-1], x)
                s_set.update(new_s)
        if len(vals) > best_len:
            best_seq = vals[:]
            best_len = len(vals)
            best_sums = s_set.copy()

    # 3. Local search: try to insert elements into best_seq
    def try_insert(seq, sums_set, max_time):
        """Try to insert elements into seq to make it longer."""
        current_seq = seq[:]
        current_sums = sums_set.copy()
        
        improved = True
        while improved and time.time() < max_time:
            improved = False
            for i in range(len(current_seq) + 1):
                if time.time() > max_time:
                    break
                prev_val = current_seq[i-1] if i > 0 else 0
                next_val = current_seq[i] if i < len(current_seq) else 257
                for x in range(prev_val + 1, min(next_val, 257)):
                    # Check if inserting x at position i is valid
                    test_seq = current_seq[:i] + [x] + current_seq[i:]
                    
                    # Compute all sums for test_seq
                    test_sums = set()
                    valid = True
                    n = len(test_seq)
                    for a in range(n):
                        s = 0
                        for b in range(a, n):
                            s += test_seq[b]
                            if s in test_sums:
                                valid = False
                                break
                            test_sums.add(s)
                        if not valid:
                            break
                    
                    if valid:
                        current_seq = test_seq
                        current_sums = test_sums
                        improved = True
                        break
                if improved:
                    break
        
        return current_seq, current_sums

    # 4. Apply local search to best_seq
    if time.time() < time_limit - 5:
        improved_seq, improved_sums = try_insert(best_seq, best_sums, time_limit - 5)
        if len(improved_seq) > best_len:
            best_seq = improved_seq
            best_len = len(best_seq)
            best_sums = improved_sums

    # 5. Try to extend from the end of best_seq
    if time.time() < time_limit - 3:
        current_seq = best_seq[:]
        current_sums = best_sums.copy()
        last_val = current_seq[-1] if current_seq else 0
        
        for x in range(last_val + 1, 257):
            if time.time() > time_limit - 3:
                break
            if not check_append_conflict(current_seq, x, current_sums):
                current_seq.append(x)
                new_s = get_suffix_sums(current_seq[:-1], x)
                current_sums.update(new_s)
        
        if len(current_seq) > best_len:
            best_seq = current_seq
            best_len = len(best_seq)
            best_sums = current_sums

    # 6. Final local search pass
    if time.time() < time_limit - 2:
        final_seq, final_sums = try_insert(best_seq, best_sums, time_limit - 2)
        if len(final_seq) > best_len:
            best_seq = final_seq
            best_len = len(best_seq)

    return best_seq
