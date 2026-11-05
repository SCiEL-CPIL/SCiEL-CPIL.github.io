import time
import random
import sys

sys.setrecursionlimit(10000)

def construct():
    start_time = time.time()
    deadline = start_time + 55

    def get_suffix_sums(seq):
        """Get suffix sums: suffix_sums[i] = sum(seq[i:])"""
        L = len(seq)
        suffix = [0] * L
        if L > 0:
            suffix[-1] = seq[-1]
            for i in range(L - 2, -1, -1):
                suffix[i] = suffix[i + 1] + seq[i]
        return suffix

    def get_all_sums(seq):
        """Get all consecutive block sums efficiently."""
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
        if not seq:
            return True
        sums = get_all_sums(seq)
        return len(sums) == len(seq) * (len(seq) + 1) // 2

    def check_append_valid(seq, x, sums_set, suffix_sums):
        """Check if appending x to seq is valid. Returns (valid, new_sums, new_suffix_sums)."""
        new_suffix_sums = [x]
        for ss in suffix_sums:
            new_suffix_sums.append(x + ss)
        
        # Check if any new sum conflicts
        for ns in new_suffix_sums:
            if ns in sums_set:
                return False, None, None
        
        new_sums = set(sums_set)
        for ns in new_suffix_sums:
            new_sums.add(ns)
        
        return True, new_sums, new_suffix_sums

    def check_prepend_valid(seq, x, sums_set, prefix_sums):
        """Check if prepending x to seq is valid."""
        new_prefix_sums = [x]
        for ps in prefix_sums:
            new_prefix_sums.append(x + ps)
        
        for ns in new_prefix_sums:
            if ns in sums_set:
                return False, None, None
        
        new_sums = set(sums_set)
        for ns in new_prefix_sums:
            new_sums.add(ns)
        
        return True, new_sums, new_prefix_sums

    def greedy_increasing():
        """Greedy: try to add smallest available value."""
        values = []
        sums_set = set()
        suffix_sums = []
        value_set = set()
        
        for x in range(1, 257):
            if x in value_set:
                continue
            valid, new_sums, new_suffix = check_append_valid(values, x, sums_set, suffix_sums)
            if valid:
                values.append(x)
                value_set.add(x)
                sums_set = new_sums
                suffix_sums = new_suffix
        
        return values

    def greedy_decreasing():
        """Greedy: try to add largest available value first (prepend)."""
        values = []
        sums_set = set()
        prefix_sums = []
        value_set = set()
        
        for x in range(256, 0, -1):
            if x in value_set:
                continue
            valid, new_sums, new_prefix = check_prepend_valid(values, x, sums_set, prefix_sums)
            if valid:
                values.insert(0, x)
                value_set.add(x)
                sums_set = new_sums
                prefix_sums = new_prefix
        
        return values

    def greedy_random_order(seed_offset=0):
        """Greedy: try values in randomized order."""
        rng = random.Random(seed_offset)
        values = []
        sums_set = set()
        suffix_sums = []
        value_set = set()
        candidates = list(range(1, 257))
        rng.shuffle(candidates)
        
        for x in candidates:
            if x in value_set:
                continue
            valid, new_sums, new_suffix = check_append_valid(values, x, sums_set, suffix_sums)
            if valid:
                values.append(x)
                value_set.add(x)
                sums_set = new_sums
                suffix_sums = new_suffix
        values.sort()
        return values

    # Get initial sequences from different greedy strategies
    seq1 = greedy_increasing()
    seq2 = greedy_decreasing()
    seq3 = greedy_random_order(42)
    seq4 = greedy_random_order(123)
    seq5 = greedy_random_order(999)
    
    candidates = [seq1, seq2, seq3, seq4, seq5]
    best_seq = max(candidates, key=len)
    best_len = len(best_seq)

    # Local search with incremental sum tracking
    current_seq = list(best_seq)
    current_sums = get_all_sums(current_seq)
    current_suffix = get_suffix_sums(current_seq)
    current_value_set = set(current_seq)
    
    all_candidates = set(range(1, 257))
    
    # Simulated annealing style local search with improved moves
    temperature = 100.0
    cooling_rate = 0.997
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.55:  # Try to add an element
            available = [x for x in all_candidates if x not in current_value_set]
            if available:
                # Try multiple candidates and pick the best
                best_add = None
                best_new_len = len(current_seq)
                
                # Sample up to 30 candidates for efficiency
                sample_size = min(30, len(available))
                sampled = random.sample(available, sample_size)
                
                for x in sampled:
                    new_seq = current_seq + [x]
                    new_seq.sort()
                    if is_valid(new_seq):
                        if len(new_seq) > best_new_len:
                            best_new_len = len(new_seq)
                            best_add = x
                
                if best_add is not None:
                    current_seq.append(best_add)
                    current_seq.sort()
                    current_value_set.add(best_add)
                    current_sums = get_all_sums(current_seq)
                    current_suffix = get_suffix_sums(current_seq)
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq = list(current_seq)
        
        elif r < 0.75:  # Try to swap: replace one element with another
            if current_seq and len(current_seq) > 1:
                available = [x for x in all_candidates if x not in current_value_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    new_seq = [x for x in current_seq if x != x_rem]
                    new_seq.append(x_add)
                    new_seq.sort()
                    if is_valid(new_seq):
                        current_seq = new_seq
                        current_value_set.discard(x_rem)
                        current_value_set.add(x_add)
                        current_sums = get_all_sums(current_seq)
                        current_suffix = get_suffix_sums(current_seq)
                        if len(current_seq) > best_len:
                            best_len = len(current_seq)
                            best_seq = list(current_seq)
        
        elif r < 0.88:  # Try to remove then add (2-opt style)
            if current_seq and len(current_seq) > 2:
                x_rem = random.choice(current_seq)
                temp_seq = [x for x in current_seq if x != x_rem]
                temp_sums = get_all_sums(temp_seq)
                temp_suffix = get_suffix_sums(temp_seq)
                temp_value_set = current_value_set - {x_rem}
                
                available = [x for x in all_candidates if x not in temp_value_set]
                if available:
                    sample_size = min(15, len(available))
                    sampled = random.sample(available, sample_size)
                    found = False
                    for x_add in sampled:
                        new_seq = temp_seq + [x_add]
                        new_seq.sort()
                        if is_valid(new_seq):
                            current_seq = new_seq
                            current_value_set = temp_value_set | {x_add}
                            current_sums = get_all_sums(current_seq)
                            current_suffix = get_suffix_sums(current_seq)
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq = list(current_seq)
                            found = True
                            break
                    if not found:
                        # Keep the removal if it helps escape local minima
                        if random.random() < 0.3:
                            current_seq = temp_seq
                            current_value_set = temp_value_set
                            current_sums = temp_sums
                            current_suffix = temp_suffix
        
        else:  # Try to remove (to escape local minima)
            if current_seq and len(current_seq) > 2:
                x_rem = random.choice(current_seq)
                current_seq.remove(x_rem)
                current_value_set.discard(x_rem)
                current_sums = get_all_sums(current_seq)
                current_suffix = get_suffix_sums(current_seq)
        
        # Cool down
        temperature *= cooling_rate
    
    return sorted(best_seq)
