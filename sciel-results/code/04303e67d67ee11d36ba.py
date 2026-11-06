import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 57.0  # 3s margin for 60s limit

    def is_valid(seq):
        n = len(seq)
        if n == 0:
            return True
        prods = set()
        for i in range(n):
            p = 1
            for j in range(i, n):
                p *= seq[j]
                if p in prods:
                    return False
                prods.add(p)
        return True

    def try_insert(seq, val):
        # seq is sorted
        # find position
        pos = 0
        while pos < len(seq) and seq[pos] < val:
            pos += 1
        # check if val is already in seq
        if pos < len(seq) and seq[pos] == val:
            return None
        new_seq = seq[:pos] + [val] + seq[pos:]
        if is_valid(new_seq):
            return new_seq
        return None

    def try_insert_sorted(seq, candidates):
        """Try to insert candidates in sorted order into seq (sorted)."""
        result = seq[:]
        remaining = sorted(candidates)
        for val in remaining:
            if val in result:
                continue
            pos = 0
            while pos < len(result) and result[pos] < val:
                pos += 1
            if pos < len(result) and result[pos] == val:
                continue
            new_seq = result[:pos] + [val] + result[pos:]
            if is_valid(new_seq):
                result = new_seq
        return result

    def greedy_construct(exclude_set=None):
        if exclude_set is None:
            exclude_set = set()
        values = []
        for x in range(2, 129):
            if x in exclude_set:
                continue
            new_seq = try_insert(values, x)
            if new_seq is not None:
                values = new_seq
        return values

    def polish(seq):
        """Systematic polish: try inserting each excluded element in sorted order,
        then try single swaps, repeat until no improvement."""
        all_values = set(range(2, 129))
        current = seq[:]
        improved = True
        while improved:
            improved = False
            # Phase A: Try to insert each excluded element (sorted order)
            excluded = sorted(all_values - set(current))
            for e in excluded:
                if e in current:
                    continue
                new_seq = try_insert(current, e)
                if new_seq is not None:
                    current = new_seq
                    improved = True
                    break
            if improved:
                continue
            # Phase B: Try single swaps (remove one, add one)
            n = len(current)
            for i in range(n):
                if time.time() > deadline:
                    return current
                removed = current[i]
                without = current[:i] + current[i+1:]
                excluded = sorted(all_values - set(without))
                for e in excluded:
                    if e == removed:
                        continue
                    new_seq = try_insert(without, e)
                    if new_seq is not None:
                        current = new_seq
                        improved = True
                        break
                if improved:
                    break
        return current

    # 1. Initial Greedy Construction
    best_seq = greedy_construct()
    best_len = len(best_seq)
    
    # 2. Polish the initial solution
    polished = polish(best_seq)
    if len(polished) >= best_len:
        best_seq = polished
        best_len = len(best_seq)
    
    # 3. Local Search with Kicks
    all_values = set(range(2, 129))
    
    def get_excluded(seq):
        return all_values - set(seq)
    
    current_seq = best_seq
    current_excluded = get_excluded(current_seq)
    
    max_kicks = 50000
    kick_count = 0
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() < deadline and kick_count < max_kicks:
        kick_count += 1
        
        # Phase 1: Try to insert any excluded element (sorted for determinism in polish-like behavior)
        improved = False
        excl_list = sorted(current_excluded)
        
        for e in excl_list:
            if time.time() > deadline:
                break
            new_seq = try_insert(current_seq, e)
            if new_seq is not None:
                current_seq = new_seq
                current_excluded.discard(e)
                improved = True
                break
        
        if improved:
            if len(current_seq) > best_len:
                best_seq = current_seq
                best_len = len(best_seq)
            no_improve_count = 0
            continue
        
        # Phase 2: If no direct insertion, try a "kick"
        k = random.randint(1, 3)
        if len(current_seq) < k:
            k = max(1, len(current_seq))
            
        if k > 0:
            to_remove = random.sample(current_seq, k)
            to_remove_set = set(to_remove)
            
            kicked_seq = [x for x in current_seq if x not in to_remove_set]
            
            candidates = current_excluded | to_remove_set
            cand_list = sorted(candidates)
            
            new_seq = kicked_seq
            new_excluded = all_values - set(new_seq)
            
            for e in cand_list:
                if time.time() > deadline:
                    break
                if e in new_excluded:
                    ins_seq = try_insert(new_seq, e)
                    if ins_seq is not None:
                        new_seq = ins_seq
                        new_excluded.discard(e)
            
            # Accept if length is not reduced
            if len(new_seq) >= len(current_seq):
                current_seq = new_seq
                current_excluded = new_excluded
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(best_seq)
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Try to polish the kicked version even if shorter, in case it leads to improvement
                if len(new_seq) >= len(current_seq) - 2:
                    polished_kick = polish(new_seq)
                    if len(polished_kick) >= len(current_seq):
                        current_seq = polished_kick
                        current_excluded = all_values - set(current_seq)
                        if len(current_seq) > best_len:
                            best_seq = current_seq
                            best_len = len(best_seq)
                            no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    no_improve_count += 1
        
        # Periodic polish to escape local optima
        if kick_count % 200 == 0 and time.time() < deadline:
            polished = polish(current_seq)
            if len(polished) >= len(current_seq):
                current_seq = polished
                current_excluded = all_values - set(current_seq)
                if len(current_seq) > best_len:
                    best_seq = current_seq
                    best_len = len(best_seq)
                    no_improve_count = 0
    
    # Final polish
    if time.time() < deadline:
        final_polished = polish(best_seq)
        if len(final_polished) >= best_len:
            best_seq = final_polished
            best_len = len(best_seq)
    
    return best_seq
