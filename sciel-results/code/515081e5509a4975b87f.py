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

    def get_sums(seq):
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

    def is_valid_fast(seq):
        if not seq:
            return True
        L = len(seq)
        P = [0] * (L + 1)
        for i in range(L):
            P[i + 1] = P[i] + seq[i]
        seen = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                s = P[j] - P[i]
                if s in seen:
                    return False
                seen.add(s)
        return True

    def greedy(start_val, end_val, step):
        values = []
        for x in range(start_val, end_val, step):
            new_vals = values + [x]
            new_vals.sort()
            if is_valid_fast(new_vals):
                values = new_vals
        return values

    seq1 = greedy(1, 257, 1)
    seq2 = greedy(256, 0, -1)
    
    best_seq = seq1 if len(seq1) >= len(seq2) else seq2

    def dfs_backtrack(current_seq, sums_set, suffix_sums, max_depth, time_limit):
        if time.time() > time_limit:
            return current_seq
        
        L = len(current_seq)
        if L >= max_depth:
            return current_seq
        
        last_val = current_seq[-1] if current_seq else 0
        best_result = list(current_seq)
        
        for x in range(last_val + 1, 257):
            if time.time() > time_limit:
                break
            
            new_suffix_sums = [x]
            for ss in suffix_sums:
                new_suffix_sums.append(x + ss)
            
            valid = True
            for ns in new_suffix_sums:
                if ns in sums_set:
                    valid = False
                    break
            
            if not valid:
                continue
            
            new_sums_set = set(sums_set)
            for ns in new_suffix_sums:
                new_sums_set.add(ns)
            
            new_seq = current_seq + [x]
            result = dfs_backtrack(new_seq, new_sums_set, new_suffix_sums, max_depth, time_limit)
            if len(result) > len(best_result):
                best_result = result
            
            if len(best_result) >= max_depth:
                break
        
        return best_result

    best_dfs_seq = []
    for start in range(1, 20):
        if time.time() > deadline:
            break
        initial_suffix = [start]
        initial_sums = {start}
        result = dfs_backtrack([start], initial_sums, initial_suffix, 20, deadline)
        if len(result) > len(best_dfs_seq):
            best_dfs_seq = result
    
    if len(best_dfs_seq) > len(best_seq):
        best_seq = best_dfs_seq

    current_seq = list(best_seq)
    best_len = len(current_seq)
    best_seq_final = list(current_seq)
    
    all_candidates = set(range(1, 257))
    
    def compute_prefix_sums(seq):
        P = [0] * (len(seq) + 1)
        for i in range(len(seq)):
            P[i + 1] = P[i] + seq[i]
        return P
    
    def compute_all_sums(seq):
        P = compute_prefix_sums(seq)
        L = len(seq)
        sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                sums.add(P[j] - P[i])
        return sums
    
    cur_P = compute_prefix_sums(current_seq)
    cur_sums = compute_all_sums(current_seq)
    
    def try_insert_sorted(x, seq, P, sums_set):
        if x in seq:
            return None
        
        pos = 0
        while pos < len(seq) and seq[pos] < x:
            pos += 1
        
        L = len(seq)
        
        new_sums = set()
        
        for i in range(pos + 1):
            s = (P[pos] - P[i]) + x
            new_sums.add(s)
        
        for j in range(pos, L):
            s = x + (P[j+1] - P[pos])
            new_sums.add(s)
        
        for i in range(pos):
            for j in range(pos, L):
                s = (P[pos] - P[i]) + x + (P[j+1] - P[pos])
                new_sums.add(s)
        
        for ns in new_sums:
            if ns in sums_set:
                return None
        
        new_seq = seq[:pos] + [x] + seq[pos:]
        
        new_P = [0] * (L + 2)
        for i in range(pos):
            new_P[i+1] = P[i+1]
        new_P[pos+1] = P[pos] + x
        for i in range(pos, L):
            new_P[i+2] = P[i+1] + x
        
        new_sums_set = set(sums_set)
        new_sums_set.update(new_sums)
        
        return (new_seq, new_P, new_sums_set)
    
    def try_remove_sorted(x, seq, P, sums_set):
        if x not in seq:
            return None
        
        idx = seq.index(x)
        new_seq = seq[:idx] + seq[idx+1:]
        
        if not new_seq:
            return (new_seq, [0], set())
        
        new_P = [0] * (len(new_seq) + 1)
        for i in range(len(new_seq)):
            new_P[i + 1] = new_P[i] + new_seq[i]
        
        L = len(new_seq)
        new_sums = set()
        for i in range(L):
            for j in range(i + 1, L + 1):
                new_sums.add(new_P[j] - new_P[i])
        
        return (new_seq, new_P, new_sums)
    
    while time.time() < deadline:
        r = random.random()
        
        if r < 0.7:
            current_set = set(current_seq)
            available = [x for x in all_candidates if x not in current_set]
            if available:
                x = random.choice(available)
                result = try_insert_sorted(x, current_seq, cur_P, cur_sums)
                if result is not None:
                    current_seq, cur_P, cur_sums = result
                    if len(current_seq) > best_len:
                        best_len = len(current_seq)
                        best_seq_final = list(current_seq)
                        
        elif r < 0.9:
            if current_seq:
                current_set = set(current_seq)
                available = [x for x in all_candidates if x not in current_set]
                if available:
                    x_rem = random.choice(current_seq)
                    x_add = random.choice(available)
                    rem_result = try_remove_sorted(x_rem, current_seq, cur_P, cur_sums)
                    if rem_result is not None:
                        seq_no_rem, P_no_rem, sums_no_rem = rem_result
                        add_result = try_insert_sorted(x_add, seq_no_rem, P_no_rem, sums_no_rem)
                        if add_result is not None:
                            current_seq, cur_P, cur_sums = add_result
                            if len(current_seq) > best_len:
                                best_len = len(current_seq)
                                best_seq_final = list(current_seq)
                            
        else:
            if current_seq and len(current_seq) > 1:
                x_rem = random.choice(current_seq)
                result = try_remove_sorted(x_rem, current_seq, cur_P, cur_sums)
                if result is not None:
                    current_seq, cur_P, cur_sums = result

    return sorted(best_seq_final)
