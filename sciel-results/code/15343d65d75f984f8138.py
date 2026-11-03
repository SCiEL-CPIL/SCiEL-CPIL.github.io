import time
import random
import math

def _precompute_neighbors(q, n, radius):
    total = q ** n
    neigh = [[] for _ in range(total)]
    for idx in range(total):
        neigh[idx].append(idx)
        for i in range(n):
            pow_q_i = q ** i
            digit = (idx // pow_q_i) % q
            for new_digit in range(q):
                if new_digit != digit:
                    new_idx = idx + (new_digit - digit) * pow_q_i
                    neigh[idx].append(new_idx)
    return neigh, total

def construct(q, n, radius):
    deadline = time.monotonic() + 115.0
    total = q ** n
    rng = random.Random(42)
    
    neigh, _ = _precompute_neighbors(q, n, radius)
    
    # Precompute neighbor sets for faster intersection
    neigh_sets = [frozenset(nb) for nb in neigh]
    
    best_code = None
    best_size = total + 1
    
    def greedy_init(seed=None):
        if seed is not None:
            local_rng = random.Random(seed)
        else:
            local_rng = rng
        
        code = []
        code_set = set()
        counts = [0] * total
        remaining_set = set(range(total))
        rem_count = total
        
        def add_to_code(c):
            nonlocal rem_count
            code.append(c)
            code_set.add(c)
            for u in neigh[c]:
                counts[u] += 1
                if counts[u] == 1:
                    remaining_set.discard(u)
                    rem_count -= 1
        
        while rem_count > 0:
            if time.monotonic() > deadline:
                return code, counts, remaining_set, rem_count
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in code_set:
                    continue
                gain = len(neigh_sets[c] & remaining_set)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            
            if best_c == -1 or best_gain == 0:
                break
                
            add_to_code(best_c)
            
        return code, counts, remaining_set, rem_count

    def local_search(code, counts, remaining_set, rem_count):
        changed = True
        while changed:
            changed = False
            for c in code[:]:
                removable = True
                for u in neigh[c]:
                    if counts[u] == 1:
                        removable = False
                        break
                if removable:
                    code.remove(c)
                    for u in neigh[c]:
                        counts[u] -= 1
                        if counts[u] == 0:
                            remaining_set.add(u)
                            rem_count += 1
                    changed = True
        return code, counts, remaining_set, rem_count

    def try_swap(code, counts, remaining_set, rem_count):
        """Try to find a 2-for-1 move: remove one codeword, add another that covers all newly uncovered."""
        code_set = set(code)
        
        for c_remove in code[:]:
            # Temporarily remove c_remove
            for u in neigh[c_remove]:
                counts[u] -= 1
            
            newly_uncovered = []
            for u in neigh[c_remove]:
                if counts[u] == 0:
                    newly_uncovered.append(u)
            
            if not newly_uncovered:
                code.remove(c_remove)
                return code, counts, remaining_set, rem_count
            
            uncovered_set = frozenset(newly_uncovered)
            best_c_add = -1
            
            for c_add in range(total):
                if c_add in code_set:
                    continue
                if neigh_sets[c_add] >= uncovered_set:
                    best_c_add = c_add
                    break
            
            if best_c_add != -1:
                code.remove(c_remove)
                code.append(best_c_add)
                for u in neigh[best_c_add]:
                    counts[u] += 1
                    if counts[u] == 1:
                        remaining_set.discard(u)
                        rem_count -= 1
                return code, counts, remaining_set, rem_count
            else:
                for u in neigh[c_remove]:
                    counts[u] += 1
        
        return code, counts, remaining_set, rem_count

    def perturb_and_repair(code, counts, remaining_set, rem_count):
        if not code:
            return code, counts, remaining_set, rem_count
        
        c_to_remove = rng.choice(code)
        code.remove(c_to_remove)
        for u in neigh[c_to_remove]:
            counts[u] -= 1
            if counts[u] == 0:
                remaining_set.add(u)
                rem_count += 1
        
        new_code = code[:]
        new_counts = counts[:]
        new_remaining_set = set(remaining_set)
        new_rem_count = rem_count
        new_code_set = set(new_code)
        
        while new_rem_count > 0:
            if time.monotonic() > deadline:
                return new_code, new_counts, new_remaining_set, new_rem_count
            
            best_c = -1
            best_gain = -1
            for c in range(total):
                if c in new_code_set:
                    continue
                gain = len(neigh_sets[c] & new_remaining_set)
                if gain > best_gain:
                    best_gain = gain
                    best_c = c
            if best_c == -1:
                break
            new_code.append(best_c)
            new_code_set.add(best_c)
            for u in neigh[best_c]:
                new_counts[u] += 1
                if new_counts[u] == 1:
                    new_remaining_set.discard(u)
                    new_rem_count -= 1
            
        return new_code, new_counts, new_remaining_set, new_rem_count

    restart_num = 0
    while time.monotonic() < deadline:
        code, counts, remaining_set, rem_count = greedy_init(seed=restart_num * 137)
        code, counts, remaining_set, rem_count = local_search(code, counts, remaining_set, rem_count)
        
        if len(code) < best_size:
            best_size = len(code)
            best_code = code[:]
        
        current_code = code[:]
        current_counts = counts[:]
        current_remaining_set = set(remaining_set)
        current_rem_count = rem_count
        
        for step in range(100):
            if time.monotonic() > deadline:
                break
            
            if step % 10 == 0:
                p_code, p_counts, p_remaining_set, p_rem_count = try_swap(
                    current_code[:], current_counts[:], set(current_remaining_set), current_rem_count
                )
                p_code, p_counts, p_remaining_set, p_rem_count = local_search(p_code, p_counts, p_remaining_set, p_rem_count)
            else:
                p_code, p_counts, p_remaining_set, p_rem_count = perturb_and_repair(
                    current_code[:], current_counts[:], set(current_remaining_set), current_rem_count
                )
                p_code, p_counts, p_remaining_set, p_rem_count = local_search(p_code, p_counts, p_remaining_set, p_rem_count)
            
            if len(p_code) <= len(current_code):
                current_code = p_code
                current_counts = p_counts
                current_remaining_set = p_remaining_set
                current_rem_count = p_rem_count
                
                if len(current_code) < best_size:
                    best_size = len(current_code)
                    best_code = current_code[:]
                    
            if best_size <= 71:
                break
        
        restart_num += 1

    result = []
    for idx in best_code:
        word = [0] * n
        temp_idx = idx
        for i in range(n):
            word[i] = temp_idx % q
            temp_idx //= q
        result.append(word)
        
    return result
