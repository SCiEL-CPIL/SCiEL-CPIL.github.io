import time
import sys
import random
from itertools import permutations

def construct(n):
    start_time = time.time()
    deadline = start_time + 140.0
    
    # 1. Recursive Construction
    word = [1]
    
    for m in range(2, n + 1):
        perms = []
        seen = set()
        L = len(word)
        for i in range(L - (m - 1) + 1):
            window = tuple(word[i:i + m - 1])
            if len(set(window)) == m - 1 and window not in seen:
                seen.add(window)
                perms.append(window)
        
        if m == n and n > 4:
            def calc_overlap(u, v):
                max_k = min(len(u), len(v))
                for k in range(max_k, 0, -1):
                    if u[-k:] == v[:k]:
                        return k
                return 0

            base_order = perms
            base_score = 0
            for i in range(len(base_order) - 1):
                base_score += calc_overlap(base_order[i], base_order[i+1])
                
            current_order = list(base_order)
            current_score = base_score
            improved = True
            max_iterations = 500 
            iter_count = 0
            
            while improved and time.time() < deadline - 10.0 and iter_count < max_iterations:
                improved = False
                iter_count += 1
                
                if len(current_order) < 2:
                    break
                
                i = random.randint(0, len(current_order) - 2)
                j = random.randint(i + 1, len(current_order) - 1)
                
                new_order = current_order[:i] + current_order[i:j+1][::-1] + current_order[j+1:]
                
                score = 0
                for k in range(len(new_order) - 1):
                    score += calc_overlap(new_order[k], new_order[k+1])
                    
                if score > current_score:
                    current_score = score
                    current_order = new_order
                    improved = True
                    
            if current_score > base_score:
                perms = current_order
            else:
                perms = base_order

        new_word = []
        for p in perms:
            block = list(p) + [m] + list(p)
            
            max_overlap = 0
            limit = min(len(new_word), len(block))
            for o in range(limit, -1, -1):
                if o == 0:
                    break
                if new_word[-o:] == block[:o]:
                    max_overlap = o
                    break
            
            new_word.extend(block[max_overlap:])
        
        word = new_word
        
        if time.time() > deadline:
            break
    
    if n <= 4:
        return "".join(map(str, word))

    # 2. Polishing Phase: Character Deletion with multiple strategies
    all_perms = list(permutations(range(1, n + 1)))
    perm_to_id = {p: i for i, p in enumerate(all_perms)}
    num_perms = len(all_perms)
    
    def compute_counts(w):
        cnts = [0] * num_perms
        L = len(w)
        for i in range(L - n + 1):
            p = tuple(w[i:i+n])
            if p in perm_to_id:
                cnts[perm_to_id[p]] += 1
        return cnts
    
    counts = compute_counts(word)
    if any(c == 0 for c in counts):
        return "".join(map(str, word))
    
    def is_deletion_valid(i, current_word, current_counts):
        L = len(current_word)
        if L <= n:
            return False
            
        old_start_min = max(0, i - n + 1)
        old_start_max = min(i, L - n)
        
        changes = []
        
        for k in range(old_start_min, old_start_max + 1):
            p = tuple(current_word[k:k+n])
            pid = perm_to_id.get(p)
            if pid is not None:
                current_counts[pid] -= 1
                changes.append((pid, -1))
                if current_counts[pid] < 0:
                    for cp, delta in changes:
                        current_counts[cp] -= delta
                    return False

        new_start_min = max(0, i - n + 1)
        new_start_max = min(i - 1, L - n - 1)
        
        for k in range(new_start_min, new_start_max + 1):
            w1 = current_word[k:i]
            w2 = current_word[i+1:k+n+1]
            p = tuple(w1 + w2)
            pid = perm_to_id.get(p)
            if pid is not None:
                current_counts[pid] += 1
                changes.append((pid, 1))
        
        valid = True
        for pid, delta in changes:
            if delta == -1:
                if current_counts[pid] == 0:
                    valid = False
                    break
        
        if not valid:
            for pid, delta in changes:
                current_counts[pid] -= delta
            return False
        
        return True

    # Strategy 1: Left-to-right greedy deletion
    def delete_left_to_right(w, cnts, time_limit):
        i = 0
        while i < len(w):
            if time.time() > time_limit:
                break
            if is_deletion_valid(i, w, cnts):
                w.pop(i)
            else:
                i += 1
        return w, cnts

    # Strategy 2: Right-to-left greedy deletion
    def delete_right_to_left(w, cnts, time_limit):
        i = len(w) - 1
        while i >= 0:
            if time.time() > time_limit:
                break
            if is_deletion_valid(i, w, cnts):
                w.pop(i)
            else:
                i -= 1
        return w, cnts

    # Strategy 3: Multiple passes with random restarts
    def delete_with_restarts(w, cnts, time_limit, num_restarts=3):
        best_w = list(w)
        best_cnts = list(cnts)
        best_len = len(w)
        
        for r in range(num_restarts):
            if time.time() > time_limit - 2.0:
                break
            cur_w = list(w)
            cur_cnts = list(cnts)
            
            # Randomize the order of checking positions
            positions = list(range(len(cur_w)))
            random.shuffle(positions)
            
            for i in positions:
                if time.time() > time_limit - 1.0:
                    break
                if i >= len(cur_w):
                    continue
                if is_deletion_valid(i, cur_w, cur_cnts):
                    cur_w.pop(i)
            
            if len(cur_w) < best_len:
                best_w = cur_w
                best_cnts = cur_cnts
                best_len = len(cur_w)
        
        return best_w, best_cnts

    # Apply strategies in sequence
    # First: left-to-right
    word, counts = delete_left_to_right(word, counts, deadline - 20.0)
    
    # Second: right-to-left
    word, counts = delete_right_to_left(word, counts, deadline - 10.0)
    
    # Third: multiple restarts if time remains
    if time.time() < deadline - 5.0:
        word, counts = delete_with_restarts(word, counts, deadline - 2.0, num_restarts=5)
    
    # Final: one more left-to-right pass
    if time.time() < deadline - 1.0:
        word, counts = delete_left_to_right(word, counts, deadline - 0.5)
    
    return "".join(map(str, word))

# Keep the function name and signature as required
# The task says: Interface: construct(n) -> a word ...
# The current program defines construct(n).
