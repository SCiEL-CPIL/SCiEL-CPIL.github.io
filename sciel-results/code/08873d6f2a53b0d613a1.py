import time
import sys

def _compute_table():
    """Pre-compute minimal antichain representations for targets 1..128."""
    start_time = time.time()
    deadline = 85.0  # Leave 5s safety margin before 90s kill
    
    smooth = set()
    for a in range(8):
        for b in range(5):
            for c in range(4):
                val = (2 ** a) * (3 ** b) * (5 ** c)
                if val <= 128:
                    smooth.add(val)
    
    smooth_list = sorted(smooth)
    reps = [None] * 129
    
    # k=1: smooth numbers
    for t in range(1, 129):
        if t in smooth:
            reps[t] = (t,)
    
    # k=2: pairs
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        for i in range(len(smooth_list)):
            a = smooth_list[i]
            if a >= t:
                break
            b = t - a
            if b < a:
                continue
            if b in smooth and a % b != 0 and b % a != 0:
                reps[t] = (a, b)
                break
    
    # k=3: triples
    for t in range(1, 129):
        if reps[t] is not None:
            continue
        found = False
        for i in range(len(smooth_list)):
            a = smooth_list[i]
            if a >= t:
                break
            for j in range(i + 1, len(smooth_list)):
                b = smooth_list[j]
                if a + b >= t:
                    break
                c = t - a - b
                if c <= b:
                    continue
                if c in smooth:
                    if (b % a != 0) and (c % a != 0) and (c % b != 0):
                        reps[t] = (a, b, c)
                        found = True
                        break
            if found:
                break
    
    # Check if we need to do DFS
    unresolved = [t for t in range(1, 129) if reps[t] is None]
    
    if unresolved:
        # k=4 and above: DFS with time guard and pruning
        def dfs(rem, start_idx, current_terms, k_remaining):
            if time.time() > deadline:
                return None
            if k_remaining == 0:
                return current_terms if rem == 0 else None
            if rem <= 0:
                return None
            # Pruning: need at least k_remaining terms, each >= smooth_list[start_idx]
            if start_idx + k_remaining > len(smooth_list):
                return None
            # Pruning: sum of smallest k_remaining available terms must be <= rem
            min_possible = sum(smooth_list[start_idx:start_idx + k_remaining]) if start_idx + k_remaining <= len(smooth_list) else float('inf')
            if min_possible > rem:
                return None
            # Pruning: sum of largest k_remaining available terms must be >= rem
            max_start = len(smooth_list) - k_remaining
            if max_start < start_idx:
                return None
            max_possible = sum(smooth_list[max_start:max_start + k_remaining])
            if max_possible < rem:
                return None
            
            for idx in range(start_idx, len(smooth_list)):
                val = smooth_list[idx]
                if val > rem:
                    break
                valid = True
                for existing in current_terms:
                    if val % existing == 0 or existing % val == 0:
                        valid = False
                        break
                if not valid:
                    continue
                res = dfs(rem - val, idx + 1, current_terms + [val], k_remaining - 1)
                if res is not None:
                    return res
            return None
        
        for t in unresolved:
            if time.time() > deadline:
                break
            for k in range(4, 11):
                res = dfs(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(res)
                    break
    
    # Fallback: if any targets still unresolved, use a more thorough DFS
    for t in range(1, 129):
        if reps[t] is None:
            found = False
            for k in range(4, 31):
                if time.time() > deadline:
                    break
                def dfs_fb(rem, start_idx, current_terms, k_remaining):
                    if time.time() > deadline:
                        return None
                    if k_remaining == 0:
                        return current_terms if rem == 0 else None
                    if rem <= 0:
                        return None
                    if start_idx >= len(smooth_list):
                        return None
                    for idx in range(start_idx, len(smooth_list)):
                        val = smooth_list[idx]
                        if val > rem:
                            break
                        valid = True
                        for existing in current_terms:
                            if val % existing == 0 or existing % val == 0:
                                valid = False
                                break
                        if not valid:
                            continue
                        res = dfs_fb(rem - val, idx + 1, current_terms + [val], k_remaining - 1)
                        if res is not None:
                            return res
                    return None
                res = dfs_fb(t, 0, [], k)
                if res is not None:
                    reps[t] = tuple(res)
                    found = True
                    break
    
    return reps

_TABLE = _compute_table()

def construct():
    return [list(_TABLE[t]) for t in range(1, 129)]
