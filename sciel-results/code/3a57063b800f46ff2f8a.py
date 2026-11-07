import time
import random

N = 128
MAX_TIME = 85.0  # safety margin for 90s wall-clock limit

class SetSolver:
    def __init__(self):
        self.S_mask = 0
        self.SumMasks = [0] * (N + 1) # SumMasks[x] has bit y set if x+y in S
        self.members = set()
        
    def add(self, x):
        if x in self.members:
            return
        self.members.add(x)
        self.S_mask |= (1 << (x - 1))
        for y in range(1, x):
            z = x - y
            self.SumMasks[y] |= (1 << (z - 1))
            
    def remove(self, x):
        if x not in self.members:
            return
        self.members.remove(x)
        self.S_mask &= ~(1 << (x - 1))
        for y in range(1, x):
            z = x - y
            self.SumMasks[y] &= ~(1 << (z - 1))

    def is_valid(self):
        members = sorted(self.members)
        n = len(members)
        
        for i in range(n):
            a = members[i]
            if i + 1 >= n: break
            if a + members[i+1] > N: break
            
            mask_a = self.SumMasks[a]
            
            for j in range(i + 1, n):
                b = members[j]
                if not (mask_a & (1 << (b - 1))):
                    continue
                
                mask_b = self.SumMasks[b]
                candidates = mask_a & mask_b
                candidates &= ~((1 << b) - 1)
                
                if candidates:
                    return False
        
        return True

    def count_conflicts(self):
        counts = {x: 0 for x in self.members}
        members = sorted(self.members)
        n = len(members)
        
        for i in range(n):
            a = members[i]
            if i + 1 >= n: break
            if a + members[i+1] > N: break
            
            mask_a = self.SumMasks[a]
            
            for j in range(i + 1, n):
                b = members[j]
                if not (mask_a & (1 << (b - 1))):
                    continue
                
                mask_b = self.SumMasks[b]
                candidates = mask_a & mask_b
                candidates &= ~((1 << b) - 1)
                
                if candidates:
                    c_mask = candidates
                    while c_mask:
                        lsb = c_mask & -c_mask
                        c_val = lsb.bit_length()
                        counts[a] += 1
                        counts[b] += 1
                        counts[c_val] += 1
                        c_mask &= c_mask - 1
                        
        return counts

def _fast_validity_check(s):
    solver = SetSolver()
    for x in s:
        solver.add(x)
    return solver.is_valid()

def _count_conflicts(s_set):
    solver = SetSolver()
    for x in s_set:
        solver.add(x)
    return solver.count_conflicts()

def _greedy_remove_until_valid(s_set, start_time, deadline_margin=5.0):
    s_set = set(s_set)
    while True:
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        counts = _count_conflicts(s_set)
        if not counts or all(v == 0 for v in counts.values()):
            break
        
        max_count = max(counts.values())
        candidates = [x for x, v in counts.items() if v == max_count]
        remove_el = min(candidates)
        s_set.remove(remove_el)
        
    return s_set

def _try_greedy_add(s_set, start_time, deadline_margin=5.0):
    s_set = set(s_set)
    candidates = [x for x in range(1, N+1) if x not in s_set]
    candidates.sort()
    
    improved = True
    while improved and time.time() - start_time < MAX_TIME - deadline_margin:
        improved = False
        for cand in candidates[:]:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            new_s = s_set | {cand}
            if _fast_validity_check(new_s):
                s_set = new_s
                candidates = [x for x in candidates if x != cand]
                improved = True
    
    return sorted(s_set)

def _try_remove_add(s_set, start_time, deadline_margin=5.0):
    s_set = set(s_set)
    best_s = set(s_set)
    best_len = len(s_set)
    
    for remove_el in sorted(s_set):
        if time.time() - start_time > MAX_TIME - deadline_margin:
            break
        remaining_set = s_set - {remove_el}
        candidates = [x for x in range(1, N+1) if x not in remaining_set]
        candidates.sort()
        
        current = set(remaining_set)
        for cand in candidates:
            if time.time() - start_time > MAX_TIME - deadline_margin:
                break
            new_s = current | {cand}
            if _fast_validity_check(new_s):
                current = new_s
        
        if len(current) > best_len:
            best_s = current
            best_len = len(current)
    
    return sorted(best_s)

def construct():
    start_time = time.time()
    
    full_set = set(range(1, N+1))
    s_set = _greedy_remove_until_valid(full_set, start_time)
    
    if time.time() - start_time < MAX_TIME - 5.0:
        s_set = set(_try_greedy_add(s_set, start_time))
        
    if time.time() - start_time < MAX_TIME - 5.0:
        s_set = set(_try_remove_add(s_set, start_time))
        
    best_s = sorted(s_set)
    best_len = len(best_s)
    
    if time.time() - start_time < MAX_TIME - 10.0:
        for _ in range(20):
            if time.time() - start_time > MAX_TIME - 5.0:
                break
            perturbed_set = set(full_set)
            for _ in range(random.randint(1, 3)):
                if perturbed_set:
                    random_remove = random.choice(list(perturbed_set))
                    perturbed_set.remove(random_remove)
            
            s_set = _greedy_remove_until_valid(perturbed_set, start_time)
            
            if time.time() - start_time < MAX_TIME - 5.0:
                s_set = set(_try_greedy_add(s_set, start_time))
                
            if time.time() - start_time < MAX_TIME - 2.0:
                s_set = set(_try_remove_add(s_set, start_time))
                
            if len(s_set) > best_len:
                best_s = sorted(s_set)
                best_len = len(s_set)
                
    if not _fast_validity_check(set(best_s)):
        best_s = list(range(63, 129))
        
    return [int(x) for x in best_s]
