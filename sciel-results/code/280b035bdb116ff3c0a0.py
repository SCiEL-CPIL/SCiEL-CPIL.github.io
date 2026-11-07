import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    candidates = list(range(1, 257))
    
    def get_subset_sums(elements):
        sums = {0}
        for e in elements:
            new_sums = set()
            for s in sums:
                new_sums.add(s + e)
            sums.update(new_sums)
        sums.discard(0)
        return sums

    def count_violations(sums, max_sum):
        if not sums:
            return 0
        arr = np.zeros(max_sum + 1, dtype=bool)
        arr[list(sums)] = True
        violations = 0
        sum_list = sorted(sums)
        for s in sum_list:
            if s * 2 > max_sum:
                continue
            multiples = arr[2*s : max_sum+1 : s]
            if multiples.any():
                violations += 1
        return violations

    def is_valid(elements, max_sum):
        if not elements:
            return True
        sums = get_subset_sums(elements)
        return count_violations(sums, max_sum) == 0

    # Greedy initialization
    def greedy_init():
        current = []
        for c in sorted(candidates, reverse=True):
            new_set = current + [c]
            if is_valid(new_set, 256 * len(new_set)):
                current = new_set
            if len(current) >= 10:
                break
        return current

    best_set = []
    best_size = 0
    
    for restart in range(1000):
        if time.time() > deadline:
            break
        
        if restart % 2 == 0:
            init_set = greedy_init()
        else:
            large_candidates = [c for c in candidates if c > 100]
            random.shuffle(large_candidates)
            init_set = []
            for c in large_candidates:
                new_set = init_set + [c]
                if is_valid(new_set, 256 * len(new_set)):
                    init_set = new_set
                if len(init_set) >= 5:
                    break
        
        if not init_set:
            continue
            
        current_set = init_set[:]
        current_sums = get_subset_sums(current_set)
        current_max_sum = sum(current_set)
        current_violations = count_violations(current_sums, current_max_sum)
        
        best_local = current_set[:]
        best_local_size = len(current_set) if current_violations == 0 else 0
        
        T = 1.0
        cooling = 0.99
        
        for step in range(2000):
            if time.time() > deadline:
                break
                
            move_type = random.choice(['add', 'swap', 'remove'])
            new_set = current_set[:]
            
            if move_type == 'add':
                avail = [c for c in candidates if c not in current_set]
                if not avail: continue
                c = random.choice(avail)
                new_set.append(c)
            elif move_type == 'remove':
                if not current_set: continue
                c = random.choice(current_set)
                new_set.remove(c)
            elif move_type == 'swap':
                if not current_set: continue
                c_in = random.choice(current_set)
                avail = [c for c in candidates if c not in current_set]
                if not avail: continue
                c_out = random.choice(avail)
                new_set.remove(c_in)
                new_set.append(c_out)
            
            new_sums = get_subset_sums(new_set)
            new_max_sum = sum(new_set)
            new_violations = count_violations(new_sums, new_max_sum)
            
            if current_violations == 0:
                if new_violations == 0:
                    if len(new_set) >= len(current_set):
                        current_set = new_set
                        current_sums = new_sums
                        current_max_sum = new_max_sum
                        if len(current_set) > best_local_size:
                            best_local_size = len(current_set)
                            best_local = current_set[:]
                else:
                    if random.random() < 0.01:
                        current_set = new_set
                        current_sums = new_sums
                        current_max_sum = new_max_sum
                        current_violations = new_violations
            else:
                if new_violations < current_violations:
                    current_set = new_set
                    current_sums = new_sums
                    current_max_sum = new_max_sum
                    current_violations = new_violations
                    if current_violations == 0 and len(current_set) > best_local_size:
                        best_local_size = len(current_set)
                        best_local = current_set[:]
                elif new_violations == current_violations:
                    if random.random() < 0.5:
                        current_set = new_set
                        current_sums = new_sums
                        current_max_sum = new_max_sum
                else:
                    if random.random() < 0.05:
                        current_set = new_set
                        current_sums = new_sums
                        current_max_sum = new_max_sum
                        current_violations = new_violations
            
            T *= cooling
            
        if best_local_size > best_size:
            best_size = best_local_size
            best_set = best_local
            if best_size >= 12:
                break

    return best_set
