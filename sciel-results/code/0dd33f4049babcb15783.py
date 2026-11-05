import time
import random
import bisect

def construct():
    start_time = time.time()
    deadline = start_time + 110  # 10s margin
    
    prescribed = [1, 2, 5, 11]
    target_size = 20
    max_val = 284
    
    best_set = prescribed[:]
    best_size = len(best_set)
    
    def compute_diffs(marks):
        diffs = set()
        n = len(marks)
        for i in range(n):
            for j in range(i + 1, n):
                diffs.add(marks[j] - marks[i])
        return diffs

    def can_add(x, marks, diffs):
        for m in marks:
            d = abs(x - m)
            if d in diffs:
                return False
        return True

    def attempt_search():
        nonlocal best_set, best_size
        
        current_set = prescribed[:]
        current_diffs = compute_diffs(current_set)
        
        # Phase 1: Greedy Insertion
        all_candidates = [x for x in range(1, max_val + 1) if x not in prescribed]
        
        while len(current_set) < target_size:
            added = False
            random.shuffle(all_candidates)
            
            for x in all_candidates:
                if x in current_set:
                    continue
                if can_add(x, current_set, current_diffs):
                    pos = bisect.bisect_left(current_set, x)
                    current_set.insert(pos, x)
                    for m in current_set:
                        if m != x:
                            current_diffs.add(abs(x - m))
                    added = True
                    if len(current_set) > best_size:
                        best_set = current_set[:]
                        best_size = len(best_set)
                    break
            
            if not added:
                break
        
        # Phase 2: Local Search
        no_progress_count = 0
        max_no_progress = 20000
        
        while time.time() < deadline:
            if len(current_set) == target_size:
                break
            
            non_prescribed_in_set = [x for x in current_set if x not in prescribed]
            
            if not non_prescribed_in_set:
                break

            x_to_remove = random.choice(non_prescribed_in_set)
            
            current_set.remove(x_to_remove)
            current_diffs = compute_diffs(current_set)
            
            added = False
            # Try more candidates per swap to increase chance of finding a valid one
            for _ in range(200):
                x = random.randint(1, max_val)
                if x in current_set:
                    continue
                if can_add(x, current_set, current_diffs):
                    pos = bisect.bisect_left(current_set, x)
                    current_set.insert(pos, x)
                    for m in current_set:
                        if m != x:
                            current_diffs.add(abs(x - m))
                    added = True
                    if len(current_set) > best_size:
                        best_set = current_set[:]
                        best_size = len(best_set)
                    break
                    
            if not added:
                pos = bisect.bisect_left(current_set, x_to_remove)
                current_set.insert(pos, x_to_remove)
                for m in current_set:
                    if m != x_to_remove:
                        current_diffs.add(abs(x_to_remove - m))
                
                no_progress_count += 1
                if no_progress_count > max_no_progress:
                    break
            else:
                no_progress_count = 0

        return best_set

    attempt_search()
    
    return best_set
