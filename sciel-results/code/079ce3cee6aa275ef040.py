import random
import time
from collections import defaultdict

def construct():
    start_time = time.time()
    DEADLINE = start_time + 80  # 10s safety margin
    
    def elapsed():
        return time.time() - start_time
    
    def within_deadline():
        return elapsed() < DEADLINE
    
    # Precompute all possible consecutive block sums for a given sequence
    # Returns a set of all sums
    def compute_all_sums(seq):
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                sums.add(s)
        return sums
    
    def check_validity(seq):
        if not seq:
            return True
        if len(seq) != len(set(seq)):
            return False
        if seq != sorted(seq):
            return False
        if seq[0] < 1 or seq[-1] > 256:
            return False
        sums = set()
        n = len(seq)
        for i in range(n):
            s = 0
            for j in range(i, n):
                s += seq[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    # Fast check: can we add x to the end of seq?
    def can_append(seq, x, sums_set):
        # Compute new tails (sums of suffixes ending at x)
        tails = [x]
        total = x
        for a in reversed(seq):
            total += a
            tails.append(total)
        for t in tails:
            if t in sums_set:
                return False
        return True
    
    # Get the new sums when appending x
    def get_new_sums(seq, x):
        new_sums = set()
        total = x
        new_sums.add(total)
        for a in reversed(seq):
            total += a
            new_sums.add(total)
        return new_sums
    
    # Greedy: always pick smallest valid next element
    def greedy_smallest(start_val=1):
        values = []
        sums = set()
        last = start_val - 1
        for x in range(start_val, 257):
            if can_append(values, x, sums):
                values.append(x)
                sums.update(get_new_sums(values[:-1], x))
                last = x
        return values
    
    # Greedy: always pick largest valid next element (build from right)
    def greedy_largest(start_val=256):
        values = []
        sums = set()
        # We build in reverse: pick largest, then next largest, etc.
        # But we need increasing order, so let's think differently.
        # Actually, let's just try: pick 256, then find largest x < 256 such that [x, 256] is valid, etc.
        # This is complex. Let's instead do a greedy that tries to pack densely from the top.
        # Alternative: start with 256, then for each subsequent element, try values from 255 down to 1.
        values = [256]
        sums = {256}
        while True:
            found = False
            for x in range(255, 0, -1):
                # Check if [x] + values is valid (i.e., x can be prepended)
                # When prepending x, new sums are: x, x+v[0], x+v[0]+v[1], ...
                new_sums = set()
                total = x
                new_sums.add(total)
                for a in values:
                    total += a
                    new_sums.add(total)
                if new_sums.isdisjoint(sums):
                    values.insert(0, x)
                    sums.update(new_sums)
                    found = True
                    break
            if not found:
                break
        return values
    
    # Multi-start greedy with different strategies
    def multi_start_greedy():
        results = []
        
        # Standard smallest-first
        results.append(greedy_smallest(1))
        
        # Start from different initial values
        for start in [2, 3, 5, 10, 20, 50]:
            results.append(greedy_smallest(start))
        
        # Largest-first
        results.append(greedy_largest())
        
        # Randomized greedy: pick random valid next element
        for _ in range(50):
            values = []
            sums = set()
            last = 0
            while True:
                found = False
                # Try a few random candidates
                candidates = list(range(last + 1, 257))
                random.shuffle(candidates)
                for x in candidates[:20]:  # Try up to 20 random candidates
                    if can_append(values, x, sums):
                        values.append(x)
                        sums.update(get_new_sums(values[:-1], x))
                        last = x
                        found = True
                        break
                if not found:
                    break
            results.append(values)
        
        # Lookahead greedy: when choosing next element, consider how many future options it leaves
        for _ in range(30):
            values = []
            sums = set()
            last = 0
            while True:
                best_x = None
                best_score = -1
                # Try all valid candidates and score them by how many further elements they allow
                for x in range(last + 1, 257):
                    if not can_append(values, x, sums):
                        continue
                    # Score: how many more elements can we add after x?
                    # Approximate by checking a few random continuations
                    temp_values = values + [x]
                    temp_sums = sums | get_new_sums(values, x)
                    # Quick check: how many random valid extensions?
                    score = 0
                    for _ in range(5):
                        # Try to extend
                        ext_count = 0
                        for y in random.sample(range(x + 1, 257), min(10, 256 - x)):
                            if can_append(temp_values, y, temp_sums):
                                ext_count += 1
                        score += ext_count
                    if score > best_score:
                        best_score = score
                        best_x = x
                if best_x is None:
                    break
                values.append(best_x)
                sums.update(get_new_sums(values[:-1], best_x))
                last = best_x
            results.append(values)
        
        return results
    
    # Local search: try to insert an element into a gap in an existing sequence
    def local_search_insert(seq, max_iters=100):
        if not seq:
            return seq
        
        best = list(seq)
        best_sums = compute_all_sums(best)
        best_len = len(best)
        
        for _ in range(max_iters):
            if not within_deadline():
                break
            
            # Try to insert a new element at a random position
            # Positions: 0 to len(best)
            pos = random.randint(0, len(best))
            
            # Determine valid range for the new element
            if pos == 0:
                min_val = 1
                max_val = best[0] - 1 if best else 256
            elif pos == len(best):
                min_val = best[-1] + 1
                max_val = 256
            else:
                min_val = best[pos - 1] + 1
                max_val = best[pos] - 1
            
            if min_val > max_val:
                continue
            
            # Try to find a value that can be inserted
            for x in range(min_val, max_val + 1):
                # Check if inserting x at pos keeps all sums distinct
                new_seq = best[:pos] + [x] + best[pos:]
                new_sums = compute_all_sums(new_seq)
                if len(new_sums) == len(new_seq) * (len(new_seq) + 1) // 2:
                    # All sums are distinct
                    best = new_seq
                    best_sums = new_sums
                    best_len = len(best)
                    break
        
        return best
    
    # Try to remove an element and replace with two elements to increase length
    def local_search_replace(seq, max_iters=100):
        if len(seq) < 3:
            return seq
        
        best = list(seq)
        best_len = len(best)
        
        for _ in range(max_iters):
            if not within_deadline():
                break
            
            # Pick a random element to remove
            if len(best) < 2:
                break
            idx = random.randint(0, len(best) - 1)
            removed = best[idx]
            temp = best[:idx] + best[idx + 1:]
            
            # Now try to insert two elements in the gap around where removed was
            # The gap is between temp[idx-1] and temp[idx] (if they exist)
            # Actually, we removed one element, so we have a gap of size 1
            # We want to fill it with 2 elements to net +1 length
            
            # Determine the range
            if idx == 0:
                lo = 1
                hi = temp[0] - 1 if temp else 256
            elif idx >= len(temp):
                lo = temp[-1] + 1 if temp else 1
                hi = 256
            else:
                lo = temp[idx - 1] + 1
                hi = temp[idx] - 1
            
            if hi - lo + 1 < 2:
                continue
            
            # Try pairs (a, b) with lo <= a < b <= hi
            found = False
            for a in range(lo, hi):
                for b in range(a + 1, hi + 1):
                    new_seq = temp[:idx] + [a, b] + temp[idx:]
                    if check_validity(new_seq):
                        best = new_seq
                        best_len = len(best)
                        found = True
                        break
                if found:
                    break
            
            if not found:
                # Try single replacement
                for x in range(lo, hi + 1):
                    new_seq = temp[:idx] + [x] + temp[idx:]
                    if check_validity(new_seq) and len(new_seq) > best_len:
                        best = new_seq
                        best_len = len(best)
                        break
        
        return best
    
    # Main algorithm
    best_seq = []
    best_len = 0
    
    def update_best(seq):
        nonlocal best_seq, best_len
        if len(seq) > best_len:
            best_seq = list(seq)
            best_len = len(seq)
    
    # Phase 1: Multi-start greedy
    candidates = multi_start_greedy()
    for seq in candidates:
        if check_validity(seq):
            update_best(seq)
    
    # Phase 2: Local search on best sequence
    if within_deadline():
        improved = local_search_insert(best_seq, max_iters=200)
        if check_validity(improved):
            update_best(improved)
    
    # Phase 3: More randomized search
    while within_deadline():
        # Randomized greedy with different biases
        for bias in ['uniform', 'larger', 'smaller']:
            if not within_deadline():
                break
            
            values = []
            sums = set()
            last = 0
            while True:
                found = False
                for _ in range(50):
                    min_val = last + 1
                    max_val = 256
                    if min_val > max_val:
                        break
                    
                    if bias == 'uniform':
                        x = random.randint(min_val, max_val)
                    elif bias == 'larger':
                        x = random.randint((min_val + max_val) // 2, max_val)
                    else:  # 'smaller'
                        x = random.randint(min_val, (min_val + max_val) // 2)
                    
                    if can_append(values, x, sums):
                        values.append(x)
                        sums.update(get_new_sums(values[:-1], x))
                        last = x
                        found = True
                        break
                if not found:
                    break
            
            if check_validity(values):
                update_best(values)
        
        # Local search on current best
        if within_deadline():
            improved = local_search_insert(best_seq, max_iters=100)
            if check_validity(improved):
                update_best(improved)
        
        # Try replacement search
        if within_deadline():
            improved = local_search_replace(best_seq, max_iters=50)
            if check_validity(improved):
                update_best(improved)
    
    return best_seq
