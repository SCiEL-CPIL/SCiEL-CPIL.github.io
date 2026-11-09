import time
import random

N = 256

def construct():
    # Deadline guard - keep safety margin
    deadline = time.time() + 25.0
    
    def is_valid_sidon(s):
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                val = s[i] + s[j]
                if val in sums:
                    return False
                sums.add(val)
        return True
    
    def is_maximal(s):
        if not is_valid_sidon(s):
            return False
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        
        for x in range(1, N + 1):
            if x in s:
                continue
            # Check if adding x breaks Sidon property
            if 2 * x in sums:
                continue
            can_add = True
            for a in s:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True
    
    def try_remove(s, sums):
        """Try to remove elements from s while maintaining maximality."""
        s = list(s)
        s_set = set(s)
        changed = True
        while changed:
            changed = False
            for i in range(len(s)):
                # Check if removing s[i] keeps the set maximal
                new_s = s[:i] + s[i+1:]
                if not new_s:
                    continue
                new_sums = set()
                valid = True
                for a in range(len(new_s)):
                    for b in range(a, len(new_s)):
                        val = new_s[a] + new_s[b]
                        if val in new_sums:
                            valid = False
                            break
                        new_sums.add(val)
                    if not valid:
                        break
                if not valid:
                    continue
                
                # Check maximality
                is_max = True
                for x in range(1, N + 1):
                    if x in new_s:
                        continue
                    if 2 * x in new_sums:
                        continue
                    can_add = True
                    for a in new_s:
                        if (x + a) in new_sums:
                            can_add = False
                            break
                    if can_add:
                        is_max = False
                        break
                
                if is_max:
                    s = new_s
                    changed = True
                    break
        return s
    
    # Known good candidate
    known_set = [1, 3, 7, 15, 31, 63, 127]
    if is_valid_sidon(known_set) and is_maximal(known_set):
        best_set = list(known_set)
        best_size = len(best_set)
    else:
        # Fallback greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            new_sums = {2 * x} | {x + a for a in selected}
            if not new_sums.intersection(sums):
                selected.append(x)
                sums.update(new_sums)
        best_set = selected
        best_size = len(best_set)
    
    # Try local reduction on the known set
    reduced = try_remove(best_set, None)
    if len(reduced) < best_size:
        best_set = reduced
        best_size = len(best_set)
    
    # Randomized search with multiple strategies
    rng = random.Random(42)
    
    while time.time() < deadline:
        # Strategy 1: Random permutation greedy
        perm = list(range(1, N + 1))
        rng.shuffle(perm)
        
        current_set = []
        sums = set()
        
        for x in perm:
            if 2 * x in sums:
                continue
            can_add = True
            for a in current_set:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                current_set.append(x)
                sums.add(2 * x)
                for a in current_set[:-1]:
                    sums.add(x + a)
        
        # Try to reduce
        reduced = try_remove(current_set, None)
        if len(reduced) < best_size:
            best_set = reduced
            best_size = len(best_set)
        
        # Strategy 2: Weighted random - prefer larger elements
        if time.time() < deadline:
            weights = [1.0 / x for x in range(1, N + 1)]
            total_w = sum(weights)
            perm = list(range(1, N + 1))
            # Custom shuffle with bias
            for i in range(len(perm) - 1, 0, -1):
                # Pick random index with bias
                r = rng.random() * total_w
                cum = 0
                chosen = 0
                for j in range(i + 1):
                    cum += weights[perm[j] - 1]
                    if r <= cum:
                        chosen = j
                        break
                perm[i], perm[chosen] = perm[chosen], perm[i]
                total_w -= weights[perm[i] - 1]
            
            current_set = []
            sums = set()
            
            for x in perm:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in current_set:
                    if (x + a) in sums:
                        can_add = False
                        break
                if can_add:
                    current_set.append(x)
                    sums.add(2 * x)
                    for a in current_set[:-1]:
                        sums.add(x + a)
            
            reduced = try_remove(current_set, None)
            if len(reduced) < best_size:
                best_set = reduced
                best_size = len(best_set)
        
        # Strategy 3: Start with large elements, then fill in
        if time.time() < deadline and best_size > 5:
            current_set = []
            sums = set()
            # Start with some large elements
            large = list(range(N, max(N - 20, 1), -1))
            rng.shuffle(large)
            for x in large:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in current_set:
                    if (x + a) in sums:
                        can_add = False
                        break
                if can_add:
                    current_set.append(x)
                    sums.add(2 * x)
                    for a in current_set[:-1]:
                        sums.add(x + a)
            
            # Then try to add smaller elements
            small = list(range(1, N - 20))
            rng.shuffle(small)
            for x in small:
                if 2 * x in sums:
                    continue
                can_add = True
                for a in current_set:
                    if (x + a) in sums:
                        can_add = False
                        break
                if can_add:
                    current_set.append(x)
                    sums.add(2 * x)
                    for a in current_set[:-1]:
                        sums.add(x + a)
            
            reduced = try_remove(current_set, None)
            if len(reduced) < best_size:
                best_set = reduced
                best_size = len(best_set)
    
    return best_set
