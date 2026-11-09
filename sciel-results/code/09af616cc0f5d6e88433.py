import time
import random

N = 256

def construct():
    # Deadline guard
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
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sums.add(s[i] + s[j])
        
        for x in range(1, N + 1):
            if x in s:
                continue
            # Check if adding x breaks Sidon property
            # New sums: 2x and x+a for a in s
            # Check if any of these are in sums
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

    # Known good candidate from knowledge: {1, 3, 7, 15, 31, 63, 127}
    known_set = [1, 3, 7, 15, 31, 63, 127]
    if is_valid_sidon(known_set) and is_maximal(known_set):
        best_set = known_set
        best_size = len(best_set)
    else:
        # Fallback to greedy
        selected = []
        sums = set()
        for x in range(1, N + 1):
            new_sums = {2 * x} | {x + a for a in selected}
            if not new_sums.intersection(sums):
                selected.append(x)
                sums.update(new_sums)
        best_set = selected
        best_size = len(best_set)

    # Randomized search to try to beat the known size
    while time.time() < deadline:
        # Random permutation
        perm = list(range(1, N + 1))
        random.shuffle(perm)
        
        current_set = []
        sums = set()
        
        for x in perm:
            # Check if x can be added
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
        
        if len(current_set) < best_size:
            best_size = len(current_set)
            best_set = current_set
            # If we find a very small set, we might want to continue
            if best_size <= 5:
                # It is very unlikely to find < 5, but if we do, we can break or continue
                # Let's continue to see if we can get even smaller, but 5 is a strong lower bound candidate
                pass

    return best_set
