import time
import sys
from itertools import combinations

def construct():
    start_time = time.time()
    timeout = 85.0  # Safety margin before 90s kill
    
    # Generate 5-smooth numbers up to 128
    smooth_numbers = []
    for i in range(1, 129):
        x = i
        while x % 2 == 0: x //= 2
        while x % 3 == 0: x //= 3
        while x % 5 == 0: x //= 5
        if x == 1:
            smooth_numbers.append(i)
    
    smooth_set = set(smooth_numbers)
    n = len(smooth_numbers)
    
    # Precompute incompatibility (divisibility)
    incompatible = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i+1, n):
            a, b = smooth_numbers[i], smooth_numbers[j]
            if a % b == 0 or b % a == 0:
                incompatible[i].add(j)
                incompatible[j].add(i)
    
    # For each target, find the MINIMUM size antichain representation
    # Use DP-style: for each target, try sizes 1, 2, 3, ... until found
    min_size = [0] * 129  # min_size[t] = minimum number of terms for target t
    min_rep = [None] * 129  # min_rep[t] = the actual representation (list of values)
    
    # Size 1: target itself must be smooth
    for i in range(n):
        v = smooth_numbers[i]
        if v <= 128 and min_size[v] == 0:
            min_size[v] = 1
            min_rep[v] = [v]
    
    # Size 2: two non-dividing smooth numbers summing to target
    for i in range(n):
        vi = smooth_numbers[i]
        for j in range(i+1, n):
            if j in incompatible[i]:
                continue
            s = vi + smooth_numbers[j]
            if s <= 128 and min_size[s] == 0:
                min_size[s] = 2
                min_rep[s] = [vi, smooth_numbers[j]]
    
    # Size 3
    for i in range(n):
        vi = smooth_numbers[i]
        for j in range(i+1, n):
            if j in incompatible[i]:
                continue
            vj = smooth_numbers[j]
            for k in range(j+1, n):
                if k in incompatible[i] or k in incompatible[j]:
                    continue
                s = vi + vj + smooth_numbers[k]
                if s <= 128 and min_size[s] == 0:
                    min_size[s] = 3
                    min_rep[s] = [vi, vj, smooth_numbers[k]]
    
    # Size 4 - only for targets not yet covered
    uncovered4 = [t for t in range(1, 129) if min_size[t] == 0]
    if uncovered4:
        uncovered_set = set(uncovered4)
        for i in range(n):
            vi = smooth_numbers[i]
            for j in range(i+1, n):
                if j in incompatible[i]:
                    continue
                vj = smooth_numbers[j]
                for k in range(j+1, n):
                    if k in incompatible[i] or k in incompatible[j]:
                        continue
                    vk = smooth_numbers[k]
                    for m in range(k+1, n):
                        if m in incompatible[i] or m in incompatible[j] or m in incompatible[k]:
                            continue
                        s = vi + vj + vk + smooth_numbers[m]
                        if s <= 128 and s in uncovered_set:
                            min_size[s] = 4
                            min_rep[s] = [vi, vj, vk, smooth_numbers[m]]
                            uncovered_set.discard(s)
    
    # Size 5 - for remaining targets
    uncovered5 = [t for t in range(1, 129) if min_size[t] == 0]
    if uncovered5:
        uncovered_set = set(uncovered5)
        for i in range(n):
            vi = smooth_numbers[i]
            for j in range(i+1, n):
                if j in incompatible[i]:
                    continue
                vj = smooth_numbers[j]
                for k in range(j+1, n):
                    if k in incompatible[i] or k in incompatible[j]:
                        continue
                    vk = smooth_numbers[k]
                    for m in range(k+1, n):
                        if m in incompatible[i] or m in incompatible[j] or m in incompatible[k]:
                            continue
                        vm = smooth_numbers[m]
                        for p in range(m+1, n):
                            if p in incompatible[i] or p in incompatible[j] or p in incompatible[k] or p in incompatible[m]:
                                continue
                            s = vi + vj + vk + vm + smooth_numbers[p]
                            if s <= 128 and s in uncovered_set:
                                min_size[s] = 5
                                min_rep[s] = [vi, vj, vk, vm, smooth_numbers[p]]
                                uncovered_set.discard(s)
    
    # Build answers
    answers = []
    total_terms = 0
    for target in range(1, 129):
        if min_rep[target] is not None:
            rep = min_rep[target]
        else:
            # Fallback: brute force
            rep = _brute_force(target, smooth_numbers, incompatible, n)
            if rep is None:
                # Absolute fallback
                if target in smooth_set:
                    rep = [target]
                else:
                    rep = [target]  # Should not happen
        answers.append(rep)
        total_terms += len(rep)
    
    return answers

def _brute_force(target, smooth_numbers, incompatible, n):
    """Brute force search for a valid antichain summing to target."""
    for size in range(1, 8):
        for combo in combinations(range(n), size):
            s = sum(smooth_numbers[i] for i in combo)
            if s != target:
                continue
            # Check antichain
            valid = True
            for a_idx in range(size):
                for b_idx in range(a_idx+1, size):
                    i, j = combo[a_idx], combo[b_idx]
                    if j in incompatible[i]:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                return [smooth_numbers[i] for i in combo]
    return None
