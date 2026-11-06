<<<<<<< SEARCH
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (evaluator kills at 90s, but let's be safe)
    
    def time_left():
        return deadline - time.time()

    def check_candidate(selected, x, sums_set):
        """Check if adding x to selected keeps all triple sums unique.
        selected is sorted. sums_set contains all triple sums of selected.
        Returns (True, new_sums) if valid, else (False, None)."""
        n = len(selected)
        
        # 1. (x, x, x) = 3x
        s = 3 * x
        if s in sums_set:
            return False, None
        new_sums = {s}
        
        # 2. (x, x, y) for y in selected -> 2x + y
        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # 3. (x, y, z) for y <= z in selected -> x + y + z
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums

    def greedy_forward(start, end, selected, sums_set):
        """Greedy forward from start to end."""
        for x in range(start, end + 1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    def greedy_backward(start, end, selected, sums_set):
        """Greedy backward from start down to end."""
        for x in range(start, end - 1, -1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    def greedy_random(selected, sums_set, candidates):
        """Greedy with random candidate selection."""
        for x in candidates:
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected, x, sums_set)
            if valid:
                selected.append(x)
                sums_set.update(new_sums)
        return selected, sums_set

    # Strategy 1: Greedy forward from 1 to 4096
    selected1 = []
    sums1 = set()
    selected1, sums1 = greedy_forward(1, 4096, selected1, sums1)
    best = selected1[:]
    
    if time_left() < 5.0:
        return best

    # Strategy 2: Greedy backward from 4096 to 1
    selected2 = []
    sums2 = set()
    selected2, sums2 = greedy_backward(4096, 1, selected2, sums2)
    if len(selected2) > len(best):
        best = selected2[:]
    
    if time_left() < 5.0:
        return best

    # Strategy 3: Base-7 Construction (Inspiration 2)
    # This is a deterministic construction that is guaranteed to work and is very fast.
    # It uses base 7 expansion with digits {0, 1, 2}.
    # Values: sum(d_i * 7^i) for i in 0..3, d_i in {0,1,2}.
    # Total 3^4 = 81 values. Remove 0 to get 80 positive integers.
    # Max value: 2 * (1 + 7 + 49 + 343) = 2 * 400 = 800 <= 4096.
    # This should yield exactly 80 elements.
    
    B = 7
    digits = [0, 1, 2]
    L = 4
    values = []
    for i1 in digits:
        for i2 in digits:
            for i3 in digits:
                for i4 in digits:
                    val = i1 + i2 * B + i3 * (B * B) + i4 * (B * B * B)
                    if val > 0:
                        values.append(val)
    
    values.sort()
    
    # Verify quickly
    if time_left() > 10.0:
        valid = True
        sums_seen = set()
        n = len(values)
        for i in range(n):
            for j in range(i, n):
                for k in range(j, n):
                    s = values[i] + values[j] + values[k]
                    if s in sums_seen:
                        valid = False
                        break
                    sums_seen.add(s)
                if not valid:
                    break
            if not valid:
                break
        
        if valid and len(values) > len(best):
            best = values[:]

    if time_left() < 5.0:
        return best

    # Strategy 4: Greedy with randomization - try multiple random orders
    rng = random.Random(42)
    for trial in range(20):
        if time_left() < 5.0:
            break
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        selected3 = []
        sums3 = set()
        selected3, sums3 = greedy_random(selected3, sums3, candidates)
        if len(selected3) > len(best):
            best = selected3[:]
    
    if time_left() < 5.0:
        return best

    # Strategy 5: Local search / Insertion into gaps
    # Start from the best set found so far.
    # Try to insert additional elements by checking all possible values 1..4096 not in the set.
    # This is O(N * 4096 * K) where K is cost of check_candidate.
    # Since best is likely around 20-30 elements (greedy) or 80 (base-7), 
    # if best is 80, we can't add more. 
    # If best is smaller, we can try to fill gaps.
    
    current = best[:]
    current_set = set(current)
    
    # Rebuild sums_set for current
    current_sums = set()
    for triple in combinations_with_replacement(current, 3):
        current_sums.add(sum(triple))
    
    # Try to add any valid x from 1 to 4096
    improved = True
    while improved and time_left() > 5.0:
        improved = False
        for x in range(1, 4097):
            if time_left() < 5.0:
                break
            if x in current_set:
                continue
            valid, new_sums = check_candidate(sorted(current), x, current_sums)
            if valid:
                current.append(x)
                current_set.add(x)
                current_sums.update(new_sums)
                improved = True
                if len(current) > len(best):
                    best = current[:]
    
    return best
# EVOLVE-BLOCK-END
=======
"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit (evaluator kills at 90s, but let's be safe)
    
    def time_left():
        return deadline - time.time()

    def check_candidate(selected, x, sums_set):
        """Check if adding x to selected keeps all triple sums unique.
        selected is sorted. sums_set contains all triple sums of selected.
        Returns (True, new_sums) if valid, else (False, None)."""
        n = len(selected)
        
        # 1. (x, x, x) = 3x
        s = 3 * x
        if s in sums_set:
            return False, None
        new_sums = {s}
        
        # 2. (x, x, y) for y in selected -> 2x + y
        for y in selected:
            s = 2 * x + y
            if s in sums_set or s in new_sums:
                return False, None
            new_sums.add(s)
        
        # 3. (x, y, z) for y <= z in selected -> x + y + z
        for i in range(n):
            y = selected[i]
            for j in range(i, n):
                z = selected[j]
                s = x + y + z
                if s in sums_set or s in new_sums:
                    return False, None
                new_sums.add(s)
        
        return True, new_sums

    def build_base7_set():
        """Build the base-7 B_3[1] set with 80 elements."""
        B = 7
        digits = [0, 1, 2]
        values = []
        for d0 in digits:
            for d1 in digits:
                for d2 in digits:
                    for d3 in digits:
                        val = d0 + d1 * B + d2 * (B * B) + d3 * (B * B * B)
                        if val > 0:
                            values.append(val)
        values.sort()
        return values

    # Strategy 1: Base-7 Construction - guaranteed 80 elements
    base7_set = build_base7_set()
    best = base7_set[:]
    best_size = len(best)
    
    # Strategy 2: Greedy forward from 1 to 4096 (might find larger)
    if time_left() > 5.0:
        selected1 = []
        sums1 = set()
        for x in range(1, 4097):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected1, x, sums1)
            if valid:
                selected1.append(x)
                sums1.update(new_sums)
        if len(selected1) > best_size:
            best = selected1[:]
            best_size = len(best)
    
    # Strategy 3: Greedy backward from 4096 to 1
    if time_left() > 5.0:
        selected2 = []
        sums2 = set()
        for x in range(4096, 0, -1):
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected2, x, sums2)
            if valid:
                selected2.append(x)
                sums2.update(new_sums)
        if len(selected2) > best_size:
            best = selected2[:]
            best_size = len(best)
    
    # Strategy 4: Randomized greedy with multiple trials
    rng = random.Random(42)
    for trial in range(30):
        if time_left() < 5.0:
            break
        candidates = list(range(1, 4097))
        rng.shuffle(candidates)
        selected3 = []
        sums3 = set()
        for x in candidates:
            if time_left() < 2.0:
                break
            valid, new_sums = check_candidate(selected3, x, sums3)
            if valid:
                selected3.append(x)
                sums3.update(new_sums)
        if len(selected3) > best_size:
            best = selected3[:]
            best_size = len(best)
    
    # Strategy 5: Local search - try to add more elements to best set
    if time_left() > 5.0 and best_size < 80:
        current = best[:]
        current_set = set(current)
        current_sums = set()
        for triple in combinations_with_replacement(current, 3):
            current_sums.add(sum(triple))
        
        improved = True
        while improved and time_left() > 5.0:
            improved = False
            for x in range(1, 4097):
                if time_left() < 5.0:
                    break
                if x in current_set:
                    continue
                valid, new_sums = check_candidate(sorted(current), x, current_sums)
                if valid:
                    current.append(x)
                    current_set.add(x)
                    current_sums.update(new_sums)
                    improved = True
                    if len(current) > best_size:
                        best = current[:]
                        best_size = len(best)
    
    # Strategy 6: Skeleton + gap filling approach
    # Use a sparse geometric progression as skeleton, then fill gaps
    if time_left() > 10.0:
        # Try different bases for the skeleton
        for base in [4, 5, 6]:
            if time_left() < 10.0:
                break
            
            # Build skeleton using powers of base
            skeleton = []
            max_val = 4096
            power = 1
            while power <= max_val:
                skeleton.append(power)
                power *= base
            skeleton.sort()
            
            # Build sums for skeleton
            skel_sums = set()
            for i in range(len(skeleton)):
                for j in range(i, len(skeleton)):
                    for k in range(j, len(skeleton)):
                        skel_sums.add(skeleton[i] + skeleton[j] + skeleton[k])
            
            # Try to add elements in gaps
            skel_set = set(skeleton)
            current = skeleton[:]
            current_sums = skel_sums.copy()
            
            # Try adding smaller numbers first
            for x in range(1, 4097):
                if time_left() < 5.0:
                    break
                if x in skel_set:
                    continue
                valid, new_sums = check_candidate(sorted(current), x, current_sums)
                if valid:
                    current.append(x)
                    skel_set.add(x)
                    current_sums.update(new_sums)
            
            if len(current) > best_size:
                best = current[:]
                best_size = len(best)
    
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
