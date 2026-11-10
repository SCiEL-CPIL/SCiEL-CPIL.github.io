import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 85.0

    N = 24
    def lcm(a, b):
        return a // gcd(a, b) * b
    
    global_lcm = 1
    for i in range(1, 25):
        global_lcm = lcm(global_lcm, i)
    
    recip_int = [0] + [global_lcm // i for i in range(1, 25)]
    
    def is_valid_bitmask(m):
        elems = [i for i in range(1, 25) if (m >> (i-1)) & 1]
        n = len(elems)
        if n < 2:
            return True
        
        for x in elems:
            target = recip_int[x]
            others = [recip_int[y] for y in elems if y != x]
            no = len(others)
            if no == 0:
                continue
            
            total_others = sum(others)
            if total_others < target:
                continue
            if total_others == target:
                return False
            
            mid = no // 2
            left = others[:mid]
            right = others[mid:]
            
            left_sums = {0}
            for val in left:
                new_sums = set()
                for s in left_sums:
                    new_sums.add(s + val)
                left_sums.update(new_sums)
            
            right_sums = {0}
            for val in right:
                new_sums = set()
                for s in right_sums:
                    new_sums.add(s + val)
                right_sums.update(new_sums)
            
            for s in left_sums:
                if (target - s) in right_sums:
                    return False
        
        return True

    # Start with known good seed
    best_mask = 0
    for i in range(9, 25):
        best_mask |= (1 << (i-1))
    best_size = 16
    
    if not is_valid_bitmask(best_mask):
        best_mask = 0
        for i in range(13, 25):
            best_mask |= (1 << (i-1))
        best_size = 12
    
    elements = list(range(1, 25))
    
    # Systematic passes first
    orderings = [
        list(range(1, 25)),
        list(range(24, 0, -1)),
        sorted(range(1, 25), key=lambda x: 1/x),  # largest reciprocals first
        sorted(range(1, 25), key=lambda x: -1/x),  # smallest reciprocals first
    ]
    
    for ordering in orderings:
        if time.time() > deadline:
            break
        S_mask = 0
        for x in ordering:
            if time.time() > deadline:
                break
            bit = 1 << (x-1)
            new_mask = S_mask | bit
            if is_valid_bitmask(new_mask):
                S_mask = new_mask
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask
    
    # Randomized greedy passes
    while time.time() < deadline - 10:
        random.shuffle(elements)
        S_mask = 0
        
        for x in elements:
            if time.time() > deadline:
                break
            bit = 1 << (x-1)
            new_mask = S_mask | bit
            if is_valid_bitmask(new_mask):
                S_mask = new_mask
        
        sz = bin(S_mask).count('1')
        if sz > best_size:
            best_size = sz
            best_mask = S_mask
            
            # Local search on the new best
            if time.time() < deadline - 5:
                current = S_mask
                improved = True
                while improved and time.time() < deadline - 3:
                    improved = False
                    for y in range(1, 25):
                        if time.time() > deadline - 3:
                            break
                        if (current >> (y-1)) & 1:
                            continue
                        new_mask = current | (1 << (y-1))
                        if is_valid_bitmask(new_mask):
                            current = new_mask
                            improved = True
                            if bin(current).count('1') > best_size:
                                best_size = bin(current).count('1')
                                best_mask = current
                    if not improved:
                        cur_elems = [i for i in range(1, 25) if (current >> (i-1)) & 1]
                        random.shuffle(cur_elems)
                        for x in cur_elems:
                            if time.time() > deadline - 3:
                                break
                            for y in range(1, 25):
                                if time.time() > deadline - 3:
                                    break
                                if (current >> (y-1)) & 1:
                                    continue
                                new_mask = (current & ~(1 << (x-1))) | (1 << (y-1))
                                if is_valid_bitmask(new_mask):
                                    current = new_mask
                                    improved = True
                                    break
                            if improved:
                                break
                    if not improved:
                        remaining = [x for x in range(1, 25) if not (current >> (x-1)) & 1]
                        random.shuffle(remaining)
                        for i in range(len(remaining)):
                            if time.time() > deadline - 3:
                                break
                            for j in range(i+1, len(remaining)):
                                if time.time() > deadline - 3:
                                    break
                                new_mask = current | (1 << (remaining[i]-1)) | (1 << (remaining[j]-1))
                                if is_valid_bitmask(new_mask):
                                    current = new_mask
                                    if bin(current).count('1') > best_size:
                                        best_size = bin(current).count('1')
                                        best_mask = current
                                    improved = True
                                    break
                            if improved:
                                break
                # Final additions
                if time.time() < deadline - 1:
                    for y in range(1, 25):
                        if time.time() > deadline - 1:
                            break
                        if (current >> (y-1)) & 1:
                            continue
                        new_mask = current | (1 << (y-1))
                        if is_valid_bitmask(new_mask):
                            current = new_mask
                            if bin(current).count('1') > best_size:
                                best_size = bin(current).count('1')
                                best_mask = current
    
    best_set = [i for i in range(1, 25) if (best_mask >> (i-1)) & 1]
    
    if not is_valid_bitmask(best_mask):
        best_set = list(range(13, 25))
    
    return best_set
