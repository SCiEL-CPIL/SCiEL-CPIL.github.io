import time
import random

def construct():
    N = 1000
    max_val = 1000
    deadline = time.time() + 110  # Safety margin for 120s limit
    
    best_S = []
    best_size = 0
    
    def run_one_trial():
        S = []
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        
        def try_add(x):
            if counts[2 * x] >= 2:
                return False
            for y in S:
                s = x + y
                if counts[s] >= 2:
                    return False
            S.append(x)
            in_S[x] = True
            counts[2 * x] += 1
            for y in S:
                if y != x:
                    counts[x + y] += 1
            return True

        def try_remove(x):
            S.remove(x)
            in_S[x] = False
            counts[2 * x] -= 1
            for y in S:
                if y != x:
                    counts[x + y] -= 1

        # 1. Randomized Greedy Initialization
        candidates = list(range(1, max_val + 1))
        random.shuffle(candidates)
        
        for c in candidates:
            if try_add(c):
                pass
        
        current_size = len(S)
        
        # 2. Local Search: Try to add random elements, use swaps when blocked
        while time.time() < deadline - 5:
            # Try direct additions first
            added_any = False
            for _ in range(10):
                c = random.randint(1, max_val)
                if not in_S[c]:
                    if try_add(c):
                        current_size += 1
                        added_any = True
                        break
            
            if added_any:
                continue
            
            # If no direct addition, try swaps
            c = random.randint(1, max_val)
            if in_S[c]:
                continue
            
            # Find all valid swaps: y in S such that removing y allows adding c
            valid_swaps = []
            for y in S:
                try_remove(y)
                if try_add(c):
                    valid_swaps.append(y)
                    try_remove(c)
                try_add(y)
            
            if valid_swaps:
                y = random.choice(valid_swaps)
                try_remove(y)
                try_add(c)
            
            # Occasionally try a double swap
            if random.random() < 0.1 and len(S) >= 2:
                y1 = random.choice(S)
                others = [y for y in S if y != y1]
                if not others:
                    continue
                y2 = random.choice(others)
                c1 = random.randint(1, max_val)
                c2 = random.randint(1, max_val)
                if in_S[c1] or in_S[c2] or c1 == c2:
                    continue
                
                try_remove(y1)
                try_remove(y2)
                if try_add(c1) and try_add(c2):
                    pass
                else:
                    if not in_S[c1]:
                        try_remove(c1)
                    if not in_S[c2]:
                        try_remove(c2)
                    try_add(y1)
                    try_add(y2)
        
        return S
    
    # Run multiple trials
    trial_count = 0
    while time.time() < deadline - 10:
        S = run_one_trial()
        trial_count += 1
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
        
        if best_size >= 52 and trial_count >= 3:
            break
        if best_size >= 50 and trial_count >= 5:
            break
    
    # Final polish: try to add remaining elements
    if best_S and time.time() < deadline - 2:
        in_S = [False] * (max_val + 1)
        counts = [0] * (2 * max_val + 1)
        for x in best_S:
            in_S[x] = True
            counts[2 * x] += 1
        for i in range(len(best_S)):
            for j in range(i + 1, len(best_S)):
                counts[best_S[i] + best_S[j]] += 1
        
        new_elements = []
        for c in range(1, max_val + 1):
            if in_S[c]:
                continue
            can_add = True
            if counts[2 * c] >= 2:
                can_add = False
            else:
                for x in best_S:
                    if counts[c + x] >= 2:
                        can_add = False
                        break
            if can_add:
                new_elements.append(c)
        
        for c in new_elements:
            best_S.append(c)
            in_S[c] = True
            counts[2 * c] += 1
            for x in best_S:
                if x != c:
                    counts[c + x] += 1
    
    return best_S
