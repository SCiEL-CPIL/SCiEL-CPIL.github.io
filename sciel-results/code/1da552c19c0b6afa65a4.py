import time
import random
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    def time_left():
        return (deadline - time.time()) > 0

    S = set()
    counts = defaultdict(int)
    
    def add(x):
        if x in S:
            return True
        if counts[2*x] >= 2:
            return False
        for y in S:
            if y == x:
                continue
            if counts[x+y] >= 2:
                return False
        
        S.add(x)
        counts[2*x] += 1
        for y in S:
            if y != x:
                counts[x+y] += 1
        return True

    def remove(x):
        if x not in S:
            return
        S.remove(x)
        counts[2*x] -= 1
        for y in S:
            if y != x:
                counts[x+y] -= 1
        if counts[2*x] == 0:
            del counts[2*x]
        for y in S:
            if y != x:
                s = x+y
                if counts[s] == 0:
                    del counts[s]

    # 1. Greedy Initialization
    for x in range(1, 1001):
        add(x)
        if not time_left():
            break

    # 2. Local Search
    while time_left():
        added = False
        # Try to add the smallest valid element
        for x in range(1, 1001):
            if x in S:
                continue
            # Quick check
            if counts[2*x] >= 2:
                continue
            valid = True
            for y in S:
                if y == x:
                    continue
                if counts[x+y] >= 2:
                    valid = False
                    break
            if valid:
                add(x)
                added = True
                break
        
        if added:
            continue

        # If stuck, try to resolve conflicts via swaps
        resolved = False
        # Try a few random candidates
        for _ in range(100):
            y = random.randint(1, 1000)
            if y in S:
                continue
            
            # Check validity and find conflicts
            conflicts = []
            if counts[2*y] >= 2:
                conflicts.append(2*y)
            
            for z in S:
                if z == y:
                    continue
                s = y + z
                if counts[s] >= 2:
                    conflicts.append(s)
                    if len(conflicts) > 1:
                        break
            
            if not conflicts:
                add(y)
                resolved = True
                break
            
            if len(conflicts) == 1:
                s_conf = conflicts[0]
                # Find pairs in S summing to s_conf
                pairs = []
                for a in S:
                    b = s_conf - a
                    if b in S and a <= b:
                        pairs.append((a, b))
                
                candidates = set()
                for a, b in pairs:
                    candidates.add(a)
                    candidates.add(b)
                
                for c in candidates:
                    remove(c)
                    if add(y):
                        resolved = True
                        break
                    else:
                        remove(y)
                        add(c)
                    if resolved:
                        break
                
                if resolved:
                    break
        
        if resolved:
            continue

    return list(S)
