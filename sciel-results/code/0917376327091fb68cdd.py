import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin for 90s limit
    
    n = 4096
    best_set = []
    
    candidates = list(range(1, n + 1))
    
    # Run multiple random restarts
    while time.time() < deadline:
        random.shuffle(candidates)
        
        S = []
        T = set()  # Set of all triple sums
        
        for x in candidates:
            # Check if adding x creates a conflict
            # New sums involving x:
            # 1. x+x+x
            # 2. x+x+s for s in S
            # 3. x+s1+s2 for s1,s2 in S, s1<=s2
            
            new_sums = []
            
            # 1. 3x
            new_sums.append(3*x)
            
            # 2. 2x + s
            for s in S:
                new_sums.append(2*x + s)
                
            # 3. x + s1 + s2
            for s1, s2 in combinations_with_replacement(S, 2):
                new_sums.append(x + s1 + s2)
                
            # Check for conflicts with existing T
            conflict = False
            seen_new_sums = set()
            for s_val in new_sums:
                if s_val in T:
                    conflict = True
                    break
                if s_val in seen_new_sums:
                    conflict = True
                    break
                seen_new_sums.add(s_val)
                
            if not conflict:
                S.append(x)
                T.update(new_sums)
                
        if len(S) > len(best_set):
            best_set = S
            
    return best_set
