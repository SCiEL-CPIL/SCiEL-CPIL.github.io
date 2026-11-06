import random
import time
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin before 90s kill
    
    best_set = []
    best_size = 0
    
    def check_validity(S):
        """Check if set S has all unique triple sums."""
        triple_sums = set()
        for a, b, c in combinations_with_replacement(S, 3):
            s = a + b + c
            if s in triple_sums:
                return False
            triple_sums.add(s)
        return True
    
    def run_greedy(order):
        """Greedy: try to add each x in order if valid."""
        S = []
        triple_sums = set()
        
        for x in order:
            if x in set(S):
                continue
            # Check all new triples involving x
            valid = True
            # New triples: (x,x,x), (x,x,y) for y in S, (x,y,z) for y<=z in S
            # (x,x,x)
            if 3*x in triple_sums:
                valid = False
            else:
                # (x,x,y) for y in S
                for y in S:
                    if 2*x + y in triple_sums:
                        valid = False
                        break
                if valid:
                    # (x,y,z) for y<=z in S
                    for i in range(len(S)):
                        for j in range(i, len(S)):
                            if x + S[i] + S[j] in triple_sums:
                                valid = False
                                break
                        if not valid:
                            break
            
            if valid:
                S.append(x)
                # Add new triple sums
                triple_sums.add(3*x)
                for y in S[:-1]:
                    triple_sums.add(2*x + y)
                    for z in S[:-1]:
                        if z >= y:
                            triple_sums.add(x + y + z)
                        elif x >= z:
                            # x is the largest, so x+y+z where y<=z
                            pass
                        else:
                            # y is largest: y+x+z
                            pass
                # Actually, let's just add all triples involving x properly
                # Remove the incorrect additions and redo
                triple_sums = set()
                for a, b, c in combinations_with_replacement(S, 3):
                    triple_sums.add(a + b + c)
        
        return S
    
    def run_greedy_efficient(order):
        """More efficient greedy with proper incremental updates."""
        S = []
        triple_sums = set()
        
        for x in order:
            if x in set(S):
                continue
            
            # Check if x can be added
            # New triples involving x: all (a,b,c) where at least one is x
            # Type 1: (x,x,x) -> sum = 3x
            # Type 2: (x,x,y) for y in S, y != x -> sum = 2x+y
            # Type 3: (x,y,z) for y,z in S, y<=z -> sum = x+y+z
            
            can_add = True
            
            if 3*x in triple_sums:
                can_add = False
            else:
                for y in S:
                    if 2*x + y in triple_sums:
                        can_add = False
                        break
                if can_add:
                    n = len(S)
                    for i in range(n):
                        for j in range(i, n):
                            if x + S[i] + S[j] in triple_sums:
                                can_add = False
                                break
                        if not can_add:
                            break
            
            if can_add:
                S.append(x)
                # Add all new triple sums involving x
                triple_sums.add(3*x)
                for y in S[:-1]:
                    triple_sums.add(2*x + y)
                n = len(S[:-1])
                for i in range(n):
                    for j in range(i, n):
                        triple_sums.add(x + S[i] + S[j])
        
        return S
    
    def local_search(S, max_time=5.0):
        """Try to improve S by adding or swapping elements."""
        S = list(S)
        S.sort()
        max_time = min(max_time, 10.0)
        
        # Precompute triple sums
        triple_sums = set()
        for a, b, c in combinations_with_replacement(S, 3):
            triple_sums.add(a + b + c)
        
        used = set(S)
        improved = True
        
        while improved and time.time() - start_time < time_limit - max_time:
            improved = False
            
            # Try adding a new element
            candidates = [x for x in range(1, 4097) if x not in used]
            random.shuffle(candidates)
            
            for x in candidates:
                can_add = True
                if 3*x in triple_sums:
                    can_add = False
                else:
                    for y in S:
                        if 2*x + y in triple_sums:
                            can_add = False
                            break
                    if can_add:
                        n = len(S)
                        for i in range(n):
                            for j in range(i, n):
                                if x + S[i] + S[j] in triple_sums:
                                    can_add = False
                                    break
                            if not can_add:
                                break
                
                if can_add:
                    S.append(x)
                    S.sort()
                    used.add(x)
                    triple_sums.add(3*x)
                    for y in S[:-1]:
                        triple_sums.add(2*x + y)
                    n = len(S[:-1])
                    for i in range(n):
                        for j in range(i, n):
                            triple_sums.add(x + S[i] + S[j])
                    improved = True
                    break
            
            if not improved:
                # Try swaps: remove y, add x
                for y in S:
                    S_temp = [z for z in S if z != y]
                    # Compute triple sums for S_temp
                    triple_sums_temp = set()
                    for a, b, c in combinations_with_replacement(S_temp, 3):
                        triple_sums_temp.add(a + b + c)
                    
                    candidates = [x for x in range(1, 4097) if x not in set(S_temp) and x != y]
                    random.shuffle(candidates[:100])
                    candidates = candidates[:100]
                    
                    for x in candidates:
                        can_add = True
                        if 3*x in triple_sums_temp:
                            can_add = False
                        else:
                            for z in S_temp:
                                if 2*x + z in triple_sums_temp:
                                    can_add = False
                                    break
                            if can_add:
                                n = len(S_temp)
                                for i in range(n):
                                    for j in range(i, n):
                                        if x + S_temp[i] + S_temp[j] in triple_sums_temp:
                                            can_add = False
                                            break
                                    if not can_add:
                                        break
                        
                        if can_add:
                            S = S_temp + [x]
                            S.sort()
                            used = set(S)
                            triple_sums = triple_sums_temp
                            triple_sums.add(3*x)
                            for z in S_temp:
                                triple_sums.add(2*x + z)
                            n = len(S_temp)
                            for i in range(n):
                                for j in range(i, n):
                                    triple_sums.add(x + S_temp[i] + S_temp[j])
                            improved = True
                            break
                    
                    if improved:
                        break
        
        return S
    
    # Run multiple greedy attempts with different orderings
    for order in [range(1, 4097), range(4096, 0, -1)]:
        current_set = run_greedy_efficient(order)
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Randomized greedy attempts
    while time.time() - start_time < time_limit - 5.0:
        candidates = list(range(1, 4097))
        random.shuffle(candidates)
        current_set = run_greedy_efficient(candidates)
        
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
    
    # Local search on best set
    if best_size > 0 and time.time() - start_time < time_limit - 10.0:
        improved_set = local_search(best_set, max_time=10.0)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    return best_set
