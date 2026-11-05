import random
import time
from math import lcm, gcd
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 50  # 50s safety margin
    
    # Candidate numbers
    candidates = list(range(2, 257))
    
    # Precompute Omega (total number of prime factors with multiplicity)
    def compute_omega(n):
        count = 0
        d = 2
        while d * d <= n:
            while n % d == 0:
                count += 1
                n //= d
            d += 1
        if n > 1:
            count += 1
        return count
    
    omega = {x: compute_omega(x) for x in candidates}
    
    # Partition by parity of Omega
    even_omega = [x for x in candidates if omega[x] % 2 == 0]
    odd_omega = [x for x in candidates if omega[x] % 2 == 1]
    
    # Precompute conflict graph: conflict[x] = set of y where lcm(x,y) <= 256
    conflict = defaultdict(set)
    for i, x in enumerate(candidates):
        for y in candidates[i+1:]:
            if lcm(x, y) <= 256:
                conflict[x].add(y)
                conflict[y].add(x)
    
    def is_valid_addition(x, current_set):
        for y in current_set:
            if y in conflict[x]:
                return False
        return True
        
    def get_score(s):
        return sum(1.0/x for x in s)
        
    def local_search(initial_set):
        current_set = list(initial_set)
        current_score = get_score(current_set)
        current_set_set = set(current_set)
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try adding
            for x in candidates:
                if x in current_set_set:
                    continue
                if is_valid_addition(x, current_set):
                    new_score = current_score + 1.0/x
                    if new_score > current_score + 1e-12:
                        current_set.append(x)
                        current_set_set.add(x)
                        current_score = new_score
                        improved = True
            
            # Try swapping
            for i in range(len(current_set)):
                x = current_set[i]
                for y in candidates:
                    if y in current_set_set:
                        continue
                    
                    # Check validity of y against all others except x
                    valid = True
                    for k in range(len(current_set)):
                        if k == i:
                            continue
                        if current_set[k] in conflict[y]:
                            valid = False
                            break
                    
                    if valid:
                        new_score = current_score - 1.0/x + 1.0/y
                        if new_score > current_score + 1e-12:
                            current_set[i] = y
                            current_set_set.remove(x)
                            current_set_set.add(y)
                            current_score = new_score
                            improved = True
        
        return current_set, current_score
    
    best_set = []
    best_score = 0.0
    
    # Strategy 1: Solve each partition independently, then combine
    def solve_partition(partition_list):
        """Solve MWIS on a partition using greedy + local search"""
        if not partition_list:
            return [], 0.0
        
        # Greedy: sort by 1/x descending (x ascending)
        sorted_part = sorted(partition_list)
        greedy_set = []
        greedy_set_set = set()
        for x in sorted_part:
            if is_valid_addition(x, greedy_set):
                greedy_set.append(x)
                greedy_set_set.add(x)
        
        # Local search
        s, score = local_search(greedy_set)
        return s, score
    
    # Solve even and odd partitions
    even_set, even_score = solve_partition(even_omega)
    odd_set, odd_score = solve_partition(odd_omega)
    
    # Combine and check validity
    combined = even_set + odd_set
    combined_set = set(combined)
    
    # Verify combined is valid
    valid_combined = True
    for i, x in enumerate(combined):
        for y in combined[i+1:]:
            if y in conflict[x]:
                valid_combined = False
                break
        if not valid_combined:
            break
    
    if valid_combined:
        combined_score = even_score + odd_score
        if combined_score > best_score:
            best_score = combined_score
            best_set = combined
    else:
        # If not valid, use the better of the two
        if even_score > odd_score:
            best_score = even_score
            best_set = even_set
        else:
            best_score = odd_score
            best_set = odd_set
    
    # Strategy 2: Standard greedy + local search from full set
    init_greedy = []
    init_set = set()
    for x in sorted(candidates):
        if is_valid_addition(x, init_greedy):
            init_greedy.append(x)
            init_set.add(x)
    
    s, score = local_search(init_greedy)
    if score > best_score:
        best_score = score
        best_set = s
    
    # Strategy 3: Random restarts with bias toward high-weight numbers
    while time.time() < deadline - 5:
        # Sample from high-weight region (small numbers)
        high_weight = [x for x in candidates if x <= 50]
        random.shuffle(high_weight)
        init = []
        for x in high_weight[:15]:
            if is_valid_addition(x, init):
                init.append(x)
        
        s, score = local_search(init)
        if score > best_score:
            best_score = score
            best_set = s
    
    return best_set
