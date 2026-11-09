import time
import random
from math import gcd
from itertools import combinations

def _lcm(a, b):
    return a * b // gcd(a, b)

def _is_bad_triple(a, b, c):
    return _lcm(a, b) == _lcm(a, c) == _lcm(b, c)

def construct():
    start_time = time.time()
    deadline = start_time + 82.0  # Safety margin for 90s limit
    
    N = 128
    
    # Precompute LCM table
    lcm_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            lcm_table[i][j] = _lcm(i, j)
            lcm_table[j][i] = lcm_table[i][j]
    
    # Precompute bad_pair as bitmasks for fast lookup
    # bad_pair[a][b] is a bitmask of c values that form a bad triple with a and b
    bad_pair_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            lab = lcm_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if lcm_table[a][c] == lab and lcm_table[b][c] == lab:
                    mask |= (1 << c)
            bad_pair_mask[a][b] = mask
            bad_pair_mask[b][a] = mask
    
    # For each x, precompute which pairs (a,b) have x in their bad set
    # conflict_with[x] = bitmask of "pair indices" where x conflicts
    # But we need to check against current set S, so let's use a different approach:
    # For each x, precompute bad_pairs_of_x[a] = bitmask of b such that x is in bad_pair[a][b]
    # Then can_add(x, S) checks: for each a in S, if (bad_pairs_of_x[a] & S_mask) != 0, conflict
    
    # bad_pairs_of_x[x][a] = bitmask of b values such that x is in bad_pair[a][b]
    bad_pairs_of_x = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            mask = bad_pair_mask[a][b]
            # For each c in mask, add b to bad_pairs_of_x[c][a] and a to bad_pairs_of_x[c][b]
            m = mask
            while m:
                c = (m & -m).bit_length() - 1
                m &= m - 1
                if c >= 1 and c <= N:
                    bad_pairs_of_x[c][a] |= (1 << b)
                    bad_pairs_of_x[c][b] |= (1 << a)
    
    def _can_add(x, S_mask, S_list):
        """Check if x can be added to S (represented by S_mask and S_list) without creating a bad triple."""
        # For each a in S, check if there's a b in S such that x is in bad_pair[a][b]
        for a in S_list:
            if (bad_pairs_of_x[x][a] & S_mask) != 0:
                return False
        return True
    
    def _local_search(S, time_limit):
        """Try to improve S by removing one element and adding others."""
        S = set(S)
        S_list = sorted(S)
        S_mask = 0
        for x in S:
            S_mask |= (1 << x)
        best_size = len(S)
        best_set = set(S)
        
        candidates = [x for x in range(1, N + 1) if x not in S]
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for remove_x in S_list:
                if time.time() > time_limit:
                    break
                S_trial = S - {remove_x}
                S_trial_list = [x for x in S_list if x != remove_x]
                S_trial_mask = S_mask & ~(1 << remove_x)
                
                # Try to greedily add candidates
                for add_x in candidates:
                    if time.time() > time_limit:
                        break
                    if _can_add(add_x, S_trial_mask, S_trial_list):
                        S_trial.add(add_x)
                        S_trial_list.append(add_x)
                        S_trial_mask |= (1 << add_x)
                        if len(S_trial) > best_size:
                            best_size = len(S_trial)
                            best_set = set(S_trial)
                            S = set(S_trial)
                            S_list = sorted(S)
                            S_mask = S_trial_mask
                            candidates = [x for x in range(1, N + 1) if x not in S]
                            improved = True
                            break
                if improved:
                    break
        return best_set
    
    # 1. Build the base set of prime powers and 1
    base_set = set()
    base_set.add(1)
    
    # Add prime powers
    for p in range(2, 129):
        is_prime = True
        if p < 2:
            is_prime = False
        else:
            for i in range(2, int(p**0.5) + 1):
                if p % i == 0:
                    is_prime = False
                    break
        if is_prime:
            power = p
            while power <= 128:
                base_set.add(power)
                power *= p
                
    remaining = [x for x in range(2, 129) if x not in base_set]
    
    # Greedy with bitmask tracking
    S = set(base_set)
    S_mask = 0
    for x in S:
        S_mask |= (1 << x)
    S_list = sorted(S)
    
    # Greedy pass 1: Increasing order
    for x in sorted(remaining):
        if _can_add(x, S_mask, S_list):
            S.add(x)
            S_mask |= (1 << x)
            S_list.append(x)
            
    best_set = set(S)
    best_size = len(S)
    
    # Try multiple random orderings
    all_candidates = list(remaining)
    
    for trial in range(50):
        if time.time() > deadline:
            break
        random.shuffle(all_candidates)
        S_current = set(base_set)
        S_cur_mask = 0
        for x in S_current:
            S_cur_mask |= (1 << x)
        S_cur_list = sorted(S_current)
        
        for x in all_candidates:
            if _can_add(x, S_cur_mask, S_cur_list):
                S_current.add(x)
                S_cur_mask |= (1 << x)
                S_cur_list.append(x)
        
        if len(S_current) > best_size:
            best_size = len(S_current)
            best_set = set(S_current)
    
    # Local search to try to improve
    if time.time() < deadline - 3:
        improved_set = _local_search(best_set, deadline - 1)
        if len(improved_set) > best_size:
            best_size = len(improved_set)
            best_set = improved_set
    
    # Final validation
    S_list = sorted(best_set)
    valid = True
    for a, b, c in combinations(S_list, 3):
        if _is_bad_triple(a, b, c):
            valid = False
            break
            
    if not valid:
        return list(base_set)
        
    if not best_set:
        return [1]
        
    return list(best_set)
