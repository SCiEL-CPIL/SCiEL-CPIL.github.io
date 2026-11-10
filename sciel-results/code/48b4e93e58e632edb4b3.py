"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # safety margin before 90s kill
    
    # Precompute for each number a the set of sums of triples from [1..192]\{a} that are divisible by a
    # But we need to check: for a selected set S, for each a in S, no triple from S\{a} sums to a multiple of a.
    # 
    # Strategy: 
    # 1. Start with a greedy construction (like the seed) to get a good initial set.
    # 2. Use local search with swaps to improve.
    # 3. Use random restarts if time permits.
    
    def is_valid_set(S):
        """Check if set S is valid: no element divides the sum of any 3 distinct other elements."""
        S_list = list(S)
        n = len(S_list)
        for i, a in enumerate(S_list):
            # Check all triples from S_list excluding a
            others = [x for x in S_list if x != a]
            for triple in combinations(others, 3):
                s = sum(triple)
                if s % a == 0:
                    return False
        return True
    
    def is_valid_set_fast(S):
        """Faster validity check using precomputed triple sums."""
        S_list = list(S)
        n = len(S_list)
        if n < 4:
            return True
        for i, a in enumerate(S_list):
            others = [x for x in S_list if x != a]
            m = len(others)
            for j in range(m):
                for k in range(j+1, m):
                    for l in range(k+1, m):
                        s = others[j] + others[k] + others[l]
                        if s % a == 0:
                            return False
        return True
    
    def greedy_construct():
        """Greedy construction: add numbers from high to low if valid."""
        selected = []
        for x in range(192, 0, -1):
            candidate = selected + [x]
            valid = True
            # Check new element x against existing
            for a in selected:
                others = [v for v in candidate if v != a]
                for t in combinations(others, 3):
                    if sum(t) % a == 0:
                        valid = False
                        break
                if not valid:
                    break
            if valid:
                # Also check x itself
                others = [v for v in candidate if v != x]
                for t in combinations(others, 3):
                    if sum(t) % x == 0:
                        valid = False
                        break
            if valid:
                selected.append(x)
        return selected
    
    # Phase 1: Greedy construction
    best_set = greedy_construct()
    best_size = len(best_set)
    
    # Phase 2: Local search with swaps
    def local_search(S, time_limit):
        """Improve set S via swap operations."""
        S = list(S)
        current_size = len(S)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            random.shuffle(S)  # Randomize order for diversification
            for i in range(len(S)):
                if time.time() >= time_limit:
                    break
                for j in range(192, 0, -1):
                    if time.time() >= time_limit:
                        break
                    if j in S:
                        continue
                    # Try swapping S[i] with j
                    new_S = S.copy()
                    new_S[i] = j
                    new_S = list(set(new_S))  # Ensure distinct
                    if len(new_S) != current_size:
                        continue
                    if is_valid_set_fast(new_S):
                        S = new_S
                        improved = True
                        break
        return S
    
    # Phase 3: Try to add more elements
    def try_expand(S, time_limit):
        """Try to add elements to S while maintaining validity."""
        S = list(S)
        added = True
        while added and time.time() < time_limit:
            added = False
            for x in range(192, 0, -1):
                if time.time() >= time_limit:
                    break
                if x in S:
                    continue
                new_S = S + [x]
                if is_valid_set_fast(new_S):
                    S = new_S
                    added = True
                    break
        return S
    
    # Main loop with random restarts
    random.seed(42)
    num_restarts = 0
    while time.time() < deadline:
        num_restarts += 1
        
        # Random restart with perturbation
        if num_restarts == 1:
            current_set = list(best_set)
        else:
            # Start from best_set and perturb
            current_set = list(best_set)
            # Remove some random elements
            num_remove = random.randint(1, min(3, len(current_set) - 4))
            for _ in range(num_remove):
                idx = random.randint(0, len(current_set) - 1)
                current_set.pop(idx)
        
        # Try to expand
        current_set = try_expand(current_set, deadline - 2.0)
        
        # Local search
        current_set = local_search(current_set, deadline - 2.0)
        
        # Update best
        if len(current_set) > best_size:
            best_size = len(current_set)
            best_set = current_set
            # Try more local search
            current_set = local_search(current_set, deadline - 2.0)
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = current_set
        
        # If we've hit the maximum possible (48), stop
        if best_size >= 48:
            break
    
    return best_set
# EVOLVE-BLOCK-END
