"""Improved program: branch-and-bound on hypergraph independence."""
import time
import random
from itertools import combinations

# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    """
    Returns a list of distinct positive integers <= 192 (max 48) such that
    no selected integer divides the sum of any 3 distinct OTHER selected integers.
    Maximizes cardinality.
    """
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s kill
    
    N = 192
    MAX_K = 48
    
    def is_valid_addition(S, x):
        """Check if adding x to a valid set S keeps it valid."""
        for a in S:
            others = [v for v in S if v != a]
            target = (-x) % a
            residue_count = {}
            for v in others:
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            for b in others:
                needed = (target - b) % a
                cnt = residue_count.get(needed, 0)
                if b % a == needed:
                    cnt -= 1
                if cnt > 0:
                    return False
        
        if len(S) >= 3:
            for triple in combinations(S, 3):
                if sum(triple) % x == 0:
                    return False
        
        return True
    
    def is_valid_full(S):
        """Full validation of set S."""
        S_list = list(S)
        for a in S_list:
            others = [x for x in S_list if x != a]
            if len(others) < 3:
                continue
            residue_count = {}
            for v in others:
                r = v % a
                residue_count[r] = residue_count.get(r, 0) + 1
            
            for i in range(len(others)):
                b = others[i]
                for j in range(i + 1, len(others)):
                    c = others[j]
                    target = (-b - c) % a
                    cnt = residue_count.get(target, 0)
                    if b % a == target:
                        cnt -= 1
                    if c % a == target:
                        cnt -= 1
                    if cnt > 0:
                        return False
        return True
    
    def greedy_construct(order):
        """Greedy construction: add numbers in given order if valid."""
        S = []
        for x in order:
            if x in S:
                continue
            if is_valid_addition(S, x):
                S.append(x)
            if len(S) >= MAX_K:
                break
        return S
    
    def local_search(S, remaining_time):
        """Local search: try to improve by adding or swapping elements."""
        S_set = set(S)
        
        while time.time() - start_time < remaining_time:
            improved = False
            
            # Try to add any number not in S
            for x in range(N, 0, -1):
                if x in S_set:
                    continue
                if is_valid_addition(S, x):
                    S.append(x)
                    S_set.add(x)
                    improved = True
                    break
            
            if improved:
                continue
            
            # Try 1-swap moves
            swapped = False
            for i, x in enumerate(list(S)):
                if time.time() - start_time > remaining_time:
                    break
                S_without = [v for j, v in enumerate(S) if j != i]
                S_without_set = set(S_without)
                for y in range(N, 0, -1):
                    if y in S_without_set or y == x:
                        continue
                    if is_valid_addition(S_without, y):
                        S_candidate = S_without + [y]
                        S_candidate_set = set(S_candidate)
                        extra_added = 0
                        for z in range(N, 0, -1):
                            if z in S_candidate_set:
                                continue
                            if is_valid_addition(S_candidate, z):
                                S_candidate.append(z)
                                S_candidate_set.add(z)
                                extra_added += 1
                                if len(S_candidate) >= MAX_K:
                                    break
                        if extra_added > 0:
                            S = S_candidate
                            S_set = set(S)
                            improved = True
                            swapped = True
                            break
                if swapped:
                    break
            
            if not improved:
                break
        
        return S
    
    # Phase 1: Greedy construction with multiple orders
    best_S = []
    best_len = 0
    
    # Descending order
    S1 = greedy_construct(range(N, 0, -1))
    if len(S1) > best_len:
        best_S = S1
        best_len = len(S1)
    
    # Ascending order
    if time.time() - start_time < DEADLINE:
        S2 = greedy_construct(range(1, N + 1))
        if len(S2) > best_len:
            best_S = S2
            best_len = len(S2)
    
    # Random orders
    random.seed(42)
    for trial in range(200):
        if time.time() - start_time > DEADLINE - 20:
            break
        order = list(range(1, N + 1))
        random.shuffle(order)
        S3 = greedy_construct(order)
        if len(S3) > best_len:
            best_S = S3
            best_len = len(S3)
    
    # Phase 2: Local search on best solution
    if time.time() - start_time < DEADLINE:
        best_S = local_search(best_S, DEADLINE - 5)
    
    # Phase 3: Branch-and-bound search
    if time.time() - start_time < DEADLINE - 10:
        def branch_and_bound(available, current_set, best_size):
            """
            Branch-and-bound: try to build a larger set than best_size.
            available: list of numbers not yet decided
            current_set: list of numbers in current solution
            best_size: current best size found
            """
            if time.time() - start_time > DEADLINE - 5:
                return current_set
            
            if len(current_set) + len(available) <= best_size:
                return current_set
            
            if not available:
                return current_set
            
            # Sort available by "constraint" - numbers that conflict with more current elements first
            def conflict_count(x):
                count = 0
                for a in current_set:
                    others = [v for v in current_set if v != a]
                    if len(others) >= 2:
                        # Check if x conflicts with a
                        for b in others:
                            for c in others:
                                if b != c and (x + b + c) % a == 0:
                                    count += 1
                                    break
                return count
            
            available_sorted = sorted(available, key=conflict_count, reverse=True)
            
            # Try including the first available number
            x = available_sorted[0]
            remaining = available_sorted[1:]
            
            if is_valid_addition(current_set, x):
                new_set = current_set + [x]
                result = branch_and_bound(remaining, new_set, best_size)
                if len(result) > best_size:
                    return result
            
            # Try excluding x
            result = branch_and_bound(remaining, current_set, best_size)
            return result
        
        # Run branch-and-bound from multiple starting points
        for start_offset in range(0, 10):
            if time.time() - start_time > DEADLINE - 10:
                break
            
            # Create a partial solution using greedy, then branch from there
            order = list(range(N, 0, -1))
            random.shuffle(order)
            partial = greedy_construct(order)
            
            if len(partial) >= best_len:
                continue
            
            # Numbers not in partial
            partial_set = set(partial)
            available = [x for x in range(1, N + 1) if x not in partial_set]
            
            result = branch_and_bound(available, partial, best_len)
            if len(result) > best_len:
                best_S = result
                best_len = len(result)
    
    # Phase 4: More local search
    if time.time() - start_time < DEADLINE:
        best_S = local_search(best_S, DEADLINE - 2)
    
    # Final verification
    if not is_valid_full(best_S):
        best_S = greedy_construct(range(N, 0, -1))
    
    return best_S

# EVOLVE-BLOCK-END
