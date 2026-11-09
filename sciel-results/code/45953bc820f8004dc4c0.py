"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 80  # 80s safety margin before 90s limit
    
    def is_valid_set(S):
        """Check if no reciprocal of a selected element equals sum of reciprocals of a subset of others."""
        S_set = set(S)
        for x in S_set:
            target = 1.0 / x
            others = [y for y in S_set if y != x]
            # Check all subsets of others
            n = len(others)
            for mask in range(1, 1 << n):
                s = 0.0
                for i in range(n):
                    if mask & (1 << i):
                        s += 1.0 / others[i]
                        if s > target + 1e-12:
                            break
                if abs(s - target) < 1e-12:
                    return False
        return True
    
    def try_add(S, candidate):
        """Check if we can add candidate to S."""
        S_set = set(S)
        if candidate in S_set:
            return False
        # Check that adding candidate doesn't create a violation
        # For the new element: 1/candidate should not equal sum of reciprocals of subset of S
        target_new = 1.0 / candidate
        n = len(S)
        for mask in range(1, 1 << n):
            s = 0.0
            for i in range(n):
                if mask & (1 << i):
                    s += 1.0 / S[i]
                    if s > target_new + 1e-12:
                        break
            if abs(s - target_new) < 1e-12:
                return False
        # For each existing element x: 1/x should not equal sum of reciprocals of subset of (S + candidate)
        for x in S:
            target = 1.0 / x
            others = [y for y in S if y != x] + [candidate]
            n2 = len(others)
            for mask in range(1, 1 << n2):
                s = 0.0
                for i in range(n2):
                    if mask & (1 << i):
                        s += 1.0 / others[i]
                        if s > target + 1e-12:
                            break
                if abs(s - target) < 1e-12:
                    return False
        return True
    
    # Start with large numbers (less likely to cause subsumption)
    best_set = list(range(13, 25))  # 12 elements
    best_size = len(best_set)
    
    # Try to add elements from 1 to 12
    for candidate in range(1, 13):
        if time.time() > deadline:
            break
        if try_add(best_set, candidate):
            best_set.append(candidate)
            best_size += 1
    
    # Local search: try to swap elements to increase size
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            for j in range(1, 25):
                if time.time() > deadline:
                    break
                if j in best_set:
                    continue
                # Try removing best_set[i] and adding j
                new_set = best_set[:i] + best_set[i+1:] + [j]
                if is_valid_set(new_set):
                    # Check if this actually helps (it has same size, but might allow future additions)
                    # For now, just accept if valid
                    # Actually, we want to maximize size, so same-size swaps don't help directly
                    # But they might enable adding more elements later
                    pass
        
        # Try to add any element from 1-24 not in best_set
        for candidate in range(1, 25):
            if time.time() > deadline:
                break
            if candidate in best_set:
                continue
            if try_add(best_set, candidate):
                best_set.append(candidate)
                best_size += 1
                improved = True
                break
    
    # Final check
    if not is_valid_set(best_set):
        # Fallback: use a known valid set
        best_set = list(range(13, 25))
    
    return best_set
# EVOLVE-BLOCK-END
