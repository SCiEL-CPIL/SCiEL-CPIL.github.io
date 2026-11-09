"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 256
    max_size = 64
    
    # Helper to check Sidon property
    def is_sidon(S):
        if len(S) == 0:
            return True
        sums = set()
        S_sorted = sorted(S)
        for i in range(len(S_sorted)):
            for j in range(i, len(S_sorted)):
                s = S_sorted[i] + S_sorted[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    # Helper to get difference set
    def get_diffs(S):
        diffs = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                diffs.add(abs(S_list[i] - S_list[j]))
        return diffs

    # Check if adding x to S keeps it Sidon
    def can_add_sidon(S, x):
        # Check if x+x or x+s creates a duplicate sum
        # Existing sums in S
        sums = set()
        S_list = list(S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                sums.add(S_list[i] + S_list[j])
        
        # New sums involving x
        for s in S_list:
            if x + s in sums:
                return False
        if 2 * x in sums:
            return False
        return True

    # Check if adding x to S keeps diffs disjoint from other_diffs
    def can_add_diff(S, x, other_diffs):
        for s in S:
            d = abs(x - s)
            if d in other_diffs:
                return False
        return True

    # Full validity check for a pair (A, B)
    def is_valid(A, B):
        if not is_sidon(A) or not is_sidon(B):
            return False
        diff_A = get_diffs(A)
        diff_B = get_diffs(B)
        if diff_A.intersection(diff_B):
            return False
        return True

    best_total = 0
    best_A = set()
    best_B = set()
    
    start_time = time.time()
    time_limit = 55.0 # Leave 5s for safety
    
    # Strategy: Local Search with Swaps
    # 1. Greedy initialization
    # 2. Iterative improvement:
    #    - Try to add elements
    #    - If stuck, try swaps to rearrange elements to potentially allow more additions later
    
    for restart in range(20):
        A = set()
        B = set()
        diff_A = set()
        diff_B = set()
        
        # Greedy fill
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        for x in candidates:
            # Try A
            if can_add_sidon(A, x) and can_add_diff(A, x, diff_B):
                # Update diff_A
                new_diffs = {abs(x - a) for a in A}
                if not new_diffs.intersection(diff_B):
                    A.add(x)
                    diff_A.update(new_diffs)
            
            # Try B
            if can_add_sidon(B, x) and can_add_diff(B, x, diff_A):
                new_diffs = {abs(x - b) for b in B}
                if not new_diffs.intersection(diff_A):
                    B.add(x)
                    diff_B.update(new_diffs)
        
        current_total = len(A) + len(B)
        if current_total > best_total:
            best_total = current_total
            best_A = A.copy()
            best_B = B.copy()
            
        # Local Search Phase
        # We try to improve by swapping elements
        # A swap (remove a, add x) is valid if:
        # 1. A\{a} U {x} is Sidon
        # 2. Diff(A\{a} U {x}) is disjoint from Diff(B)
        
        improved = True
        while improved and (time.time() - start_time < time_limit):
            improved = False
            
            # Try adding to A
            if len(A) < max_size:
                for x in range(1, n + 1):
                    if x not in A:
                        if can_add_sidon(A, x) and can_add_diff(A, x, diff_B):
                            new_diffs = {abs(x - a) for a in A}
                            if not new_diffs.intersection(diff_B):
                                A.add(x)
                                diff_A.update(new_diffs)
                                improved = True
                                break
            
            # Try adding to B
            if len(B) < max_size:
                for x in range(1, n + 1):
                    if x not in B:
                        if can_add_sidon(B, x) and can_add_diff(B, x, diff_A):
                            new_diffs = {abs(x - b) for b in B}
                            if not new_diffs.intersection(diff_A):
                                B.add(x)
                                diff_B.update(new_diffs)
                                improved = True
                                break
            
            if improved:
                current_total = len(A) + len(B)
                if current_total > best_total:
                    best_total = current_total
                    best_A = A.copy()
                    best_B = B.copy()
                continue # Restart loop to try more additions
            
            # If no additions, try swaps
            # Randomly pick a set to swap in
            if random.random() < 0.5:
                # Swap in A
                if A:
                    a_to_remove = random.choice(list(A))
                    # Find a candidate x
                    candidates_for_swap = [x for x in range(1, n + 1) if x not in A]
                    random.shuffle(candidates_for_swap)
                    
                    for x in candidates_for_swap:
                        # Check if A' = A - {a} + {x} is Sidon
                        A_temp = A.copy()
                        A_temp.remove(a_to_remove)
                        A_temp.add(x)
                        
                        if is_sidon(A_temp):
                            # Check diffs
                            diff_A_temp = get_diffs(A_temp)
                            if not diff_A_temp.intersection(diff_B):
                                # Perform swap
                                A = A_temp
                                diff_A = diff_A_temp
                                improved = True
                                break
            else:
                # Swap in B
                if B:
                    b_to_remove = random.choice(list(B))
                    candidates_for_swap = [x for x in range(1, n + 1) if x not in B]
                    random.shuffle(candidates_for_swap)
                    
                    for x in candidates_for_swap:
                        B_temp = B.copy()
                        B_temp.remove(b_to_remove)
                        B_temp.add(x)
                        
                        if is_sidon(B_temp):
                            diff_B_temp = get_diffs(B_temp)
                            if not diff_B_temp.intersection(diff_A):
                                B = B_temp
                                diff_B = diff_B_temp
                                improved = True
                                break

    return [list(best_A), list(best_B)]

# EVOLVE-BLOCK-END
