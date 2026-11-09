"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import random
import time

def construct():
    n = 256
    max_size = 64
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    def is_sidon(set_S):
        if len(set_S) < 2:
            return True
        sums = set()
        S_list = sorted(set_S)
        for i in range(len(S_list)):
            for j in range(i, len(S_list)):
                s = S_list[i] + S_list[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def get_diffs(set_S):
        diffs = set()
        S_list = sorted(set_S)
        for i in range(len(S_list)):
            for j in range(i + 1, len(S_list)):
                diffs.add(S_list[j] - S_list[i])
        return diffs

    def can_add_to(A, B, x, target):
        # Check if adding x to target set maintains Sidon and disjoint diffs
        # We need to check Sidon for the new set
        # And check if new diffs conflict with other set's diffs or existing diffs in target
        
        if target == 'A':
            new_set = set(A) | {x}
            if not is_sidon(new_set):
                return False
            new_diffs = get_diffs(new_set)
            # Old diffs of A are subset of new_diffs
            # We need new_diffs to be disjoint from diff_B
            # And new_diffs must be disjoint from diff_A (which is true if Sidon, as diffs are unique)
            # Wait, Sidon implies unique diffs. So we just need new_diffs disjoint from diff_B.
            # But we also need to ensure that the *new* diffs introduced don't conflict with diff_B.
            # Actually, if we just check `new_diffs.isdisjoint(diff_B)`, that's sufficient?
            # Yes, because diff_A (old) was already disjoint from diff_B.
            # And new_diffs are unique within A (Sidon).
            # So we just need `new_diffs.isdisjoint(diff_B)`.
            # However, we also need to ensure that the new diffs don't conflict with each other?
            # Sidon check handles that.
            
            # Optimization: We don't need to recompute all diffs if we are careful, but for small sets it's fine.
            return new_diffs.isdisjoint(diff_B)
            
        else: # target == 'B'
            new_set = set(B) | {x}
            if not is_sidon(new_set):
                return False
            new_diffs = get_diffs(new_set)
            return new_diffs.isdisjoint(diff_A)

    # Phase 1: Greedy Initialization
    # Try a few random orders to get a decent start
    best_A = []
    best_B = []
    best_total = 0
    
    for trial in range(20):
        if time.time() - start_time > time_limit * 0.2:
            break
        random.seed(42 + trial)
        A = []
        B = []
        candidates = list(range(1, n + 1))
        random.shuffle(candidates)
        
        diff_A = set()
        diff_B = set()
        
        for x in candidates:
            # Try A
            if can_add_to(A, B, x, 'A'):
                # Verify Sidon quickly? can_add_to does it.
                A.append(x)
                diff_A = get_diffs(A)
            # Try B
            if can_add_to(A, B, x, 'B'):
                B.append(x)
                diff_B = get_diffs(B)
                
        total = len(A) + len(B)
        if total > best_total:
            best_total = total
            best_A = A[:]
            best_B = B[:]
            
    # Phase 2: Local Search Improvement
    # We try to add elements or swap elements to increase total size
    # We will work on a copy of the best solution found
    A = best_A[:]
    B = best_B[:]
    diff_A = get_diffs(A)
    diff_B = get_diffs(B)
    
    # Helper to check if a specific add is valid
    def try_add(A, B, diff_A, diff_B, x, target):
        if target == 'A':
            new_A = A + [x]
            if not is_sidon(new_A):
                return False
            new_diffs = get_diffs(new_A)
            if not new_diffs.isdisjoint(diff_B):
                return False
            # Also need to check if new_diffs conflicts with old diff_A?
            # Sidon ensures new_diffs are unique within A.
            # But do new_diffs intersect old diff_A?
            # If d is in new_diffs and d is in old diff_A, then d = a_i - a_j = a_k - a_l.
            # This is allowed as long as it's the same pair? No, Sidon prevents distinct pairs having same sum.
            # Sidon prevents distinct pairs having same sum.
            # Does Sidon prevent distinct pairs having same difference?
            # Yes, as proved before.
            # So new_diffs and old_diffs are disjoint (except for 0 which is not in diff sets).
            # So we just need new_diffs disjoint from diff_B.
            return True
        else:
            new_B = B + [x]
            if not is_sidon(new_B):
                return False
            new_diffs = get_diffs(new_B)
            if not new_diffs.isdisjoint(diff_A):
                return False
            return True

    # Local Search Loop
    improved = True
    while improved and (time.time() - start_time < time_limit):
        improved = False
        # Try to add any element from 1..256 not in A or B (or even in one, if adding to other)
        # Actually, elements can be in both.
        # Let's try adding to A or B for all x in 1..256
        for x in range(1, n + 1):
            if time.time() - start_time > time_limit:
                return [A, B]
            
            # Try adding to A
            if x not in A:
                if try_add(A, B, diff_A, diff_B, x, 'A'):
                    A.append(x)
                    diff_A = get_diffs(A)
                    improved = True
                    # Restart loop to pick up new opportunities
                    break
            
            # Try adding to B
            if x not in B:
                if try_add(A, B, diff_A, diff_B, x, 'B'):
                    B.append(x)
                    diff_B = get_diffs(B)
                    improved = True
                    break
                    
    # If we have time, try some swaps?
    # Swaps are complex. Adding is primary.
    
    return [A, B]

# EVOLVE-BLOCK-END
