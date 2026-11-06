# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement, combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s kill
    
    N = 256
    MAX_SIZE = 64
    
    # Precompute pair sums and differences for potential conflicts?
    # No, dynamic is better for local search.
    # But we can precompute all possible pairs? No, 256*256 is small enough to check on fly with sets.
    
    def get_sums_and_diffs(S):
        sums = set()
        diffs = set()
        for a, b in combinations_with_replacement(S, 2):
            sums.add(a + b)
        for a, b in combinations(S, 2):
            if a != b:
                diffs.add(b - a)
        return sums, diffs

    def is_valid_addition(S, x, other_diffs):
        # Check if adding x to S keeps S Sidon and diffs disjoint from other_diffs
        new_diffs = set()
        for a in S:
            d = abs(x - a)
            if d == 0:
                continue
            if d in other_diffs:
                return False
            new_diffs.add(d)
        
        # Check new sums
        new_sums = set()
        for a in S:
            s = x + a
            if s in new_sums:
                return False
            new_sums.add(s)
        # Check x+x
        s_xx = 2 * x
        if s_xx in new_sums:
            return False
        
        # Check existing sums against new sums
        # Actually, easier to just recompute sums of S+{x} and check uniqueness
        # But we can be smarter.
        # Sum conflict if x + a1 == a2 + a3 for some a1, a2, a3 in S
        # Or x + x == a1 + a2
        
        # Let's just do the full check for correctness first, optimize if needed.
        # Given N=256 and size <= 64, |S| is small.
        candidate = S + [x]
        sums = set()
        for a, b in combinations_with_replacement(candidate, 2):
            s = a + b
            if s in sums:
                return False
            sums.add(s)
        
        # Check diff disjointness
        # We already checked new diffs against other_diffs.
        # Do we need to check if new diffs conflict with existing diffs of S?
        # No, S is already Sidon, so its diffs are unique within itself.
        # The only constraint is disjoint from OTHER set's diffs.
        
        return True

    def greedy_construct(order, A_diffs, B_diffs):
        A = []
        B = []
        A_diffs_cur = set()
        B_diffs_cur = set()
        
        for x in order:
            # Try adding to A
            if len(A) < MAX_SIZE:
                if is_valid_addition(A, x, B_diffs_cur):
                    A.append(x)
                    for a in A[:-1]:
                        A_diffs_cur.add(abs(x - a))
                    continue
            # Try adding to B
            if len(B) < MAX_SIZE:
                if is_valid_addition(B, x, A_diffs_cur):
                    B.append(x)
                    for b in B[:-1]:
                        B_diffs_cur.add(abs(x - b))
                    continue
            # If neither, skip
            
        return A, B, A_diffs_cur, B_diffs_cur

    def local_search(A, B, A_diffs, B_diffs, max_iters=2000):
        A = A[:]
        B = B[:]
        A_diffs = set(A_diffs)
        B_diffs = set(B_diffs)
        
        best_size = len(A) + len(B)
        no_improve_count = 0
        
        for _ in range(max_iters):
            if time.time() > deadline - 5.0:
                break
                
            improved = False
            
            # 1. Try adding an element
            for side in range(2):
                S = A if side == 0 else B
                Other_diffs = B_diffs if side == 0 else A_diffs
                if len(S) >= MAX_SIZE:
                    continue
                S_set = set(S)
                candidates = [x for x in range(1, N + 1) if x not in S_set]
                random.shuffle(candidates)
                for x in candidates:
                    if is_valid_addition(S, x, Other_diffs):
                        S.append(x)
                        for y in S[:-1]:
                            d = abs(x - y)
                            if side == 0:
                                A_diffs.add(d)
                            else:
                                B_diffs.add(d)
                        improved = True
                        best_size = len(A) + len(B)
                        no_improve_count = 0
                        break
                if improved:
                    break
            
            if improved:
                continue

            # 2. Try swapping an element
            for side in range(2):
                S = A if side == 0 else B
                Other_diffs = B_diffs if side == 0 else A_diffs
                if len(S) == 0:
                    continue
                S_list = S[:]
                random.shuffle(S_list)
                for i, old_x in enumerate(S_list):
                    S_set = set(S)
                    candidates = [x for x in range(1, N + 1) if x not in S_set]
                    random.shuffle(candidates)
                    for new_x in candidates:
                        # Temporarily remove old_x
                        S.remove(old_x)
                        # Recalculate diffs for S
                        # This is expensive. Let's do a simpler check.
                        # We need to check if (S - {old_x}) + {new_x} is valid.
                        # And diffs of new set disjoint from Other_diffs.
                        
                        # Quick check: new_x not in Other_diffs? No, new_x is an element, not a diff.
                        # The diffs of the new set must be disjoint from Other_diffs.
                        
                        # Let's just try the full validation on the candidate set.
                        candidate_set = S + [new_x]
                        valid = True
                        # Check Sidon
                        sums = set()
                        for a, b in combinations_with_replacement(candidate_set, 2):
                            s = a + b
                            if s in sums:
                                valid = False
                                break
                            sums.add(s)
                        if not valid:
                            S.append(old_x)
                            continue
                        
                        # Check Diffs
                        new_diffs = set()
                        for a, b in combinations(candidate_set, 2):
                            if a != b:
                                d = abs(b - a)
                                new_diffs.add(d)
                        
                        if new_diffs & Other_diffs:
                            S.append(old_x)
                            continue
                        
                        # If valid, perform swap
                        S.remove(new_x) # It was added in candidate_set? No, S was modified.
                        # Wait, S currently is S - {old_x}.
                        # We want S - {old_x} + {new_x}.
                        S.append(new_x)
                        
                        # Update diffs
                        if side == 0:
                            A_diffs = new_diffs
                        else:
                            B_diffs = new_diffs
                        
                        improved = True
                        no_improve_count = 0
                        break
                    
                    if improved:
                        break
                
                if improved:
                    break
            
            if not improved:
                no_improve_count += 1
                if no_improve_count > 100:
                    break

        return A, B, A_diffs, B_diffs

    best_A = []
    best_B = []
    best_score = 0
    
    # Run multiple restarts
    while time.time() < deadline:
        # Random ordering
        order = list(range(1, N + 1))
        random.shuffle(order)
        
        A, B, A_diffs, B_diffs = greedy_construct(order, set(), set())
        
        # Local search
        A, B, A_diffs, B_diffs = local_search(A, B, A_diffs, B_diffs)
        
        score = len(A) + len(B)
        if score > best_score:
            best_score = score
            best_A = A
            best_B = B
            
        # If we hit the max possible (128), we can stop? 
        # Max size is 64 each. 128 total.
        if best_score >= 128:
            break

    # Ensure non-empty
    if not best_A:
        best_A = [1]
    if not best_B:
        best_B = [2]
        
    # Final validation to be safe
    # (Optional but good for robustness)
    
    return [best_A, best_B]
# EVOLVE-BLOCK-END
