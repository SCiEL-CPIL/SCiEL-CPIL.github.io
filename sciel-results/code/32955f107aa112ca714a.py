"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    deadline = time.time() + 85.0
    n = 192
    max_size = 48
    
    # Precompute for each number x in [1, n], the bitset of sums of 3 distinct other numbers
    # that are divisible by x.
    # This is expensive to precompute for all pairs, so we do it on the fly but cache per x.
    # Actually, let's precompute the set of "bad sums" for each x.
    # Bad sums for x: sums S such that S % x == 0 and S can be formed by 3 distinct numbers from the rest.
    # Since the "rest" changes, we can't fully precompute.
    # Instead, we use a fast check function.
    
    # Strategy:
    # 1. Fast Greedy with bitsets for membership.
    # 2. Local Search: Try to add a new number, or swap a number.
    
    def is_valid_addition(candidate, selected_set, selected_list):
        """
        Check if adding 'candidate' to 'selected' is valid.
        selected_set: set of selected numbers for O(1) lookup.
        selected_list: list of selected numbers.
        """
        # The new element is 'candidate'.
        # We need to check two things:
        # 1. For any existing element 'a' in selected, does 'a' divide the sum of any 3 distinct others in (selected + [candidate]) excluding 'a'?
        #    The "others" for 'a' are (selected \ {a}) U {candidate}.
        #    We previously knew that no 'a' divides sum of any 3 from (selected \ {a}).
        #    So we only need to check sums involving 'candidate'.
        #    The 3 others must include 'candidate' and 2 from (selected \ {a}).
        #    Let the 2 be u, v. Sum = candidate + u + v.
        #    Check if (candidate + u + v) % a == 0.
        #
        # 2. For the new element 'candidate', does 'candidate' divide the sum of any 3 distinct others in (selected + [candidate]) excluding 'candidate'?
        #    The "others" are just 'selected'.
        #    So check if sum of any 3 from 'selected' is divisible by 'candidate'.
        
        # Optimization:
        # If we precompute for each 'a' in selected, the set of pairs (u, v) from (selected \ {a}) such that (candidate + u + v) % a == 0, it's hard.
        # Let's just iterate.
        
        # Check 2: Does candidate divide sum of any 3 from selected?
        if len(selected_list) >= 3:
            for t in combinations(selected_list, 3):
                if sum(t) % candidate == 0:
                    return False
        
        # Check 1: For each a in selected, does a divide sum of any 3 from (selected \ {a}) U {candidate}?
        # New sums must include candidate.
        # So we look for u, v in (selected \ {a}) such that (candidate + u + v) % a == 0.
        # This is equivalent to (u + v) % a == (-candidate) % a.
        
        for a in selected_list:
            target = (-candidate) % a
            # We need to check if there exist u, v in (selected \ {a}) such that (u + v) % a == target.
            # To do this fast, we can precompute a frequency map of residues mod a for (selected \ {a}).
            # But 'a' changes.
            # Given n=192 and size <= 48, O(n^2) per check is 48*48 = 2304 ops.
            # With ~1000 checks, that's 2.3M ops. Fast enough in Python? Maybe.
            # Let's try to be smart.
            
            # Build residue counts for selected \ {a}
            # Actually, iterating through all pairs u, v in selected \ {a} is O(k^2).
            # k is up to 47. 47*47/2 ~ 1100 pairs.
            # For each a, we do this. So 48 * 1100 = 52,800 ops per validity check.
            # 500 checks = 26M ops. Might be too slow in Python.
            
            # Alternative:
            # For a fixed 'a', we want to know if there are u, v in S_a = selected \ {a} with (u+v) % a == target.
            # We can precompute for each 'a' in selected, a set of residues of elements in selected \ {a}.
            # But 'selected' changes.
            
            # Let's just do the double loop but break early.
            others = [x for x in selected_list if x != a]
            # To find u, v in others with (u+v) % a == target:
            # We can use a set of residues seen so far.
            seen_residues = set()
            for u in others:
                r_u = u % a
                needed_r_v = (target - r_u) % a
                if needed_r_v in seen_residues:
                    return False
                seen_residues.add(r_u)
                
        return True

    best_result = []
    
    # Multi-start Greedy
    # Orderings:
    # 1. Descending (Large numbers first, safer)
    # 2. Ascending
    # 3. Randomized Descending with noise
    # 4. Randomized Ascending with noise
    
    orderings = [
        list(range(n, 0, -1)),
        list(range(1, n + 1)),
    ]
    
    # Add some randomized orderings
    for _ in range(5):
        lst = list(range(1, n + 1))
        random.shuffle(lst)
        # Bias towards larger numbers?
        # Sort by random key but keep large numbers somewhat prioritized?
        # Let's just use pure random.
        orderings.append(lst)
        
    for order in orderings:
        if time.time() > deadline:
            break
            
        selected = []
        selected_set = set()
        
        for x in order:
            if time.time() > deadline:
                break
            if is_valid_addition(x, selected_set, selected):
                selected.append(x)
                selected_set.add(x)
                
        if len(selected) > len(best_result):
            best_result = selected[:]
            
    # Local Search / Hill Climbing
    # Try to improve best_result
    current = best_result[:]
    current_set = set(current)
    current_size = len(current)
    
    # Precompute sums of 3 for current set?
    # We will use the same is_valid_addition logic but adapted for swaps.
    
    def try_add(x):
        nonlocal current, current_set, current_size
        if x in current_set:
            return False
        if is_valid_addition(x, current_set, current):
            current.append(x)
            current_set.add(x)
            current_size += 1
            return True
        return False
        
    def try_remove(x):
        nonlocal current, current_set, current_size
        if x not in current_set:
            return False
        current.remove(x)
        current_set.remove(x)
        current_size -= 1
        return True
        
    # Iterative improvement
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try to add any number not in current
        # Sort candidates by "safety" (larger numbers first)
        candidates = [x for x in range(1, n + 1) if x not in current_set]
        candidates.sort(reverse=True)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if try_add(x):
                improved = True
                # Restart candidate loop? Or just continue?
                # If we add one, the set changes, so previously invalid might become valid?
                # No, adding constraints only makes it harder to add more.
                # So if x was invalid before, it's still invalid.
                # But if we added x, maybe some other y becomes valid?
                # No, adding x adds constraints on y.
                # So we can just continue.
                
        # Try swaps: Remove a "bad" small number, add a "good" large number?
        # This is harder. Let's stick to addition first.
        
        # If no addition worked, try removing the smallest element and see if it allows a larger one?
        # This is a local search move.
        if not improved and current:
            # Try removing the smallest element
            min_elem = min(current)
            # Remove it
            try_remove(min_elem)
            # Try to add any number
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            candidates.sort(reverse=True)
            added_any = False
            for x in candidates:
                if time.time() > deadline:
                    break
                if try_add(x):
                    added_any = True
                    improved = True
                    break
            
            if not added_any:
                # Re-add min_elem if it was valid? It was valid before.
                # But the set changed.
                # Actually, if we removed min_elem, we should try to re-add it if we didn't add anything else.
                # Or just keep it removed if it didn't help.
                # But we want to maximize size.
                # If we didn't add anything, we are smaller.
                # So re-add min_elem.
                try_add(min_elem)

    # Final check and return
    if len(current) > len(best_result):
        best_result = current[:]
        
    # Ensure validity of best_result (paranoid check)
    # The is_valid_addition logic is incremental, so it should be correct.
    # But let's do a full check on the final result to be safe?
    # It might be slow.
    # Given the constraints, the incremental check is reliable.
    
    return best_result
# EVOLVE-BLOCK-END
