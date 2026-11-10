import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s safety margin

    def is_valid_sidon(arr):
        arr_set = set(arr)
        if len(arr_set) != len(arr):
            return False
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                b = arr[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True

    def is_maximal(arr, sums):
        arr_set = set(arr)
        for x in range(1, N + 1):
            if x in arr_set:
                continue
            conflict = False
            for a in arr:
                if (x + a) in sums:
                    conflict = True
                    break
            if not conflict:
                if (2 * x) in sums:
                    conflict = True
            if not conflict:
                return False
        return True

    def get_sums(arr):
        sums = set()
        for i in range(len(arr)):
            a = arr[i]
            for j in range(i, len(arr)):
                sums.add(a + arr[j])
        return sums

    # Strategy: 
    # 1. Pick a small random set of "anchors" (size k).
    # 2. Check if anchors are Sidon. If not, restart.
    # 3. Greedily add elements to the set to ensure maximality.
    #    An element x is "blocked" if adding it would violate Sidon.
    #    If x is NOT blocked (i.e., it can be added), we MUST add it to maintain maximality?
    #    No, that would make the set larger.
    #    Wait, maximality means NO element can be added.
    #    So, if we start with a small set S, and there exists x not in S that CAN be added, S is not maximal.
    #    To make it maximal, we have two choices:
    #    A) Add x to S (increases size).
    #    B) Modify S such that x cannot be added (might be hard).
    #    
    #    The "greedy fill" approach:
    #    Start with empty set? No, start with random anchors.
    #    Iterate through candidates x in [1, N].
    #    If x is in S, skip.
    #    Check if x can be added to S (Sidon check).
    #    If x CAN be added, then S is not maximal with respect to x.
    #    To fix this, we have to add x? 
    #    If we add every x that can be added, we get a maximal set.
    #    This is essentially the "greedy saturation" from the current program, but starting from anchors.
    #    
    #    Does starting from anchors help?
    #    If anchors are well-chosen, they might block many x's naturally.
    #    Then we only need to add a few more to block the rest.
    #    The final size is |anchors| + |filled_elements|.
    #    
    #    Let's try to minimize |anchors| + |filled_elements|.
    
    best_set = None
    best_size = N + 1
    
    rng = random.Random(42)

    # Try different anchor sizes
    # If anchor size is too small, we might need to add many elements.
    # If anchor size is too large, we start with a large set.
    # Optimal is somewhere in between.
    
    # Heuristic: Anchors should be spread out to maximize sum coverage.
    # Random sampling is a decent proxy.
    
    for k_anchors in range(1, 15):
        if time.time() > deadline:
            break
            
        found_in_k = False
        attempts = 0
        max_attempts = 100 # Increase attempts for smaller k
        
        while attempts < max_attempts and time.time() < deadline:
            attempts += 1
            
            # Generate random anchors
            anchors = rng.sample(range(1, N + 1), k_anchors)
            
            # Check if anchors form a Sidon set
            if not is_valid_sidon(anchors):
                continue
                
            # Start with anchors
            current_set = list(anchors)
            sums = get_sums(current_set)
            
            # Greedily fill to ensure maximality
            # We need to ensure that for every x not in current_set, x cannot be added.
            # "Cannot be added" means: adding x creates a duplicate sum.
            # Duplicate sum s = x + a = b + c (with a,b,c in current_set, a<=b, c<=d? No, just sums).
            # Actually, x can be added if {x+a for a in S} U {2x} is disjoint from Sums(S).
            # If it is NOT disjoint, x is blocked.
            # If it IS disjoint, x is NOT blocked.
            # If x is not blocked, we MUST add it to make the set maximal?
            # Yes, because if we don't add it, the set is not maximal (x can be added).
            # So, we iterate through all x in [1, N] \ S.
            # If x can be added, we add it.
            
            # Optimization: Iterate in a specific order?
            # Adding x changes Sums(S), which might block or unblock other y's?
            # Adding x adds new sums. It can only BLOCK other elements (by creating new collisions), never unblock them.
            # So the order of addition doesn't change the final set?
            # Let's verify:
            # If x is addable to S, and y is addable to S.
            # Add x -> S' = S U {x}.
            # Is y addable to S'?
            # If y was addable to S, {y+a} U {2y} disjoint from Sums(S).
            # Sums(S') = Sums(S) U {x+a for a in S} U {2x}.
            # If {y+a} U {2y} intersects {x+a' for a' in S} U {2x}, then y is NOT addable to S'.
            # So adding x might block y.
            # Therefore, the final set depends on the order of addition!
            # To minimize size, we want to add elements that block as many others as possible.
            # This is a Set Cover / Dominating Set problem on the "addability" graph.
            
            # Greedy Heuristic for filling:
            # At each step, find the x that, if added, blocks the most other candidates.
            # Or simpler: just add the smallest x that is addable?
            # Or random order?
            
            # Let's try: Add the smallest addable x.
            # This might be slow if we check all x every time.
            # N=256 is small. 256 * 256 * 256 is manageable?
            # 256 candidates. For each, check 256 sums. 256*256 = 65k ops per step.
            # Max steps 256. Total ~16M ops. In Python, this might take ~1-2 seconds per attempt.
            # We have 55s budget. 100 attempts * 2s = 200s. Too slow.
            
            # We need a faster check or fewer attempts.
            # Let's limit attempts for larger k_anchors.
            
            # Fast check for "is x addable":
            # Precompute Sums(S).
            # For x, check if (x+a) in Sums or 2x in Sums for any a in S.
            
            # Let's try a simpler greedy:
            # Iterate x from 1 to N.
            # If x not in S:
            #   Check if x is addable.
            #   If yes, add x.
            
            # This is deterministic given the anchors.
            # Let's run this and see the size.
            
            # To improve, we can try different orders of filling (e.g., random, descending).
            # But let's start with ascending (smallest first) as it might create small sums that block others.
            
            # Reset for filling
            current_set = list(anchors)
            sums = get_sums(current_set)
            in_set = set(current_set)
            
            for x in range(1, N + 1):
                if x in in_set:
                    continue
                
                # Check if x is addable
                addable = True
                for a in current_set:
                    if (x + a) in sums:
                        addable = False
                        break
                if addable:
                    if (2 * x) in sums:
                        addable = False
                
                if addable:
                    # Add x
                    current_set.append(x)
                    in_set.add(x)
                    # Update sums
                    for a in current_set[:-1]: # Exclude x itself? No, sums are a+b with a<=b.
                        # When adding x, new sums are x+a for all a in old S, and 2x.
                        # Wait, my loop `for a in current_set` includes x if I'm not careful.
                        # Let's just recompute sums? No, too slow.
                        # Update:
                        pass # Placeholder, let's do it properly
                    
                    # Proper update:
                    # New sums are {x + a for a in S_old} U {2x}
                    # S_old is current_set before adding x.
                    # Let's keep a list of elements.
                    
            # The above loop is flawed because updating sums incrementally is tricky if we don't store S_old.
            # Let's just recompute sums after each addition?
            # Size of S grows. Max 256.
            # Recomputing sums is O(|S|^2).
            # Total cost sum_{i=1}^{M} i^2 ~ M^3.
            # M ~ 20-50. 50^3 = 125,000. Very fast.
            
            # Let's redo the filling logic cleanly.
            current_set = list(anchors)
            in_set = set(current_set)
            
            for x in range(1, N + 1):
                if x in in_set:
                    continue
                
                # Check if x is addable to current_set
                sums = get_sums(current_set)
                addable = True
                for a in current_set:
                    if (x + a) in sums:
                        addable = False
                        break
                if addable and (2 * x) in sums:
                    addable = False
                
                if addable:
                    current_set.append(x)
                    in_set.add(x)
            
            # Now current_set is maximal (by construction, we added all addable elements in ascending order).
            # Is it Sidon? Yes, we only added if addable.
            # Verify validity
            if is_valid_sidon(current_set):
                if len(current_set) < best_size:
                    best_size = len(current_set)
                    best_set = current_set[:]
                    found_in_k = True
                    # If we find a very small one, we can break early?
                    # Theoretical lower bound is ~sqrt(N) / something?
                    # Let's just keep searching for smaller k_anchors.
        
        # If we found a good set for this k, we might want to try to reduce it?
        # No, the filling process is greedy. 
        # Different anchor sets might yield different final sizes.
        # The loop above tries `max_attempts` random anchor sets for each k_anchors.
        
        # If we found a set, do we break out of k_anchors loop?
        # No, larger k_anchors might lead to smaller final sets if they block more efficiently.
        # But usually, starting small is better for minimization.
        # However, if k_anchors=1, we might fill in 100 elements.
        # If k_anchors=10, we might fill in 5 elements. Total 15.
        # So we must check all k_anchors.

    if best_set is None:
        # Fallback: Greedy from empty
        current_set = []
        in_set = set()
        for x in range(1, N + 1):
            sums = get_sums(current_set)
            addable = True
            for a in current_set:
                if (x + a) in sums:
                    addable = False
                    break
            if addable and (2 * x) in sums:
                addable = False
            if addable:
                current_set.append(x)
                in_set.add(x)
        best_set = current_set

    return best_set

# EVOLVE-BLOCK-END
