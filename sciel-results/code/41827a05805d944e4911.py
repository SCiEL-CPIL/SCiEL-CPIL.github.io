import random
import time

def construct():
    N = 512
    # Precompute all 4-APs
    # AP is (a, a+d, a+2d, a+3d)
    # a >= 1, a+3d <= N
    # d >= 1
    aps = []
    # index: element -> list of AP indices
    elem_to_aps = {i: [] for i in range(1, N+1)}
    
    ap_id = 0
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps.append((a, b, c, e))
            for x in (a, b, c, e):
                elem_to_aps[x].append(ap_id)
            ap_id += 1
    
    total_aps = len(aps)
    
    best_set = set(range(1, N+1))
    best_size = 0
    
    # Run multiple restarts of the greedy removal + add-back
    for _ in range(20): # 20 restarts
        S = set(range(1, N+1))
        
        # Track for each AP how many of its elements are in S
        # Initially all are in S, so count is 4 for all
        ap_count = [4] * total_aps
        
        # Count for each element how many active APs (count==4) it is in
        # This helps pick the best element to remove
        elem_active_count = {i: 0 for i in range(1, N+1)}
        
        # Initialize elem_active_count
        for ap_idx in range(total_aps):
            a, b, c, e = aps[ap_idx]
            elem_active_count[a] += 1
            elem_active_count[b] += 1
            elem_active_count[c] += 1
            elem_active_count[e] += 1
        
        # Greedy Removal Phase
        # We want to remove elements to break all APs.
        # At each step, remove the element in the most active APs.
        # To avoid O(N^2) scanning for max, we can just scan N elements (512 is small).
        
        while True:
            # Find element with max active AP count
            max_count = 0
            candidate = -1
            # Randomize tie-breaking
            items = list(range(1, N+1))
            random.shuffle(items)
            for x in items:
                if x in S:
                    if elem_active_count[x] > max_count:
                        max_count = elem_active_count[x]
                        candidate = x
            
            if max_count == 0:
                break
            
            # Remove candidate
            S.remove(candidate)
            
            # Update counts for APs containing candidate
            for ap_idx in elem_to_aps[candidate]:
                if ap_count[ap_idx] > 0:
                    # Decrement count for this AP
                    ap_count[ap_idx] -= 1
                    
                    # If this AP was active (count was 4) and is no longer active (count < 4)
                    # We need to decrement elem_active_count for the other 3 elements
                    if ap_count[ap_idx] == 3:
                        a, b, c, e = aps[ap_idx]
                        for y in (a, b, c, e):
                            if y != candidate:
                                elem_active_count[y] -= 1
                    # If count becomes 0, it's dead, no further updates needed for active status
                    # But we already handled the transition from 4->3 which is the only one that affects "active" status for removal logic?
                    # Wait. "Active" means count == 4.
                    # If count was 4 and becomes 3, it is no longer active.
                    # The elements in this AP (except the removed one) had their active_count decremented.
                    # This is correct.
                    
            # Note: We don't need to update if ap_count was < 4 before, because it wasn't active.
            # But what if we remove an element from an AP that was already broken (count < 4)?
            # ap_count decreases, but it doesn't affect elem_active_count because it wasn't active.
            # So the logic holds.

        # Add-back Phase
        # Try to add elements back that don't create new APs
        # We need to check if adding x creates an AP.
        # An AP is created if there exists an AP containing x such that the other 3 elements are in S.
        
        candidates = [x for x in range(1, N+1) if x not in S]
        random.shuffle(candidates)
        
        for x in candidates:
            can_add = True
            for ap_idx in elem_to_aps[x]:
                a, b, c, e = aps[ap_idx]
                # Check if the other 3 are in S
                # We can use a set for S for O(1) lookup, but S is a set already.
                # However, we need to be careful: if we add x, we modify S.
                # But we are just checking.
                
                # Get the other 3
                others = [y for y in (a, b, c, e) if y != x]
                if all(y in S for y in others):
                    can_add = False
                    break
            
            if can_add:
                S.add(x)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = set(S)
            
        # Optional: Local search / Simulated Annealing to improve best_set
        # Given time constraints, the greedy + add-back is likely sufficient for a jump.
        # But let's add a simple hill climbing on the best_set found so far.
        
    # Final Polish: Try to add any remaining elements
    # The add-back loop above already did this, but maybe we missed some due to order?
    # Re-run add-back on best_set
    S = best_set
    candidates = [x for x in range(1, N+1) if x not in S]
    random.shuffle(candidates)
    for x in candidates:
        can_add = True
        for ap_idx in elem_to_aps[x]:
            a, b, c, e = aps[ap_idx]
            others = [y for y in (a, b, c, e) if y != x]
            if all(y in S for y in others):
                can_add = False
                break
        if can_add:
            S.add(x)
            
    return sorted(S)

# EVOLVE-BLOCK-START
# (The code above is the complete program logic, but we need to wrap it in the correct structure)
# The prompt asks to rewrite the program. The current program has EVOLVE-BLOCK-START/END.
# I will place the entire logic inside the EVOLVE-BLOCK or replace the whole file content.
# The instruction says: "Rewrite the program from scratch... Keep the program's entry-point function name(s)... Output the COMPLETE program in a single python code block."
# So I will output the whole file.
# EVOLVE-BLOCK-END
