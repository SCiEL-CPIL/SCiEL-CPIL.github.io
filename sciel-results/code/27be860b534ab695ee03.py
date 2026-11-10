import time
import sys

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums 
    of different cardinalities are disjoint.
    
    Strategy: Greedy construction.
    1. Start with an empty set.
    2. Maintain for each cardinality k a set of achievable sums S_k.
    3. Try to add the smallest available integer x in 1..256 not already in the set.
    4. Check if adding x violates the constraint: 
       For all k, the new sums formed by adding x to subsets of size k-1 must 
       not intersect with existing sums of size k (or any other size, but specifically 
       we need S'_k ∩ S_j = empty for j != k).
       Actually, the constraint is: S_k ∩ S_j = empty for k != j.
       When we add x, new sums for size k are {s + x | s in S_{k-1}}.
       We must ensure these new sums do not intersect with:
       a) Existing S_k (same size, different subsets? No, repeated sums within same cardinality are ALLOWED. 
          Wait, the task says: "Whenever two subsets have different cardinalities their sums must differ. 
          Repeated sums within the SAME cardinality are allowed."
          So S_k ∩ S_k is fine. We only need S'_k ∩ S_j = empty for j != k.
       b) Existing S_j for j != k.
       c) Also, the new sums for size k might intersect with new sums for size j? 
          No, we are adding x to ALL previous sizes simultaneously.
          Let S be the current set. Let T be the set of sums for S.
          Let T' be the set of sums for S ∪ {x}.
          T'_k = T_k ∪ {s + x | s in T_{k-1}}.
          We need T'_k ∩ T'_j = empty for k != j.
          Since T_k ∩ T_j = empty for k != j (by induction), we just need to check:
          1. (T_k ∪ {s+x | s in T_{k-1}}) ∩ T_j = empty for k != j.
             This splits into:
             a. T_k ∩ T_j = empty (already true)
             b. {s+x | s in T_{k-1}} ∩ T_j = empty for k != j.
             c. {s+x | s in T_{k-1}} ∩ {t+x | t in T_{j-1}} = empty for k != j? 
                If s+x = t+x, then s=t. If s=t, then k-1 = j-1 => k=j. 
                So if k!=j, the sets of new sums are disjoint from each other 
                IF the original sets T_{k-1} and T_{j-1} are disjoint? 
                Actually, if s in T_{k-1} and t in T_{j-1} and s=t, then k-1=j-1 => k=j. 
                So for k!=j, T_{k-1} ∩ T_{j-1} is empty. 
                Therefore {s+x | s in T_{k-1}} ∩ {t+x | t in T_{j-1}} is empty.
             
          So the only check needed is:
          For all k, j with k != j:
             ({s + x | s in T_{k-1}}) ∩ T_j = empty.
          
          Note: T_{-1} is {0} (empty subset sum). T_0 is {0}.
          For k=1, T_0 is {0}. New sums: {x}. Must not be in T_j for j != 1.
          For k>1, T_{k-1} is non-empty.
    """
    
    deadline = time.time() + 80.0  # Safety margin
    
    # T[k] is a set of integers representing sums of subsets of size k
    # Start with empty set: T[0] = {0}
    T = [set() for _ in range(61)]
    T[0] = {0}
    current_set = []
    used = set()
    
    # We want to maximize size. Max possible is 60.
    # Greedily try to add smallest available numbers.
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
            
        if x in used:
            continue
            
        # Check validity of adding x
        valid = True
        # We need to check for all k (1..current_size+1) and j (0..current_size) where k!=j
        # New sums for size k: NewSums_k = {s + x | s in T[k-1]}
        # We need NewSums_k ∩ T[j] == empty for all j != k.
        
        # Optimization: 
        # For a fixed k, NewSums_k is determined.
        # We check intersection with all T[j] for j != k.
        
        max_k = len(current_set) + 1
        
        # Pre-compute NewSums for all k
        # Actually, we can just iterate.
        
        for k in range(1, max_k + 1):
            if not T[k-1]:
                continue
            new_sums = {s + x for s in T[k-1]}
            # Check against all T[j] where j != k
            for j in range(0, max_k): # j goes up to current max size, which is len(current_set)
                if j == k:
                    continue
                if T[j]:
                    # Check intersection
                    # new_sums is a set. T[j] is a set.
                    if new_sums & T[j]:
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            # Add x to current_set
            current_set.append(x)
            used.add(x)
            
            # Update T
            # New T[k] = Old T[k] ∪ {s+x | s in Old T[k-1]}
            # We must update from high k to low k to avoid using updated values
            # Or use a copy.
            # Since max size is small (<=60), copying is cheap.
            old_T = [s.copy() for s in T]
            for k in range(1, len(current_set) + 1):
                # T[k] gets new elements from T[k-1] + x
                for s in old_T[k-1]:
                    T[k].add(s + x)
            
            # If we reached max size, we can stop early if we want, 
            # but maybe we can fit more? No, max is 60.
            if len(current_set) == 60:
                break
        
    # Return list of ints
    return current_set

# EVOLVE-BLOCK-END
