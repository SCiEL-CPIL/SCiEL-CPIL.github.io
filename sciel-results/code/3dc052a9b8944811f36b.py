import time
import itertools

def construct():
    """
    Construct a set of distinct integers in 1..256 such that subset sums
    of different cardinalities are disjoint.
    
    Strategy: Greedy construction with exact cardinality-indexed subset-sum tracking.
    1. Start with empty set.
    2. Maintain for each cardinality k a set of achievable sums S_k.
    3. Try to add the smallest available integer x in 1..256 not already in the set.
    4. Check if adding x violates the constraint:
       For all k >= 1, new sums formed by adding x to subsets of size k-1 must
       not intersect with existing sums of size j for any j != k.
       Actually, the constraint is: S_k ∩ S_j = empty for k != j.
       When we add x, new sums for size k are {s + x : s in S_{k-1}}.
       We need these new sums to not be in S_j for any j != k.
       Also, existing S_k must not contain s+x for any s in S_{k-1} where the new sum would be in S_j for j != k.
       
       More precisely: After adding x, the new sum sets are:
       S'_0 = S_0 = {0}
       S'_k = S_k ∪ {s + x : s in S_{k-1}} for k >= 1
       
       We need S'_k ∩ S'_j = empty for all k != j.
       Since S_k ∩ S_j = empty was already satisfied, we only need to check:
       1. {s + x : s in S_{k-1}} ∩ S_j = empty for all j != k (and j != k-1 is not relevant since S_{k-1} is separate)
          Wait, we need S'_k ∩ S'_j = empty.
          S'_k = S_k ∪ New_k where New_k = {s+x : s in S_{k-1}}
          S'_j = S_j ∪ New_j where New_j = {t+x : t in S_{j-1}}
          
          So we need:
          - S_k ∩ S_j = empty (already true)
          - S_k ∩ New_j = empty for k != j
          - New_k ∩ S_j = empty for k != j
          - New_k ∩ New_j = empty for k != j
          
          But since x is positive and all elements are positive, the sums are distinct in many cases.
          Let's just compute the new sets and check for intersections.
    """
    start_time = time.time()
    deadline = start_time + 85  # 85 second deadline with safety margin
    
    # S[k] is the set of subset sums of cardinality k
    S = [set() for _ in range(61)]
    S[0] = {0}
    
    current_set = []
    used = set()
    
    for x in range(1, 257):
        if time.time() > deadline:
            break
        
        # Try to add x
        # Compute new sums
        new_S = [set() for _ in range(len(S))]
        new_S[0] = {0}
        
        valid = True
        for k in range(1, len(S)):
            # New sums of size k: old S[k] plus {s + x : s in S[k-1]}
            new_S[k] = set(S[k])
            if S[k-1]:
                new_S[k].update(s + x for s in S[k-1])
        
        # Check disjointness: new_S[k] ∩ new_S[j] = empty for k != j
        for k in range(len(new_S)):
            for j in range(k+1, len(new_S)):
                if new_S[k] & new_S[j]:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            S = new_S
            current_set.append(x)
            used.add(x)
    
    return current_set
