"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def construct():
    """
    Construct a set of distinct integers in 1..256 such that whenever two subsets
    have different cardinalities, their sums must differ.
    
    We use a greedy approach:
    Iterate x from 1 to 256.
    If adding x to the current set A preserves the property, add it.
    The property is preserved if x is not in the set of differences between sums of different cardinalities.
    Specifically, if A has sums S_k for cardinality k, we need x not in S_i - S_j for any i != j.
    This is equivalent to checking if (S_j + x) intersects S_i for any i != j.
    """
    start_time = time.time()
    deadline = 58.0  # Safety margin
    
    # S[k] is a set of sums of k-element subsets of the current set A
    S = [set() for _ in range(20)]  # Max size 20 is safe, we'll expand if needed
    S[0] = {0}
    current_size = 0
    A = []
    
    # Pre-allocate or extend S as needed
    # We'll just use a list of sets and extend it when current_size increases
    
    for x in range(1, 257):
        if time.time() - start_time > deadline:
            break
            
        # Check if x can be added
        # Condition: For all i != j, (S[j] + x) and S[i] must be disjoint.
        # i.e., no s in S[j] such that s + x is in S[i].
        
        valid = True
        # We only need to check j from 0 to current_size
        # and i from 0 to current_size, i != j
        # Note: S[k] for k > current_size is empty, so we can ignore them.
        
        for j in range(current_size + 1):
            if not S[j]:
                continue
            # Shift S[j] by x
            # Instead of creating a new set, iterate and check membership in S[i]
            for s in S[j]:
                candidate_sum = s + x
                # Check if candidate_sum is in any S[i] for i != j
                # We can optimize by breaking early
                for i in range(current_size + 1):
                    if i == j:
                        continue
                    if candidate_sum in S[i]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                break
        
        if valid:
            # Add x to A
            A.append(x)
            current_size += 1
            
            # Update S
            # New S[k] = Old S[k] U (Old S[k-1] + x)
            # We must iterate k in reverse to avoid using updated values in the same step
            # But we need to be careful with the list size
            
            # Extend S if necessary
            while len(S) <= current_size:
                S.append(set())
                
            for k in range(current_size, 0, -1):
                # S[k] gets elements from S[k-1] + x
                # S[k] already has the old elements (sums not using x)
                # We add sums using x
                new_sums = {s + x for s in S[k-1]}
                S[k].update(new_sums)
            # S[0] remains {0}
            
            # Optional: Prune S? No, we need exact sums.
            
    return [int(x) for x in A]

# EVOLVE-BLOCK-END
