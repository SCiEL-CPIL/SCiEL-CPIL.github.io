"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    """
    Greedy construction of a set of distinct integers in 1..256 such that
    subset sums of different cardinalities are disjoint.
    """
    MAX_VAL = 256
    result = []
    used = set()
    
    # We try to add elements from 1 to 256.
    # For each candidate, we check if adding it violates the condition.
    # Condition: For any k != m, S_k (sums of size k) and S_m (sums of size m) must be disjoint.
    # This is equivalent to: No sum of a subset of size k equals a sum of a subset of size m (k!=m).
    
    # To efficiently check, we can maintain the set of all possible sums for each cardinality.
    # However, the number of subsets grows as 2^n. For n=9, 2^9=512, which is small.
    # For n=10, 2^10=1024.
    
    # Let's use a dictionary: sums_by_cardinality[k] = set of sums of size k.
    sums_by_cardinality = {0: {0}}
    current_max_cardinality = 0
    
    for candidate in range(1, MAX_VAL + 1):
        if candidate in used:
            continue
            
        # Tentatively add candidate
        # We need to compute new sums_by_cardinality
        new_sums = {}
        valid = True
        
        # Copy existing sums
        for k in range(current_max_cardinality + 1):
            new_sums[k] = set(sums_by_cardinality[k])
        
        # Add sums involving the candidate
        # A subset of size k+1 containing candidate has sum = sum_of_subset_of_size_k_from_old + candidate
        for k in range(current_max_cardinality + 1):
            new_k = k + 1
            if new_k not in new_sums:
                new_sums[new_k] = set()
            
            for s in sums_by_cardinality[k]:
                new_sum = s + candidate
                new_sums[new_k].add(new_sum)
        
        # Check for collisions between different cardinalities
        # We only need to check if any sum in new_sums[k] is in new_sums[m] for k != m.
        # Since old sums were valid, we only need to check:
        # 1. New sums (involving candidate) against old sums (not involving candidate) of different cardinality.
        # 2. New sums against other new sums of different cardinality? 
        #    New sums of size k+1 are formed from old sums of size k.
        #    Old sums of size k are in new_sums[k].
        #    So we check new_sums[k+1] against new_sums[j] for all j != k+1.
        
        for k in range(1, current_max_cardinality + 2):
            if k not in new_sums:
                continue
            sums_k = new_sums[k]
            if not sums_k:
                continue
            for m in range(0, current_max_cardinality + 2):
                if m == k:
                    continue
                if m not in new_sums:
                    continue
                sums_m = new_sums[m]
                # Check intersection
                # Optimization: Only check if intersection is non-empty
                # Since sets can be small, direct intersection is fine.
                if sums_k.intersection(sums_m):
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            used.add(candidate)
            sums_by_cardinality = new_sums
            current_max_cardinality += 1
            result.append(candidate)
            
            # If we have 10 elements, we can stop early if we assume we can't get more?
            # But let's try to get as many as possible.
            # Max possible is bounded by MAX_VAL.
    
    return [int(x) for x in result]

# EVOLVE-BLOCK-END
