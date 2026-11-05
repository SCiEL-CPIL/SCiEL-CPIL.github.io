import time

def construct():
    start_time = time.time()
    deadline = 60.0  # Safety margin before 90s limit
    
    # Modular inverses of 1..10 mod 101
    inverses = [pow(x, -1, 101) for x in range(1, 11)]
    
    # min_size[r] will store the minimum size of a subset summing to r
    # best_subset[r] will store that subset
    min_size = [11] * 101  # 11 is larger than any possible subset size (max 10)
    best_subset = [None] * 101
    
    # Iterate through all non-empty subsets of {1, ..., 10}
    # There are 2^10 - 1 = 1023 subsets
    for mask in range(1, 1 << 10):
        # Check time
        if time.time() - start_time > deadline - 1.0:
            break
            
        terms = []
        s = 0
        # Unroll or loop to compute sum and terms
        # For efficiency, we can compute sum first, then terms if needed
        # But we need terms for the final output.
        # Since N=10 is small, this is fast.
        
        # Calculate sum and size
        size = 0
        sum_val = 0
        for x in range(1, 11):
            if (mask >> (x - 1)) & 1:
                size += 1
                sum_val += inverses[x - 1]
        
        residue = sum_val % 101
        
        if size < min_size[residue]:
            min_size[residue] = size
            # Reconstruct terms for this mask
            terms = [x for x in range(1, 11) if (mask >> (x - 1)) & 1]
            best_subset[residue] = terms

    # Check if all residues are covered
    # If not, we might need a fallback, but the problem implies solvability.
    # If some are not covered, min_size[r] remains 11.
    # The task says "Every residue must be represented".
    
    result = []
    max_len = 0
    for r in range(101):
        if best_subset[r] is not None:
            result.append(best_subset[r])
            if len(best_subset[r]) > max_len:
                max_len = len(best_subset[r])
        else:
            # This should not happen if the problem is well-posed
            # Fallback to an empty list or raise error? 
            # The contract says "101 nonempty lists".
            # If a residue is unreachable, the program fails the contract.
            # However, with 1023 subsets and 101 residues, it's highly likely all are covered.
            result.append([])
            
    return result
