import time

def construct():
    """
    Returns a set of n=12 distinct positive integers such that all subset sums are distinct.
    Uses a greedy construction to minimize the largest element.
    """
    n = 12
    start_time = time.time()
    deadline = start_time + 80.0  # Safety margin before 90s limit
    
    # Greedy construction
    # Start with the smallest possible positive integer
    # At each step, add the smallest positive integer not yet used
    # that preserves distinct subset sums
    
    # Current set of numbers
    current_set = []
    # All subset sums of current_set
    # Start with just the empty subset sum (0)
    subset_sums = {0}
    used_numbers = set()
    
    for i in range(n):
        # Try numbers starting from 1, increasing
        # But we can be smarter: we need to find the smallest number
        # that doesn't create duplicate subset sums
        
        # The new subset sums when adding x are:
        # old_sums ∪ {s + x for s in old_sums}
        # We need all these to be distinct
        
        # Start from 1 and go up
        # But we can estimate a lower bound:
        # If we have k numbers with max sum S, we need x > S - min_possible_new_sum
        # Actually, the simplest approach: just try from 1 upward
        
        # Optimization: we can start from a reasonable lower bound
        # The sum of all current elements is max_sum
        # New elements will be in [1, max_sum + new_element]
        # To avoid collision, we need new_element such that:
        # {s + new_element for s in subset_sums} ∩ subset_sums = ∅
        # i.e., new_element not in {s1 - s2 for s1, s2 in subset_sums}
        
        # For small sets, just iterate from 1
        # But for efficiency, we can use the fact that:
        # If max(current_set) = M, then all subset sums are in [0, sum(current_set)]
        # The new sums will be in [new_element, sum(current_set) + new_element]
        # Collision occurs if there exist s1, s2 in subset_sums with s1 - s2 = new_element
        # i.e., new_element = s1 - s2 for some s1 > s2
        
        # So we need new_element NOT in {s1 - s2 for s1, s2 in subset_sums, s1 > s2}
        
        # Let's compute the set of "forbidden" differences
        # But this could be large. For small n, it's fine.
        
        # Alternative: just try from 1 upward and check
        
        # For efficiency with n=12, let's use a smarter approach
        # The greedy algorithm for distinct subset sums is known to produce
        # the "Stanley sequence" or similar
        
        # Let's just do the simple greedy: try from 1 upward
        # But we can start from a lower bound
        
        # Lower bound: if current max is M, we might need to go higher
        # Actually, the optimal greedy produces: 1, 2, 4, 8, ... (powers of 2)
        # But let's verify if a different greedy order gives better results
        
        # Let's implement the simple greedy with pruning
        # Try candidate from 1 upward
        # But we can skip: if candidate <= max(current_set) and candidate is not in used_numbers
        # we still need to check
        
        # Actually, let's just try from 1 upward, but with a time check
        
        found = False
        candidate = 1
        while not found:
            if time.time() > deadline:
                # Return best effort: just use powers of 2
                return [1 << i for i in range(n)]
            
            if candidate not in used_numbers:
                # Check if adding candidate preserves distinct subset sums
                # New subset sums = subset_sums ∪ {s + candidate for s in subset_sums}
                # These must all be distinct
                
                # Check for collision:
                # A collision occurs if there exist s1, s2 in subset_sums
                # such that s1 + candidate = s2 (impossible if candidate > 0 and s1 >= 0, s2 >= 0, unless s1=0 and s2=candidate)
                # Or s1 = s2 + candidate
                # More generally, we need:
                # {s + candidate for s in subset_sums} ∩ subset_sums = ∅
                
                # So we need: for all s in subset_sums, s + candidate not in subset_sums
                # Equivalently: candidate not in {s1 - s2 for s1, s2 in subset_sums}
                
                # Let's check directly
                valid = True
                for s in subset_sums:
                    if s + candidate in subset_sums:
                        valid = False
                        break
                
                if valid:
                    # Add candidate
                    current_set.append(candidate)
                    used_numbers.add(candidate)
                    # Update subset sums
                    new_sums = set()
                    for s in subset_sums:
                        new_sums.add(s + candidate)
                    subset_sums.update(new_sums)
                    found = True
            candidate += 1
    
    return current_set
