"""Program to construct a set of 12 distinct positive integers with distinct subset sums, minimizing the largest element."""
import time

# EVOLVE-BLOCK-START
def construct():
    """
    Greedy minimal extension followed by local optimization to minimize the largest element.
    """
    deadline = time.time() + 75  # safety margin before 90s kill

    # Start with the smallest possible: 1
    elements = [1]
    # All subset sums of current elements (start with empty subset sum = 0)
    subset_sums = {0, 1}

    for _ in range(11):  # need 12 total, already have 1
        candidate = 2  # smallest unused positive integer (1 is taken)
        while True:
            # Check if adding 'candidate' creates any collision
            has_collision = False
            for s in subset_sums:
                if (s + candidate) in subset_sums:
                    has_collision = True
                    break

            if not has_collision:
                break

            candidate += 1
            # Deadline check
            if time.time() > deadline:
                return [1 << i for i in range(12)]

        # Add candidate
        elements.append(candidate)
        # Update subset sums
        new_sums = set()
        for s in subset_sums:
            new_sums.add(s + candidate)
        subset_sums = subset_sums | new_sums

    # Local optimization: try to reduce the largest element
    # Try replacing the largest element with a smaller unused value
    best_elements = elements[:]
    best_max = max(elements)
    
    # Sort elements for easier manipulation
    elements_sorted = sorted(elements)
    
    # Try to reduce the largest element
    for target_idx in range(len(elements_sorted) - 1, -1, -1):
        if time.time() > deadline:
            break
            
        current_val = elements_sorted[target_idx]
        
        # Try all smaller values that aren't already used
        used = set(elements_sorted)
        for new_val in range(1, current_val):
            if new_val in used:
                continue
            
            # Check if replacing current_val with new_val maintains distinct subset sums
            # Remove current_val's contribution from subset_sums
            # This is complex, so we'll rebuild from scratch for correctness
            
            # Create new element list
            new_elements = elements_sorted[:]
            new_elements[target_idx] = new_val
            new_elements = sorted(new_elements)
            
            # Verify all subset sums are distinct
            new_subset_sums = {0}
            valid = True
            for elem in new_elements:
                new_sums = set()
                for s in new_subset_sums:
                    ns = s + elem
                    if ns in new_subset_sums:
                        valid = False
                        break
                    new_sums.add(ns)
                if not valid:
                    break
                new_subset_sums = new_subset_sums | new_sums
            
            if valid and len(new_subset_sums) == 4096:
                # Found a better solution
                if new_val < best_max:
                    best_max = new_val
                    best_elements = new_elements[:]
                    elements_sorted = new_elements[:]
                    break  # Restart from this improved state
    
    return best_elements
# EVOLVE-BLOCK-END
