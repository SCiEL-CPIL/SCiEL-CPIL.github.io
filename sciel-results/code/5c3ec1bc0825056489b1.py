"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin before 60s limit

    def compute_sums(elements):
        """
        Returns a list of bitsets (integers) where bit s is set if sum s is achievable
        with the given cardinality.
        """
        n = len(elements)
        max_sum = sum(elements)
        # Use Python integers as bitsets. Bit i represents sum i.
        sums = [0] * (n + 1)
        sums[0] = 1  # Bit 0 set (sum 0)
        
        for val in elements:
            # Iterate backwards to avoid using val multiple times
            for k in range(n - 1, -1, -1):
                if sums[k]:
                    # Shift sums[k] by val bits
                    sums[k + 1] |= (sums[k] << val)
        
        return sums

    def count_conflicts(sums):
        """
        Returns the total number of cross-cardinality conflicts.
        A conflict is a sum s that is achievable by two different cardinalities.
        """
        conflicts = 0
        n = len(sums) - 1
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] and sums[j]:
                    # Count bits in intersection
                    inter = sums[i] & sums[j]
                    if inter:
                        # Popcount
                        conflicts += bin(inter).count('1')
        return conflicts

    def is_valid(sums):
        n = len(sums) - 1
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums[i] & sums[j]:
                    return False
        return True

    # Start with a random set of size 50
    # We want to maximize size, so start high and remove
    best_set = []
    best_conflicts = float('inf')
    
    # Try multiple random starts
    random.seed(42)
    for attempt in range(20):
        if time.time() > deadline:
            break
            
        # Random subset of 1..256
        # Bias towards smaller numbers? Smaller numbers create denser sums, 
        # which might mean fewer distinct sums, but more collisions?
        # Let's try uniform first.
        elements = random.sample(range(1, 257), 50)
        
        current_conflicts = float('inf')
        while time.time() < deadline - 5:
            sums = compute_sums(elements)
            current_conflicts = count_conflicts(sums)
            
            if current_conflicts == 0:
                # Valid set found
                if len(elements) > len(best_set):
                    best_set = elements[:]
                break
            
            # Find element to remove
            # Try removing each element and see if conflicts decrease
            best_elem_to_remove = -1
            min_conflicts_after_removal = current_conflicts
            
            # Optimization: Only check a subset of candidates if N is large?
            # With N=50, checking all 50 is fine.
            for idx, val in enumerate(elements):
                # Temporarily remove
                new_elements = elements[:idx] + elements[idx+1:]
                new_sums = compute_sums(new_elements)
                new_conflicts = count_conflicts(new_sums)
                
                if new_conflicts < min_conflicts_after_removal:
                    min_conflicts_after_removal = new_conflicts
                    best_elem_to_remove = idx
            
            if best_elem_to_remove == -1:
                # No single removal helps? 
                # This might happen if conflicts are distributed.
                # Force remove the largest element?
                # Or break and try a new random start?
                # Let's force remove the largest element to reduce sum range.
                max_idx = max(range(len(elements)), key=lambda i: elements[i])
                best_elem_to_remove = max_idx
                # If even that doesn't help, we might be stuck.
                # But removing reduces N, which reduces potential collisions.
                
            elements.pop(best_elem_to_remove)
            
        if len(elements) > len(best_set):
            best_set = elements[:]

    # If best_set is empty or small, fallback to greedy
    if len(best_set) < 10:
        # Fallback to the previous greedy logic or simple power of 2
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
        
    # Final validation
    sums = compute_sums(best_set)
    if not is_valid(sums):
        # Fallback to known valid set
        best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]

    return best_set
# EVOLVE-BLOCK-END
