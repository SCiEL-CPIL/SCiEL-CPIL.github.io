import time
import sys

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    # Best valid set found so far
    # Powers of 2 are a known valid set
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    def compute_sums_by_cardinality(s):
        """
        Returns a list of bitsets. sums_by_k[k] is a bitset where bit i is set 
        if sum i is achievable by a subset of cardinality k.
        """
        n = len(s)
        sums_by_k = [0] * (n + 1)
        sums_by_k[0] = 1  # bit 0 is set
        
        for v in s:
            # Update in reverse order to avoid using the same element twice
            for k in range(n, 0, -1):
                sums_by_k[k] |= (sums_by_k[k-1] << v)
                
        return sums_by_k

    def check_validity(sums_by_k):
        """
        Check if there are any cross-cardinality collisions.
        Returns True if valid (no collisions).
        """
        n = len(sums_by_k) - 1
        for i in range(n + 1):
            for j in range(i + 1, n + 1):
                if sums_by_k[i] & sums_by_k[j]:
                    return False
        return True

    def add_element(sums_by_k, v):
        """
        Returns the new sums_by_k after adding element v, or None if invalid.
        """
        n = len(sums_by_k) - 1
        new_sums_by_k = [0] * (n + 2)
        
        # Copy existing sums
        for k in range(n + 1):
            new_sums_by_k[k] = sums_by_k[k]
        
        # Add new sums involving v
        # Subsets of size k+1 in new set = (subsets of size k in old set) + v
        for k in range(n, -1, -1):
            new_sums_by_k[k+1] |= (sums_by_k[k] << v)
            
        # Check validity
        if not check_validity(new_sums_by_k):
            return None
            
        return new_sums_by_k

    def backtrack(current_sums, current_set, candidates):
        nonlocal best_set, best_size, deadline, start_time
        
        if time.time() > deadline:
            return
            
        # Pruning: if even adding all remaining candidates doesn't beat best_size
        if len(current_set) + len(candidates) <= best_size:
            return

        if len(current_set) > best_size:
            best_set = list(current_set)
            best_size = len(current_set)
            
        if not candidates:
            return
            
        for i in range(len(candidates)):
            v = candidates[i]
            
            new_sums = add_element(current_sums, v)
            if new_sums is not None:
                backtrack(new_sums, current_set + [v], candidates[i+1:])
                
            if time.time() > deadline:
                return

    # Initialize sums for empty set
    initial_sums = [1]  # sums_by_k[0] = 1 (bit 0 set)
    
    # Candidates: 256 down to 1
    # Trying larger numbers first is a strong heuristic for this problem
    candidates = list(range(256, 0, -1))
    
    # Start backtracking
    backtrack(initial_sums, [], candidates)
    
    # Ensure types are correct
    return [int(x) for x in best_set]

if __name__ == "__main__":
    t0 = time.time()
    result = construct()
    print(f"Result: {len(result)} elements, took {time.time()-t0:.2f}s")
    print(f"Set: {result}")
