from fractions import Fraction
import random
import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # Safety margin
    
    def is_dissociated(S):
        if not S:
            return True
        # Use a set of Fractions
        sums = set()
        n = len(S)
        # Iterate over all 2^n subsets
        # For small n (<= 15), this is fine.
        # 2^15 = 32768.
        # Fraction addition is somewhat slow, but 32768 * 15 ops is ~500k ops.
        # Might be too slow if done in a tight loop with many restarts.
        
        # Optimization: 
        # We can check collisions incrementally or use a faster check.
        # But for now, let's stick to exactness and see if it's fast enough.
        # If S has size k, we compute 2^k sums.
        
        # Actually, we can just generate all subset sums and check for duplicates.
        # To make it faster, we can use a list and sort? No, set is better for dup check.
        
        # Let's optimize the sum generation.
        # Instead of iterating 0 to 2^n-1 and summing from scratch,
        # we can use DP: sums[i] = sums[i-1] + val[i]? No, that's for contiguous.
        # Standard subset sum enumeration:
        # current_sum = 0
        # for mask in range(1 << n):
        #    ...
        
        # Let's try a recursive/iterative generation that is efficient.
        # Or simply:
        all_sums = [Fraction(0, 1)]
        for x in S:
            new_sums = []
            val = Fraction(1, x)
            for s in all_sums:
                new_sums.append(s + val)
            all_sums.extend(new_sums)
            
        # Check for duplicates
        if len(all_sums) != len(set(all_sums)):
            return False
        return True

    # Initial set from seed
    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    best_size = len(best_set)
    
    # Universe of numbers
    universe = list(range(1, 23))
    
    # Strategy 1: Greedy addition to existing best
    # Try to add any element not in best_set
    # If it fails, try swapping
    
    # Strategy 2: Randomized Greedy Construction
    # Start with empty, add random elements if valid
    
    # Strategy 3: Local Search
    # Swap elements to try to increase size or maintain size while enabling further additions
    
    def try_extend(S):
        # Try to add elements from universe not in S
        # Sort candidates by value? Or random?
        # Adding smaller numbers is harder (more collisions).
        # Adding larger numbers is easier.
        # So maybe try adding largest first?
        candidates = [x for x in universe if x not in S]
        # Sort by value descending to try "safe" large numbers first
        candidates.sort(reverse=True)
        
        current_S = list(S)
        for x in candidates:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated(new_S):
                current_S = new_S
        return current_S

    # Try extending the seed
    extended_seed = try_extend(best_set)
    if len(extended_seed) > best_size:
        best_set = extended_seed
        best_size = len(best_set)

    # Randomized Search
    random.seed(42)
    
    for i in range(1000): # Max iterations, but bounded by time
        if time.time() > deadline:
            break
            
        # Random start
        # Pick a random subset of size k
        # Or start with a random permutation and greedily add
        perm = universe.copy()
        random.shuffle(perm)
        
        current_S = []
        for x in perm:
            if time.time() > deadline:
                break
            new_S = current_S + [x]
            if is_dissociated(new_S):
                current_S = new_S
        
        if len(current_S) > best_size:
            best_set = current_S
            best_size = len(best_set)
            
        # Local search on current_S
        # Try to swap elements to allow more additions
        # Or just try to extend current_S further with the remaining elements
        # The greedy loop above already did this for the specific order.
        # Let's try a different order for extension
        # Sort remaining by value
        remaining = [x for x in universe if x not in current_S]
        remaining.sort(reverse=True) # Large first
        
        temp_S = list(current_S)
        for x in remaining:
             if time.time() > deadline:
                 break
             new_S = temp_S + [x]
             if is_dissociated(new_S):
                 temp_S = new_S
                 
        if len(temp_S) > best_size:
            best_set = temp_S
            best_size = len(best_set)

    # Final Polish: Try to swap elements in best_set to see if we can get a larger set
    # This is complex. The greedy + random restarts should be sufficient for small N.
    
    return best_set

if __name__ == "__main__":
    # Quick test
    print(construct())
