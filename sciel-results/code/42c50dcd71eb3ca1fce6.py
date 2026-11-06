"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import random
import time

N = 256

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a small inclusion-maximal Sidon set in [1, N].
    
    Strategy:
    1. Start with a greedy ascending construction to get a valid Sidon set.
    2. Use local search (removing elements) to minimize cardinality while maintaining
       the maximal property.
    3. Use a deadline guard to ensure we finish within time limits.
    """
    start_time = time.time()
    deadline = start_time + 25  # 25s safety margin within 30s limit
    
    def get_sums(S):
        """Compute all pairwise sums a+b with a<=b."""
        sums = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                sums.add(a + b)
        return sums
    
    def is_sidon(S):
        """Check if S is a Sidon set."""
        sums = set()
        n = len(S)
        for i in range(n):
            a = S[i]
            for j in range(i, n):
                b = S[j]
                s = a + b
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(S, sums):
        """Check if S is inclusion-maximal (no element in [1,N] can be added)."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if adding x preserves Sidon property
            new_sums = set(sums)
            for a in S:
                s = a + x
                if s in new_sums:
                    break
                new_sums.add(s)
            else:
                # Also check 2*x
                s = 2 * x
                if s in new_sums:
                    break
                # If we get here, x could be added
                return False
        return True
    
    # Step 1: Greedy ascending construction
    def greedy_construct():
        selected = []
        sums = set()
        for x in range(1, N + 1):
            added = [x + a for a in selected] + [2 * x]
            if len(set(added)) == len(added) and not sums.intersection(added):
                selected.append(x)
                sums.update(added)
        return selected, sums
    
    # Initial greedy set
    best_set, best_sums = greedy_construct()
    best_len = len(best_set)
    
    # Local search: try removing elements to reduce size while maintaining maximality
    for iteration in range(2000):
        if time.time() > deadline:
            break
        
        # Try each element for removal
        for i in range(len(best_set)):
            if time.time() > deadline:
                break
            
            # Create a new set without element i
            new_set = best_set[:i] + best_set[i+1:]
            new_sums = get_sums(new_set)
            
            # Check if it's still maximal
            if is_maximal(new_set, new_sums):
                best_set = new_set
                best_sums = new_sums
                best_len = len(best_set)
                # After removal, continue checking for more removals
                break
    
    # Additional optimization: try random removals and re-optimization
    random.seed(42)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Start from a perturbed version
        current_set = list(best_set)
        
        # Try random removals
        for _ in range(20):
            if time.time() > deadline:
                break
            if len(current_set) <= 1:
                break
            
            idx = random.randint(0, len(current_set) - 1)
            new_set = current_set[:idx] + current_set[idx+1:]
            new_sums = get_sums(new_set)
            
            if is_maximal(new_set, new_sums):
                current_set = new_set
        
        # If we found a smaller set, update best
        if len(current_set) < best_len:
            best_set = current_set
            best_sums = get_sums(best_set)
            best_len = len(best_set)
    
    return best_set
# EVOLVE-BLOCK-END
