"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0  # Safety margin before 30s limit
    
    # Helper: check if adding x to set S (with sum set sums) keeps Sidon property
    def can_add(x, S, sums):
        # Check 2x
        if 2 * x in sums:
            return False
        # Check x + a for a in S
        for a in S:
            if x + a in sums:
                return False
        return True

    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums

    def is_maximal(S):
        """Check if S is inclusion-maximal: no element in [1,N]\S can be added."""
        if not S:
            return False
        sums = get_sums(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            if 2 * x in sums:
                continue
            addable = True
            for a in S:
                if x + a in sums:
                    addable = False
                    break
            if addable:
                return False  # x can be added, so not maximal
        return True

    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            if can_add(x, S, sums):
                S.append(x)
                # Update sums
                sums.add(2 * x)
                for a in S[:-1]:  # S[:-1] is S without the newly added x
                    sums.add(x + a)
        return S

    def prune(S):
        """Try to remove elements while maintaining maximality."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            # Iterate in reverse to maintain index validity if we remove
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                if is_maximal(new_S):
                    S = new_S
                    improved = True
                    break # Restart loop as indices changed
        return S

    # Phase 1: Deterministic Descending Greedy
    order_desc = list(range(N, 0, -1))
    S_init = greedy_construct(order_desc)
    S_init = prune(S_init)
    best_set = S_init
    best_size = len(S_init)
    
    # Phase 2: Randomized Restarts with more aggressive pruning
    rng = random.Random(42)
    
    # Precompute some useful structures
    all_elements = list(range(1, N + 1))
    
    while time.time() < deadline:
        # Try different random permutations
        order = list(all_elements)
        rng.shuffle(order)
        
        S = greedy_construct(order)
        
        # More aggressive pruning: try multiple removal strategies
        # Strategy 1: Standard prune
        S_pruned = prune(S)
        
        # Strategy 2: Try removing elements in different orders
        # Sort by some heuristic - elements that are "more central" might be harder to remove
        # or elements with fewer conflicts
        S_sorted = sorted(S_pruned, key=lambda x: (x, -x))
        
        # Try to remove from the sorted list
        improved = True
        while improved and time.time() < deadline:
            improved = False
            for i in range(len(S_sorted) - 1, -1, -1):
                new_S = S_sorted[:i] + S_sorted[i+1:]
                if not new_S:
                    continue
                if is_maximal(new_S):
                    S_sorted = new_S
                    improved = True
                    break
        
        if len(S_sorted) < best_size:
            best_size = len(S_sorted)
            best_set = S_sorted
    
    # Final verification
    if not is_maximal(best_set):
        # Fallback: use the initial greedy result which should be maximal
        best_set = greedy_construct(order_desc)
    
    return best_set

# EVOLVE-BLOCK-END
