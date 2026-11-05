"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
from bisect import bisect_left

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0  # Safety margin before 30s limit
    
    # Bitmask helpers
    # Sums range from 1+1=2 to 256+256=512.
    # We need bits 0..512.
    MAX_SUM = 512
    
    def get_masks(S):
        """Return (S_mask, sums_mask) for set S."""
        S_mask = 0
        sums_mask = 0
        S_list = list(S)
        n = len(S_list)
        for i in range(n):
            a = S_list[i]
            S_mask |= (1 << a)
            sums_mask |= (1 << (2 * a))
            for j in range(i + 1, n):
                b = S_list[j]
                sums_mask |= (1 << (a + b))
        return S_mask, sums_mask

    def can_add_bitmask(x, S_mask, sums_mask):
        """Check if x can be added to S defined by masks."""
        # 1. 2x must not be in sums
        if (sums_mask >> (2 * x)) & 1:
            return False
        # 2. (sums >> x) & S_mask must be 0
        # This checks if there exists a in S such that x+a is in sums
        if (sums_mask >> x) & S_mask:
            return False
        return True

    def is_maximal_bitmask(S):
        """Check if S is inclusion-maximal using bitmasks."""
        S_mask, sums_mask = get_masks(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add_bitmask(x, S_mask, sums_mask):
                return False
        return True

    def greedy_construct(order):
        S = []
        S_mask = 0
        sums_mask = 0
        for x in order:
            if can_add_bitmask(x, S_mask, sums_mask):
                S.append(x)
                S_mask |= (1 << x)
                # Update sums_mask
                sums_mask |= (1 << (2 * x))
                for a in S[:-1]:
                    sums_mask |= (1 << (x + a))
        return S

    def prune_aggressive(S):
        """Try to remove elements while maintaining maximality."""
        S = list(S)
        if not S:
            return S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 1:
                break
            
            # Try removing each element
            # Randomize order to avoid bias
            indices = list(range(n))
            random.shuffle(indices)
            
            for idx in indices:
                if time.time() > deadline:
                    break
                new_S = S[:idx] + S[idx+1:]
                if not new_S:
                    continue
                if is_maximal_bitmask(new_S):
                    S = new_S
                    improved = True
                    break  # Restart with new S
            
        return S

    def prune_double_removal(S):
        """Try removing two elements at a time."""
        S = list(S)
        if not S:
            return S
        
        improved = True
        while improved and time.time() < deadline:
            improved = False
            n = len(S)
            if n <= 2:
                break
            
            # Try all pairs
            for i in range(n):
                if time.time() > deadline:
                    break
                for j in range(i + 1, n):
                    if time.time() > deadline:
                        break
                    new_S = S[:i] + S[i+1:j] + S[j+1:]
                    if not new_S:
                        continue
                    if is_maximal_bitmask(new_S):
                        S = new_S
                        improved = True
                        break  # Restart with new S
                if improved:
                    break
        
        return S

    # Phase 1: Deterministic Descending Greedy
    order_desc = list(range(N, 0, -1))
    S_init = greedy_construct(order_desc)
    S_init = prune_aggressive(S_init)
    best_set = S_init
    best_size = len(S_init)
    
    # Phase 2: Randomized Restarts
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    iteration = 0
    while time.time() < deadline:
        iteration += 1
        
        order = list(all_elements)
        rng.shuffle(order)
        
        S = greedy_construct(order)
        
        # Aggressive pruning
        S_pruned = prune_aggressive(S)
        
        # Double removal
        S_double = prune_double_removal(S_pruned)
        
        if len(S_double) < best_size:
            best_size = len(S_double)
            best_set = S_double
            
    # Final verification
    if not is_maximal_bitmask(best_set):
        best_set = greedy_construct(order_desc)
        
    return best_set

# EVOLVE-BLOCK-END
