"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # Safety margin before 30s limit
    
    # Precompute a bitset for fast sum membership checks
    # Max possible sum is 256 + 255 = 511, so we need bits 0..511
    MAX_SUM = 2 * N  # 512, safe upper bound
    
    def can_add_fast(x, sums_bitset, S):
        """Check if x can be added to S given sums as a bitset (int with bits set)."""
        # Check 2x
        if (sums_bitset >> (2 * x)) & 1:
            return False
        # Check x + a for a in S
        for a in S:
            if (sums_bitset >> (x + a)) & 1:
                return False
        return True

    def get_sums_bitset(S):
        """Compute the sum bitset for set S."""
        sums = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums |= (1 << (S[i] + S[j]))
        return sums

    def is_maximal_fast(S, sums_bitset=None):
        """Check if S is inclusion-maximal using bitset for sums."""
        if not S:
            return False
        if sums_bitset is None:
            sums_bitset = get_sums_bitset(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            # Check if x can be added
            if (sums_bitset >> (2 * x)) & 1:
                continue
            addable = True
            for a in S:
                if (sums_bitset >> (x + a)) & 1:
                    addable = False
                    break
            if addable:
                return False
        return True

    def greedy_construct(order):
        S = []
        sums = 0  # bitset
        for x in order:
            if can_add_fast(x, sums, S):
                S.append(x)
                # Update sums bitset
                sums |= (1 << (2 * x))
                for a in S[:-1]:
                    sums |= (1 << (x + a))
        return S

    def prune_optimized(S):
        """Try to remove elements while maintaining maximality, using cached sums."""
        S = list(S)
        sums = get_sums_bitset(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return S
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                # Recompute sums for new_S
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    sums = new_sums
                    improved = True
                    break
        return S

    # Phase 1: Deterministic Descending Greedy
    order_desc = list(range(N, 0, -1))
    S_init = greedy_construct(order_desc)
    S_init = prune_optimized(S_init)
    best_set = S_init
    best_size = len(S_init)
    
    # Phase 2: Randomized Restarts
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    # Also try some structured orders
    structured_orders = [
        list(range(N, 0, -1)),  # descending
        list(range(1, N + 1)),  # ascending
        list(range(N // 2, 0, -1)) + list(range(N // 2 + 1, N + 1)),  # high-mid then low-mid
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),  # odds then evens
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),  # evens then odds
    ]
    
    for order in structured_orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        S_pruned = prune_optimized(S)
        if len(S_pruned) < best_size:
            best_size = len(S_pruned)
            best_set = S_pruned
    
    # Random restarts
    while time.time() < deadline:
        # Mix: sometimes fully random, sometimes with bias
        r = rng.random()
        if r < 0.3:
            order = list(all_elements)
            rng.shuffle(order)
        elif r < 0.5:
            # Bias toward larger numbers first (they create larger sums, fewer conflicts)
            order = sorted(all_elements, key=lambda x: -x + rng.gauss(0, 50))
        elif r < 0.7:
            # Bias toward smaller numbers first
            order = sorted(all_elements, key=lambda x: x + rng.gauss(0, 50))
        else:
            # Fully random
            order = list(all_elements)
            rng.shuffle(order)
        
        S = greedy_construct(order)
        S_pruned = prune_optimized(S)
        
        if len(S_pruned) < best_size:
            best_size = len(S_pruned)
            best_set = S_pruned
    
    # Final verification
    if not is_maximal_fast(best_set):
        best_set = greedy_construct(order_desc)
    
    return best_set

# EVOLVE-BLOCK-END
