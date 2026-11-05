"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # safety margin before 30s limit
    
    best_size = N + 1
    best_set = None
    
    # Use integer bitset for sums: sums range from 2 to 512
    # sums_bit is a Python int where bit s is set if sum s is present
    
    def compute_sums_bitset(S):
        """Compute bitset of all pair sums a+b for a<=b in S."""
        sums_bit = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums_bit |= (1 << (S[i] + S[j]))
        return sums_bit
    
    def can_add_to_bitset(S, sums_bit, x):
        """Check if x can be added to Sidon set S with sums_bit bitset."""
        if (sums_bit >> (2 * x)) & 1:
            return False
        for a in S:
            if (sums_bit >> (x + a)) & 1:
                return False
        return True
    
    def is_maximal_fast(S, sums_bit):
        """Check if S is inclusion-maximal given its sums bitset."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add_to_bitset(S, sums_bit, x):
                return False
        return True
    
    def greedy_construct(order):
        """Build a Sidon set by greedily adding elements in given order."""
        S = []
        sums_bit = 0
        for x in order:
            if can_add_to_bitset(S, sums_bit, x):
                S.append(x)
                sums_bit |= (1 << (2 * x))
                for a in S[:-1]:
                    sums_bit |= (1 << (x + a))
        return S, sums_bit
    
    def prune(S):
        """Try to remove elements while maintaining maximality."""
        sums_bit = compute_sums_bitset(S)
        improved = True
        while improved:
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums_bit = compute_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums_bit):
                    S = new_S
                    sums_bit = new_sums_bit
                    improved = True
        return S
    
    # Phase 1: Ascending greedy baseline
    S0, sums0 = greedy_construct(range(1, N + 1))
    S0 = prune(S0)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = list(S0)
    
    # Phase 2: Randomized greedy constructions with pruning
    rng = random.Random(42)
    
    for trial in range(20000):
        if time.time() > deadline:
            break
        
        # Random permutation
        order = list(range(1, N + 1))
        rng.shuffle(order)
        
        S, sums = greedy_construct(order)
        S.sort()
        
        # Prune
        S = prune(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Phase 3: Local search - try to find smaller maximal sets
    if best_set is not None and time.time() < deadline - 3:
        S = list(best_set)
        S.sort()
        
        # Try removing elements and checking maximality more aggressively
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline - 2:
                    break
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums_bit = compute_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums_bit):
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = list(S)
        
        # Try swap-based local search: remove one element, add another
        if time.time() < deadline - 1:
            for _ in range(50):
                if time.time() > deadline - 1:
                    break
                S_copy = list(S)
                S_copy.sort()
                improved_swap = False
                for i in range(len(S_copy)):
                    if time.time() > deadline - 1:
                        break
                    S_without = S_copy[:i] + S_copy[i+1:]
                    if not S_without:
                        continue
                    sums_without = compute_sums_bitset(S_without)
                    S_without_set = set(S_without)
                    # Try to find an element x that can be added and results in a maximal set
                    for x in range(1, N + 1):
                        if x in S_without_set:
                            continue
                        if can_add_to_bitset(S_without, sums_without, x):
                            new_S = sorted(S_without + [x])
                            new_sums_bit = compute_sums_bitset(new_S)
                            if is_maximal_fast(new_S, new_sums_bit):
                                if len(new_S) < len(S_copy):
                                    S = new_S
                                    improved_swap = True
                                    if len(S) < best_size:
                                        best_size = len(S)
                                        best_set = list(S)
                                break
                if not improved_swap:
                    break
    
    if best_set is None:
        # Fallback
        S0, _ = greedy_construct(range(1, N + 1))
        S0 = prune(S0)
        best_set = S0
    
    # Final verification and repair
    S = sorted(best_set)
    sums_bit = compute_sums_bitset(S)
    
    # Check Sidon property
    seen_sums = set()
    for i in range(len(S)):
        for j in range(i, len(S)):
            s = S[i] + S[j]
            if s in seen_sums:
                # Invalid - fallback
                S0, _ = greedy_construct(range(1, N + 1))
                S0 = prune(S0)
                return S0
            seen_sums.add(s)
    
    # Check maximality, repair if needed
    if not is_maximal_fast(S, sums_bit):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add_to_bitset(S, sums_bit, x):
                    S.append(x)
                    S_set.add(x)
                    sums_bit |= (1 << (2 * x))
                    for a in S[:-1]:
                        sums_bit |= (1 << (x + a))
                    changed = True
                    break
        S = sorted(S)
        best_set = S
    
    return best_set

# EVOLVE-BLOCK-END
