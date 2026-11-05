"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # safety margin before 30s limit
    
    # Bitset representation: sums_bitset[s] = True if s is in the sum set
    # Sums range from 2 to 2*N = 512, so we need indices 0..512
    # Use a Python set for sums (fast membership)
    
    best_size = N + 1
    best_set = None
    
    def get_sums(S):
        """Compute all pair sums a+b for a<=b in S."""
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums.add(S[i] + S[j])
        return sums
    
    def can_add_to(S, sums, x):
        """Check if x can be added to Sidon set S with sum set sums."""
        if 2 * x in sums:
            return False
        for a in S:
            if x + a in sums:
                return False
        # Also check that new sums are distinct from each other
        # 2x is already checked against sums
        # x+a for a in S: need x+a != x+b for a!=b, which is true since a!=b
        # x+a != 2x iff a != x, which is true since x not in S
        return True
    
    def is_maximal_fast(S, sums):
        """Check if S is inclusion-maximal given its sum set."""
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add_to(S, sums, x):
                return False
        return True
    
    def greedy_construct(order):
        """Build a Sidon set by greedily adding elements in given order."""
        S = []
        sums = set()
        for x in order:
            if can_add_to(S, sums, x):
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S, sums
    
    def prune(S):
        """Try to remove elements while maintaining maximality."""
        sums = get_sums(S)
        improved = True
        while improved:
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    sums = new_sums
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
                new_sums = get_sums(new_S)
                if is_maximal_fast(new_S, new_sums):
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
                    sums_without = get_sums(S_without)
                    S_without_set = set(S_without)
                    # Try to find an element x that can be added and results in a maximal set
                    for x in range(1, N + 1):
                        if x in S_without_set:
                            continue
                        if can_add_to(S_without, sums_without, x):
                            new_S = sorted(S_without + [x])
                            new_sums = get_sums(new_S)
                            if is_maximal_fast(new_S, new_sums):
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
    sums = get_sums(S)
    
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
    if not is_maximal_fast(S, sums):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add_to(S, sums, x):
                    S.append(x)
                    S_set.add(x)
                    sums.add(2 * x)
                    for a in S[:-1]:
                        sums.add(x + a)
                    changed = True
                    break
        S = sorted(S)
        best_set = S
    
    return best_set

# EVOLVE-BLOCK-END
