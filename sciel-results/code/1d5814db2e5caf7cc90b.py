"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # safety margin before 30s limit
    
    best_size = N + 1
    best_set = None
    
    # Precompute for fast blocking checks
    # sums_arr is a boolean array of size 513 (indices 0..512)
    # where sums_arr[s] = True if sum s is present
    
    def compute_sums_array(S):
        """Compute boolean array of all pair sums a+b for a<=b in S."""
        sums_arr = np.zeros(513, dtype=bool)
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums_arr[S[i] + S[j]] = True
        return sums_arr
    
    def can_add_fast(S, sums_arr, x):
        """Check if x can be added to Sidon set S using numpy array."""
        if sums_arr[2 * x]:
            return False
        if len(S) > 0:
            # Check all sums x + a for a in S
            candidate_sums = x + np.array(S, dtype=np.int32)
            if np.any(sums_arr[candidate_sums]):
                return False
        return True
    
    def is_maximal_fast(S, sums_arr):
        """Check if S is inclusion-maximal given its sums array."""
        S_set = set(S)
        S_arr = np.array(S, dtype=np.int32)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add_fast(S, sums_arr, x):
                return False
        return True
    
    def greedy_construct(order):
        """Build a Sidon set by greedily adding elements in given order."""
        S = []
        S_arr = np.array([], dtype=np.int32)
        sums_arr = np.zeros(513, dtype=bool)
        for x in order:
            if sums_arr[2 * x]:
                continue
            if len(S) > 0:
                candidate_sums = x + S_arr
                if np.any(sums_arr[candidate_sums]):
                    continue
            S.append(x)
            S_arr = np.append(S_arr, x)
            sums_arr[2 * x] = True
            if len(S) > 1:
                sums_arr[x + S_arr[:-1]] = True
        return S, sums_arr
    
    def prune(S):
        """Try to remove elements while maintaining maximality."""
        sums_arr = compute_sums_array(S)
        improved = True
        while improved:
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums_arr = compute_sums_array(new_S)
                if is_maximal_fast(new_S, new_sums_arr):
                    S = new_S
                    sums_arr = new_sums_arr
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
    
    for trial in range(50000):
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
                new_sums_arr = compute_sums_array(new_S)
                if is_maximal_fast(new_S, new_sums_arr):
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = list(S)
        
        # Try swap-based local search: remove one element, add another
        if time.time() < deadline - 1:
            for _ in range(100):
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
                    sums_without = compute_sums_array(S_without)
                    S_without_set = set(S_without)
                    # Try to find an element x that can be added and results in a maximal set
                    for x in range(1, N + 1):
                        if x in S_without_set:
                            continue
                        if can_add_fast(S_without, sums_without, x):
                            new_S = sorted(S_without + [x])
                            new_sums_arr = compute_sums_array(new_S)
                            if is_maximal_fast(new_S, new_sums_arr):
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
    sums_arr = compute_sums_array(S)
    
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
    if not is_maximal_fast(S, sums_arr):
        changed = True
        while changed and time.time() < deadline:
            changed = False
            S_set = set(S)
            S_arr = np.array(S, dtype=np.int32)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add_fast(S, sums_arr, x):
                    S.append(x)
                    S_set.add(x)
                    S_arr = np.append(S_arr, x)
                    sums_arr[2 * x] = True
                    if len(S) > 1:
                        sums_arr[x + S_arr[:-1]] = True
                    changed = True
                    break
        S = sorted(S)
        best_set = S
    
    return best_set

# EVOLVE-BLOCK-END
