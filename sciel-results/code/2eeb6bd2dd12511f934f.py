"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random
import numpy as np

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # safety margin before 30s limit
    
    # Precompute for vectorized maximality check
    # For a Sidon set S, x can be added iff:
    #   1. 2x not in sumset
    #   2. For all a in S: x+a not in sumset
    # We can check this with numpy vectorization
    
    def get_sums_np(S):
        """Compute all pair sums a+b for a<=b in S, return as numpy array."""
        S_arr = np.array(S, dtype=np.int32)
        n = len(S_arr)
        if n == 0:
            return np.array([], dtype=np.int32)
        # Generate all pairs (i,j) with i<=j
        ii, jj = np.triu_indices(n)
        sums = S_arr[ii] + S_arr[jj]
        return sums
    
    def is_sidon(S):
        """Check if S is a valid Sidon set."""
        S_arr = np.array(S, dtype=np.int32)
        n = len(S_arr)
        if n == 0:
            return True
        ii, jj = np.triu_indices(n)
        sums = S_arr[ii] + S_arr[jj]
        # Check for duplicates
        if len(sums) != len(np.unique(sums)):
            return False
        return True
    
    def is_maximal_vectorized(S):
        """Check if S is inclusion-maximal using vectorized operations."""
        S_arr = np.array(S, dtype=np.int32)
        n = len(S_arr)
        
        # Compute sumset
        ii, jj = np.triu_indices(n)
        sums = S_arr[ii] + S_arr[jj]
        sumset = set(sums.tolist())
        
        S_set = set(S_arr.tolist())
        
        # For each missing x, check if it can be added
        missing = [x for x in range(1, N+1) if x not in S_set]
        if not missing:
            return True
        
        missing_arr = np.array(missing, dtype=np.int32)
        
        # Check condition 1: 2x not in sumset
        # Vectorize: for each x in missing, check 2x in sumset
        double_x = 2 * missing_arr
        cond1_blocked = np.array([s in sumset for s in double_x.tolist()])
        
        # Check condition 2: for all a in S, x+a not in sumset
        # For each missing x, check if any (x+a) is in sumset for a in S
        # We can do this with broadcasting
        # missing_arr shape: (m,), S_arr shape: (n,)
        # We want to check: for each x in missing, is there any a in S such that x+a in sumset?
        
        # Vectorized approach: create all pairs (x, a) and check
        # m = len(missing), n = len(S)
        # pairs = missing_arr[:, None] + S_arr[None, :]  # shape (m, n)
        # Then check each element against sumset
        
        m = len(missing_arr)
        if m > 0 and n > 0:
            # Create all pair sums
            pair_sums = missing_arr[:, None] + S_arr[None, :]  # shape (m, n)
            # Check which are in sumset
            # For efficiency, check row by row or use a different approach
            # Actually, let's just do a Python loop over missing elements but vectorize over S
            for idx in range(m):
                if cond1_blocked[idx]:
                    continue  # Already blocked by 2x
                x = missing_arr[idx]
                # Check if any x+a is in sumset
                # Vectorized check over a in S
                xa_sums = x + S_arr  # shape (n,)
                # Check if any of these are in sumset
                if any(s in sumset for s in xa_sums.tolist()):
                    continue  # Blocked
                # x can be added, so not maximal
                return False
        else:
            # No missing elements or empty S
            for idx in range(m):
                if cond1_blocked[idx]:
                    continue
                x = missing_arr[idx]
                # Check if any x+a is in sumset (S is empty, so no a)
                # Only 2x matters, already checked
                # So x can be added
                return False
        
        return True
    
    def can_add_to(S, sums_set, x):
        """Check if x can be added to Sidon set S with sum set sums_set."""
        if 2 * x in sums_set:
            return False
        for a in S:
            if x + a in sums_set:
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
        sums = set()
        S_arr = np.array(S, dtype=np.int32)
        n = len(S_arr)
        ii, jj = np.triu_indices(n)
        sums = set((S_arr[ii] + S_arr[jj]).tolist())
        
        improved = True
        while improved:
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                # Check maximality of new_S
                if is_maximal_vectorized(new_S):
                    S = new_S
                    # Recompute sums
                    new_arr = np.array(S, dtype=np.int32)
                    nn = len(new_arr)
                    ii2, jj2 = np.triu_indices(nn)
                    sums = set((new_arr[ii2] + new_arr[jj2]).tolist())
                    improved = True
        return S
    
    best_size = N + 1
    best_set = None
    
    # Phase 1: Ascending greedy baseline
    S0, sums0 = greedy_construct(range(1, N + 1))
    S0 = prune(S0)
    if len(S0) < best_size:
        best_size = len(S0)
        best_set = list(S0)
    
    # Phase 2: Randomized greedy constructions with pruning
    rng = random.Random(42)
    
    for trial in range(50000):
        if time.time() > deadline - 5:
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
    if best_set is not None and time.time() < deadline - 5:
        S = list(best_set)
        S.sort()
        
        # Try removing elements and checking maximality more aggressively
        improved = True
        while improved and time.time() < deadline - 3:
            improved = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline - 3:
                    break
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                if is_maximal_vectorized(new_S):
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_set = list(S)
        
        # Try swap-based local search: remove one element, add another
        if time.time() < deadline - 2:
            for _ in range(100):
                if time.time() > deadline - 2:
                    break
                S_copy = list(S)
                S_copy.sort()
                improved_swap = False
                for i in range(len(S_copy)):
                    if time.time() > deadline - 2:
                        break
                    S_without = S_copy[:i] + S_copy[i+1:]
                    if not S_without:
                        continue
                    # Compute sums for S_without
                    S_without_arr = np.array(S_without, dtype=np.int32)
                    nw = len(S_without_arr)
                    ii_w, jj_w = np.triu_indices(nw)
                    sums_without = set((S_without_arr[ii_w] + S_without_arr[jj_w]).tolist())
                    S_without_set = set(S_without)
                    # Try to find an element x that can be added and results in a maximal set
                    for x in range(1, N + 1):
                        if x in S_without_set:
                            continue
                        if can_add_to(S_without, sums_without, x):
                            new_S = sorted(S_without + [x])
                            if is_maximal_vectorized(new_S):
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
    
    # Check Sidon property
    S_arr = np.array(S, dtype=np.int32)
    n = len(S_arr)
    ii, jj = np.triu_indices(n)
    sums_arr = S_arr[ii] + S_arr[jj]
    if len(sums_arr) != len(np.unique(sums_arr)):
        # Invalid - fallback
        S0, _ = greedy_construct(range(1, N + 1))
        S0 = prune(S0)
        return S0
    
    sums = set(sums_arr.tolist())
    
    # Check maximality, repair if needed
    if not is_maximal_vectorized(S):
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
