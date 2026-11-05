"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 27.0  # Safety margin before 30s limit
    
    # Precompute for fast operations
    MAX_SUM = 2 * N  # 512

    def get_sums_bitset(S):
        """Compute the sum bitset for set S."""
        sums = 0
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                sums |= (1 << (S[i] + S[j]))
        return sums

    def get_blocked_elements(S, sums_bitset):
        """Compute the set of elements blocked by S.
        An element y is blocked if 2y is in sums or y+s is in sums for some s in S.
        """
        blocked = 0
        for y in range(1, N + 1):
            if (sums_bitset >> (2 * y)) & 1:
                blocked |= (1 << y)
                continue
            for s in S:
                if (sums_bitset >> (y + s)) & 1:
                    blocked |= (1 << y)
                    break
        return blocked

    def is_maximal_fast(S, sums_bitset=None):
        """Check if S is inclusion-maximal using bitset for sums."""
        if not S:
            return False
        if sums_bitset is None:
            sums_bitset = get_sums_bitset(S)
        S_mask = 0
        for s in S:
            S_mask |= (1 << s)
        
        # All elements in [1, N] not in S must be blocked
        all_mask = 0
        for i in range(1, N + 1):
            all_mask |= (1 << i)
        
        missing_mask = all_mask & ~S_mask
        if missing_mask == 0:
            return True
            
        blocked_mask = get_blocked_elements(S, sums_bitset)
        
        # Check if missing_mask is a subset of blocked_mask
        if (missing_mask & ~blocked_mask) != 0:
            return False
        return True

    def can_add_fast(x, sums_bitset, S):
        """Check if x can be added to S given sums as a bitset."""
        if (sums_bitset >> (2 * x)) & 1:
            return False
        for a in S:
            if (sums_bitset >> (x + a)) & 1:
                return False
        return True

    def get_removable_elements(S, sums_bitset):
        """Find elements in S that can be removed while maintaining maximality."""
        removable = []
        for i in range(len(S)):
            new_S = S[:i] + S[i+1:]
            if not new_S:
                continue
            new_sums = get_sums_bitset(new_S)
            if is_maximal_fast(new_S, new_sums):
                removable.append(i)
        return removable

    def greedy_coverage():
        """Greedy construction maximizing coverage (blocking)."""
        S = []
        sums = 0
        S_set = set()
        
        while True:
            if time.time() > deadline:
                break
            
            best_x = -1
            best_gain = -1
            
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                if not can_add_fast(y, sums, S):
                    continue
                
                # Calculate how many new elements this would block
                new_sums = sums | (1 << (2 * y))
                for s in S:
                    new_sums |= (1 << (y + s))
                
                # Count new blocked elements among those currently addable
                gain = 0
                for z in range(1, N + 1):
                    if z in S_set:
                        continue
                    # Was z addable before?
                    if not can_add_fast(z, sums, S):
                        continue
                    # Is z addable after adding y?
                    if can_add_fast(z, new_sums, S + [y]):
                        continue
                    gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = y
            
            if best_x == -1:
                break
            
            S.append(best_x)
            S_set.add(best_x)
            sums |= (1 << (2 * best_x))
            for s in S[:-1]:
                sums |= (1 << (best_x + s))
            
            if is_maximal_fast(S, sums):
                break
        
        return S

    def greedy_construct(order):
        """Greedy construction with given order."""
        S = []
        sums = 0  # bitset
        for x in order:
            if can_add_fast(x, sums, S):
                S.append(x)
                sums |= (1 << (2 * x))
                for a in S[:-1]:
                    sums |= (1 << (x + a))
        return S

    def prune_aggressive(S):
        """Aggressively prune: try removing each element, check maximality."""
        S = list(S)
        improved = True
        while improved:
            improved = False
            if time.time() > deadline:
                return S
            # Try removing each element
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i+1:]
                if not new_S:
                    continue
                new_sums = get_sums_bitset(new_S)
                if is_maximal_fast(new_S, new_sums):
                    S = new_S
                    improved = True
                    break  # Restart with new indices
        return S

    def local_search_removability(S, max_iters=200):
        """
        Local search guided by 'removability'.
        1. If any element is removable, remove it.
        2. Else, try 1-swaps to find a state with higher removability.
        """
        S = list(S)
        sums = get_sums_bitset(S)
        best_S = list(S)
        best_size = len(S)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            
            # Step 1: Check for removable elements
            removable = get_removable_elements(S, sums)
            if removable:
                # Remove the first one found (or one that leads to best next state)
                # For now, just remove the first one
                idx = removable[0]
                S = S[:idx] + S[idx+1:]
                sums = get_sums_bitset(S)
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                continue
            
            # Step 2: No removable elements. Try 1-swaps
            best_swap = None
            best_removability = 0
            
            S_set = set(S)
            
            for i in range(len(S)):
                elem_out = S[i]
                new_S_base = S[:i] + S[i+1:]
                new_sums_base = get_sums_bitset(new_S_base)
                
                for j in range(1, N + 1):
                    if j in S_set:
                        continue
                    if not can_add_fast(j, new_sums_base, new_S_base):
                        continue
                    
                    trial_S = new_S_base + [j]
                    trial_sums = new_sums_base | (1 << (2 * j))
                    for a in new_S_base:
                        trial_sums |= (1 << (j + a))
                    
                    if not is_maximal_fast(trial_S, trial_sums):
                        continue
                    
                    # Calculate removability of trial_S
                    rem_count = len(get_removable_elements(trial_S, trial_sums))
                    
                    if rem_count > best_removability:
                        best_removability = rem_count
                        best_swap = (i, j, trial_S, trial_sums)
            
            if best_swap:
                i, j, new_S, new_sums = best_swap
                S = new_S
                sums = new_sums
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
            else:
                # No improving swap found. Break or random restart?
                # For now, break this local search
                break
                
        return best_S

    # Phase 0: Try greedy coverage first
    if time.time() < deadline - 5.0:
        S_cov = greedy_coverage()
        S_cov = prune_aggressive(S_cov)
        best_set = S_cov
        best_size = len(S_cov)
    else:
        best_set = None
        best_size = N + 1
    
    # Phase 1: Multiple deterministic orders
    orders = [
        list(range(N, 0, -1)),  # descending
        list(range(1, N + 1)),  # ascending
        list(range(N // 2, 0, -1)) + list(range(N // 2 + 1, N + 1)),
        list(range(1, N + 1, 2)) + list(range(2, N + 1, 2)),
        list(range(2, N + 1, 2)) + list(range(1, N + 1, 2)),
    ]
    
    for order in orders:
        if time.time() > deadline:
            break
        S = greedy_construct(order)
        S = prune_aggressive(S)
        if len(S) < best_size:
            best_size = len(S)
            best_set = S
    
    # Phase 2: Randomized restarts with local search
    rng = random.Random(42)
    all_elements = list(range(1, N + 1))
    
    iterations = 0
    while time.time() < deadline - 3.0:  # Leave 3s for final checks
        iterations += 1
        # Various random order strategies
        r = rng.random()
        if r < 0.25:
            order = list(all_elements)
            rng.shuffle(order)
        elif r < 0.4:
            # Bias toward larger numbers
            order = sorted(all_elements, key=lambda x: -x + rng.gauss(0, 80))
        elif r < 0.55:
            # Bias toward smaller numbers
            order = sorted(all_elements, key=lambda x: x + rng.gauss(0, 80))
        else:
            # Random with occasional reversal
            order = list(all_elements)
            rng.shuffle(order)
            if rng.random() < 0.3:
                order.reverse()
        
        S = greedy_construct(order)
        S = prune_aggressive(S)
        
        # Use removability-guided local search
        if len(S) < best_size:
            S_ls = local_search_removability(S, max_iters=100)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
        elif len(S) == best_size:
            # Even if same size, local search might find a structure that allows reduction
            S_ls = local_search_removability(S, max_iters=50)
            if len(S_ls) < best_size:
                best_size = len(S_ls)
                best_set = S_ls
    
    # Final: try more aggressive local search on best found
    if best_set is None:
        best_set = greedy_construct(list(range(N, 0, -1)))
    
    # One more round of local search with more iterations
    if time.time() < deadline - 1.0:
        final_ls = local_search_removability(best_set, max_iters=200)
        if len(final_ls) < len(best_set):
            best_set = final_ls
        
        # Also try pure removal search as a final polish
        final_prune = prune_aggressive(best_set)
        if len(final_prune) < len(best_set):
            best_set = final_prune
    
    # Final verification
    if not is_maximal_fast(best_set):
        # Fallback to a known valid construction
        best_set = greedy_construct(list(range(N, 0, -1)))
        if not is_maximal_fast(best_set):
            best_set = greedy_construct(list(range(1, N + 1)))
    
    return best_set

# EVOLVE-BLOCK-END
