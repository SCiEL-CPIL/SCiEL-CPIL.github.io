"""Improved search for small inclusion-maximal Sidon sets in [1,256]."""
import time
import random
import math

N = 256

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 28.0  # safety margin before 30s kill

    best_size = N + 1
    best_set = None

    def is_sidon(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True

    def compute_sums(S):
        sums = set()
        for i in range(len(S)):
            for j in range(i, len(S)):
                sums.add(S[i] + S[j])
        return sums

    def is_maximal(S, sums):
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                return False
        return True

    def count_blocked(S, sums, x):
        """Count how many remaining elements would be blocked if x is added."""
        S_set = set(S)
        new_sums = set(sums)
        new_sums.add(2 * x)
        for a in S:
            new_sums.add(x + a)

        blocked = 0
        for y in range(1, N + 1):
            if y == x or y in S_set:
                continue
            if 2 * y in new_sums:
                blocked += 1
                continue
            y_blocked = False
            for a in S:
                if (y + a) in new_sums:
                    y_blocked = True
                    break
            if y_blocked:
                blocked += 1
        return blocked

    def get_addable(S, sums):
        """Get all elements that can be added to S without breaking Sidon property."""
        S_set = set(S)
        addable = []
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if can_add:
                addable.append(x)
        return addable

    def greedy_construct(order=None, max_size=None):
        """Greedy construction with blocking heuristic.
        
        If order is None, uses natural order 1..N.
        If max_size is set, stops when we reach that size if maximal.
        """
        S = []
        S_set = set()
        sums = set()
        
        if order is None:
            order = list(range(1, N + 1))
        
        for x in order:
            if x in S_set:
                continue
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            S.append(x)
            S_set.add(x)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
            
            if max_size is not None and len(S) >= max_size:
                # Check if we're already maximal
                if is_maximal(S, sums):
                    break
        
        return S, sums

    def local_search_remove(S, sums):
        """Try to remove elements from a maximal Sidon set while maintaining maximality.
        
        Returns (new_S, new_sums) if a smaller maximal set is found, else (S, sums).
        """
        S = list(S)
        sums = compute_sums(S)
        
        improved = True
        while improved:
            improved = False
            for i in range(len(S)):
                # Try removing S[i]
                candidate = S[:i] + S[i+1:]
                if not candidate:
                    continue
                cand_sums = compute_sums(candidate)
                if is_maximal(candidate, cand_sums):
                    S = candidate
                    sums = cand_sums
                    improved = True
                    break
        return S, sums

    def backtrack_search(S, sums, depth, max_depth):
        """Backtracking search to find minimal maximal Sidon set."""
        nonlocal best_size, best_set
        
        if time.time() > deadline - 0.5:
            return False
        
        if len(S) >= best_size:
            return False
        
        if len(S) >= max_depth:
            return False
        
        # Check if maximal
        if is_maximal(S, sums):
            if len(S) < best_size:
                best_size = len(S)
                best_set = list(S)
                return True
            return False
        
        # Get addable elements
        addable = get_addable(S, sums)
        if not addable:
            return False
        
        # Score each candidate by how many elements it blocks
        scored = []
        for x in addable:
            blocked = count_blocked(S, sums, x)
            scored.append((blocked, x))
        
        # Sort by blocking power descending
        scored.sort(reverse=True)
        
        # Limit branching
        max_branch = min(len(scored), 6)
        
        for blocked, x in scored[:max_branch]:
            S.append(x)
            old_sums = set(sums)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
            
            if backtrack_search(S, sums, depth + 1, max_depth):
                S.pop()
                sums.clear()
                sums.update(old_sums)
                return True
            
            S.pop()
            sums.clear()
            sums.update(old_sums)
        
        return False

    # Phase 1: Greedy with different orderings
    # Natural order
    S, sums = greedy_construct()
    if S:
        best_size = len(S)
        best_set = list(S)
    
    # Reverse order
    S, sums = greedy_construct(order=list(range(N, 0, -1)))
    if S and len(S) < best_size:
        best_size = len(S)
        best_set = list(S)
    
    # Random orderings
    for seed in range(10):
        if time.time() > deadline - 5.0:
            break
        random.seed(seed)
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, sums = greedy_construct(order=order)
        if S and len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
    
    # Phase 2: Local search (removal) on best set found
    if best_set is not None:
        improved_S, improved_sums = local_search_remove(best_set, compute_sums(best_set))
        if len(improved_S) < best_size:
            best_size = len(improved_S)
            best_set = improved_S
    
    # Phase 3: Backtracking search with random starts
    for seed in range(15):
        if time.time() > deadline - 2.0:
            break
        
        random.seed(seed + 100)
        
        # Start with a small random seed set
        S = []
        sums = set()
        num_initial = random.randint(1, 4)
        for _ in range(num_initial):
            candidates = [x for x in range(1, N + 1) if x not in S]
            if not candidates:
                break
            x = random.choice(candidates)
            
            if 2 * x in sums:
                continue
            can_add = True
            for a in S:
                if (x + a) in sums:
                    can_add = False
                    break
            if not can_add:
                continue
            
            S.append(x)
            sums.add(2 * x)
            for a in S[:-1]:
                sums.add(x + a)
        
        max_depth = best_size - 1 if best_size <= N else 12
        backtrack_search(S, sums, 0, max_depth)
        
        # Also try local search on any new best
        if best_set is not None and time.time() < deadline - 1.0:
            improved_S, improved_sums = local_search_remove(best_set, compute_sums(best_set))
            if len(improved_S) < best_size:
                best_size = len(improved_S)
                best_set = improved_S
    
    # Phase 4: More aggressive local search with multiple restarts
    for seed in range(5):
        if time.time() > deadline - 0.5:
            break
        
        random.seed(seed + 200)
        order = list(range(1, N + 1))
        random.shuffle(order)
        S, sums = greedy_construct(order=order)
        if S:
            improved_S, improved_sums = local_search_remove(S, compute_sums(S))
            if len(improved_S) < best_size:
                best_size = len(improved_S)
                best_set = improved_S
    
    if best_set is not None:
        return best_set
    else:
        # Fallback: simple greedy
        S, sums = greedy_construct()
        return S
# EVOLVE-BLOCK-END
