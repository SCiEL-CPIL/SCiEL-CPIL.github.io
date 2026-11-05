"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0
    start = time.time()
    
    def get_sums(S):
        sums = set()
        n = len(S)
        for i in range(n):
            si = S[i]
            sums.add(si + si)
            for j in range(i + 1, n):
                sums.add(si + S[j])
        return sums
    
    def can_add(x, S, sums):
        if 2 * x in sums:
            return False
        for a in S:
            if x + a in sums:
                return False
        return True
    
    def is_sidon(S):
        sums = set()
        n = len(S)
        for i in range(n):
            for j in range(i, n):
                s = S[i] + S[j]
                if s in sums:
                    return False
                sums.add(s)
        return True
    
    def is_maximal(S):
        sums = get_sums(S)
        S_set = set(S)
        for x in range(1, N + 1):
            if x in S_set:
                continue
            if can_add(x, S, sums):
                return False
        return True
    
    def greedy_construct(order):
        S = []
        sums = set()
        for x in order:
            ok = True
            if 2 * x in sums:
                ok = False
            else:
                for a in S:
                    if x + a in sums:
                        ok = False
                        break
            if ok:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S
    
    def prune_for_maximality(S):
        """Remove elements while maintaining maximality."""
        S = list(S)
        changed = True
        while changed:
            changed = False
            for i in range(len(S) - 1, -1, -1):
                if time.time() > deadline:
                    return S
                new_S = S[:i] + S[i + 1:]
                if len(new_S) == 0:
                    continue
                if is_maximal(new_S):
                    S = new_S
                    changed = True
                    break
        return S
    
    def try_remove_pair(S, i, j):
        """Try removing elements at positions i and j, check if still maximal."""
        if i == j:
            return None
        new_S = S[:min(i,j)] + S[min(i,j)+1:max(i,j)] + S[max(i,j)+1:]
        if len(new_S) == 0:
            return None
        if is_maximal(new_S):
            return new_S
        return None
    
    # Phase 1: Build initial Sidon set using ascending greedy
    S = greedy_construct(range(1, N + 1))
    S = prune_for_maximality(S)
    best_S = list(S)
    best_size = len(S)
    
    # Phase 2: Try different greedy orderings
    rng = random.Random(12345)
    
    for trial in range(200):
        if time.time() > deadline:
            break
        
        # Various ordering strategies
        if trial % 5 == 0:
            # Random
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif trial % 5 == 1:
            # Descending
            order = list(range(N, 0, -1))
        elif trial % 5 == 2:
            # Start from middle
            order = list(range(N // 2, N + 1)) + list(range(1, N // 2))
        elif trial % 5 == 3:
            # Primes-like (just use a fixed pattern)
            order = list(range(1, N + 1))
            for i in range(0, len(order), 2):
                order[i], order[i + 1] = order[i + 1], order[i]
        else:
            # Random with bias toward larger numbers
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        
        S = greedy_construct(order)
        S = prune_for_maximality(S)
        
        if len(S) < best_size:
            best_size = len(S)
            best_S = list(S)
    
    # Phase 3: Local search - try swapping elements
    S = list(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i + 1:]
            if not S_without:
                continue
            
            sums_without = get_sums(S_without)
            S_without_set = set(S_without)
            
            # Try to find a replacement that keeps it maximal
            for x in range(1, N + 1):
                if x in S_without_set:
                    continue
                if not can_add(x, S_without, sums_without):
                    continue
                new_S = sorted(S_without + [x])
                if is_maximal(new_S):
                    S = new_S
                    improved = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
    
    # Phase 4: More aggressive removal with different strategies
    S = list(best_S)
    for attempt in range(50):
        if time.time() > deadline:
            break
        
        # Try removing multiple elements at once
        if len(S) > 10:
            removed = set()
            new_S = list(S)
            # Try removing elements from the end
            for i in range(len(new_S) - 1, -1, -1):
                if time.time() > deadline:
                    break
                candidate = new_S[:i] + new_S[i + 1:]
                if is_maximal(candidate):
                    new_S = candidate
                    removed.add(S[i])
            
            if len(new_S) < len(best_S):
                best_S = new_S
                best_size = len(new_S)
    
    # Phase 5: Pair removal - try removing two elements at once
    S = list(best_S)
    for attempt in range(30):
        if time.time() > deadline:
            break
        if len(S) <= 12:
            break
        
        found_removal = False
        for i in range(len(S) - 1):
            if time.time() > deadline:
                break
            for j in range(i + 1, len(S)):
                if time.time() > deadline:
                    break
                result = try_remove_pair(S, i, j)
                if result is not None:
                    S = result
                    found_removal = True
                    if len(S) < best_size:
                        best_size = len(S)
                        best_S = list(S)
                    break
            if found_removal:
                break
    
    # Phase 6: Final aggressive single-element removal pass
    S = list(best_S)
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for i in range(len(S) - 1, -1, -1):
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i + 1:]
            if len(new_S) == 0:
                continue
            if is_maximal(new_S):
                S = new_S
                changed = True
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                break
    
    # Final verification and repair
    S = list(best_S)
    if not is_sidon(S):
        # Fall back to ascending greedy
        S = greedy_construct(range(1, N + 1))
        S = prune_for_maximality(S)
    
    if not is_maximal(S):
        # Repair: add elements that can be added
        changed = True
        while changed:
            changed = False
            sums = get_sums(S)
            S_set = set(S)
            for x in range(1, N + 1):
                if x in S_set:
                    continue
                if can_add(x, S, sums):
                    S.append(x)
                    S_set.add(x)
                    sums.add(2 * x)
                    for a in S[:-1]:
                        sums.add(x + a)
                    changed = True
                    break
        S = sorted(S)
    
    return S

# EVOLVE-BLOCK-END
