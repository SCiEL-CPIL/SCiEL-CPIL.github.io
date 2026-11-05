"""Construct a small inclusion-maximal Sidon set in [1,256]."""
import time
import random

N = 256

# EVOLVE-BLOCK-START

def construct():
    """Return an inclusion-maximal Sidon set in [1,256] with small cardinality."""
    deadline = time.time() + 25.0
    
    def get_sums_set(S):
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
        if not S:
            return False
        sums = get_sums_set(S)
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
            if 2 * x in sums:
                continue
            conflict = False
            for a in S:
                if x + a in sums:
                    conflict = True
                    break
            if not conflict:
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
    
    def fast_greedy(order):
        """Fast greedy using set-based sums."""
        S = []
        sums = set()
        for x in order:
            if 2 * x in sums:
                continue
            conflict = False
            for a in S:
                if x + a in sums:
                    conflict = True
                    break
            if not conflict:
                S.append(x)
                sums.add(2 * x)
                for a in S[:-1]:
                    sums.add(x + a)
        return S
    
    # Phase 1: Build initial Sidon set using ascending greedy
    S = greedy_construct(range(1, N + 1))
    S = prune_for_maximality(S)
    best_S = list(S)
    best_size = len(S)
    
    # Phase 2: High-frequency randomized restarts with various orderings
    rng = random.Random(42)
    base_order = list(range(1, N + 1))
    
    for trial in range(50000):
        if time.time() > deadline:
            break
        
        # Fast random shuffle
        order = base_order[:]
        rng.shuffle(order)
        
        # Try different strategies based on trial
        if trial % 10 == 0:
            # Pure random
            pass
        elif trial % 10 == 1:
            # Descending
            order = order[::-1]
        elif trial % 10 == 2:
            # Random with high bias: put larger numbers first
            order = sorted(order, key=lambda x: -x)
            # Partial shuffle of first half
            half = len(order) // 2
            for i in range(half - 1, 0, -1):
                j = rng.randint(0, i)
                order[i], order[j] = order[j], order[i]
        elif trial % 10 == 3:
            # Start from ends
            order = []
            left, right = 1, N
            while left <= right:
                order.append(left)
                if left != right:
                    order.append(right)
                left += 1
                right -= 1
        elif trial % 10 == 4:
            # Random but with some structure: alternate low/high
            low = [x for x in order if x <= N // 2]
            high = [x for x in order if x > N // 2]
            order = []
            li, hi = 0, 0
            for i in range(N):
                if li < len(low):
                    order.append(low[li])
                    li += 1
                if hi < len(high):
                    order.append(high[hi])
                    hi += 1
        elif trial % 10 == 5:
            # Quadratic-like ordering: prioritize numbers with certain properties
            # Use a simple hash to create varied orderings
            seed_val = trial * 6151
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1))
        elif trial % 10 == 6:
            # Reverse quadratic
            seed_val = trial * 8191
            order = sorted(range(1, N + 1), key=lambda x: (x * seed_val) % (N + 1), reverse=True)
        elif trial % 10 == 7:
            # Random pairs: shuffle then keep pairs together
            order = list(range(1, N + 1))
            rng.shuffle(order)
            # Swap within pairs to create variation
            for i in range(0, N, 2):
                if i + 1 < N and rng.random() < 0.5:
                    order[i], order[i + 1] = order[i + 1], order[i]
        elif trial % 10 == 8:
            # Weighted random: bias toward middle values
            weights = [1.0 + abs(x - N/2) / (N/2) for x in range(1, N + 1)]
            order = list(range(1, N + 1))
            for i in range(len(order) - 1, 0, -1):
                # Weighted selection
                total_w = sum(weights[k] for k in range(i + 1))
                r = rng.random() * total_w
                cum = 0
                for k in range(i + 1):
                    cum += weights[k]
                    if cum >= r:
                        order[i], order[k] = order[k], order[i]
                        break
        else:
            # Pure random with different seed
            rng2 = random.Random(trial * 104729 + 7)
            order = list(range(1, N + 1))
            rng2.shuffle(order)
        
        S = fast_greedy(order)
        if S:
            S_pruned = prune_for_maximality(S)
            if len(S_pruned) < best_size:
                best_size = len(S_pruned)
                best_S = list(S_pruned)
    
    # Phase 3: Local search - try removing elements from best
    S = list(best_S)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            new_S = S[:i] + S[i + 1:]
            if not new_S:
                continue
            if is_maximal(new_S):
                S = new_S
                improved = True
                if len(S) < best_size:
                    best_size = len(S)
                    best_S = list(S)
                break
    
    # Phase 4: Try replacing elements
    S = list(best_S)
    for _ in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(len(S)):
            if time.time() > deadline:
                break
            S_without = S[:i] + S[i + 1:]
            if not S_without:
                continue
            sums_without = get_sums_set(S_without)
            S_without_set = set(S_without)
            
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
            if improved:
                break
    
    # Final verification and repair
    S = list(best_S)
    if not S or not is_sidon(S):
        S = greedy_construct(range(1, N + 1))
        S = prune_for_maximality(S)
    
    if not is_maximal(S):
        changed = True
        while changed:
            changed = False
            sums = get_sums_set(S)
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
