import time
import random
import numpy as np

N = 256

def construct():
    best_size = N + 1
    best_set = []
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s kill

    def compute_sums(S):
        """Compute all pairwise sums a+b for a<=b in S."""
        s = set()
        for i in range(len(S)):
            s.add(2 * S[i])
            for j in range(i + 1, len(S)):
                s.add(S[i] + S[j])
        return s

    def is_valid_add(x, S, sums):
        """Check if x can be added to Sidon set S."""
        if 2 * x in sums:
            return False
        for s in S:
            if x + s in sums:
                return False
        return True

    def is_maximal(S, sums):
        """Check if S is inclusion-maximal: no element from [1,N]\S can be added."""
        S_set = set(S)
        for y in range(1, N + 1):
            if y in S_set:
                continue
            if is_valid_add(y, S, sums):
                return False
        return True

    def local_search_reduce(S, sums, deadline):
        """Try to remove elements while maintaining maximality.
        Returns a smaller maximal Sidon set if found."""
        S = list(S)
        sums = compute_sums(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Try removing each element
            for i in range(len(S)):
                if time.time() >= deadline:
                    break
                # Temporarily remove S[i]
                S_temp = S[:i] + S[i+1:]
                sums_temp = compute_sums(S_temp)
                if is_maximal(S_temp, sums_temp):
                    S = S_temp
                    sums = sums_temp
                    improved = True
                    break  # Restart scan after a removal
        return S, sums

    def greedy_construct(order):
        """Greedy construct Sidon set following given order."""
        S = []
        sums = set()
        for x in order:
            if is_valid_add(x, S, sums):
                S.append(x)
                sums.add(2 * x)
                for s in S[:-1]:
                    sums.add(x + s)
        return S, sums

    def weighted_order(rng):
        """Create a weighted order favoring spread-out elements."""
        weights = []
        for x in range(1, N + 1):
            # Favor numbers that are spread out - extremes and middle
            dist_from_center = abs(x - N / 2)
            if dist_from_center > N * 0.35:
                w = 2.0  # Favor extremes
            elif dist_from_center > N * 0.2:
                w = 1.5  # Favor middle-ish
            else:
                w = 1.0  # Center
            weights.append(w)
        
        # Sample without replacement using weights
        order = []
        remaining = list(range(1, N + 1))
        rem_weights = list(weights)
        while remaining:
            total_w = sum(rem_weights)
            r = rng.random() * total_w
            cum = 0
            for i, w in enumerate(rem_weights):
                cum += w
                if r <= cum:
                    order.append(remaining[i])
                    remaining.pop(i)
                    rem_weights.pop(i)
                    break
        return order

    # Strategy 1: Ascending greedy (baseline)
    order = list(range(1, N + 1))
    S, sums = greedy_construct(order)
    if len(S) < best_size:
        if is_maximal(S, sums):
            S_reduced, sums_reduced = local_search_reduce(S, sums, deadline)
            if len(S_reduced) < best_size:
                best_size = len(S_reduced)
                best_set = S_reduced[:]

    # Strategy 2: Multiple random orders with local search
    for iteration in range(3000):
        if time.time() >= deadline:
            break
        
        rng = random.Random(iteration * 7919 + 12345)
        strategy = iteration % 4
        
        if strategy == 0:
            # Random order
            order = list(range(1, N + 1))
            rng.shuffle(order)
        elif strategy == 1:
            # Weighted order favoring spread
            order = weighted_order(rng)
        elif strategy == 2:
            # Hybrid: start with spread elements, then random
            spread = [1, N//4, N//2, 3*N//4, N]
            order = list(range(1, N + 1))
            rng.shuffle(order)
            for s in spread:
                if s in order:
                    order.remove(s)
            order = spread + order
        else:
            # Bias towards middle values
            order = list(range(1, N + 1))
            rng.shuffle(order)
            order.sort(key=lambda x: -abs(x - N/2) + rng.random() * 30)
        
        S, sums = greedy_construct(order)
        
        if len(S) < best_size:
            if is_maximal(S, sums):
                S_reduced, sums_reduced = local_search_reduce(S, sums, deadline)
                if len(S_reduced) < best_size:
                    best_size = len(S_reduced)
                    best_set = S_reduced[:]
    
    # Final attempt: try to reduce the best set found further
    if best_set:
        sums_best = compute_sums(best_set)
        S_final, sums_final = local_search_reduce(best_set, sums_best, deadline)
        if len(S_final) < best_size:
            best_size = len(S_final)
            best_set = S_final[:]
    
    return best_set
