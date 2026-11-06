import time
import random

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

    def local_search_reduce(S, sums, time_limit=None):
        """Try to remove elements while maintaining maximality.
        Returns a smaller maximal Sidon set if found."""
        S = list(S)
        sums = compute_sums(S)
        improved = True
        while improved:
            improved = False
            if time_limit is not None and time.time() > time_limit:
                break
            # Try removing each element in random order
            indices = list(range(len(S)))
            random.shuffle(indices)
            for i in indices:
                if time_limit is not None and time.time() > time_limit:
                    break
                # Temporarily remove S[i]
                S_temp = S[:i] + S[i+1:]
                if not S_temp:
                    continue
                sums_temp = compute_sums(S_temp)
                if is_maximal(S_temp, sums_temp):
                    S = S_temp
                    sums = sums_temp
                    improved = True
                    break
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

    def coverage_based_construct():
        """Construct a Sidon set using coverage scoring.
        At each step, pick the element that 'blocks' the most other elements
        while maintaining the Sidon property."""
        S = []
        sums = set()
        remaining = set(range(1, N + 1))
        
        while remaining:
            # Find the best candidate based on coverage
            best_x = None
            best_score = -1
            best_new_sums = None
            
            # Sample a subset for efficiency if remaining is large
            if len(remaining) > 30:
                candidates = random.sample(list(remaining), 30)
            else:
                candidates = list(remaining)
            
            for x in candidates:
                if not is_valid_add(x, S, sums):
                    continue
                
                # Calculate coverage: how many other elements would be blocked
                # by adding x
                new_sums = sums | {2 * x} | {x + s for s in S}
                blocked_count = 0
                for y in remaining:
                    if y == x:
                        continue
                    if not is_valid_add(y, S, new_sums):
                        blocked_count += 1
                
                # Score: prefer elements that block more, but also consider
                # the value itself for diversity
                score = blocked_count * 10 - abs(x - N/2) * 0.1 + random.random() * 5
                
                if score > best_score:
                    best_score = score
                    best_x = x
                    best_new_sums = new_sums
            
            if best_x is None:
                break
            
            S.append(best_x)
            sums = best_new_sums
            remaining.remove(best_x)
            
            # Check if we've achieved maximality
            if len(S) >= 10:
                if is_maximal(S, sums):
                    break
        
        return S, sums

    # Strategy 1: Coverage-based construction with reduction
    for iteration in range(5000):
        if time.time() >= deadline:
            break
        
        strategy = iteration % 4
        
        if strategy == 0:
            # Coverage-based construction
            S, sums = coverage_based_construct()
        elif strategy == 1:
            # Random order greedy
            order = list(range(1, N + 1))
            random.shuffle(order)
            S, sums = greedy_construct(order)
        elif strategy == 2:
            # Hybrid: start with some structured elements, then random
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Put some spread-out elements first
            spread = [1, N//4, N//2, 3*N//4, N]
            for s in spread:
                if s in order:
                    order.remove(s)
            order = spread + order
            S, sums = greedy_construct(order)
        else:
            # Prioritize elements that create many conflicts (high blocking power)
            order = list(range(1, N + 1))
            random.shuffle(order)
            # Sort by "blocking potential" - elements in middle range block more
            order.sort(key=lambda x: -abs(x - N/2) + random.random() * 50)
            S, sums = greedy_construct(order)
        
        if not S:
            continue
            
        # Check if maximal
        if is_maximal(S, sums):
            # Try to reduce further
            S_reduced, sums_reduced = local_search_reduce(S, sums, deadline)
            if len(S_reduced) < best_size:
                best_size = len(S_reduced)
                best_set = S_reduced[:]
        else:
            # Try to make it maximal by adding elements
            S_temp = list(S)
            sums_temp = set(sums)
            S_set = set(S_temp)
            for y in range(1, N + 1):
                if y in S_set:
                    continue
                if is_valid_add(y, S_temp, sums_temp):
                    S_temp.append(y)
                    S_set.add(y)
                    sums_temp.add(2 * y)
                    for s in S_temp[:-1]:
                        sums_temp.add(y + s)
            
            if is_maximal(S_temp, sums_temp):
                S_reduced, sums_reduced = local_search_reduce(S_temp, sums_temp, deadline)
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
