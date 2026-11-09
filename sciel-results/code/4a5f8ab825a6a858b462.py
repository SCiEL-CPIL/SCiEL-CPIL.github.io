def construct():
    import time
    import random
    
    deadline = time.time() + 140
    
    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x != 0:
                values.add(x)
        return sorted(values)
    
    def is_sum_free(S):
        if not S:
            return True
        S_set = set(S)
        for i in range(len(S)):
            for j in range(i, len(S)):
                if S[i] + S[j] in S_set:
                    return False
        return True
    
    def greedy_sum_free(S, order):
        result = []
        result_set = set()
        for x in order:
            if x in result_set:
                continue
            valid = True
            if x + x in result_set:
                valid = False
            if valid:
                for a in result_set:
                    if x + a in result_set:
                        valid = False
                        break
            if valid:
                result.append(x)
                result_set.add(x)
        return result
    
    def try_add(S_current, x, S_full):
        """Try to add x to S_current. Return True if successful, modifying S_current in place."""
        S_set = set(S_current)
        if x in S_set:
            return False
        # Check if x + x is in S_current
        if x + x in S_set:
            return False
        # Check if x + a is in S_current for any a in S_current
        for a in S_set:
            if x + a in S_set:
                return False
        S_current.append(x)
        return True
    
    def local_search_improve(S, S_full, max_rounds=100, rng=None):
        """Improve S by trying to add elements from S_full \\ S."""
        if rng is None:
            rng = random.Random(123)
        
        S_set = set(S)
        S_remaining = [x for x in S_full if x not in S_set]
        
        for round_num in range(max_rounds):
            improved = False
            for x in S_remaining:
                if try_add(S, x, S_full):
                    S_set.add(x)
                    S_remaining.remove(x)
                    improved = True
                    break
            if not improved:
                break
        
        return S
    
    def solve_instance(S, time_limit=30):
        n = len(S)
        if n == 0:
            return []
        
        start_time = time.time()
        max_abs = max(abs(x) for x in S)
        
        candidates = []
        
        # Structural classes
        odds = [x for x in S if x % 2 != 0]
        candidates.append(odds)
        
        mod1 = [x for x in S if x % 3 == 1]
        candidates.append(mod1)
        
        mod2 = [x for x in S if x % 3 == 2]
        candidates.append(mod2)
        
        high_pos = [x for x in S if x > max_abs // 2]
        candidates.append(high_pos)
        
        high_neg = [x for x in S if x < -max_abs // 2]
        candidates.append(high_neg)
        
        pos = [x for x in S if x > 0]
        candidates.append(pos)
        
        neg = [x for x in S if x < 0]
        candidates.append(neg)
        
        # Greedy with various orderings
        orders = [
            sorted(S, key=lambda x: -abs(x)),
            sorted(S, key=lambda x: abs(x)),
            sorted(S),
            sorted(S, reverse=True),
            sorted(S, key=lambda x: x),
        ]
        for order in orders:
            candidates.append(greedy_sum_free(S, order))
        
        # Find best initial candidate
        best = max(candidates, key=len)
        best_size = len(best)
        
        # Try local search on the best few candidates
        for cand in sorted(candidates, key=len, reverse=True)[:5]:
            if time.time() - start_time > time_limit:
                break
            improved = local_search_improve(list(cand), S, max_rounds=50)
            if len(improved) > best_size:
                best = improved
                best_size = len(best)
        
        # Verify
        if is_sum_free(best):
            return best
        else:
            return [x for x in S if x % 2 != 0]
    
    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline - 5:
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
            continue
        
        S = instance(size, seed)
        result = solve_instance(S)
        results.append(result)
    
    return results
