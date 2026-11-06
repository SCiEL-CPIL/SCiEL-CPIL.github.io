import random
from itertools import combinations_with_replacement

def construct():
    n = 4096
    max_k = 40  # Theoretical upper bound
    
    def check_and_add(S, T, x):
        """
        Fast check if x can be added to S.
        S: list of elements.
        T: set of triple sums.
        x: candidate.
        Returns True if valid, and updates S and T in place if valid? 
        No, better to return new_T or just modify S,T if we pass them as mutable and commit.
        For safety in parallel/retry logic, let's just return a boolean and the new sums set if valid.
        """
        # We need to check:
        # 1. Internal collisions among new sums.
        # 2. Collisions with T.
        
        # Expected number of new sums:
        # 1 (xxx) + len(S) (xxs) + len(S)*(len(S)+1)/2 (xss)
        k = len(S)
        expected = 1 + k + k * (k + 1) // 2
        
        new_sums = set()
        
        # 1. 3x
        new_sums.add(3 * x)
        
        # 2. 2x + s
        for s in S:
            new_sums.add(2 * x + s)
            
        # 3. x + s1 + s2
        # Optimization: iterate i, j
        for i in range(k):
            s1 = S[i]
            # s1 + s2 where s2 >= s1
            # We can just iterate all pairs if k is small, but combinations_with_replacement is slow in Python.
            # Manual loop is faster.
            for j in range(i, k):
                new_sums.add(x + s1 + S[j])
                
        # If any collision occurred, len(new_sums) < expected
        if len(new_sums) < expected:
            return False, None
            
        # Check against existing T
        if new_sums & T:
            return False, None
            
        return True, new_sums

    def build_greedy(candidates_order, initial_S=None):
        if initial_S is None:
            S = []
            T = set()
        else:
            S = list(initial_S)
            T = set()
            for t in combinations_with_replacement(S, 3):
                T.add(sum(t))
        
        for x in candidates_order:
            if x in S: # O(k) check, k<=40, so fast. Or use a set for S?
                continue
            # Using a set for S membership check is O(1). Let's maintain S_set.
            # Actually, we can just check `if x in S_set`.
            
            valid, new_sums = check_and_add(S, T, x)
            if valid:
                S.append(x)
                S_set.add(x)
                T.update(new_sums)
                if len(S) >= max_k:
                    break
                    
        return S

    # We need S_set in build_greedy. Let's refactor.
    
    def build_greedy_fast(candidates_order):
        S = []
        S_set = set()
        T = set()
        
        for x in candidates_order:
            if x in S_set:
                continue
            
            k = len(S)
            expected = 1 + k + k * (k + 1) // 2
            new_sums = set()
            
            # 3x
            new_sums.add(3 * x)
            
            # 2x + s
            for s in S:
                new_sums.add(2 * x + s)
                
            # x + s1 + s2
            for i in range(k):
                s1 = S[i]
                for j in range(i, k):
                    new_sums.add(x + s1 + S[j])
                    
            if len(new_sums) < expected:
                continue
                
            if new_sums & T:
                continue
                
            # Valid
            S.append(x)
            S_set.add(x)
            T.update(new_sums)
            
            if len(S) >= max_k:
                break
                
        return S

    all_candidates = list(range(1, n + 1))
    best_size = 0
    best_set = []
    
    # Strategy 1: Pure Random Greedy (Many trials)
    # Since the problem is small, we can run many trials.
    # 100 trials * 4096 candidates * ~100 ops/check = 40M ops. Should be < 5s.
    
    random.seed(42)
    
    # Run 200 trials of random order
    for i in range(200):
        cands = all_candidates[:]
        random.shuffle(cands)
        S = build_greedy_fast(cands)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            if best_size == max_k:
                break
                
    # Strategy 2: Random start with a few large numbers?
    # Or just more random trials.
    
    # If we didn't find 40, try a local search?
    # Or just return best_set.
    
    return best_set
