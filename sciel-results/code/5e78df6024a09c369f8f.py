import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115.0

    def instance(size, seed, limit=10000):
        rng = random.Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_valid_add(x, A):
        if 2 * x in A:
            return False
        for a in A:
            if x + a in A:
                return False
        return True

    def greedy_fill(S, A, order=None):
        # A is a set, S is the full set
        # Add elements from S \ A to A if valid
        # order: list of elements to try first
        if order is None:
            order = [x for x in S if x not in A]
        for x in order:
            if x not in A and is_valid_add(x, A):
                A.add(x)
        return A

    def local_search(S, A):
        # Try to improve A by removing one element and refilling
        improved = True
        while improved and time.time() < deadline:
            improved = False
            best_A = set(A)
            best_size = len(A)
            
            # Try removing each element
            for x in list(A):
                A_trial = set(A)
                A_trial.remove(x)
                # Greedily refill
                # To make it deterministic and fast, we can just try to add any valid element
                # But the order matters. Let's use a fixed order (e.g., sorted by abs value) for refill
                # Or just the "fill" logic which iterates S.
                # To avoid O(N^2) inside the loop, we can just call greedy_fill
                greedy_fill(S, A_trial)
                
                if len(A_trial) > best_size:
                    best_size = len(A_trial)
                    best_A = A_trial
                    improved = True
                    # We can break early if we find a better one? 
                    # Or keep looking for the best improvement?
                    # Let's keep looking for the best improvement in this pass.
            
            if improved:
                A = best_A
        
        return A

    def solve_instance(S):
        if not S:
            return []
        
        # 1. Initial Heuristic: Odds
        A1 = set(x for x in S if x % 2 != 0)
        greedy_fill(S, A1)
        
        # 2. Initial Heuristic: Greedy by abs value (descending)
        A2 = set()
        for x in sorted(S, key=abs, reverse=True):
            if is_valid_add(x, A2):
                A2.add(x)
        greedy_fill(S, A2)
        
        # 3. Initial Heuristic: Greedy by abs value (ascending)
        A3 = set()
        for x in sorted(S, key=abs):
            if is_valid_add(x, A3):
                A3.add(x)
        greedy_fill(S, A3)
        
        # 4. Randomized Greedy
        A4 = set()
        rng = random.Random(123)
        for _ in range(5):
            A_temp = set()
            shuffled = S[:]
            rng.shuffle(shuffled)
            for x in shuffled:
                if is_valid_add(x, A_temp):
                    A_temp.add(x)
            greedy_fill(S, A_temp)
            if len(A_temp) > len(A4):
                A4 = A_temp

        # Pick the best
        A = max([A1, A2, A3, A4], key=len)
        
        # Local Search
        A = local_search(S, A)
        
        return list(A)

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        if time.time() > deadline:
            # Fallback to odds if time runs out
            S = instance(size, seed)
            results.append([x for x in S if x % 2 != 0])
            continue
            
        S = instance(size, seed)
        subset = solve_instance(S)
        results.append(subset)
        
    return results
