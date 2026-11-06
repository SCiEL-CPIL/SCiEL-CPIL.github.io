"""Rewritten program: maximize reciprocal sum of 3-AP-free subset of [1,256]."""
import time

# EVOLVE-BLOCK-START
def construct():
    N = 256
    deadline = time.time() + 82.0  # safety margin before 90s kill
    
    def creates_ap(S, y):
        """Check if adding y to S creates a 3-term arithmetic progression."""
        for a in S:
            c = 2 * y - a
            if c != a and c != y and c in S:
                return True
        for b in S:
            c = 2 * b - y
            if c != b and c != y and c in S:
                return True
        return False
    
    def reciprocal_sum(S):
        return sum(1.0 / x for x in S)
    
    def greedy(order):
        S = set()
        for x in order:
            if not creates_ap(S, x):
                S.add(x)
        return S
    
    def local_search(S):
        """1-add and 1-swap local search."""
        S = set(S)
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try 1-add moves
            best_gain = 0
            best_y = None
            for y in range(1, N + 1):
                if y in S:
                    continue
                if not creates_ap(S, y):
                    gain = 1.0 / y
                    if gain > best_gain:
                        best_gain = gain
                        best_y = y
            
            if best_y is not None and best_gain > 0:
                S.add(best_y)
                improved = True
                continue
            
            # Try 1-swap moves
            best_gain = 0
            best_x = None
            best_y = None
            for x in S:
                if time.time() > deadline:
                    break
                for y in range(1, N + 1):
                    if y in S:
                        continue
                    S_temp = S - {x}
                    if not creates_ap(S_temp, y):
                        gain = 1.0 / y - 1.0 / x
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_y = y
                if time.time() > deadline:
                    break
            
            if best_x is not None and best_y is not None and best_gain > 0:
                S.remove(best_x)
                S.add(best_y)
                improved = True
        
        return S
    
    # Try multiple initializations
    best_S = set()
    best_score = 0.0
    
    # Greedy in ascending order
    S1 = greedy(range(1, N + 1))
    S1 = local_search(S1)
    score1 = reciprocal_sum(S1)
    if score1 > best_score:
        best_S = set(S1)
        best_score = score1
    
    if time.time() > deadline:
        return list(best_S)
    
    # Try greedy with some randomization
    import random
    rng = random.Random(42)
    for trial in range(20):
        if time.time() > deadline:
            break
        # Create a shuffled order biased toward small numbers
        order = list(range(1, N + 1))
        # Sort by 1/x with some noise
        order.sort(key=lambda x: -1.0 / x + rng.uniform(0, 0.5) / x)
        S = greedy(order)
        S = local_search(S)
        score = reciprocal_sum(S)
        if score > best_score:
            best_S = set(S)
            best_score = score
    
    return list(best_S)
# EVOLVE-BLOCK-END
