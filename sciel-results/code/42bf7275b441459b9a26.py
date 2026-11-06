from fractions import Fraction
import itertools

def construct():
    # Meet-in-the-middle: split 1..30 into 1..15 and 16..30
    # For each half, enumerate all 3^15 sign assignments
    # Find pair with smallest nonzero |sum1 + sum2|
    
    half1 = list(range(1, 16))   # 1..15
    half2 = list(range(16, 31))  # 16..30
    
    # Enumerate all assignments for half1
    # Use iterative generation to avoid recursion depth
    def enumerate_half(indices):
        results = []
        n = len(indices)
        # Generate all 3^n tuples
        for tup in itertools.product([-1, 0, 1], repeat=n):
            s = 0.0
            for i, c in zip(indices, tup):
                if c != 0:
                    s += c / i
            results.append((s, tup))
        return results
    
    # This is 3^15 = 14.3M, might be slow in Python
    # Let's try a smarter approach: dynamic programming or greedy
    
    # Actually, let's just do a local search with random restarts
    import random
    random.seed(42)
    
    best_val = float('inf')
    best_c = [0]*30
    
    def eval_c(c):
        s = sum(c[i] / (i+1) for i in range(30))
        return abs(s)
    
    # Start from current solution
    c = [0]*28 + [1, -1]
    best_val = eval_c(c)
    best_c = c[:]
    
    # Local search: try flipping each coefficient
    for iteration in range(100000):
        improved = False
        for i in range(30):
            for new_val in [-1, 0, 1]:
                if new_val == c[i]:
                    continue
                c[i] = new_val
                val = eval_c(c)
                if 0 < val < best_val:
                    best_val = val
                    best_c = c[:]
                    improved = True
                c[i] = c[i]  # revert
        if not improved:
            # Random restart
            c = [random.choice([-1, 0, 1]) for _ in range(30)]
            val = eval_c(c)
            if 0 < val < best_val:
                best_val = val
                best_c = c[:]
    
    return best_c
