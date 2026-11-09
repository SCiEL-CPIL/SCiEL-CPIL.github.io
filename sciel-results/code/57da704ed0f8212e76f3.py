import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin
    
    target = Fraction(5, 31)
    
    # Known solution
    best_solution = [8, 31, 248]
    
    # Search for better solutions (max denominator < 248)
    # Denominators must be >= 7
    
    for M in range(247, 6, -1):
        if time.time() > deadline:
            break
        
        # Precompute suffix sums of reciprocals for pruning
        # suffix[i] = sum(1/j for j in range(i, M+1))
        suffix = [Fraction(0, 1)] * (M + 2)
        for i in range(M, 6, -1):
            suffix[i] = suffix[i + 1] + Fraction(1, i)
        
        path = []
        
        def dfs(start, remaining):
            if remaining == 0:
                return True
            if remaining < 0 or start > M:
                return False
            # Pruning: Can we achieve 'remaining' with denominators start..M?
            if remaining > suffix[start]:
                return False
            
            # Try not picking 'start'
            if dfs(start + 1, remaining):
                return True
            
            # Try picking 'start'
            path.append(start)
            if dfs(start + 1, remaining - Fraction(1, start)):
                return True
            path.pop()
            
            return False
        
        if dfs(7, target):
            # Verify solution
            if len(path) > 0 and len(path) <= 256:
                s = sum(Fraction(1, d) for d in path)
                if s == target:
                    return path
                    
    return best_solution
