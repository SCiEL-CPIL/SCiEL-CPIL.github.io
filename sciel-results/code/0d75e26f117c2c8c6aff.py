import time
from fractions import Fraction

def construct():
    start_time = time.time()
    deadline = start_time + 85.0
    
    target = Fraction(5, 31)
    
    # Known valid solution: [7, 62, 434]
    # We try to find a solution with a smaller maximum denominator.
    
    def check_D_max(D_max):
        # Returns a list of denominators if a solution exists with max <= D_max, else None
        class Solver:
            def __init__(self, D_max):
                self.D_max = D_max
                self.found = False
                self.sol = []
                
            def dfs(self, rem, last_denom, path):
                if time.time() > deadline:
                    return
                if rem == 0:
                    self.found = True
                    self.sol = path
                    return
                    
                # k must be > 1/rem
                # 1/k < rem => k > 1/rem
                # k_min = floor(1/rem) + 1
                # Also k > last_denom
                # rem = n/d. 1/rem = d/n.
                k_min = max(last_denom + 1, (rem.denominator // rem.numerator) + 1)
                
                if k_min > self.D_max:
                    return
                
                # Pruning: 
                # The maximum sum of distinct unit fractions from k_min to D_max is H(D_max) - H(k_min-1).
                # If rem > this sum, it's impossible.
                # For D_max <= 1000, we can approximate or just let the loop run (it's fast).
                # For larger D_max, we might need better pruning, but we expect D_max to be small.
                
                for k in range(k_min, self.D_max + 1):
                    # 1/k must be strictly less than rem
                    if Fraction(1, k) >= rem:
                        continue
                    
                    new_rem = rem - Fraction(1, k)
                    
                    # Optional pruning: 
                    # If new_rem > 0, we need to represent it with denominators > k.
                    # The smallest possible next denominator is k+1.
                    # If new_rem > 1/(k+1) + 1/(k+2) + ... + 1/D_max, fail.
                    # This is the same as the max sum check for the next step.
                    
                    self.dfs(new_rem, k, path + [k])
                    if self.found:
                        return

        solver = Solver(D_max)
        solver.dfs(target, 0, [])
        if solver.found:
            return solver.sol
        return None

    # Search for the minimal D_max
    # Start from a reasonable lower bound.
    # 1/7 is the largest term in our known solution.
    # If we use a term 1/6, remainder is negative or too large?
    # 5/31 - 1/6 = (30-31)/186 = -1/186. So 1/6 is too big.
    # So first term must be >= 7.
    # If first term is 7, we found a solution with max 434.
    # If first term is 8, max was 1736.
    # If first term is 9, max was 5580.
    # So 7 seems promising.
    
    # We iterate D_max from 100 upwards.
    # If we find a solution, we return it.
    for D_max in range(100, 1000):
        if time.time() > deadline:
            break
        sol = check_D_max(D_max)
        if sol:
            return sol
            
    # Fallback to the known solution
    return [7, 62, 434]
