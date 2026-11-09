import math

def construct():
    def find_solution(k):
        # DFS with integer arithmetic
        # State: num, den, next_d, rem, path
        # Current sum = num/den
        # Target = 1/1
        
        def dfs(num, den, next_d, rem, path):
            if rem == 0:
                if num == den:
                    return path
                return None
            
            # Deficit = 1 - num/den = (den - num) / den
            def_num = den - num
            if def_num <= 0:
                return None
            
            # We need sum of rem terms = def_num / def_den
            # Min d: 1/d <= def_num/def_den => d >= def_den/def_num
            min_d = (def_den + def_num - 1) // def_num
            min_d = max(next_d, min_d)
            
            # Max d: rem/d >= def_num/def_den => d <= rem * def_den / def_num
            max_d = (rem * def_den) // def_num
            
            if min_d > max_d:
                return None
                
            for d in range(min_d, max_d + 1):
                # Pruning: even if we pick the smallest possible denominators d, d+1, ..., d+rem-1
                # their sum must be >= deficit
                # Approx check: rem/d >= deficit is already used for max_d.
                # Stronger check: sum(1/(d+i)) >= deficit
                
                new_num = num * d + den
                new_den = den * d
                g = math.gcd(new_num, new_den)
                if g > 1:
                    new_num //= g
                    new_den //= g
                    
                res = dfs(new_num, new_den, d + 1, rem - 1, path + [d])
                if res is not None:
                    return res
            return None
        
        return dfs(0, 1, 5, k, [])

    for k in range(8, 12):
        sol = find_solution(k)
        if sol:
            return sol
            
    raise RuntimeError("No solution found")
