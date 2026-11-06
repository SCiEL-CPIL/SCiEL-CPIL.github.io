import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Leave 10s margin

    def dfs(num, den, last_d, k_left):
        if time.time() > deadline:
            return None

        if k_left == 1:
            if den % num == 0:
                d = den // num
                if d > last_d and d <= 10000000:
                    return [d]
            return None

        if k_left == 0:
            return [] if num == 0 else None

        d_min = max(last_d + 1, den // num + 1)
        d_max = (k_left * den) // num
        
        if d_min > d_max:
            return None

        for d in range(d_min, d_max + 1):
            if d > 10000000:
                break
            
            # Check if max possible sum is less than target
            # MaxSum = 1/d + (k_left-1)/(d+1)
            # If MaxSum < num/den, break
            # den * (d + 1 + d*(k_left-1)) < num * d * (d+1)
            
            lhs = den * (d + 1 + d * (k_left - 1))
            rhs = num * d * (d + 1)
            
            if lhs < rhs:
                break

            new_num = num * d - den
            new_den = den * d
            
            if new_num < 0:
                continue

            g = gcd(new_num, new_den)
            if g > 1:
                new_num //= g
                new_den //= g
            
            res = dfs(new_num, new_den, d, k_left - 1)
            if res is not None:
                return [d] + res

        return None

    for k in [8, 9, 10]:
        res = dfs(1, 1, 4, k)
        if res is not None:
            # Verify
            s_num = 0
            s_den = 1
            for d in res:
                g = gcd(s_den, d)
                s_num = s_num * (d // g) + (s_den // g)
                s_den = s_den * (d // g)
                g2 = gcd(s_num, s_den)
                if g2 > 1:
                    s_num //= g2
                    s_den //= g2
            
            if s_num == s_den and all(5 <= d <= 10000000 for d in res) and len(set(res)) == len(res):
                return res

    raise RuntimeError("No solution found")
