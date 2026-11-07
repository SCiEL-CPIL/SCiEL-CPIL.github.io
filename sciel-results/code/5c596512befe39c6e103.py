import math

def squarefree_kernel(n):
    if n <= 1:
        return n
    result = 1
    d = 2
    while d * d <= n:
        count = 0
        while n % d == 0:
            n //= d
            count += 1
        if count % 2 == 1:
            result *= d
        d += 1 if d == 2 else 2
    if n > 1:
        result *= n
    return result

def construct():
    target = 210
    sfk = [0] * 10211
    for i in range(211, 10211):
        sfk[i] = squarefree_kernel(i)
    
    for max_val in range(211, 10211):
        dp = {1: None}
        found = False
        for num in range(211, max_val + 1):
            s = sfk[num]
            new_dp = {}
            for k in dp:
                new_k = squarefree_kernel(k * s)
                if new_k not in new_dp:
                    new_dp[new_k] = (k, num)
            for new_k, val in new_dp.items():
                if new_k not in dp:
                    dp[new_k] = val
            if target in dp:
                found = True
                break
        if found:
            result = []
            k = target
            while k != 1:
                prev_k, num = dp[k]
                result.append(num)
                k = prev_k
            prod = 1
            for x in result:
                prod *= x
            prod *= 210
            sq = math.isqrt(prod)
            if sq * sq == prod:
                return [int(x) for x in result]
    return [224, 240]
