def construct():
    import math
    from math import isqrt
    
    LIMIT = 10**18
    
    # Sieve primes up to 1000
    def primes_upto(n):
        sieve = bytearray([1])*(n+1)
        sieve[0]=sieve[1]=0
        for i in range(2, isqrt(n)+1):
            if sieve[i]:
                for j in range(i*i, n+1, i):
                    sieve[j]=0
        return [i for i in range(2,n+1) if sieve[i]]
    
    all_primes = primes_upto(1000)
    p1 = [p for p in all_primes if p % 4 == 1]
    
    # DFS to maximize D = prod(e_i+1) with prod p_i^e_i <= LIMIT, non-increasing exponents
    best = {'D': 0, 'N': 0, 'exp': []}
    
    def dfs(idx, max_e, cur_prod, cur_D, exps):
        if cur_D > best['D'] or (cur_D == best['D'] and cur_prod > best['N']):
            # only record if we have at least one prime (cur_prod>1)
            if cur_prod >= 2:
                best['D'] = cur_D
                best['N'] = cur_prod
                best['exp'] = list(exps)
        if idx >= len(p1):
            return
        p = p1[idx]
        val = 1
        for e in range(0, max_e+1):
            if e > 0:
                val *= p
                if cur_prod * val > LIMIT:
                    break
            exps.append(e)
            dfs(idx+1, e, cur_prod*val, cur_D*(e+1), exps)
            exps.pop()
    
    dfs(0, 60, 1, 1, [])
    
    N = best['N']
    exps = best['exp']
    # primes used are p1[0:len(exps)] with exponents exps
    used = [(p1[i], exps[i]) for i in range(len(exps)) if exps[i] > 0]
    
    # find a,b for each prime p = a^2 + b^2
    def rep_prime(p):
        for a in range(1, isqrt(p)+1):
            b2 = p - a*a
            b = isqrt(b2)
            if b*b == b2 and b >= 1:
                return (a, b)
        raise ValueError
    
    # generate all representations via Gaussian integers
    def gauss_mul(z1, z2):
        x1,y1 = z1; x2,y2 = z2
        return (x1*x2 - y1*y2, x1*y2 + y1*x2)
    
    def gauss_pow(z, e):
        r = (1,0)
        base = z
        while e:
            if e & 1:
                r = gauss_mul(r, base)
            base = gauss_mul(base, base)
            e >>= 1
        return r
    
    pairs_set = set()
    # iterate over all f_i in 0..e_i
    import itertools
    ranges = [range(e+1) for (_, e) in used]
    for fs in itertools.product(*ranges):
        z = (1,0)
        for (p, e), f in zip(used, fs):
            a,b = rep_prime(p)
            # f copies of (a,b), e-f copies of (a,-b)
            z = gauss_mul(z, gauss_pow((a,b), f))
            z = gauss_mul(z, gauss_pow((a,-b), e-f))
        x,y = z
        u = abs(x); v = abs(y)
        if u > v: u,v = v,u
        if u >= 1 and v >= 1:
            pairs_set.add((u*u, v*v))
    
    pairs = sorted([a,b] for (a,b) in pairs_set)
    # verify a+b == N
    pairs = [[a,b] for (a,b) in pairs if a+b == N and a <= b]
    return {'N': N, 'pairs': pairs}
