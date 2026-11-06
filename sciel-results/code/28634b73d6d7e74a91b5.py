def construct():
    import time
    start_time = time.time()
    deadline = 85.0
    
    N = 256
    S = set(range(171, 257))
    
    def is_valid_add(x, S):
        S_list = sorted(S)
        S_set = S
        
        # x as a: need b > x, b <= c, b,c in S, x | (b+c)
        # b+c in [2*(x+1), 512], multiples of x: 3x, 4x, ...
        for k in range(3, 513 // x + 1):
            m = k * x
            if m < 2 * (x + 1):
                continue
            if m > 512:
                break
            # Need b + c = m, b > x, b <= c, b,c in S
            # b from max(x+1, m-256) to min(256, m//2)
            b_lo = max(x + 1, m - 256)
            b_hi = min(256, m // 2)
            for b in range(b_lo, b_hi + 1):
                c = m - b
                if b <= c and b in S_set and c in S_set:
                    return False
        
        # x as b: a < x, x <= c, a,c in S, a | (x+c)
        # x+c in [x+171, x+256] (since c >= 171 initially, but S changes)
        # Actually c in S, c >= x (since c >= b = x)
        # a | (x+c), a < x
        for a in S_list:
            if a >= x:
                break
            # Need c in S, c >= x, a | (x+c)
            # x+c in [x + min(S), x + 256]
            # (x+c) % a == 0, so x+c = k*a for some k
            # c = k*a - x, need c in S, c >= x
            # k*a - x >= x => k*a >= 2x => k >= ceil(2x/a)
            # k*a - x <= 256 => k*a <= 256+x => k <= floor((256+x)/a)
            k_lo = (2 * x + a - 1) // a  # ceil(2x/a)
            k_hi = (256 + x) // a
            for k in range(k_lo, k_hi + 1):
                c = k * a - x
                if c >= x and c in S_set:
                    return False
        
        # x as c: a < b <= x, a,b in S, a | (b+x)
        # b+x in [a+1+x, x+x] = [a+1+x, 2x]
        # a | (b+x), so b+x = k*a
        # b = k*a - x, need b in S, a < b <= x
        for a in S_list:
            if a >= x:
                break
            # b = k*a - x, a < b <= x
            # a < k*a - x => k*a > a+x => k > 1 + x/a => k >= floor(x/a) + 2 (roughly)
            # Actually: k*a - x > a => k*a > a + x => k > (a+x)/a = 1 + x/a
            # k*a - x <= x => k*a <= 2x => k <= 2x/a
            k_lo = (a + x) // a + 1  # k > (a+x)/a, so k >= floor((a+x)/a) + 1
            k_hi = (2 * x) // a
            for k in range(k_lo, k_hi + 1):
                b = k * a - x
                if a < b <= x and b in S_set:
                    return False
        
        return True
    
    candidates = list(range(1, 171))
    
    for x in sorted(candidates, reverse=True):
        if time.time() > deadline - 10:
            break
        if is_valid_add(x, S):
            S.add(x)
    
    return sorted(S)
