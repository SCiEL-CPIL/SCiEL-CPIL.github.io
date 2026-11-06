    cache = {}
    def eval_n(n, exps_tuple=None):
        nonlocal best_n, best_score
        if n <= 1 or n > 10**18:
            return 0
        if exps_tuple and exps_tuple in cache:
            return cache[exps_tuple]
        s = _tau_perp_fast(n)
        if exps_tuple:
            cache[exps_tuple] = s
        if s > best_score:
            best_score = s
            best_n = n
        return s
