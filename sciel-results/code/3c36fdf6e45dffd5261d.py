def construct():
    import random
    
    def compute_sizes(A):
        sums = set()
        prods = set()
        for i, a in enumerate(A):
            for j, b in enumerate(A):
                sums.add(a + b)
                prods.add(a * b)
        return len(sums), len(prods)
    
    def score(A):
        s, p = compute_sizes(A)
        return max(s, p)
    
    # Start with a geometric progression-like set
    # Powers of 2 up to 8192, then fill in
    gp = [2**i for i in range(14)]  # 1,2,4,...,8192
    # Need 6 more elements
    # Try adding elements that create few new products
    candidates = [3, 5, 6, 7, 9, 10, 11, 12, 14, 15, 18, 20, 21, 24, 28, 30, 35, 40, 42, 48, 56, 60, 70, 72, 80, 84, 90, 96, 105, 112, 120, 140, 144, 160, 168, 180, 192, 210, 224, 240, 280, 288, 320, 336, 360, 384, 420, 448, 480, 560, 576, 640, 672, 720, 768, 840, 896, 960, 1120, 1152, 1280, 1344, 1440, 1536, 1680, 1792, 1920, 2240, 2304, 2560, 2688, 2880, 3072, 3360, 3584, 3840, 4480, 4608, 5120, 5376, 5760, 6144, 6720, 7168, 7680, 8960, 9216, 10240]
    candidates = [c for c in candidates if c <= 10000 and c not in gp]
    
    random.seed(42)
    A = gp[:]
    for c in candidates:
        if len(A) >= 20:
            break
        A.append(c)
    A = A[:20]
    A = sorted(set(A))
    
    if len(A) < 20:
        # Fill with random
        for v in range(1, 10001):
            if len(A) >= 20:
                break
            if v not in A:
                A.append(v)
        A = sorted(A)
    
    best_A = A[:]
    best_score = score(best_A)
    
    # Local search
    for iteration in range(3000):
        i = random.randint(0, 19)
        old_val = best_A[i]
        improved = False
        for _ in range(50):
            new_val = random.randint(1, 10000)
            if new_val == old_val or new_val in best_A:
                continue
            best_A[i] = new_val
            s_val = score(best_A)
            if s_val < best_score:
                best_score = s_val
                improved = True
            else:
                best_A[i] = old_val
        if not improved and iteration % 100 == 0:
            # Try a bigger perturbation
            for _ in range(5):
                i1 = random.randint(0, 19)
                i2 = random.randint(0, 19)
                if i1 == i2:
                    continue
                old1, old2 = best_A[i1], best_A[i2]
                new1 = random.randint(1, 10000)
                new2 = random.randint(1, 10000)
                if new1 == new2 or new1 in best_A or new2 in best_A:
                    continue
                best_A[i1] = new1
                best_A[i2] = new2
                s_val = score(best_A)
                if s_val < best_score:
                    best_score = s_val
                else:
                    best_A[i1] = old1
                    best_A[i2] = old2
    
    return sorted(best_A)
