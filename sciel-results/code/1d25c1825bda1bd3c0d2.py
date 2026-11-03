import time
import itertools

def construct(n):
    start_time = time.time()
    time_limit = 115.0 # Safety margin
    
    def get_overlap(a, b):
        # a, b are tuples
        max_ov = 0
        # Max overlap is len(a)-1
        for o in range(len(a)-1, -1, -1):
            if o == 0:
                max_ov = 0
                break
            if a[-o:] == b[:o]:
                max_ov = o
                break
        return max_ov

    def optimize_order(perms, m, time_limit):
        # perms is list of tuples
        k = len(perms)
        if k <= 2:
            return perms
        
        # Precompute overlap matrix
        # For speed, use list of lists
        ov_mat = [[0]*k for _ in range(k)]
        for i in range(k):
            pi = perms[i]
            for j in range(k):
                if i != j:
                    ov_mat[i][j] = get_overlap(pi, perms[j])
        
        # Initial order: 0..k-1
        order = list(range(k))
        
        def calc_score(order):
            s = 0
            for i in range(k-1):
                s += ov_mat[order[i]][order[i+1]]
            return s
        
        current_score = calc_score(order)
        
        deadline = start_time + time_limit
        improved = True
        # Use simple hill climbing with swaps
        # For k=720, this might be slow if done naively.
        # But we only do it for m=7 (k=720) and m=6 (k=120).
        # 720 swaps * 720 iterations? No, we scan all i.
        
        while improved and time.time() < deadline:
            improved = False
            for i in range(k-1):
                if time.time() > deadline:
                    break
                a = order[i]
                b = order[i+1]
                
                # Delta calculation
                prev = order[i-1] if i > 0 else None
                next_ = order[i+2] if i+2 < k else None
                
                old_s = 0
                new_s = 0
                
                if prev is not None:
                    old_s += ov_mat[prev][a]
                    new_s += ov_mat[prev][b]
                
                old_s += ov_mat[a][b]
                new_s += ov_mat[b][a]
                
                if next_ is not None:
                    old_s += ov_mat[b][next_]
                    new_s += ov_mat[a][next_]
                
                if new_s > old_s:
                    order[i], order[i+1] = b, a
                    current_score += new_s - old_s
                    improved = True
        
        return [perms[i] for i in order]

    word = [1]
    
    for m in range(2, n + 1):
        symbols = list(range(1, m))
        perms = list(itertools.permutations(symbols))
        
        # Decide whether to optimize
        # Optimization is beneficial for m >= 5
        if m >= 5:
            # Allocate time
            # Remaining time
            elapsed = time.time() - start_time
            remaining = time_limit - elapsed
            # Reserve 10s for construction and validation
            opt_time = max(1.0, remaining - 10.0)
            
            # For m=7, we have 720 perms.
            # For m=6, 120 perms.
            order = optimize_order(perms, m, opt_time)
        else:
            order = perms
            
        # Build new word
        new_word = []
        for i, p in enumerate(order):
            block = list(p) + [m] + list(p)
            if i == 0:
                new_word = block
            else:
                prev_p = list(order[i-1])
                # Overlap between prev_p and p
                max_ov = 0
                for o in range(min(len(prev_p), len(p)), -1, -1):
                    if prev_p[-o:] == p[:o]:
                        max_ov = o
                        break
                new_word += block[max_ov:]
        
        word = new_word
        
        # Time check
        if time.time() > time_limit - 5.0:
            break
            
    return "".join(map(str, word))
