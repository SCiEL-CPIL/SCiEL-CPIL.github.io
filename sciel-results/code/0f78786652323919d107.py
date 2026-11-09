import time
import random

def construct():
    start = time.time()
    deadline = start + 80
    
    n = 128
    
    # Precompute all 4-APs as a list of tuples
    aps = []
    for a in range(n):
        max_d = (n - 1 - a) // 3
        for d in range(1, max_d + 1):
            aps.append((a, a+d, a+2*d, a+3*d))
    
    num_aps = len(aps)
    
    # For each position, list of AP indices it belongs to
    pos_to_aps = [[] for _ in range(n)]
    for i, ap in enumerate(aps):
        for pos in ap:
            pos_to_aps[pos].append(i)
    
    # Fast violation check
    def is_viol(idx, col):
        a, b, c, e = aps[idx]
        ca, cb, cc, ce = col[a], col[b], col[c], col[e]
        if ca == cb and ca == cc and ca == ce:
            return True
        if ca == cb:
            if cc == ce:
                return ca != cc
            if ca == cc:
                return cb != ce
            if ca == ce:
                return cb != cc
            return False
        if ca == cc:
            if cb == ce:
                return ca != cb
            if ca == ce:
                return cb != cc
            return False
        if ca == ce:
            if cb == cc:
                return ca != cb
            return False
        if cb == cc:
            if cb == ce:
                return ca != cb
            return False
        if cb == ce:
            if cb == cc:
                return ca != cb
            return False
        if cc == ce:
            return ca != cc
        return False
    
    best = None
    best_count = 128
    
    for k in range(3, 11):
        if time.time() > deadline - 25:
            break
        for attempt in range(300):
            if time.time() > deadline - 20:
                break
            
            if attempt % 4 == 0:
                col = [random.randint(0, k-1) for _ in range(n)]
            elif attempt % 4 == 1:
                col = [i % k for i in range(n)]
            elif attempt % 4 == 2:
                col = [(i * 3) % k for i in range(n)]
            else:
                col = [(i * 5) % k for i in range(n)]
            
            total_viol = 0
            for i in range(num_aps):
                if is_viol(i, col):
                    total_viol += 1
            
            for step in range(30000):
                if time.time() > deadline - 20:
                    break
                if total_viol == 0:
                    used = len(set(col))
                    if used < best_count:
                        best_count = used
                        best = col[:]
                    break
                
                i = random.randint(0, num_aps - 1)
                if not is_viol(i, col):
                    for _ in range(15):
                        i = random.randint(0, num_aps - 1)
                        if is_viol(i, col):
                            break
                    else:
                        for i in range(num_aps):
                            if is_viol(i, col):
                                break
                        else:
                            break
                
                a, b, c, e = aps[i]
                pos = random.choice((a, b, c, e))
                new = random.randint(0, k-1)
                if new == col[pos]:
                    continue
                
                old = col[pos]
                for j in pos_to_aps[pos]:
                    if is_viol(j, col):
                        total_viol -= 1
                col[pos] = new
                for j in pos_to_aps[pos]:
                    if is_viol(j, col):
                        total_viol += 1
            
            if best_count <= k:
                break
    
    if best is not None:
        mapping = {}
        next_id = 0
        result = []
        for c in best:
            if c not in mapping:
                mapping[c] = next_id
                next_id += 1
            result.append(mapping[c])
        
        # Verify
        for i in range(num_aps):
            a, b, c, e = aps[i]
            if len({result[a], result[b], result[c], result[e]}) < 3:
                # Invalid, use fallback
                break
        else:
            return result
    
    # Fallback: greedy
    colors = []
    for x in range(1, 129):
        for color in range(128):
            if (all(len({color}|{colors[x-1-j*d] for j in (1,2,3)})>=3 for d in range(1,(x-1)//3+1))
                and all(any(colors[x-1-j*d]!=color for j in (1,2)) for d in range(1,(x-1)//2+1))):
                colors.append(color)
                break
    return colors
