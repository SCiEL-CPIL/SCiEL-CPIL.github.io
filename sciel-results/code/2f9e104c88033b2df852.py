N = 256

def construct():
    S = set()
    sums = set()
    remaining = set(range(1, N + 1))
    
    while remaining:
        best_x = None
        best_blocked = -1
        best_new_sums = None
        for x in remaining:
            new_sums = {2 * x}
            for a in S:
                new_sums.add(x + a)
            if sums & new_sums:
                continue
            blocked = 0
            for y in remaining:
                if y == x:
                    continue
                y_sums = {2 * y}
                for a in S:
                    y_sums.add(y + a)
                y_sums.add(y + x)
                if (sums | new_sums) & y_sums:
                    blocked += 1
            if blocked > best_blocked:
                best_blocked = blocked
                best_x = x
                best_new_sums = new_sums
        
        if best_x is None:
            break
        
        S.add(best_x)
        remaining.discard(best_x)
        sums |= best_new_sums
        
        to_remove = set()
        for y in remaining:
            y_sums = {2 * y}
            for a in S:
                y_sums.add(y + a)
            if sums & y_sums:
                to_remove.add(y)
        remaining -= to_remove
    
    return list(S)
