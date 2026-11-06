import time

def construct():
    start_time = time.time()
    deadline = start_time + 85
    n = 256
    
    def would_create_4ap(selected, x):
        s = selected
        max_d = (n - x) // 3
        for d in range(1, max_d + 1):
            if x+d in s and x+2*d in s and x+3*d in s:
                return True
        max_d = min(x-1, (n-x)//2)
        for d in range(1, max_d + 1):
            if x-d in s and x+d in s and x+2*d in s:
                return True
        max_d = min((x-1)//2, n-x)
        for d in range(1, max_d + 1):
            if x-2*d in s and x-d in s and x+d in s:
                return True
        max_d = (x-1)//3
        for d in range(1, max_d + 1):
            if x-3*d in s and x-2*d in s and x-d in s:
                return True
        return False
    
    def new_3aps(selected, x):
        s = selected
        cnt = 0
        max_d = (n - x) // 2
        for d in range(1, max_d + 1):
            if x+d in s and x+2*d in s:
                cnt += 1
        max_d = min(x-1, n-x)
        for d in range(1, max_d + 1):
            if x-d in s and x+d in s:
                cnt += 1
        max_d = (x-1) // 2
        for d in range(1, max_d + 1):
            if x-2*d in s and x-d in s:
                cnt += 1
        return cnt
    
    def count_3aps(selected):
        s = selected
        cnt = 0
        for a in s:
            max_d = (n - a) // 2
            for d in range(1, max_d + 1):
                if a+d in s and a+2*d in s:
                    cnt += 1
        return cnt
    
    selected = set()
    for _ in range(n):
        if time.time() > deadline - 15:
            break
        best_x = None
        best_gain = -1
        for x in range(1, n+1):
            if x in selected:
                continue
            if would_create_4ap(selected, x):
                continue
            gain = new_3aps(selected, x)
            if gain > best_gain:
                best_gain = gain
                best_x = x
        if best_x is None:
            break
        selected.add(best_x)
    
    current_score = count_3aps(selected)
    
    if time.time() < deadline - 5:
        improved = True
        while improved and time.time() < deadline - 5:
            improved = False
            for x in list(selected):
                if time.time() > deadline - 5:
                    break
                temp = selected - {x}
                base = count_3aps(temp)
                best_y = None
                best_new = base
                for y in range(1, n+1):
                    if y in temp:
                        continue
                    if would_create_4ap(temp, y):
                        continue
                    gain = new_3aps(temp, y)
                    if base + gain > best_new:
                        best_new = base + gain
                        best_y = y
                if best_y is not None and best_new > current_score:
                    selected = temp | {best_y}
                    current_score = best_new
                    improved = True
    
    return sorted(selected)
