import time

def construct():
    deadline = time.time() + 85.0
    n = 128
    
    for num_colors in [3, 4, 5, 6, 7, 8, 10, 12, 16, 20, 30, 50, 100, 128]:
        if time.time() > deadline - 5:
            break
        result = _solve(n, num_colors, deadline)
        if result is not None:
            return result
    
    return _greedy(n)

def _solve(n, num_colors, deadline):
    colors = [-1] * n
    
    def backtrack(pos):
        if time.time() > deadline:
            return False
        if pos == n:
            return True
        
        for c in range(num_colors):
            valid = True
            for d in range(1, (pos + 1) // 3 + 1):
                i = pos - 3*d
                if i < 0:
                    break
                j = pos - 2*d
                k = pos - d
                if len({colors[i], colors[j], colors[k], c}) < 3:
                    valid = False
                    break
            if not valid:
                continue
            
            colors[pos] = c
            if backtrack(pos + 1):
                return True
            colors[pos] = -1
        
        return False
    
    if backtrack(0):
        return colors
    return None

def _greedy(n):
    colors = []
    for x in range(n):
        found = False
        for c in range(128):
            valid = True
            for d in range(1, (x + 1) // 3 + 1):
                i = x - 3*d
                if i < 0:
                    break
                j = x - 2*d
                k = x - d
                if len({colors[i], colors[j], colors[k], c}) < 3:
                    valid = False
                    break
            if valid:
                colors.append(c)
                found = True
                break
        if not found:
            colors.append(0)
    return colors
