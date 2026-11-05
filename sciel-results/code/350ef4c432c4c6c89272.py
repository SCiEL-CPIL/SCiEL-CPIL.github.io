import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    def is_sidon(s):
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d = s[j] - s[i]
                if d in diffs:
                    return False
                diffs.add(d)
        return True
    
    def get_covered(s, max_d=30):
        diffs = set()
        n = len(s)
        for i in range(n):
            for j in range(i+1, n):
                d = s[j] - s[i]
                if d <= max_d:
                    diffs.add(d)
        return diffs
    
    s = [0]
    covered = set()
    
    while len(covered) < 30:
        if time.time() > deadline:
            break
        
        missing = set(range(1, 31)) - covered
        best_x = None
        best_new_covered = set()
        best_gain = 0
        
        # Generate candidates
        candidates = set()
        for d in missing:
            for a in s:
                x1 = a + d
                x2 = a - d
                if 0 <= x1 <= 3000:
                    candidates.add(x1)
                if 0 <= x2 <= 3000:
                    candidates.add(x2)
        
        # Add small values
        for c in range(1, 400):
            candidates.add(c)
        
        for x in sorted(candidates):
            if x in s or x < 0:
                continue
            
            new_s = sorted(s + [x])
            if not is_sidon(new_s):
                continue
            
            new_covered = get_covered(new_s)
            gain = len(new_covered - covered)
            
            if gain > best_gain or (gain == best_gain and best_x is not None and x < best_x):
                best_gain = gain
                best_x = x
                best_new_covered = new_covered
        
        if best_x is None:
            # Fallback
            x = 2 * max(s) + 100
            while True:
                new_s = sorted(s + [x])
                if is_sidon(new_s):
                    break
                x += 1
            s = new_s
            covered = get_covered(s)
        else:
            s.append(best_x)
            s.sort()
            covered = best_new_covered
    
    return s
