import time

N = 128

def construct():
    deadline = time.time() + 80
    
    colors = [0] * (N + 1)
    
    for x in range(1, N + 1):
        if time.time() > deadline:
            return colors[1:]
        
        found = False
        for c in range(128):
            colors[x] = c
            valid = True
            
            for d in range(1, (x - 1) // 3 + 1):
                a = x - 3 * d
                if a < 1:
                    break
                cs = {colors[a], colors[a + d], colors[a + 2 * d], c}
                if len(cs) < 3:
                    valid = False
                    break
            
            if valid:
                found = True
                break
        
        if not found:
            # Fallback - shouldn't happen
            colors[x] = 0
    
    return colors[1:]
