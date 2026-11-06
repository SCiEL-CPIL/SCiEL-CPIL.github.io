import time

_P = 1009

def construct():
    # Precompute quadratic residues and their smallest roots
    is_res = [False] * _P
    root_map = {}
    
    for x in range(1, _P):
        r = (x * x) % _P
        if r == 0:
            continue
        if not is_res[r]:
            is_res[r] = True
            root_map[r] = x

    # Sliding window to find the smallest s
    # Window size is 5
    window_count = 0
    for i in range(1, 6):
        if is_res[i]:
            window_count += 1
            
    max_s = _P - 5
    for s in range(1, max_s + 1):
        if window_count == 5:
            roots = [root_map[s + i] for i in range(5)]
            return {'start': s, 'roots': roots}
        
        if s < max_s:
            if is_res[s]:
                window_count -= 1
            if is_res[s + 5]:
                window_count += 1

    raise RuntimeError("No run of 5 consecutive non-zero quadratic residues found")
