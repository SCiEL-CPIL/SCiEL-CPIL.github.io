import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    n = 192
    max_size = 48
    
    # Try descending order first
    selected = []
    
    for x in range(n, 0, -1):
        if time.time() > deadline:
            break
            
        valid = True
        s_len = len(selected)
        
        if s_len > 0:
            # Precompute residues of selected mod x
            # Only needed if s_len >= 2 for Check 1/2a
            if s_len >= 2:
                res_map = {}
                for v in selected:
                    r = v % x
                    if r not in res_map:
                        res_map[r] = []
                    res_map[r].append(v)
            else:
                res_map = None

            # Check 1: x does not divide sum of any 3 distinct in selected
            if s_len >= 3:
                for i in range(s_len):
                    a = selected[i]
                    for j in range(i+1, s_len):
                        b = selected[j]
                        target_res = (- (a + b)) % x
                        if target_res in res_map:
                            for c in res_map[target_res]:
                                if c != a and c != b:
                                    valid = False
                                    break
                        if not valid:
                            break
                    if not valid:
                        break
            
            # Check 2a: x does not divide (a+b) for any distinct a,b in selected
            if valid and s_len >= 2:
                for i in range(s_len):
                    a = selected[i]
                    a_res = a % x
                    target_res = (-a_res) % x
                    if target_res in res_map:
                        for b in res_map[target_res]:
                            if b != a:
                                valid = False
                                break
                    if not valid:
                        break
            
            # Check 2b: y does not divide (x+a+b) for any distinct a,b in selected, any y in selected
            if valid and s_len >= 2:
                for i in range(s_len):
                    a = selected[i]
                    for j in range(i+1, s_len):
                        b = selected[j]
                        s_ab = x + a + b
                        for y in selected:
                            if s_ab % y == 0:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
        
        if valid:
            selected.append(x)
            if len(selected) >= max_size:
                break

    return selected
