def construct():
    import time
    start_time = time.time()
    DEADLINE = 85.0  # safety margin before 90s
    
    selected = []
    selected_set = set()
    
    for x in range(1, 257):
        if time.time() - start_time > DEADLINE:
            break
        
        # Check if adding x creates a 3-term AP
        valid = True
        for a in selected:
            for b in selected:
                if a == b:
                    continue
                # Three numbers a, b, x form a 3-term AP if they can be ordered
                # as p < q < r with p + r = 2q
                # Case 1: x is smallest: x + b = 2a or x + a = 2b
                # Case 2: x is middle: a + b = 2x
                # Case 3: x is largest: a + x = 2b or b + x = 2a
                
                # More simply: for any pair (a, b) in selected, check if x completes an AP
                # The three possible APs involving x and two existing elements:
                # (i)   x, a, b where x < a < b: x + b = 2a => x = 2a - b
                # (ii)  a, x, b where a < x < b: a + b = 2x => x = (a+b)/2
                # (iii) a, b, x where a < b < x: a + x = 2b => x = 2b - a
                
                # So check:
                # 2*a - b == x (and 2*a - b < a, i.e., a < b)
                # (a + b) % 2 == 0 and (a+b)//2 == x (and a < x < b)
                # 2*b - a == x (and b < x, i.e., a < b)
                
                if a < b:
                    if 2*a - b == x:  # x smallest
                        valid = False
                        break
                    if 2*b - a == x:  # x largest
                        valid = False
                        break
                    if (a + b) % 2 == 0 and (a + b) // 2 == x:  # x middle
                        valid = False
                        break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            selected_set.add(x)
    
    return selected
