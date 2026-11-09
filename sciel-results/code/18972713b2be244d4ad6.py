import time

def construct():
    # 1. Generate seed
    selected = set()
    for x in range(1, 257):
        if all(not all(x-j*d in selected for j in (1,2,3)) for d in range(1, (x-1)//3+1)):
            selected.add(x)
    
    # Helper to count 3-APs
    def count_3ap(A):
        s = set(A)
        count = 0
        for a in s:
            for d in range(1, 257 - a):
                if a + d in s and a + 2*d in s:
                    count += 1
        return count

    # Helper to check if adding y to A makes it non-4-free
    # A is a set, y is an int not in A
    def is_4free_with_y(A, y):
        s = set(A)
        # y is 1st: y, y+d, y+2d, y+3d
        max_d = (256 - y) // 3
        for d in range(1, max_d + 1):
            if (y + d) in s and (y + 2*d) in s and (y + 3*d) in s:
                return False
        # y is 2nd: y-d, y, y+d, y+2d
        max_d = min(y - 1, (256 - y) // 2)
        for d in range(1, max_d + 1):
            if (y - d) in s and (y + d) in s and (y + 2*d) in s:
                return False
        # y is 3rd: y-2d, y-d, y, y+d
        max_d = min((y - 1) // 2, 256 - y)
        for d in range(1, max_d + 1):
            if (y - 2*d) in s and (y - d) in s and (y + d) in s:
                return False
        # y is 4th: y-3d, y-2d, y-d, y
        max_d = (y - 1) // 3
        for d in range(1, max_d + 1):
            if (y - 3*d) in s and (y - 2*d) in s and (y - d) in s:
                return False
        return True

    current_set = selected
    current_count = count_3ap(current_set)
    
    deadline = time.time() + 50.0 # Safety margin
    
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try swaps
        # To avoid O(N^3) behavior, we can iterate x in A, y in complement
        # But we need to be careful with set modification during iteration
        # We'll collect a list of potential swaps or just do one swap and restart
        A_list = list(current_set)
        comp_list = list(set(range(1, 257)) - current_set)
        
        for x in A_list:
            if time.time() > deadline:
                return sorted(current_set)
            for y in comp_list:
                if time.time() > deadline:
                    return sorted(current_set)
                
                # Check if swap is valid
                # New set is (current_set - {x}) U {y}
                # We need to check if y creates a 4-AP with current_set - {x}
                # Note: removing x cannot create a 4-AP.
                A_temp = current_set - {x}
                if is_4free_with_y(A_temp, y):
                    # Calculate gain in 3-APs
                    # Gain = 3-APs in (A_temp U {y}) - 3-APs in A_temp
                    # This is equal to number of 3-APs involving y in A_temp U {y}
                    # Minus number of 3-APs involving x in current_set (which are lost)
                    # Wait, current_count includes 3-APs with x.
                    # new_count = current_count - (3-APs with x in current_set) + (3-APs with y in A_temp U {y})
                    
                    # Calculate 3-APs with x in current_set
                    loss = 0
                    for d in range(1, 257):
                        # x is middle: x-d, x, x+d
                        if x-d in current_set and x+d in current_set:
                            loss += 1
                        # x is end: x-2d, x-d, x
                        if x-2*d in current_set and x-d in current_set:
                            loss += 1
                        # x is start: x, x+d, x+2d
                        if x+d in current_set and x+2*d in current_set:
                            loss += 1
                            
                    # Calculate 3-APs with y in A_temp U {y}
                    gain = 0
                    for d in range(1, 257):
                        # y is middle
                        if y-d in A_temp and y+d in A_temp:
                            gain += 1
                        # y is end
                        if y-2*d in A_temp and y-d in A_temp:
                            gain += 1
                        # y is start
                        if y+d in A_temp and y+2*d in A_temp:
                            gain += 1
                            
                    if gain > loss:
                        current_set = A_temp | {y}
                        current_count += gain - loss
                        improved = True
                        break # Restart outer loop
            if improved:
                break
                
    return sorted(current_set)
