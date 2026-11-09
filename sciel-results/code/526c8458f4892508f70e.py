def construct():
    N = 256
    selected = []
    in_set = [False] * (N + 1)
    
    # Greedy Initial Solution
    for x in range(1, N + 1):
        valid = True
        for a in selected:
            # Check if x completes a 3-AP with two existing elements
            # 1. a, b, x -> b = (a+x)/2
            if (x - a) % 2 == 0:
                b = (x - a) // 2
                if 1 <= b <= N and in_set[b] and b != a and b != x:
                    valid = False
                    break
            # 2. a, x, b -> b = 2x - a
            b = 2 * x - a
            if 1 <= b <= N and in_set[b] and b != x:
                valid = False
                break
            if not valid:
                break
        
        if valid:
            selected.append(x)
            in_set[x] = True

    # Local Search
    improved = True
    while improved:
        improved = False
        
        # Try to add any element
        for y in range(1, N + 1):
            if not in_set[y]:
                # Check if adding y creates a 3-AP
                valid_add = True
                for a in selected:
                    # 1. a, b, y -> b = (a+y)/2
                    if (y - a) % 2 == 0:
                        b = (y - a) // 2
                        if 1 <= b <= N and in_set[b] and b != a and b != y:
                            valid_add = False
                            break
                    # 2. a, y, b -> b = 2y - a
                    b = 2 * y - a
                    if 1 <= b <= N and in_set[b] and b != y:
                        valid_add = False
                        break
                    if not valid_add:
                        break
                
                if valid_add:
                    selected.append(y)
                    in_set[y] = True
                    improved = True
                    break
        
        if improved:
            continue

        # Try to swap one element
        for i, x in enumerate(selected):
            # Try swapping x with y
            for y in range(1, N + 1):
                if in_set[y]:
                    continue
                
                # Check if S \ {x} U {y} is valid
                valid_swap = True
                for a in selected:
                    if a == x:
                        continue
                    # Check if y forms AP with a and another element in S\{x}
                    # 1. a, b, y -> b = (a+y)/2
                    if (y - a) % 2 == 0:
                        b = (y - a) // 2
                        if 1 <= b <= N and in_set[b] and b != a and b != y and b != x:
                            valid_swap = False
                            break
                    # 2. a, y, b -> b = 2y - a
                    b = 2 * y - a
                    if 1 <= b <= N and in_set[b] and b != y and b != x:
                        valid_swap = False
                        break
                    if not valid_swap:
                        break
                
                if valid_swap:
                    # Check if gain is positive
                    if 1.0 / y > 1.0 / x:
                        # Perform swap
                        selected[i] = y
                        in_set[x] = False
                        in_set[y] = True
                        improved = True
                        break
            if improved:
                break

    return selected
