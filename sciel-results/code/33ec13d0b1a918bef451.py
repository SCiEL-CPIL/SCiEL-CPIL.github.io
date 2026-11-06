import time

def construct():
    start_time = time.time()
    deadline = start_time + 55  # 55s safety margin
    
    N = 128
    colors = [-1] * (N + 1)
    
    # Precompute APs ending at each index
    aps_ending = [[] for _ in range(N + 1)]
    for x in range(4, N + 1):
        for d in range(1, (x - 1) // 3 + 1):
            a = x - 3 * d
            if a >= 1:
                aps_ending[x].append((a, d))
                
    def is_valid(x, c):
        for (a, d) in aps_ending[x]:
            c1 = colors[a]
            c2 = colors[a + d]
            c3 = colors[a + 2 * d]
            # Check distinct count
            if c1 == c2 and c2 == c3:
                # All 3 same, need c to be different? No, need size >= 3.
                # If c1=c2=c3=S, then {S, c} has size 1 or 2. Always < 3.
                return False
            elif c1 == c2:
                # c1=c2=A, c3=B. Set is {A, B}.
                # Need {A, B, c} size >= 3 => c not in {A, B}
                if c == c1 or c == c3:
                    return False
            elif c1 == c3:
                # c1=c3=A, c2=B. Set is {A, B}.
                # Need c not in {A, B}
                if c == c1 or c == c2:
                    return False
            elif c2 == c3:
                # c2=c3=A, c1=B. Set is {A, B}.
                # Need c not in {A, B}
                if c == c2 or c == c1:
                    return False
            else:
                # All 3 distinct. Set size 3.
                # {A, B, C, c} size >= 3 is always true.
                pass
        return True

    def backtrack(x):
        if time.time() > deadline:
            return False
        if x > N:
            return True
            
        # Heuristic: Try colors in a specific order?
        # Maybe 0, 1, 2 is fine.
        for c in range(3):
            if is_valid(x, c):
                colors[x] = c
                if backtrack(x + 1):
                    return True
                colors[x] = -1
        return False

    # Symmetry breaking
    colors[1] = 0
    
    found = False
    for c2 in [0, 1]:
        colors[2] = c2
        if backtrack(3):
            found = True
            break
        colors[2] = -1
        
    if found:
        return colors[1:]
    else:
        # Fallback to greedy if backtracking fails or times out
        greedy_colors = []
        for x in range(1, N + 1):
            for c in range(128):
                valid = True
                for d in range(1, (x - 1) // 3 + 1):
                    a = x - 3 * d
                    if a < 1: continue
                    c1 = greedy_colors[a - 1]
                    c2 = greedy_colors[a - 1 + d]
                    c3 = greedy_colors[a - 1 + 2 * d]
                    if len({c1, c2, c3, c}) < 3:
                        valid = False
                        break
                if valid:
                    greedy_colors.append(c)
                    break
        return greedy_colors
