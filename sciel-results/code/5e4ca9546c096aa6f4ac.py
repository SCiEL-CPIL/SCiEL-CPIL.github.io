import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50
    
    N = 512
    compat = [[False] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            g = gcd(i, j)
            if (i * j) // g <= N:
                compat[i][j] = True
                compat[j][i] = True
            else:
                compat[i][j] = False
                compat[j][i] = False
    
    S = set(range(1, N + 1))
    
    # Greedy removal
    for i in range(1, N + 1):
        if i in S:
            for j in range(i + 1, N + 1):
                if j in S and not compat[i][j]:
                    S.remove(j)
                    
    R = set(range(1, N + 1)) - S
    
    improved = True
    iterations = 0
    while improved and time.time() < deadline:
        improved = False
        iterations += 1
        
        # Try to add
        added = False
        for x in list(R):
            compatible = True
            for y in S:
                if not compat[x][y]:
                    compatible = False
                    break
            if compatible:
                S.add(x)
                R.remove(x)
                improved = True
                added = True
                break
        
        if not added:
            # Try to swap
            swapped = False
            for x in list(R):
                conflicts = []
                for y in S:
                    if not compat[x][y]:
                        conflicts.append(y)
                if len(conflicts) == 1:
                    y = conflicts[0]
                    S.remove(y)
                    S.add(x)
                    R.remove(x)
                    R.add(y)
                    improved = True
                    swapped = True
                    break
            # If no swap with 1 conflict, we stop.
            # Swaps with >1 conflicts would reduce size, which we don't want immediately
            # unless it leads to a net gain. But that's complex.
            # Given the greedy start, 1-conflict swaps should be sufficient.

    return list(S)
