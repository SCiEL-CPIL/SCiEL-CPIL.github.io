import random

N = 256

def construct():
    best_set = None
    best_size = N + 1
    
    # Run multiple trials to escape local minima or find better random tie-breaks
    for trial in range(100):
        S = set()
        sums = set()
        
        while True:
            # Identify U: elements that can be added
            U = []
            for x in range(1, N + 1):
                if x in S:
                    continue
                # Quick check: 2x
                if 2 * x in sums:
                    continue
                # Check x + a for a in S
                conflict = False
                for a in S:
                    if x + a in sums:
                        conflict = True
                        break
                if not conflict:
                    U.append(x)
            
            if not U:
                break
            
            # Optimization: Pre-compute S list for faster iteration
            S_list = list(S)
            U_set = set(U)
            
            # For each x in U, determine which y in U are blocked by x
            # Two elements x, y in U conflict if:
            # 1. 2x = 2y (impossible if x!=y)
            # 2. 2x = y + a  => y = 2x - a
            # 3. x + b = 2y  => y = (x + b) / 2
            # 4. x + b = y + a => y = x + b - a
            # where a, b in S.
            
            # We build a graph on U where edge (x,y) exists if they conflict.
            # Then we pick the vertex with the highest degree.
            
            degrees = {x: 0 for x in U}
            
            # To avoid double counting, we can just iterate x, find conflicting y, and increment.
            # Since the relation is symmetric, we can just count unique neighbors.
            # Or simply, for each x, find all y in U that conflict, and add to a set of neighbors.
            
            # Efficiently finding neighbors:
            # For a given x, the set of conflicting y values (ignoring U membership for a moment) is:
            # C_x = { 2x - a for a in S } U { (x + b) // 2 for b in S if (x+b)%2==0 } U { x + b - a for a in S for b in S }
            
            # We can precompute the "difference set" of S: D_S = { b - a for a, b in S }
            # And "half-sums" etc.
            
            # Let's just iterate S for each x. |S| is small (< 20 usually).
            # |U| can be up to 256.
            # 256 * 20 * 20 = 100,000 ops per step. This is very fast.
            
            for x in U:
                neighbors = set()
                # Type 2: y = 2x - a
                for a in S_list:
                    y = 2 * x - a
                    if 1 <= y <= N and y in U_set and y != x:
                        neighbors.add(y)
                        
                # Type 3: y = (x + b) / 2
                for b in S_list:
                    val = x + b
                    if val % 2 == 0:
                        y = val // 2
                        if 1 <= y <= N and y in U_set and y != x:
                            neighbors.add(y)
                            
                # Type 4: y = x + b - a
                for a in S_list:
                    for b in S_list:
                        y = x + b - a
                        if 1 <= y <= N and y in U_set and y != x:
                            neighbors.add(y)
                
                degrees[x] = len(neighbors)
            
            # Find max degree
            max_deg = -1
            candidates = []
            for x in U:
                if degrees[x] > max_deg:
                    max_deg = degrees[x]
                    candidates = [x]
                elif degrees[x] == max_deg:
                    candidates.append(x)
            
            # Pick one
            chosen_y = random.choice(candidates)
            
            # Update
            S.add(chosen_y)
            sums.add(2 * chosen_y)
            for a in S:
                if a != chosen_y:
                    sums.add(chosen_y + a)
            
        if len(S) < best_size:
            best_size = len(S)
            best_set = list(S)
            
    return best_set
