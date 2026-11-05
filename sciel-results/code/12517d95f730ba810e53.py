import time
import itertools

def construct():
        start_time = time.time()
        deadline = start_time + 140  # 10s margin
        
        def is_valid_coloring(n, colors):
            # colors is a list of 0/1 for pairs (i,j) i<j
            # Build adjacency lists for color 0 and color 1
            adj0 = [set() for _ in range(n)]
            adj1 = [set() for _ in range(n)]
            idx = 0
            for i in range(n):
                for j in range(i+1, n):
                    c = colors[idx]
                    if c == 0:
                        adj0[i].add(j)
                        adj0[j].add(i)
                    else:
                        adj1[i].add(j)
                        adj1[j].add(i)
                    idx += 1
            
            # Check for K2,2,2 in color 0 (i.e., violation in adj0)
            # Violation in adj0 means adj1 on those 6 vertices has max degree <= 1
            # Violation in adj1 means adj0 on those 6 vertices has max degree <= 1
            
            vertices = list(range(n))
            for combo in itertools.combinations(vertices, 6):
                # Check adj1
                degs1 = [0]*6
                for k in range(6):
                    for m in range(k+1, 6):
                        if combo[m] in adj1[combo[k]]:
                            degs1[k] += 1
                            degs1[m] += 1
                if max(degs1) <= 1:
                    return False
                
                # Check adj0
                degs0 = [0]*6
                for k in range(6):
                    for m in range(k+1, 6):
                        if combo[m] in adj0[combo[k]]:
                            degs0[k] += 1
                            degs0[m] += 1
                if max(degs0) <= 1:
                    return False
                    
            return True

        # Generate Paley 29
        n = 29
        q = n
        squares = {x*x % q for x in range(1, q)}
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                if (j-i) % q in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        
        # Verify
        if is_valid_coloring(n, colors):
            return {'n': n, 'colors': colors}
        
        # Fallback: Paley 17
        n = 17
        q = n
        squares = {x*x % q for x in range(1, q)}
        colors = []
        for i in range(n):
            for j in range(i+1, n):
                if (j-i) % q in squares:
                    colors.append(0)
                else:
                    colors.append(1)
        return {'n': n, 'colors': colors}
