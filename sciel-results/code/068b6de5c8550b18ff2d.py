import time
import random

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110
    
    # Target: 5-regular graph, girth >= g
    # Try Cayley graphs on Z_n
    # Generators: a, -a, b, -b, c (where c = -c, so 2c = 0)
    # n must be even.
    
    # Moore bound
    # For k=5, g=9, N >= 425
    # Let's search n from 426 to 2000
    
    # To speed up, we can check the condition:
    # No sum of < g generators is 0 mod n.
    # Generators: a, -a, b, -b, c.
    # Note: c = n/2.
    # Sum = x*a + y*b + z*c = 0
    # z*c = z*(n/2).
    # If z is even, z*c is multiple of n.
    # If z is odd, z*c = n/2.
    # So we need x*a + y*b = 0 (if z even) or x*a + y*b = n/2 (if z odd).
    # Number of a's and b's in the word is at most g-1.
    
    # We can precompute all possible sums of x*a + y*b for |x|+|y| <= g-1.
    # And check if any is 0 or n/2.
    
    # This is much faster than BFS.
    
    max_sum = g - 1 # 8
    
    # List of all (x, y) such that |x| + |y| <= max_sum
    # And x, y are integers.
    # The sum is x*a + y*b.
    
    # We iterate n.
    for n in range(426, 1500, 2):
        if time.time() > deadline:
            break
            
        c = n // 2
        
        # Try random a, b
        for _ in range(20):
            if time.time() > deadline:
                break
                
            a = random.randint(1, n-1)
            b = random.randint(1, n-1)
            
            # Check a, b distinct and not 0, c
            if a in [0, c] or b in [0, c]:
                continue
            if a == b or a == -b or a == c or a == -c or b == c or b == -c:
                continue
            if 2*a == 0 or 2*b == 0:
                continue
                
            # Check if any x*a + y*b is 0 or n/2 for |x|+|y| <= 8
            # and x, y not both 0 (length 0 is not a cycle)
            # Actually, length must be >= 1.
            # And the word must be non-backtracking?
            # In a Cayley graph, a word corresponds to a walk.
            # If the walk is not a cycle (i.e., it doesn't repeat vertices), it's a path.
            # If it sums to 0, it's a closed walk.
            # If the closed walk has no repeated vertices (except start/end), it's a cycle.
            # For small girth, we just need to ensure no short closed walks.
            # If a short closed walk exists, girth is short.
            
            valid = True
            # Precompute sums?
            # Just loop x, y
            for x in range(-max_sum, max_sum + 1):
                for y in range(-max_sum, max_sum + 1):
                    if abs(x) + abs(y) == 0:
                        continue
                    if abs(x) + abs(y) > max_sum:
                        continue
                        
                    s = (x * a + y * b) % n
                    # If s == 0, we have a closed walk using x a's and y b's.
                    # But we also have c's.
                    # The total length is |x| + |y| + z.
                    # We need to check if there exists z such that total length < g.
                    # And x*a + y*b + z*c = 0.
                    # z*c = 0 if z even, n/2 if z odd.
                    
                    # Case 1: z even. x*a + y*b = 0.
                    # Total length L = |x| + |y| + z.
                    # We need L < g.
                    # Min z is 0. So if |x|+|y| < g, and s==0, we have a cycle?
                    # Not necessarily, the walk might backtrack.
                    # But if we have a closed walk of length L, the girth is at most L.
                    # So if s==0 and |x|+|y| < g, we are bad.
                    
                    # Case 2: z odd. x*a + y*b = n/2.
                    # Min z is 1.
                    # So if |x|+|y| + 1 < g, and s==n/2, we are bad.
                    
                    if s == 0:
                        if abs(x) + abs(y) < g:
                            valid = False
                            break
                    elif s == c:
                        if abs(x) + abs(y) + 1 < g:
                            valid = False
                            break
                            
                if not valid:
                    break
            
            if valid:
                # Construct edges
                edges = []
                for i in range(n):
                    for nb in [ (i+a)%n, (i-a)%n, (i+b)%n, (i-b)%n, (i+c)%n ]:
                        if i < nb:
                            edges.append((i, nb))
                return n, edges
                
    # Fallback: LUW
    # ...
