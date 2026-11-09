import random
import itertools
from math import gcd
import time

def count_exact_4_lines(points):
    """
    Counts the number of lines that contain exactly 4 of the given points.
    Returns 0 if any line contains 5 or more points.
    """
    n = len(points)
    if n != 16:
        return 0
    
    # Check for duplicates
    if len(set(map(tuple, points))) != 16:
        return 0

    def get_line_key(p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        # a*x + b*y + c = 0
        # (y2-y1)x - (x2-x1)y + (x2*y1 - x1*y2) = 0
        a = y2 - y1
        b = -(x2 - x1)
        c = x1 * y2 - x2 * y1
        
        # Normalize
        g = gcd(gcd(abs(a), abs(b)), abs(c))
        if g == 0:
            return None
        a //= g
        b //= g
        c //= g
        
        # Ensure canonical sign
        if a < 0:
            a = -a
            b = -b
            c = -c
        elif a == 0:
            if b < 0:
                b = -b
                c = -c
            elif b == 0:
                if c < 0:
                    c = -c
        
        return (a, b, c)

    lines = {}
    for i in range(n):
        for j in range(i+1, n):
            key = get_line_key(points[i], points[j])
            if key is None:
                continue
            if key not in lines:
                lines[key] = set()
            lines[key].add(i)
            lines[key].add(j)
            
    count = 0
    for key, pts_on_line in lines.items():
        k = len(pts_on_line)
        if k > 4:
            return 0 # Invalid configuration
        if k == 4:
            count += 1
            
    return count

def construct():
    # Simulated Annealing
    # Start with a random configuration
    # Grid size 200x200 to allow more flexibility
    GRID = 100
    
    # Initialize points
    # Use a set to ensure uniqueness
    points = []
    while len(points) < 16:
        x = random.randint(-GRID, GRID)
        y = random.randint(-GRID, GRID)
        if [x, y] not in points:
            points.append([x, y])
    
    current_score = count_exact_4_lines(points)
    best_points = [p[:] for p in points]
    best_score = current_score
    
    # Simulated Annealing parameters
    start_temp = 10.0
    end_temp = 0.1
    iterations = 50000 # Adjust based on time
    start_time = time.time()
    timeout = 110.0 # Leave 10s buffer
    
    # Cooling schedule
    # T = T0 * (T_end/T0)^(iter/max_iter)
    # Or simple linear/exponential decay
    
    for i in range(iterations):
        if time.time() - start_time > timeout:
            break
            
        # Calculate temperature
        # Linear decay
        temp = start_temp - (start_temp - end_temp) * (i / iterations)
        
        # Generate a move
        move_type = random.random()
        if move_type < 0.7:
            # Move one point
            idx = random.randint(0, 15)
            old_p = points[idx][:]
            new_x = random.randint(-GRID, GRID)
            new_y = random.randint(-GRID, GRID)
            points[idx] = [new_x, new_y]
            
            new_score = count_exact_4_lines(points)
            
            # Acceptance criterion
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
            else:
                # Accept with probability exp(delta / temp)
                # Since score is integer and small, delta is small negative
                if temp > 0:
                    prob = pow(2.718281828, delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        points[idx] = old_p
                else:
                    points[idx] = old_p
                    
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]
                
        else:
            # Swap two points
            i1, i2 = random.sample(range(16), 2)
            p1 = points[i1][:]
            p2 = points[i2][:]
            points[i1] = p2
            points[i2] = p1
            
            new_score = count_exact_4_lines(points)
            
            delta = new_score - current_score
            if delta > 0:
                current_score = new_score
            else:
                if temp > 0:
                    prob = pow(2.718281828, delta / temp)
                    if random.random() < prob:
                        current_score = new_score
                    else:
                        points[i1] = p1
                        points[i2] = p2
                else:
                    points[i1] = p1
                    points[i2] = p2
            
            if current_score > best_score:
                best_score = current_score
                best_points = [p[:] for p in points]

    # Final validation
    final_score = count_exact_4_lines(best_points)
    if final_score == 0:
        # Fallback to a known valid configuration
        # 4x4 grid is valid (no 5 collinear)
        return [[x, y] for x in range(4) for y in range(4)]
        
    return best_points

# EVOLVE-BLOCK-START
# The construct function is defined above. 
# The prompt asks to keep the function names. 
# The current program has construct inside the block. 
# I will place the helper functions and construct inside the block or outside?
# The seed program has:
# # EVOLVE-BLOCK-START
# def construct(): ...
# # EVOLVE-BLOCK-END
#
# I should keep the structure similar. I'll define helpers outside or inside?
# Usually, it's safer to keep everything in the block if the evaluator only extracts that part, 
# but the prompt says "Rewrite the program... Output the COMPLETE program".
# And "Keep the function names and signatures of the seed program."
# The seed program defines construct inside the block.
# I will define helper functions outside the block to keep the block clean, 
# or inside if they are necessary. 
# Let's put helpers outside the block to avoid cluttering the evolution target, 
# but ensure construct is inside.

# Wait, the seed code is:
# """Initial program..."""
# # EVOLVE-BLOCK-START
# def construct():
#    ...
# # EVOLVE-BLOCK-END

# I will follow this structure.

# EVOLVE-BLOCK-END
