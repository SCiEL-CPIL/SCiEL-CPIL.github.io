import random
from itertools import combinations
from math import gcd
import time

def count_ordinary_lines(points):
    """Count lines containing exactly 2 points."""
    n = len(points)
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            dx = x2 - x1
            dy = y2 - y1
            
            if dx == 0:
                key = ('v', x1)
            elif dy == 0:
                key = ('h', y1)
            else:
                g = gcd(abs(dx), abs(dy))
                dx_n = dx // g
                dy_n = dy // g
                if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                    dx_n = -dx_n
                    dy_n = -dy_n
                a = dy_n
                b = -dx_n
                c = a * x1 + b * y1
                g2 = gcd(abs(a), abs(b))
                if g2 > 0:
                    a //= g2
                    b //= g2
                    c //= g2
                if a < 0 or (a == 0 and b < 0):
                    a = -a
                    b = -b
                    c = -c
                key = ('g', a, b, c)
            
            if key not in line_points:
                line_points[key] = set()
            line_points[key].add(i)
            line_points[key].add(j)
    
    ordinary = 0
    for key, pts in line_points.items():
        if len(pts) == 2:
            ordinary += 1
    
    return ordinary

def construct():
    """Find 12 integer points minimizing ordinary lines via Genetic Algorithm."""
    
    # Parameters
    POP_SIZE = 50
    GENERATIONS = 200
    MUTATION_RATE = 0.1
    TOURNAMENT_SIZE = 3
    
    # Initialize population
    # Use a mix of random points and structured points (like grids) to ensure diversity
    population = []
    
    # Add some structured seeds
    # 3x4 grid
    grid1 = [[x, y] for x in range(3) for y in range(4)]
    population.append(grid1)
    
    # 4x3 grid
    grid2 = [[x, y] for x in range(4) for y in range(3)]
    population.append(grid2)
    
    # Random points
    for _ in range(POP_SIZE - 2):
        pts = []
        used = set()
        while len(pts) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            if (x, y) not in used:
                used.add((x, y))
                pts.append([x, y])
        population.append(pts)
    
    # Evaluate initial population
    scored_population = [(count_ordinary_lines(pts), pts) for pts in population]
    scored_population.sort(key=lambda x: x[0])
    
    best_score = scored_population[0][0]
    best_points = scored_population[0][1]
    
    start_time = time.time()
    
    for gen in range(GENERATIONS):
        if time.time() - start_time > 100: # Keep 20s buffer
            break
            
        next_population = []
        
        # Elitism: keep top 5
        for i in range(5):
            next_population.append(scored_population[i])
        
        # Fill the rest
        while len(next_population) < POP_SIZE:
            # Selection
            parents = []
            for _ in range(2):
                tournament = random.sample(scored_population, TOURNAMENT_SIZE)
                winner = min(tournament, key=lambda x: x[0])
                parents.append(winner)
            
            # Crossover
            p1 = parents[0][1]
            p2 = parents[1][1]
            
            child = []
            for i in range(12):
                if random.random() < 0.5:
                    child.append(p1[i][:])
                else:
                    child.append(p2[i][:])
            
            # Check for duplicates and validity
            valid = True
            seen = set()
            for p in child:
                if p[0] in seen: # Simple check, actually need tuple
                    pass # This check is flawed, need proper set of tuples
            
            # Proper duplicate check
            seen = set()
            for p in child:
                if tuple(p) in seen:
                    valid = False
                    break
                seen.add(tuple(p))
            
            if not valid:
                # If invalid, try mutation to fix or discard
                # For simplicity, if invalid, just use a parent
                child = random.choice(parents)[1][:]
                seen = set()
                for p in child:
                    seen.add(tuple(p))
                if len(seen) != 12:
                     # Still invalid? Should not happen if parents were valid
                     continue

            # Mutation
            if random.random() < MUTATION_RATE:
                idx = random.randint(0, 11)
                # Randomly perturb one point
                # Try small moves
                for _ in range(5):
                    dx = random.randint(-2, 2)
                    dy = random.randint(-2, 2)
                    if dx == 0 and dy == 0:
                        continue
                    nx = child[idx][0] + dx
                    ny = child[idx][1] + dy
                    if -10**9 <= nx <= 10**9 and -10**9 <= ny <= 10**9:
                        # Check duplicate
                        dup = False
                        for j in range(12):
                            if j != idx and child[j][0] == nx and child[j][1] == ny:
                                dup = True
                                break
                        if not dup:
                            child[idx] = [nx, ny]
                            break
            
            # Evaluate
            score = count_ordinary_lines(child)
            next_population.append((score, child))
            
            # Update global best
            if score < best_score:
                best_score = score
                best_points = child[:]
        
        # Update scored population
        scored_population = next_population
        scored_population.sort(key=lambda x: x[0])
        
        # Early stop if we found the theoretical minimum
        if best_score == 6:
            break
            
    return best_points
