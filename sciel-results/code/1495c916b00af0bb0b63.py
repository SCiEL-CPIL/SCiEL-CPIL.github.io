"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def _count_ordinary_lines(points):
    """
    Counts the number of lines containing exactly two of the given points.
    """
    n = len(points)
    if n < 2:
        return 0
    
    line_points = {}
    
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            
            dx = x2 - x1
            dy = y2 - y1
            
            g = math.gcd(dx, dy)
            if g == 0:
                continue
                
            dx_n = dx // g
            dy_n = dy // g
            
            if dx_n < 0 or (dx_n == 0 and dy_n < 0):
                dx_n = -dx_n
                dy_n = -dy_n
                
            c = dy_n * x1 - dx_n * y1
            
            key = (dx_n, dy_n, c)
            
            if key not in line_points:
                line_points[key] = set()
            
            line_points[key].add((x1, y1))
            line_points[key].add((x2, y2))
            
    ordinary_count = 0
    for pts in line_points.values():
        if len(pts) == 2:
            ordinary_count += 1
            
    return ordinary_count

def _generate_initial_populations():
    """Generate diverse initial populations using different construction strategies."""
    populations = []
    
    # Strategy 1: Three concurrent lines through origin
    # Points on y=0, y=x, y=-x
    p1 = [[0,0], [1,0], [2,0], [3,0], [0,1], [0,2], [0,3], [1,1], [2,2], [3,3], [1,-1], [2,-2]]
    if len(set(map(tuple, p1))) == 12:
        populations.append(p1)
    
    # Strategy 2: Grid with bent rows
    p2 = [[0,0], [1,0], [2,0], [3,0], [0,1], [1,2], [2,1], [3,0], [0,3], [1,4], [2,3], [3,2]]
    # Check distinctness
    if len(set(map(tuple, p2))) == 12:
        populations.append(p2)
    
    # Strategy 3: Points on a conic (parabola y=x^2)
    p3 = [[i, i*i] for i in range(-3, 5)]  # 8 points on parabola
    # Add 4 more points strategically
    p3 += [[0, 1], [1, 0], [-1, 0], [2, 4]]
    # Remove duplicates and ensure 12 distinct
    seen = set()
    unique_p3 = []
    for p in p3:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_p3.append(p)
    if len(unique_p3) == 12:
        populations.append(unique_p3)
    
    # Strategy 4: Random in small box
    for _ in range(5):
        p4 = []
        seen = set()
        while len(p4) < 12:
            x = random.randint(-5, 5)
            y = random.randint(-5, 5)
            t = (x, y)
            if t not in seen:
                seen.add(t)
                p4.append([x, y])
        populations.append(p4)
    
    # Strategy 5: Based on known orchard problem solutions
    # Try to create configurations with many 3-point lines
    p5 = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2], [3,0], [3,1], [3,2]]
    if len(set(map(tuple, p5))) == 12:
        populations.append(p5)
    
    # Strategy 6: Symmetric cross pattern
    p6 = [[0,0], [1,0], [-1,0], [0,1], [0,-1], [2,0], [-2,0], [0,2], [0,-2], [1,1], [1,-1], [-1,1]]
    if len(set(map(tuple, p6))) == 12:
        populations.append(p6)
    
    # Strategy 7: Points forming multiple 3-point lines
    # Line y=0: (0,0),(1,0),(2,0)
    # Line y=1: (0,1),(1,1),(2,1)  
    # Line x=0: (0,0),(0,1),(0,2)
    # Line x=1: (1,0),(1,1),(1,2)
    # Line x=2: (2,0),(2,1),(2,2)
    # Line y=x: (0,0),(1,1),(2,2)
    # Line y=x-1: (1,0),(2,1),(3,2)
    # Line y=x-2: (2,0),(3,1),(4,2)
    # We need 12 points. Let's use:
    p7 = [[0,0], [1,0], [2,0], [0,1], [1,1], [2,1], [0,2], [1,2], [2,2], [3,1], [3,2], [4,2]]
    if len(set(map(tuple, p7))) == 12:
        populations.append(p7)
    
    return populations

def _crossover(p1, p2):
    """Uniform crossover between two point sets."""
    child = []
    for i in range(12):
        if random.random() < 0.5:
            child.append(p1[i][:])
        else:
            child.append(p2[i][:])
    return child

def _mutate(point, mutation_rate=0.3, mutation_scale=5):
    """Mutate a single point."""
    if random.random() < mutation_rate:
        x, y = point
        new_x = x + random.randint(-mutation_scale, mutation_scale)
        new_y = y + random.randint(-mutation_scale, mutation_scale)
        return [new_x, new_y]
    return point[:]

def _mutate_population(pop, mutation_rate=0.2, mutation_scale=3):
    """Mutate all points in a population."""
    new_pop = []
    for p in pop:
        new_p = _mutate(p, mutation_rate, mutation_scale)
        new_pop.append(new_p)
    return new_pop

def _repair(pop):
    """Ensure all points are distinct."""
    seen = set()
    repaired = []
    for p in pop:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            repaired.append(p)
        else:
            # Find a new random position
            while True:
                x = random.randint(-10, 10)
                y = random.randint(-10, 10)
                t = (x, y)
                if t not in seen:
                    seen.add(t)
                    repaired.append([x, y])
                    break
    return repaired

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Generate initial populations
    initial_pops = _generate_initial_populations()
    
    if not initial_pops:
        # Fallback
        initial_pops = [[[x, y] for y in range(3) for x in range(4)]]
    
    # Evaluate initial populations
    scored_pops = []
    for pop in initial_pops:
        score = _count_ordinary_lines(pop)
        scored_pops.append((score, pop))
    
    scored_pops.sort(key=lambda x: x[0])
    
    best_score, best_points = scored_pops[0]
    
    # Genetic algorithm
    population_size = 20
    max_generations = 200
    
    # Initialize population with best + random variations
    population = [best_points[:]]
    for score, pop in scored_pops[1:]:
        population.append(pop[:])
    
    # Fill rest with random populations
    while len(population) < population_size:
        pop = []
        seen = set()
        while len(pop) < 12:
            x = random.randint(-10, 10)
            y = random.randint(-10, 10)
            t = (x, y)
            if t not in seen:
                seen.add(t)
                pop.append([x, y])
        population.append(pop)
    
    for gen in range(max_generations):
        if time.time() > deadline:
            break
        
        # Evaluate current population
        scored = []
        for pop in population:
            score = _count_ordinary_lines(pop)
            scored.append((score, pop))
        
        scored.sort(key=lambda x: x[0])
        
        # Update best
        if scored[0][0] < best_score:
            best_score = scored[0][0]
            best_points = scored[0][1][:]
        
        # If we reached the theoretical minimum, stop early
        if best_score <= 6:
            break
        
        # Selection: keep top half
        elite_size = max(2, population_size // 4)
        next_population = [pop[:] for _, pop in scored[:elite_size]]
        
        # Crossover and mutation to fill rest
        while len(next_population) < population_size:
            # Pick two parents from top 2/3
            parent1 = random.choice(scored[:population_size * 2 // 3])[1]
            parent2 = random.choice(scored[:population_size * 2 // 3])[1]
            
            child = _crossover(parent1, parent2)
            child = _mutate_population(child, mutation_rate=0.15, mutation_scale=4)
            child = _repair(child)
            
            next_population.append(child)
        
        population = next_population
    
    return best_points
# EVOLVE-BLOCK-END
