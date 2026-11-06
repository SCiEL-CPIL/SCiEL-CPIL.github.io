"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np
from itertools import combinations, permutations

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill

    def verify_coloring(colors, n):
        """Verify that no color class contains a C6 using bitset adjacency."""
        num_edges = n * (n - 1) // 2
        if len(colors) != num_edges:
            return False
        # Build per-color adjacency as bitsets
        adj = [0] * n  # for each color, we'll build separately
        for c in range(4):
            adj_c = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj_c[i] |= (1 << j)
                        adj_c[j] |= (1 << i)
                    idx += 1
            # Check for C6 in this color using bit operations
            for a0 in range(n):
                n0 = adj_c[a0]
                while n0:
                    a1 = (n0 & -n0).bit_length() - 1
                    n0 &= n0 - 1
                    if a1 <= a0:
                        continue
                    n1 = adj_c[a1] & ~(1 << a0)
                    while n1:
                        a2 = (n1 & -n1).bit_length() - 1
                        n1 &= n1 - 1
                        if a2 in (a0, a1):
                            continue
                        n2 = adj_c[a2] & ~(1 << a0) & ~(1 << a1)
                        while n2:
                            a3 = (n2 & -n2).bit_length() - 1
                            n2 &= n2 - 1
                            if a3 in (a0, a1, a2):
                                continue
                            n3 = adj_c[a3] & ~(1 << a0) & ~(1 << a1) & ~(1 << a2)
                            while n3:
                                a4 = (n3 & -n3).bit_length() - 1
                                n3 &= n3 - 1
                                if a4 in (a0, a1, a2, a3):
                                    continue
                                n4 = adj_c[a4] & ~(1 << a0) & ~(1 << a1) & ~(1 << a2) & ~(1 << a3)
                                while n4:
                                    a5 = (n4 & -n4).bit_length() - 1
                                    n4 &= n4 - 1
                                    if a5 in (a0, a1, a2, a3, a4):
                                        continue
                                    if adj_c[a5] & (1 << a0):
                                        return False
        return True

    def count_c6s(colors, n):
        """Count monochromatic C6s using bitset operations."""
        count = 0
        for c in range(4):
            adj_c = [0] * n
            idx = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if colors[idx] == c:
                        adj_c[i] |= (1 << j)
                        adj_c[j] |= (1 << i)
                    idx += 1
            for a0 in range(n):
                n0 = adj_c[a0]
                while n0:
                    a1 = (n0 & -n0).bit_length() - 1
                    n0 &= n0 - 1
                    if a1 <= a0:
                        continue
                    n1 = adj_c[a1] & ~(1 << a0)
                    while n1:
                        a2 = (n1 & -n1).bit_length() - 1
                        n1 &= n1 - 1
                        if a2 in (a0, a1):
                            continue
                        n2 = adj_c[a2] & ~(1 << a0) & ~(1 << a1)
                        while n2:
                            a3 = (n2 & -n2).bit_length() - 1
                            n2 &= n2 - 1
                            if a3 in (a0, a1, a2):
                                continue
                            n3 = adj_c[a3] & ~(1 << a0) & ~(1 << a1) & ~(1 << a2)
                            while n3:
                                a4 = (n3 & -n3).bit_length() - 1
                                n3 &= n3 - 1
                                if a4 in (a0, a1, a2, a3):
                                    continue
                                n4 = adj_c[a4] & ~(1 << a0) & ~(1 << a1) & ~(1 << a2) & ~(1 << a3)
                                while n4:
                                    a5 = (n4 & -n4).bit_length() - 1
                                    n4 &= n4 - 1
                                    if a5 in (a0, a1, a2, a3, a4):
                                        continue
                                    if adj_c[a5] & (1 << a0):
                                        count += 1
        return count

    def backtrack_local(n, time_limit):
        """Iterative backtracking with local C6 check."""
        num_edges = n * (n - 1) // 2
        assignment = [-1] * num_edges
        adj_c = [[set() for _ in range(n)] for _ in range(4)]
        edges = [(i, j) for i in range(n) for j in range(i + 1, n)]

        def would_create_c6(c, u, v):
            for a in adj_c[c][u]:
                if a == v:
                    continue
                for b in adj_c[c][a]:
                    if b in (u, v, a):
                        continue
                    for d in adj_c[c][b]:
                        if d in (u, a, b, v):
                            continue
                        for e in adj_c[c][d]:
                            if e in (u, a, b, d, v):
                                continue
                            if v in adj_c[c][e]:
                                return True
            return False

        # Symmetry breaking
        assignment[0] = 0
        adj_c[0][0].add(1)
        adj_c[0][1].add(0)

        # Iterative DFS with stack
        stack = [(1, 0)]  # (pos, next_color_to_try)
        while stack:
            if time.time() > time_limit:
                return None
            pos, color = stack[-1]
            if pos == num_edges:
                return assignment[:]
            u, v = edges[pos]
            found = False
            while color < 4:
                if not would_create_c6(color, u, v):
                    assignment[pos] = color
                    adj_c[color][u].add(v)
                    adj_c[color][v].add(u)
                    stack[-1] = (pos, color + 1)
                    stack.append((pos + 1, 0))
                    found = True
                    break
                color += 1
            if not found:
                # Backtrack
                stack.pop()
                if stack:
                    prev_pos, _ = stack[-1]
                    if prev_pos > 0:
                        pu, pv = edges[prev_pos - 1]
                        pc = assignment[prev_pos - 1]
                        if pc >= 0:
                            adj_c[pc][pu].discard(pv)
                            adj_c[pc][pv].discard(pu)
                            assignment[prev_pos - 1] = -1
        return None

    def genetic_algorithm(n, pop_size, num_generations, time_limit):
        """Genetic algorithm with bitset C6 counting."""
        num_edges = n * (n - 1) // 2
        
        # Initialize population with random colorings
        population = []
        for _ in range(pop_size):
            chrom = random.choices(range(4), k=num_edges)
            population.append(chrom)
        
        # Also add a cyclic-based seed if n is odd
        if n % 2 == 1:
            m = (n - 1) // 2
            elements = list(range(1, n))
            random.shuffle(elements)
            sets = [set(elements[i*m:(i+1)*m]) for i in range(4)]
            chrom = []
            for i in range(n):
                for j in range(i + 1, n):
                    d = (j - i) % n
                    assigned = False
                    for c in range(4):
                        if d in sets[c] or (n - d) in sets[c]:
                            chrom.append(c)
                            assigned = True
                            break
                    if not assigned:
                        chrom.append(random.randint(0, 3))
            population[0] = chrom
        
        best_chrom = None
        best_fitness = float('inf')
        
        for gen in range(num_generations):
            if time.time() > time_limit:
                break
            
            # Evaluate fitness
            fitnesses = []
            for chrom in population:
                fit = count_c6s(chrom, n)
                fitnesses.append(fit)
                if fit < best_fitness:
                    best_fitness = fit
                    best_chrom = chrom[:]
            
            if best_fitness == 0:
                return best_chrom
            
            # Selection: tournament
            new_population = []
            for _ in range(pop_size):
                # Tournament selection
                contenders = random.sample(range(pop_size), min(5, pop_size))
                winner = min(contenders, key=lambda i: fitnesses[i])
                new_population.append(population[winner][:])
            
            # Crossover and mutation
            for i in range(0, pop_size, 2):
                if i + 1 >= pop_size:
                    break
                p1, p2 = new_population[i], new_population[i+1]
                # Uniform crossover
                child1 = [p1[j] if random.random() < 0.5 else p2[j] for j in range(num_edges)]
                child2 = [p2[j] if random.random() < 0.5 else p1[j] for j in range(num_edges)]
                # Mutation
                for j in range(num_edges):
                    if random.random() < 0.02:
                        child1[j] = random.randint(0, 3)
                    if random.random() < 0.02:
                        child2[j] = random.randint(0, 3)
                new_population[i] = child1
                new_population[i+1] = child2
            
            population = new_population
        
        return best_chrom if best_chrom else None

    def try_cyclic(n, num_trials=5000):
        """Try cyclic 4-coloring of K_n where n is odd."""
        if n % 2 == 0:
            return None
        m = (n - 1) // 2
        elements = list(range(1, n))
        for trial in range(num_trials):
            random.shuffle(elements)
            sets = [frozenset(elements[i*m:(i+1)*m]) for i in range(4)]
            valid = True
            for c in range(4):
                S = sets[c]
                adj = [set() for _ in range(n)]
                for i in range(n):
                    for s in S:
                        adj[i].add((i + s) % n)
                        adj[i].add((i - s) % n)
                found = False
                for a0 in range(n):
                    if found:
                        break
                    for a1 in adj[a0]:
                        if found or a1 <= a0:
                            continue
                        for a2 in adj[a1]:
                            if found or a2 == a0:
                                continue
                            for a3 in adj[a2]:
                                if found or a3 in (a0, a1):
                                    continue
                                for a4 in adj[a3]:
                                    if found or a4 in (a0, a1, a2):
                                        continue
                                    for a5 in adj[a4]:
                                        if a5 in (a0, a1, a2, a3):
                                            continue
                                        if a0 in adj[a5]:
                                            found = True
                                            break
                if found:
                    valid = False
                    break
            if valid:
                colors = []
                for i in range(n):
                    for j in range(i + 1, n):
                        d = (j - i) % n
                        for c in range(4):
                            if d in sets[c] or (n - d) in sets[c]:
                                colors.append(c)
                                break
                return colors
        return None

    # Strategy: try multiple approaches
    # 1. Cyclic for n=17
    # 2. Backtracking for n=17
    # 3. Genetic algorithm for n=17 and n=18
    # 4. Fallback to smaller n

    # Try n=18 with genetic algorithm (most ambitious)
    if time.time() < deadline - 40:
        print("Trying n=18 with genetic algorithm...", flush=True)
        colors = genetic_algorithm(18, pop_size=200, num_generations=50, time_limit=deadline - 40)
        if colors is not None and verify_coloring(colors, 18):
            print("Found n=18!", flush=True)
            return {'n': 18, 'colors': colors}

    # Try n=17 with cyclic
    if time.time() < deadline - 30:
        print("Trying n=17 cyclic...", flush=True)
        colors = try_cyclic(17, num_trials=2000)
        if colors is not None and verify_coloring(colors, 17):
            print("Found n=17 cyclic!", flush=True)
            return {'n': 17, 'colors': colors}

    # Try n=17 with backtracking
    if time.time() < deadline - 20:
        print("Trying n=17 backtracking...", flush=True)
        colors = backtrack_local(17, deadline - 20)
        if colors is not None and verify_coloring(colors, 17):
            print("Found n=17 backtracking!", flush=True)
            return {'n': 17, 'colors': colors}

    # Try n=17 with genetic algorithm
    if time.time() < deadline - 10:
        print("Trying n=17 genetic...", flush=True)
        colors = genetic_algorithm(17, pop_size=300, num_generations=100, time_limit=deadline - 10)
        if colors is not None and verify_coloring(colors, 17):
            print("Found n=17 genetic!", flush=True)
            return {'n': 17, 'colors': colors}

    # Try n=16
    if time.time() < deadline - 5:
        colors = try_cyclic(15, num_trials=1000)
        if colors is not None and verify_coloring(colors, 15):
            return {'n': 15, 'colors': colors}
        colors = backtrack_local(16, deadline - 5)
        if colors is not None and verify_coloring(colors, 16):
            return {'n': 16, 'colors': colors}
        colors = genetic_algorithm(16, pop_size=200, num_generations=50, time_limit=deadline - 5)
        if colors is not None and verify_coloring(colors, 16):
            return {'n': 16, 'colors': colors}

    # Fallback: n=9
    n = 9
    colors = [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]
    return {'n': n, 'colors': colors}
# EVOLVE-BLOCK-END
