import time
import sys
import numpy as np
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    def _bitmask_has_c6(adj_bitmasks, n, u, v, c):
        """
        Check if adding edge (u,v) with color c creates a C6 in color c.
        adj_bitmasks[c] is a list of integers, one per vertex.
        adj_bitmasks[c][x] has bit y set iff edge (x,y) is color c.
        
        A C6 through (u,v) requires a path u-w-x-y-z-v of length 4 in color c.
        This exists iff there are w,z with (u,w) and (z,v) in color c,
        and a path of length 2 from w to z in color c (i.e., exists x with (w,x) and (x,z)).
        
        Path of length 2 from w to z exists iff (adj[w] & adj[z]) != 0 
        (there is some x adjacent to both w and z in color c).
        """
        mask_u = adj_bitmasks[c][u]
        mask_v = adj_bitmasks[c][v]
        
        # For each w neighbor of u in color c, and z neighbor of v in color c,
        # check if w and z share a common neighbor in color c.
        # We need to exclude u and v from consideration as intermediate vertices.
        # Also w != z.
        
        # Iterate over set bits in mask_u (neighbors of u)
        w = 0
        mu = mask_u
        while mu:
            if mu & 1:
                # w is a neighbor of u
                # Now iterate over set bits in mask_v (neighbors of v)
                z = 0
                mv = mask_v
                while mv:
                    if mv & 1:
                        if w != z and w != v and z != u:
                            # Check if w and z have a common neighbor in color c
                            # That common neighbor x must not be u or v
                            common = adj_bitmasks[c][w] & adj_bitmasks[c][z]
                            # Exclude u and v from common
                            if common & ~(1 << u) & ~(1 << v):
                                return True
                    z += 1
                    mv >>= 1
            w += 1
            mu >>= 1
        return False

    def _count_c6s(adj_bitmasks, n, c):
        """Count the number of C6s in color class c."""
        # For each edge (u,v) in color c, count paths of length 4 from u to v
        # This overcounts each C6 by 12 (each C6 has 6 edges, and each edge 
        # is part of 2 C6s if both directions counted... actually let's just count
        # distinct 6-cycles by checking all 6-tuples is too slow.
        # Instead, count "C6 witnesses": for each ordered pair (u,v) with edge in color c,
        # count the number of paths of length 4 from u to v in color c.
        # Each C6 will be counted 12 times (6 edges x 2 directions).
        count = 0
        for u in range(n):
            mu = adj_bitmasks[c][u]
            v = 0
            while mu:
                if mu & 1:
                    # v is neighbor of u in color c
                    # Count paths of length 4 from u to v
                    # Path u-w-x-y-z-v
                    # For each w neighbor of u (w != v), and z neighbor of v (z != u):
                    #   count common neighbors of w and z (excluding u,v)
                    w = 0
                    mw = mu & ~(1 << v)
                    while mw:
                        if mw & 1:
                            z = 0
                            mz = adj_bitmasks[c][v] & ~(1 << u)
                            while mz:
                                if mz & 1:
                                    if w != z:
                                        common = adj_bitmasks[c][w] & adj_bitmasks[c][z]
                                        common &= ~(1 << u) & ~(1 << v)
                                        count += bin(common).count('1')
                                z += 1
                                mz >>= 1
                        w += 1
                        mw >>= 1
                v += 1
                mu >>= 1
        # Each C6 is counted 12 times
        return count // 12

    def _total_c6_count(adj_bitmasks, n):
        """Total number of C6s across all 4 colors."""
        total = 0
        for c in range(4):
            total += _count_c6s(adj_bitmasks, n, c)
        return total

    def _build_adj_bitmasks(colors, n):
        """Build adjacency bitmasks from a color array (flattened, lexicographic)."""
        adj = [0] * n
        adj[0] = 0
        adj_bitmasks = [[0] * n for _ in range(4)]
        
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
                idx += 1
        return adj_bitmasks

    def _fitness(colors, n):
        """
        Fitness: negative of total C6 count. Higher is better.
        A valid coloring has 0 C6s, so fitness = 0.
        We want to maximize fitness (minimize C6 count).
        """
        adj_bitmasks = _build_adj_bitmasks(colors, n)
        return -_total_c6_count(adj_bitmasks, n)

    def _crossover(parent1, parent2):
        """Uniform crossover."""
        length = len(parent1)
        child = []
        for i in range(length):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def _mutate(colors, n, rate=0.05):
        """Random mutation: change some edges to random colors."""
        child = list(colors)
        for i in range(len(child)):
            if random.random() < rate:
                child[i] = random.randint(0, 3)
        return child

    def _ga_solve(n, time_limit, pop_size=100, max_gen=500):
        """
        Genetic algorithm to find a 4-coloring of K_n with no C6.
        Returns a valid coloring (list of colors) or None.
        """
        E = n * (n - 1) // 2
        
        # Initialize population
        population = []
        for _ in range(pop_size):
            # Random initialization
            colors = [random.randint(0, 3) for _ in range(E)]
            population.append(colors)
        
        # Track best
        best_colors = None
        best_fitness = float('-inf')
        
        gen = 0
        no_improve = 0
        while time.time() < time_limit and gen < max_gen:
            # Evaluate fitness
            fitnesses = []
            for colors in population:
                f = _fitness(colors, n)
                fitnesses.append(f)
                if f > best_fitness:
                    best_fitness = f
                    best_colors = list(colors)
                    if f == 0:
                        return best_colors
            
            # Sort by fitness
            indexed = list(zip(fitnesses, range(len(population))))
            indexed.sort(key=lambda x: -x[0])
            sorted_indices = [idx for _, idx in indexed]
            
            # Elitism: keep top 10%
            elite_count = max(1, pop_size // 10)
            elite = [list(population[idx]) for idx in sorted_indices[:elite_count]]
            
            # Generate next generation
            new_pop = list(elite)
            while len(new_pop) < pop_size and time.time() < time_limit:
                # Tournament selection
                def select():
                    a, b = random.sample(range(len(population)), 2)
                    if fitnesses[a] >= fitnesses[b]:
                        return population[a]
                    else:
                        return population[b]
                
                p1 = select()
                p2 = select()
                child = _crossover(p1, p2)
                child = _mutate(child, n, rate=0.03)
                new_pop.append(child)
            
            population = new_pop
            gen += 1
            
            # Check for stagnation
            if best_fitness > prev_best_fitness:
                no_improve = 0
            else:
                no_improve += 1
            
            prev_best_fitness = best_fitness
            
            if no_improve > 50:
                # Restart with more diversity
                for i in range(pop_size):
                    population[i] = [random.randint(0, 3) for _ in range(E)]
                no_improve = 0
        
        if best_colors is not None and best_fitness == 0:
            return best_colors
        return None

    # Try to find a valid coloring for n=17 using GA
    best_n = 0
    best_colors = None
    
    # First, try a quick greedy to get a baseline for smaller n
    def _greedy_solve(n, time_limit):
        """Simple greedy backtracking for small n."""
        E = n * (n - 1) // 2
        colors = [-1] * E
        adj_bitmasks = [[0] * n for _ in range(4)]
        
        def get_idx(i, j):
            if i > j:
                i, j = j, i
            return i * (2 * n - i - 1) // 2 - i  # This formula needs verification
        
        # Precompute edge index
        edge_to_idx = {}
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                edge_to_idx[(i, j)] = idx
                idx += 1
        
        def backtrack(eidx):
            if time.time() > time_limit:
                return False
            if eidx == E:
                return True
            
            i, j = None, None
            for (a, b), idx2 in edge_to_idx.items():
                if idx2 == eidx:
                    i, j = a, b
                    break
            
            for c in range(4):
                # Check if adding (i,j) with color c creates C6
                # Temporarily add to bitmasks
                adj_bitmasks[c][i] |= (1 << j)
                adj_bitmasks[c][j] |= (1 << i)
                
                if not _bitmask_has_c6(adj_bitmasks, n, i, j, c):
                    colors[eidx] = c
                    if backtrack(eidx + 1):
                        return True
                    colors[eidx] = -1
                else:
                    # Undo
                    adj_bitmasks[c][i] &= ~(1 << j)
                    adj_bitmasks[c][j] &= ~(1 << i)
            
            return False
        
        if backtrack(0):
            return colors
        return None

    # Try n=17 with GA
    if time.time() < deadline - 20:
        sol = _ga_solve(17, deadline - 25)
        if sol is not None:
            best_n = 17
            best_colors = sol
    
    # If GA failed, try greedy for smaller n
    if best_colors is None:
        for n in range(16, 9, -1):
            if time.time() > deadline - 10:
                break
            sol = _greedy_solve(n, deadline - 12)
            if sol is not None:
                best_n = n
                best_colors = sol
                break
    
    # Fallback: return n=1 if all else fails
    if best_colors is None:
        return {'n': 1, 'colors': []}
    
    return {'n': best_n, 'colors': best_colors}

if __name__ == "__main__":
    construct()
