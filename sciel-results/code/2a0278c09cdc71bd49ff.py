import time
import numpy as np
import random

def _has_k222_fast(adj_bits, n):
    """Fast K2,2,2 check using bitmasks."""
    for a in range(n):
        for b in range(a + 1, n):
            common_ab = adj_bits[a] & adj_bits[b]
            common_ab &= ~(1 << a)
            common_ab &= ~(1 << b)
            
            if common_ab.bit_count() < 4:
                continue
            
            cn_list = []
            mask = common_ab
            while mask:
                lsb = mask & (-mask)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                mask ^= lsb
            
            m = len(cn_list)
            for ci in range(m):
                c = cn_list[ci]
                for di in range(ci + 1, m):
                    d = cn_list[di]
                    common_cd = adj_bits[c] & adj_bits[d]
                    intersection = common_ab & common_cd
                    intersection &= ~(1 << c)
                    intersection &= ~(1 << d)
                    
                    if intersection.bit_count() >= 2:
                        return True
    return False

def _count_k222_fast(adj_bits, n):
    """Count K2,2,2 subgraphs using bitmasks. Returns count of K2,2,2 occurrences."""
    count = 0
    for a in range(n):
        for b in range(a + 1, n):
            common_ab = adj_bits[a] & adj_bits[b]
            common_ab &= ~(1 << a)
            common_ab &= ~(1 << b)
            
            if common_ab.bit_count() < 4:
                continue
            
            cn_list = []
            mask = common_ab
            while mask:
                lsb = mask & (-mask)
                idx = lsb.bit_length() - 1
                cn_list.append(idx)
                mask ^= lsb
            
            m = len(cn_list)
            for ci in range(m):
                c = cn_list[ci]
                for di in range(ci + 1, m):
                    d = cn_list[di]
                    common_cd = adj_bits[c] & adj_bits[d]
                    intersection = common_ab & common_cd
                    intersection &= ~(1 << c)
                    intersection &= ~(1 << d)
                    
                    bc = intersection.bit_count()
                    if bc >= 2:
                        # Number of ways to choose 2 from bc vertices
                        count += bc * (bc - 1) // 2
    # Each K2,2,2 is counted 3 times (once for each of its 3 pairs)
    # But we also need to account for the fact that for each pair (a,b),
    # we're counting all ways to form the remaining structure.
    # Actually this overcounts. Let's just use it as a proxy.
    return count

def _check_validity_fast(colors, n):
    """Check validity using bitmask representation."""
    adj0 = [0] * n
    adj1 = [0] * n
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            if c == 0:
                adj0[i] |= (1 << j)
                adj0[j] |= (1 << i)
            else:
                adj1[i] |= (1 << j)
                adj1[j] |= (1 << i)
            idx += 1
    return not _has_k222_fast(adj0, n) and not _has_k222_fast(adj1, n)

def _build_paley(n):
    """Build Paley coloring for prime n ≡ 1 (mod 4)."""
    squares = set()
    for x in range(1, n):
        squares.add((x * x) % n)
    
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            diff = (j - i) % n
            colors.append(0 if diff in squares else 1)
    return colors

def _build_paley_subgraph(prime, remove_vertex):
    """Build Paley coloring for K_prime with one vertex removed."""
    squares = set()
    for x in range(1, prime):
        squares.add((x * x) % prime)
    
    vertices = [v for v in range(prime) if v != remove_vertex]
    n = len(vertices)
    
    colors = []
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            old_i = vertices[i_idx]
            old_j = vertices[j_idx]
            diff = (old_j - old_i) % prime
            colors.append(0 if diff in squares else 1)
    return colors

def _genetic_search_30(colors29, n29=29, deadline=None):
    """Genetic algorithm to find a valid 30-vertex coloring extending the 29-vertex base."""
    if deadline is None:
        deadline = time.time() + 30
    
    # Build adjacency for the 29-vertex graph
    adj0_29 = [0] * n29
    adj1_29 = [0] * n29
    idx = 0
    for i in range(n29):
        for j in range(i + 1, n29):
            c = colors29[idx]
            if c == 0:
                adj0_29[i] |= (1 << j)
                adj0_29[j] |= (1 << i)
            else:
                adj1_29[i] |= (1 << j)
                adj1_29[j] |= (1 << i)
            idx += 1
    
    def _get_full_adj(new_edges):
        full_adj0 = [0] * 30
        full_adj1 = [0] * 30
        for i in range(n29):
            full_adj0[i] = adj0_29[i]
            full_adj1[i] = adj1_29[i]
        for i in range(n29):
            c = new_edges[i]
            if c == 0:
                full_adj0[29] |= (1 << i)
                full_adj0[i] |= (1 << 29)
            else:
                full_adj1[29] |= (1 << i)
                full_adj1[i] |= (1 << 29)
        return full_adj0, full_adj1

    def _penalty(new_edges):
        """Count K2,2,2 in both color classes."""
        full_adj0, full_adj1 = _get_full_adj(new_edges)
        return _count_k222_fast(full_adj0, 30) + _count_k222_fast(full_adj1, 30)

    def _crossover(parent1, parent2):
        """Uniform crossover."""
        child = []
        for i in range(n29):
            if random.random() < 0.5:
                child.append(parent1[i])
            else:
                child.append(parent2[i])
        return child

    def _mutate(individual, rate=0.1):
        """Bit-flip mutation."""
        child = individual[:]
        for i in range(n29):
            if random.random() < rate:
                child[i] = 1 - child[i]
        return child

    # Initialize population with diverse individuals
    random.seed(42)
    pop_size = 50
    population = []
    
    # Start with some heuristic initializations
    # 1. All 0s
    population.append([0] * n29)
    # 2. All 1s
    population.append([1] * n29)
    # 3. Random
    for _ in range(pop_size - 2):
        population.append([random.randint(0, 1) for _ in range(n29)])
    
    # Evaluate initial population
    fitness = [_penalty(ind) for ind in population]
    best_idx = min(range(pop_size), key=lambda i: fitness[i])
    best_fitness = fitness[best_idx]
    best_individual = population[best_idx][:]
    
    if best_fitness == 0:
        return best_individual
    
    # GA loop
    generations = 0
    while time.time() < deadline:
        generations += 1
        
        # Select parents (tournament selection)
        new_population = []
        
        # Elitism: keep best
        new_population.append(best_individual[:])
        
        while len(new_population) < pop_size:
            # Tournament selection
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            if fitness[p1_idx] <= fitness[p2_idx]:
                p1 = population[p1_idx]
            else:
                p1 = population[p2_idx]
            
            p1_idx = random.randint(0, pop_size - 1)
            p2_idx = random.randint(0, pop_size - 1)
            if fitness[p1_idx] <= fitness[p2_idx]:
                p2 = population[p1_idx]
            else:
                p2 = population[p2_idx]
            
            # Crossover
            if random.random() < 0.8:
                child = _crossover(p1, p2)
            else:
                child = p1[:]
            
            # Mutation
            child = _mutate(child, rate=0.15)
            
            new_population.append(child)
        
        population = new_population
        fitness = [_penalty(ind) for ind in population]
        
        # Update best
        current_best_idx = min(range(pop_size), key=lambda i: fitness[i])
        if fitness[current_best_idx] < best_fitness:
            best_fitness = fitness[current_best_idx]
            best_individual = population[current_best_idx][:]
            if best_fitness == 0:
                return best_individual
    
    return best_individual if best_fitness == 0 else None

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s kill
    
    # Start with Paley graph on 29 vertices
    p29 = 29
    colors29 = _build_paley(p29)
    
    # Verify
    if not _check_validity_fast(colors29, p29):
        p17 = 17
        colors17 = _build_paley(p17)
        return {'n': p17, 'colors': colors17}
    
    best_n = p29
    best_colors = colors29[:]
    
    # Try Paley on 31 with vertex removed
    p31 = 31
    for remove in range(p31):
        if time.time() > deadline - 20:
            break
        
        colors30 = _build_paley_subgraph(p31, remove)
        
        if _check_validity_fast(colors30, 30):
            return {'n': 30, 'colors': colors30}
    
    # Try genetic algorithm to extend 29 to 30
    if time.time() < deadline - 10:
        result = _genetic_search_30(colors29, deadline=deadline - 5)
        if result is not None:
            # Build full 30-vertex coloring
            full_colors = colors29[:]
            full_colors.extend(result)
            if _check_validity_fast(full_colors, 30):
                return {'n': 30, 'colors': full_colors}
    
    return {'n': best_n, 'colors': best_colors}
