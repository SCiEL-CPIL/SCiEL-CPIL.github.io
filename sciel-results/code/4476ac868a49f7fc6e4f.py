"""
Strategy: Genetic Algorithm with Block Crossover, Greedy Mutation, and Error Repair
Instead of single-point local search (remove-and-refill), we maintain a population of 
candidate solutions (bitsets). We use a block-based crossover where we partition the 
domain [1..512] into blocks and select the block from the fitter parent for each block.
We then apply a mutation phase that greedily adds any element that doesn't violate 
the 4-AP constraint, and a repair phase that removes elements to fix any violations.
"""
import time
import random
import numpy as np

N = 512
DEADLINE = 58.0  # Safety margin

# Precompute conflict information using bitsets for speed
# For each x, precompute a list of "conflict masks" or just use the forbidden triples approach
# but optimized with bit operations.

def precompute_forbidden():
    """
    Precompute for each x (0-indexed), the list of triples (u, v, w) such that 
    {x, u, v, w} forms a 4-AP.
    """
    forbidden = [[] for _ in range(N)]
    for d in range(1, (N - 1) // 3 + 1):
        for a in range(1, N - 3 * d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            # 4-AP: a, b, c, e
            forbidden[a - 1].append((b - 1, c - 1, e - 1))
            forbidden[b - 1].append((a - 1, c - 1, e - 1))
            forbidden[c - 1].append((a - 1, b - 1, e - 1))
            forbidden[e - 1].append((a - 1, b - 1, c - 1))
    return forbidden

FORBIDDEN = precompute_forbidden()

def can_add(bits, x):
    """
    Check if x (0-indexed) can be added to the set represented by bits.
    """
    for u, v, w in FORBIDDEN[x]:
        if (bits >> u) & 1 and (bits >> v) & 1 and (bits >> w) & 1:
            return False
    return True

def greedy_add(bits):
    """
    Try to add any element not in the set that can be added.
    Returns the updated bits.
    """
    # Try to add elements in a random order
    candidates = [i for i in range(N) if not (bits >> i) & 1]
    random.shuffle(candidates)
    for c in candidates:
        if can_add(bits, c):
            bits |= (1 << c)
    return bits

def repair(bits):
    """
    Remove elements to fix any 4-AP violations.
    This is a simple heuristic: find a 4-AP and remove one element from it.
    """
    # Check for violations
    # For each 4-AP, check if all four are in the set
    # If so, remove one (randomly)
    changed = True
    while changed:
        changed = False
        for d in range(1, (N - 1) // 3 + 1):
            for a in range(1, N - 3 * d + 1):
                b = a + d
                c = a + 2 * d
                e = a + 3 * d
                # Check if all four are in the set
                if ((bits >> (a-1)) & 1) and ((bits >> (b-1)) & 1) and \
                   ((bits >> (c-1)) & 1) and ((bits >> (e-1)) & 1):
                    # Remove one randomly
                    options = [a-1, b-1, c-1, e-1]
                    remove_idx = random.choice(options)
                    bits &= ~(1 << remove_idx)
                    changed = True
                    break
            if changed:
                break
    return bits

def block_crossover(parent1, parent2, size1, size2):
    """
    Block-based crossover: partition [1..N] into blocks, select the block from 
    the fitter parent for each block.
    """
    # Partition into blocks of size ~16
    block_size = 16
    num_blocks = (N + block_size - 1) // block_size
    
    # Determine which parent is fitter
    if size1 >= size2:
        fitter, less = parent1, parent2
    else:
        fitter, less = parent2, parent1
    
    child = 0
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, N)
        # Randomly choose which parent to take this block from
        # Bias towards the fitter parent
        if random.random() < 0.7:
            # Take from fitter parent
            for i in range(start, end):
                if (fitter >> i) & 1:
                    child |= (1 << i)
        else:
            # Take from less fit parent
            for i in range(start, end):
                if (less >> i) & 1:
                    child |= (1 << i)
    
    return child

def construct():
    deadline = time.time() + DEADLINE
    
    # 1. Initialize population with diverse random solutions
    pop_size = 50
    population = []
    
    # Create diverse initial solutions
    for i in range(pop_size):
        # Different strategies for initialization
        if i % 3 == 0:
            # Random greedy
            bits = 0
            order = list(range(N))
            random.shuffle(order)
            for x in order:
                if can_add(bits, x):
                    bits |= (1 << x)
        elif i % 3 == 1:
            # Greedy in increasing order
            bits = 0
            for x in range(N):
                if can_add(bits, x):
                    bits |= (1 << x)
        else:
            # Greedy in decreasing order
            bits = 0
            for x in range(N - 1, -1, -1):
                if can_add(bits, x):
                    bits |= (1 << x)
        
        # Apply mutation
        bits = greedy_add(bits)
        # Apply repair (should be no-op if greedy is correct)
        bits = repair(bits)
        population.append(bits)
    
    # Evaluate initial population
    sizes = [bin(b).count('1') for b in population]
    best_idx = max(range(pop_size), key=lambda i: sizes[i])
    best_bits = population[best_idx]
    best_size = sizes[best_idx]
    
    # 2. Genetic algorithm loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Sort population by size (descending)
        pop_with_sizes = [(population[i], sizes[i]) for i in range(pop_size)]
        pop_with_sizes.sort(key=lambda x: x[1], reverse=True)
        
        # Update best
        if pop_with_sizes[0][1] > best_size:
            best_size = pop_with_sizes[0][1]
            best_bits = pop_with_sizes[0][0]
        
        # Selection: keep top half
        top_half = pop_with_sizes[:pop_size // 2]
        
        # Create next generation
        next_pop = []
        
        # Elitism: keep the best
        next_pop.append(top_half[0][0])
        next_pop.append(top_half[0][1])
        
        # Crossover and mutation
        while len(next_pop) < pop_size and time.time() < deadline:
            # Select two parents from top half
            p1_idx = random.randint(0, len(top_half) - 1)
            p2_idx = random.randint(0, len(top_half) - 1)
            p1, s1 = top_half[p1_idx]
            p2, s2 = top_half[p2_idx]
            
            # Crossover
            child = block_crossover(p1, p2, s1, s2)
            
            # Mutation: greedy add
            child = greedy_add(child)
            
            # Repair
            child = repair(child)
            
            # Evaluate
            child_size = bin(child).count('1')
            next_pop.append((child, child_size))
        
        # Update population
        population = [x[0] for x in next_pop]
        sizes = [x[1] for x in next_pop]
        
        # If stuck, inject some random diversity
        if gen % 20 == 0 and best_size < 200:
            for _ in range(5):
                bits = 0
                order = list(range(N))
                random.shuffle(order)
                for x in order:
                    if can_add(bits, x):
                        bits |= (1 << x)
                bits = greedy_add(bits)
                bits = repair(bits)
                population[random.randint(0, pop_size - 1)] = bits
                sizes[random.randint(0, pop_size - 1)] = bin(bits).count('1')
    
    # Convert best_bits to list
    result = [i + 1 for i in range(N) if (best_bits >> i) & 1]
    return result
