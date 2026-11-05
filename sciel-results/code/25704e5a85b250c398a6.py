"""
Constructs a 12-state binary deterministic automaton maximizing shortest reset word length.
Uses a Hybrid Genetic Algorithm with optimized BFS threshold calculation.
"""
import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1

# EVOLVE-BLOCK-START

def _calculate_threshold(a, b):
    """
    Computes the exact reset threshold using BFS on the subset automaton.
    Optimized by precomputing transitions for all 2^N subsets.
    Returns -1 if not synchronizing.
    """
    size = 1 << N
    
    # Pre-compute transition tables for 'a' and 'b' for all subsets
    # trans_a[mask] = subset reached from mask by applying 'a'
    trans_a = [0] * size
    trans_b = [0] * size
    
    for mask in range(size):
        ma = 0
        mb = 0
        # Iterate through bits set in mask
        m = mask
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            ma |= 1 << a[i]
            mb |= 1 << b[i]
            m -= lsb
        trans_a[mask] = ma
        trans_b[mask] = mb
        
    visited = bytearray(size) # Faster than set for boolean checks
    visited[FULL_MASK] = 1
    
    queue = deque()
    queue.append((FULL_MASK, 0))
    
    while queue:
        mask, depth = queue.popleft()
        
        if mask & (mask - 1) == 0:
            return depth
            
        next_a = trans_a[mask]
        if not visited[next_a]:
            visited[next_a] = 1
            queue.append((next_a, depth + 1))
            
        next_b = trans_b[mask]
        if not visited[next_b]:
            visited[next_b] = 1
            queue.append((next_b, depth + 1))
            
    return -1

def _random_automaton():
    return [random.randint(0, N-1) for _ in range(N)], [random.randint(0, N-1) for _ in range(N)]

def _mutate_swap(a, b):
    """
    Mutate by swapping destinations or changing a single transition to an existing destination.
    This tends to preserve structural properties better than fully random mutation.
    """
    a_new = a[:]
    b_new = b[:]
    
    # Choose which map to mutate
    if random.random() < 0.5:
        pos = random.randint(0, N - 1)
        # Strategy 1: Swap with another state's destination in the same map
        if random.random() < 0.3:
            other = random.randint(0, N - 1)
            a_new[pos], a_new[other] = a_new[other], a_new[pos]
        else:
            # Strategy 2: Change to a random destination
            a_new[pos] = random.randint(0, N - 1)
    else:
        pos = random.randint(0, N - 1)
        if random.random() < 0.3:
            other = random.randint(0, N - 1)
            b_new[pos], b_new[other] = b_new[other], b_new[pos]
        else:
            b_new[pos] = random.randint(0, N - 1)
            
    return a_new, b_new

def _crossover(a1, b1, a2, b2):
    """
    Simple uniform crossover.
    """
    a_child = []
    b_child = []
    for i in range(N):
        if random.random() < 0.5:
            a_child.append(a1[i])
            b_child.append(b1[i])
        else:
            a_child.append(a2[i])
            b_child.append(b2[i])
    return a_child, b_child

def construct():
    """
    Construct a 12-state binary automaton maximizing reset threshold.
    Uses a population-based search with optimized evaluation.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 60s limit
    
    # Initialize population with diverse high-quality structures
    # 1. Standard Černý automaton (threshold N^2 - 3N + 4 = 144 - 36 + 4 = 112? No, for N=12, max is 112? 
    #    Actually Černý bound is (N-1)^2. For N=12, (11)^2 = 121.
    #    The standard Černý automaton has threshold (N-1)^2.
    #    Let's verify: a is a cycle, b fixes 0 and shifts others.
    #    a: i -> i+1 mod N. b: 0->0, i->i-1 mod N (or similar).
    #    Let's construct the standard one.
    
    # Standard Černý automaton:
    # a: i -> (i+1) % N
    # b: 0 -> 0, i -> (i-1) % N for i > 0? 
    # Actually, the standard one is:
    # a: cycle 0->1->...->N-1->0
    # b: 0->0, 1->0, 2->1, ..., N-1->N-2
    # This has threshold (N-1)^2.
    
    a_cerny = [(i + 1) % N for i in range(N)]
    b_cerny = [0] * N
    for i in range(1, N):
        b_cerny[i] = i - 1
        
    pop = []
    
    # Add standard Černy
    s = _calculate_threshold(a_cerny, b_cerny)
    pop.append((s, a_cerny[:], b_cerny[:]))
    
    # Add variations
    for _ in range(5):
        a, b = _random_automaton()
        s = _calculate_threshold(a, b)
        pop.append((s, a, b))
        
    # Sort by score descending
    pop.sort(key=lambda x: x[0], reverse=True)
    
    best_score = pop[0][0]
    best_a = pop[0][1][:]
    best_b = pop[0][2][:]
    
    # If best is not synchronizing, try more randoms
    if best_score < 0:
        for _ in range(100):
            a, b = _random_automaton()
            s = _calculate_threshold(a, b)
            if s > best_score:
                best_score = s
                best_a = a
                best_b = b
                pop[0] = (s, a, b)
                pop.sort(key=lambda x: x[0], reverse=True)
            if best_score > 0:
                break
        if best_score < 0:
             # Fallback: return a valid synchronizing automaton if found in randoms, else best effort
             # But we must return a valid one. The standard Cerny should sync.
             # If _calculate_threshold failed for Cerny, there is a bug.
             # Assuming Cerny works, best_score should be 121.
             pass

    population_size = 20
    # Fill population with mutations of the best
    while len(pop) < population_size:
        parent_a, parent_b = best_a, best_b
        child_a, child_b = _mutate_swap(parent_a, parent_b)
        s = _calculate_threshold(child_a, child_b)
        pop.append((s, child_a, child_b))
        
    pop.sort(key=lambda x: x[0], reverse=True)
    
    generation = 0
    while time.time() < deadline:
        generation += 1
        next_pop = []
        
        # Elitism: keep top 2
        for i in range(2):
            next_pop.append(pop[i])
            
        # Fill rest with offspring
        while len(next_pop) < population_size:
            # Select parents (tournament or random from top half)
            p1 = pop[random.randint(0, population_size // 2)]
            p2 = pop[random.randint(0, population_size // 2)]
            
            if random.random() < 0.5:
                child_a, child_b = _crossover(p1[1], p1[2], p2[1], p2[2])
            else:
                child_a, child_b = p1[1][:], p1[2][:]
            
            # Mutate
            if random.random() < 0.8:
                child_a, child_b = _mutate_swap(child_a, child_b)
            
            s = _calculate_threshold(child_a, child_b)
            next_pop.append((s, child_a, child_b))
            
        pop = next_pop
        pop.sort(key=lambda x: x[0], reverse=True)
        
        if pop[0][0] > best_score:
            best_score = pop[0][0]
            best_a = pop[0][1][:]
            best_b = pop[0][2][:]
            
        # Periodic injection of randoms to maintain diversity
        if generation % 10 == 0:
            a, b = _random_automaton()
            s = _calculate_threshold(a, b)
            if s > pop[-1][0]:
                pop[-1] = (s, a, b)
                pop.sort(key=lambda x: x[0], reverse=True)

    return [best_a, best_b]

# EVOLVE-BLOCK-END
