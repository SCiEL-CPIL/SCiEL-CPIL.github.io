"""Size-efficient sorting networks on 17 wires via bit-vector simulation and GA."""
import time
import random

N = 17
NUM_INPUTS = 1 << N
DEADLINE = 55.0  # Safety margin before 60s limit
START_TIME = time.time()

# Precompute bit-vector representation of initial states
# Each wire is represented as a Python integer where bit k is the value on that wire for input k.
# There are 2^N inputs, so we need 2^N bits per wire.
# For N=17, 131072 bits = 16KB per wire. 17 wires = 272KB.
# Python big integers handle this efficiently.

def _init_bit_vectors():
    """
    Create initial bit-vector states for each wire.
    state[wire] is an integer where bit k is set if wire has value 1 for input k.
    Input k is an integer from 0 to 2^N - 1.
    Wire w has value 1 if bit w of k is 1.
    """
    states = [0] * N
    for k in range(NUM_INPUTS):
        for w in range(N):
            if (k >> w) & 1:
                states[w] |= (1 << k)
    return states

INITIAL_STATES = _init_bit_vectors()

def _time_left():
    return DEADLINE - (time.time() - START_TIME)

def verify_network(comps):
    """
    Verify network using bit-vector simulation.
    comps: list of (i, j) tuples.
    Returns True if valid.
    """
    if not comps:
        return False
    
    # Copy initial states
    state = list(INITIAL_STATES)
    
    for i, j in comps:
        wi = state[i]
        wj = state[j]
        
        # Comparator (i, j): min to i, max to j.
        # min(wi, wj) = wi & wj
        # max(wi, wj) = wi | wj
        state[i] = wi & wj
        state[j] = wi | wj
        
    # Check if sorted: for all rows, state[:, k] <= state[:, k+1]
    # This means for all k, if state[w] has bit k set, state[w+1] must also have bit k set.
    # i.e., state[w] & ~state[w+1] must be 0 for all w.
    for w in range(N - 1):
        if state[w] & ~state[w + 1]:
            return False
            
    return True

def make_seed():
    """Odd-even transposition network."""
    return [(i, i + 1) for phase in range(N) for i in range(phase % 2, N - 1, 2)]

def greedy_remove(comps):
    """
    Try to remove each comparator if the network remains valid.
    """
    n = len(comps)
    improved = True
    while improved:
        improved = False
        for idx in range(n - 1, -1, -1):
            if _time_left() < 2.0:
                return comps
            candidate = comps[:idx] + comps[idx+1:]
            if verify_network(candidate):
                comps = candidate
                n = len(comps)
                improved = True
                break # Restart scan
    return comps

def construct():
    """Construct a sorting network for 17 wires, minimizing comparator count."""
    global START_TIME
    START_TIME = time.time()
    
    rng = random.Random(42)
    
    # Start with odd-even transposition
    comps = make_seed()
    
    # First pass: aggressive greedy removal
    comps = greedy_remove(comps)
    
    best = list(comps)
    best_len = len(best)
    
    current = list(best)
    current_len = best_len
    
    # Genetic Algorithm with depth-aware mutations
    # Population size
    pop_size = 10
    population = [list(best)]
    
    # Add some random valid networks to population
    for _ in range(pop_size - 1):
        # Start from best and apply random mutations
        pop_candidate = list(best)
        for _ in range(5):
            op = rng.choice(['swap', 'move'])
            n = len(pop_candidate)
            if n >= 2 and op == 'swap':
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    pop_candidate[i1], pop_candidate[i2] = pop_candidate[i2], pop_candidate[i1]
            elif n >= 1 and op == 'move':
                i1 = rng.randrange(n)
                i2 = rng.randrange(n)
                if i1 != i2:
                    elem = pop_candidate.pop(i1)
                    pop_candidate.insert(i2, elem)
            if verify_network(pop_candidate):
                population.append(list(pop_candidate))
                break
        else:
            population.append(list(best))
    
    # GA loop
    generations = 0
    while _time_left() > 5.0:
        generations += 1
        new_population = []
        
        # Elitism: keep the best
        new_population.append(list(best))
        
        # Crossover and mutation
        for _ in range(pop_size - 1):
            # Select parents
            p1 = rng.choice(population)
            p2 = rng.choice(population)
            
            # Crossover: uniform crossover on comparator list
            # Since length might differ, we take min length for crossover, then append remainder
            len1 = len(p1)
            len2 = len(p2)
            min_len = min(len1, len2)
            
            child = []
            for i in range(min_len):
                if rng.random() < 0.5:
                    child.append(p1[i])
                else:
                    child.append(p2[i])
            
            # Append remainder
            if len1 > min_len:
                child.extend(p1[min_len:])
            elif len2 > min_len:
                child.extend(p2[min_len:])
            
            # Mutation
            n = len(child)
            num_mutations = rng.randint(1, 3)
            for _ in range(num_mutations):
                if n == 0:
                    break
                op = rng.choice(['remove', 'swap', 'move', 'remove'])
                if op == 'remove' and n > 0:
                    idx = rng.randrange(n)
                    child.pop(idx)
                    n -= 1
                elif op == 'swap' and n >= 2:
                    i1 = rng.randrange(n)
                    i2 = rng.randrange(n)
                    if i1 != i2:
                        child[i1], child[i2] = child[i2], child[i1]
                elif op == 'move' and n >= 1:
                    i1 = rng.randrange(n)
                    i2 = rng.randrange(n)
                    if i1 != i2:
                        elem = child.pop(i1)
                        child.insert(i2, elem)
            
            # Evaluate
            if verify_network(child):
                # Greedy remove to improve
                child = greedy_remove(child)
                if len(child) < best_len:
                    best = list(child)
                    best_len = len(child)
                    new_population.insert(0, list(best))
                else:
                    new_population.append(list(child))
            else:
                # Invalid, keep old parent
                new_population.append(list(p1))
        
        population = new_population[:pop_size]
        
        # Local search on best
        if _time_left() > 10.0:
            improved_best = greedy_remove(best)
            if len(improved_best) < best_len:
                best = improved_best
                best_len = len(best)
                population[0] = list(best)
    
    # Final greedy pass
    best = greedy_remove(best)
    
    return best
