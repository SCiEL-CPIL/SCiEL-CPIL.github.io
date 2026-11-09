import random
import copy
import time

N = 16
MAX_COMPARATORS = 4096

# EVOLVE-BLOCK-START

def construct():
    """
    Returns a list of comparator pairs for a sorting network on N wires.
    Uses a genetic algorithm to find a compact network.
    """
    def count_comparators(network):
        return len(network)

    def get_depth(network):
        if not network:
            return 0
        # Calculate depth by simulating parallel layers
        # A comparator (i, j) is ready when the previous comparator on i or j is done
        # We can use a simple simulation:
        # depth[i] = time when wire i is last touched
        # For a sequence of comparators, if we assume sequential execution, depth is len(network).
        # But the task asks for "parallel depth is reported separately" and the score is based on comparator count.
        # However, the interface returns a list. The evaluator likely runs it sequentially or checks validity.
        # For the GA, we just need to minimize count.
        return len(network)

    def is_valid(network):
        """
        Verify the network sorts all 0-1 inputs.
        For N=16, this is 2^16 = 65536 checks.
        We can optimize by only checking a subset or using a faster method,
        but for correctness in GA, we might use a heuristic or a fast check.
        Given 60s limit, and GA iterations, we need a fast validity check or heuristic.
        
        A full 0-1 principle check is expensive.
        Let's try to optimize the check.
        We can represent the input as a bitmask.
        We process the comparators.
        If at any point the network fails to sort a specific input, it's invalid.
        
        To speed up, we can check:
        1. Does it sort the input (1, 1, ..., 1, 0, 0, ..., 0)? (All 1s then 0s)
           Actually, the 0-1 principle says it's sufficient to check all 2^N inputs.
           But there is a stronger condition: a network sorts all inputs iff it sorts all inputs of the form
           where the first k bits are 1 and the rest 0, for k=0..N? No, that's not sufficient.
           
           However, we can use the fact that if a network fails, it fails on some 0-1 input.
           
           For the GA, we can use a fitness function based on a random sample of 0-1 inputs,
           and only do the full check for the best candidates.
        """
        # This will be the expensive part.
        # We'll use a fast C-like loop in Python, or just accept it might be slow.
        # 65536 * len(network) operations. If len(network) is ~60, that's ~4M ops.
        # In Python, 4M ops might take 0.1-0.5s.
        # If we run 100 generations * 50 individuals * 0.5s = 2500s. Too slow.
        
        # We need a faster validity check or a heuristic.
        
        # Heuristic: Check if the output is sorted for a random subset of 0-1 inputs.
        # If it passes the subset, it's *likely* valid.
        # We will do the full check only for the final candidate or when we think it's close.
        
        pass # Placeholder, actual implementation below

    def fitness(network):
        """
        Returns a score where higher is better.
        We want to minimize comparator count and maximize validity.
        Let's define fitness = -len(network) if valid, else -10000.
        Or, if we use a heuristic validity:
        fitness = -len(network) + 10 * (fraction of random 0-1 inputs sorted correctly)
        """
        if not network:
            return 0
        
        # Quick check: does it sort a few specific hard cases?
        # Case 1: 1, 0, 1, 0...
        # Case 2: 0, 1, 0, 1...
        # Case 3: Random
        
        # Let's use a fast approximation:
        # Simulate on 100 random 0-1 inputs.
        # Count how many are sorted.
        
        import random as rnd
        sample_size = 200
        correct = 0
        
        # Pre-generate sample inputs for consistency within a fitness call?
        # Or just use global random.
        
        for _ in range(sample_size):
            # Generate random 0-1 input
            inp = [0] * N
            for i in range(N):
                if rnd.random() < 0.5:
                    inp[i] = 1
            
            # Apply network
            current = inp[:]
            for u, v in network:
                if current[u] > current[v]:
                    current[u], current[v] = current[v], current[u]
            
            # Check if sorted
            is_sorted = True
            for i in range(N - 1):
                if current[i] > current[i + 1]:
                    is_sorted = False
                    break
            
            if is_sorted:
                correct += 1
        
        fraction = correct / sample_size
        
        # If fraction is 1, it might be valid. We penalize length.
        # If fraction < 1, we heavily penalize.
        
        score = fraction * 10000 - len(network)
        
        return score

    def full_check(network):
        """
        Exhaustive 0-1 check.
        """
        if not network:
            return N <= 1
        
        for mask in range(1 << N):
            current = [0] * N
            for i in range(N):
                if mask & (1 << i):
                    current[i] = 1
            
            for u, v in network:
                if current[u] > current[v]:
                    current[u], current[v] = current[v], current[u]
            
            # Check sorted
            for i in range(N - 1):
                if current[i] > current[i + 1]:
                    return False
        
        return True

    def mutate(network, mutation_rate=0.1):
        new_network = network[:]
        # Randomly add, remove, or swap comparators
        num_mutations = int(len(new_network) * mutation_rate) + 1
        
        for _ in range(num_mutations):
            op = random.randint(0, 2)
            if op == 0 and new_network: # Remove
                idx = random.randint(0, len(new_network) - 1)
                new_network.pop(idx)
            elif op == 1: # Add
                u = random.randint(0, N - 1)
                v = random.randint(0, N - 1)
                if u != v:
                    if u > v:
                        u, v = v, u
                    new_network.append((u, v))
            elif op == 2 and len(new_network) >= 2: # Swap
                i = random.randint(0, len(new_network) - 2)
                j = random.randint(i + 1, len(new_network) - 1)
                new_network[i], new_network[j] = new_network[j], new_network[i]
        
        return new_network

    def crossover(net1, net2):
        # Simple crossover: take first half of net1 and second half of net2
        # Or splice.
        if not net1 or not net2:
            return (net1 or net2)[:], (net2 or net1)[:]
        
        p = random.randint(1, min(len(net1), len(net2)) - 1)
        child1 = net1[:p] + net2[p:]
        child2 = net2[:p] + net1[p:]
        return child1, child2

    def initialize_population(pop_size):
        population = []
        for _ in range(pop_size):
            # Start with a random network or a known valid one mutated
            # Let's start with Odd-Even Transposition and mutate
            base = [(i, i + 1) for phase in range(N)
                    for i in range(phase % 2, N - 1, 2)]
            
            # Heavily mutate
            net = base[:]
            # Randomly remove some
            for _ in range(len(net) // 2):
                if net:
                    idx = random.randint(0, len(net) - 1)
                    net.pop(idx)
            
            population.append(net)
        
        return population

    # GA Parameters
    pop_size = 50
    generations = 20
    mutation_rate = 0.1
    
    start_time = time.time()
    time_limit = 55  # Leave 5s buffer
    
    population = initialize_population(pop_size)
    
    best_network = None
    best_score = -float('inf')
    
    # Evaluate initial population
    for net in population:
        score = fitness(net)
        if score > best_score:
            best_score = score
            best_network = net[:]
    
    for gen in range(generations):
        if time.time() - start_time > time_limit:
            break
            
        new_population = []
        
        # Elitism: keep the best
        new_population.append(best_network[:])
        
        while len(new_population) < pop_size:
            # Select parents (Tournament selection)
            p1 = random.choice(population)
            p2 = random.choice(population)
            
            # Crossover
            c1, c2 = crossover(p1, p2)
            
            # Mutate
            c1 = mutate(c1, mutation_rate)
            c2 = mutate(c2, mutation_rate)
            
            # Evaluate
            s1 = fitness(c1)
            s2 = fitness(c2)
            
            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)
                
            if s1 > best_score:
                best_score = s1
                best_network = c1[:]
            if s2 > best_score:
                best_score = s2
                best_network = c2[:]
        
        population = new_population
    
    # Check if the best network is fully valid
    if best_network and full_check(best_network):
        return best_network
    
    # If not valid, try to repair or fall back
    # Let's try to add more comparators to the best invalid one and re-check
    # Or just return the best invalid one if it's close? No, it must be valid.
    
    # Fallback: Return a known valid network (Odd-Even Transposition)
    # This ensures correctness, even if score is 1.0
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

# EVOLVE-BLOCK-END
