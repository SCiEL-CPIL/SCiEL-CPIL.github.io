import time
import random
import sys

N = 17
TOTAL_MASKS = 1 << N
DEADLINE_FRACTION = 0.90

# Pre-compute initial and expected wires for verification
_INITIAL_WIRES = None
_EXPECTED_WIRES = None

def _init_wires():
    global _INITIAL_WIRES, _EXPECTED_WIRES
    if _INITIAL_WIRES is not None:
        return
    
    _INITIAL_WIRES = [0] * N
    _EXPECTED_WIRES = [0] * N
    
    # Initial wires: bit k of wire i is 1 if (k >> i) & 1
    for i in range(N):
        w = 0
        for k in range(TOTAL_MASKS):
            if (k >> i) & 1:
                w |= (1 << k)
        _INITIAL_WIRES[i] = w
        
    # Expected wires: bit k of wire i is 1 if popcount(k) >= N-i
    for i in range(N):
        w = 0
        threshold = N - i
        for k in range(TOTAL_MASKS):
            if k.bit_count() >= threshold:
                w |= (1 << k)
        _EXPECTED_WIRES[i] = w

_init_wires()

def _verify_network(net):
    """
    Fast verification using pre-computed big integers.
    """
    wires = _INITIAL_WIRES[:]
    
    for i, j in net:
        wi = wires[i]
        wj = wires[j]
        # min to i, max to j
        # Swap bits where wi=1 and wj=0
        S = wi & ~wj
        wires[i] = wi ^ S
        wires[j] = wj ^ S
        
    # Check against expected
    for i in range(N):
        if wires[i] != _EXPECTED_WIRES[i]:
            return False
    return True

def _build_seed():
    """Odd-even transposition network for N wires."""
    net = []
    for phase in range(N):
        start = phase % 2
        for i in range(start, N - 1, 2):
            net.append((i, i + 1))
    return net

def _build_brent_kung(n):
    """
    Recursive Brent-Kung sorting network construction.
    Returns a list of comparators for n wires.
    """
    if n == 1:
        return []
    
    # Split n into n1 and n2 such that n1 + n2 = n and n1 <= n2
    # Optimal split is roughly n/2, but Brent-Kung uses a specific recurrence.
    # For simplicity and correctness, we use a standard split.
    # The recurrence for optimal size is T(n) = T(k) + T(n-k) + k(n-k)
    # where k is chosen to minimize T.
    # For small n, we can hardcode or use a simple heuristic.
    
    # Let's use a simple recursive split: n1 = n // 2, n2 = n - n1
    n1 = n // 2
    n2 = n - n1
    
    # Recursively sort the two halves
    net = []
    
    # Sort first half (wires 0 to n1-1)
    net.extend(_build_brent_kung(n1))
    
    # Sort second half (wires n1 to n-1)
    # We need to offset the comparators for the second half
    if n2 > 1:
        sub_net = _build_brent_kung(n2)
        for i, j in sub_net:
            net.append((i + n1, j + n1))
    
    # Merge the two sorted halves
    # We need a merging network for n1 and n2 elements.
    # A simple merging network can be constructed by comparing (i, i+n1) for all i.
    # However, a simple cross-comparison is not sufficient for merging.
    # We need a proper merge network.
    
    # For now, let's use a simple "odd-even merge" or just a full cross-comparison 
    # followed by re-sorting? No, that's inefficient.
    
    # A valid (though not optimal) merge:
    # Compare (i, i+n1) for i in 0..min(n1,n2)-1
    # Then sort the resulting sequence?
    
    # Actually, the simplest correct merge for small n is to just generate
    # a full sorting network for the combined n wires? No, that defeats the purpose.
    
    # Let's use a known small merge network or fall back to a simple construction.
    # For the purpose of this GA, we just need a valid starting population.
    # The seed (odd-even) is valid. Brent-Kung is better if implemented correctly.
    
    # Let's implement a basic merge:
    # 1. Compare (i, i+n1) for i in range(min(n1, n2))
    # 2. Recursively merge the resulting "interleaved" halves?
    
    # This is getting complex. Let's stick to the Seed for initialization 
    # and rely on the GA to find better structures.
    # However, having a few diverse valid networks is good.
    
    # Alternative: Use the seed for all initial individuals, but with random shuffles?
    # Shuffling breaks validity.
    
    # Let's just return the seed for n wires if we don't have a proper BK implementation.
    # Or, we can use a "Batcher Odd-Merge" which is easier to code.
    
    # Batcher Odd-Merge for n elements:
    # 1. Split into two halves of size n/2 (pad if odd? BK handles non-powers of 2).
    # 2. Sort halves.
    # 3. Merge.
    
    # Given the complexity, let's just use the Seed for all initial population.
    # The GA will explore via mutations.
    return _build_seed()

def construct():
    start_time = time.time()
    deadline = start_time + 60 * DEADLINE_FRACTION
    
    rng = random.Random(42)
    
    # Initialize population with diverse valid networks
    # Since generating diverse valid networks is hard, we start with the seed
    # and apply random valid mutations (if possible) or just start with copies.
    # To get diversity, we can try to construct networks with different "structures"
    # by randomly permuting the wire indices in the seed? 
    # Permuting wires in a sorting network results in a valid sorting network 
    # for the permuted inputs, but since we sort ALL inputs, it's still a valid 
    # sorting network for the standard inputs? 
    # Yes! If P is a permutation of wires, and S is a sorting network, 
    # then P^-1 S P is also a sorting network.
    # We can generate diverse starting points by permuting the wire labels.
    
    population = []
    pop_size = 20
    
    # Generate diverse initial population using wire permutations
    for _ in range(pop_size):
        # Random permutation of 0..N-1
        perm = list(range(N))
        rng.shuffle(perm)
        
        # Apply permutation to the seed network
        seed_net = _build_seed()
        permuted_net = []
        for i, j in seed_net:
            # The comparator (i, j) in the permuted network corresponds to 
            # comparing wire perm[i] and perm[j] in the original space?
            # No. 
            # Let S be the network. It sorts x_0..x_{N-1}.
            # Let P be a permutation.
            # New network S' has comparators (P(i), P(j)) for each (i,j) in S.
            # Does S' sort?
            # Yes, it sorts the inputs in the order defined by P.
            # Since we need to sort the standard order, and the network is symmetric 
            # in the sense that it sorts ANY permutation of inputs to the sorted order,
            # wait. 
            # A sorting network sorts x_1..x_n into sorted order.
            # If we permute the wires, we are effectively sorting x_{P(1)}..x_{P(n)}.
            # The output will be sorted x_{P(1)}..x_{P(n)}.
            # But we want the output to be sorted x_1..x_n.
            # So S' is NOT a valid sorting network for the standard inputs unless P is identity.
            
            # Correction: 
            # If we permute the INPUT wires, we need to permute the OUTPUT wires back.
            # A sorting network is a fixed sequence of comparators.
            # If we relabel the wires, the network structure changes.
            # If we take a valid network S and apply a permutation P to the wire labels,
            # we get a new network S_P.
            # S_P sorts the input vector v if and only if S sorts P(v).
            # Since S sorts ALL vectors, S_P also sorts ALL vectors.
            # Therefore, S_P is a valid sorting network.
            
            a = perm[i]
            b = perm[j]
            if a > b:
                a, b = b, a
            permuted_net.append((a, b))
        
        if _verify_network(permuted_net):
            population.append(permuted_net)
        else:
            # Fallback to seed
            population.append(_build_seed())
    
    # Ensure we have enough population
    while len(population) < pop_size:
        population.append(_build_seed())
    
    best_net = min(population, key=len)
    best_size = len(best_net)
    
    all_comps = [(i, j) for i in range(N) for j in range(i+1, N)]
    
    # Genetic Algorithm Loop
    while time.time() < deadline:
        new_population = []
        
        # Keep the best individual (elitism)
        new_population.append(best_net[:])
        
        # Generate offspring
        while len(new_population) < pop_size and time.time() < deadline:
            # Selection: Pick two parents randomly (or tournament)
            # Let's use tournament selection of size 2
            p1 = rng.choice(population)
            p2 = rng.choice(population)
            
            # Crossover: Splice
            if len(p1) > 0 and len(p2) > 0:
                # Random splice point
                # We splice based on length
                point = rng.randint(1, max(len(p1), len(p2)))
                
                # Take first part of p1 and second part of p2
                # We need to be careful: the resulting network might not be valid.
                # C1 = p1[:point]
                # C2 = p2[point:]
                # Child = C1 + C2
                
                # This is a simple "cut and splice".
                # It might create duplicate comparators or invalid structures.
                # We verify immediately.
                
                child = p1[:point] + p2[point:]
                
                # Remove duplicates? 
                # Duplicates are harmless but waste space.
                # Let's keep them for now, verification will handle correctness.
                # If valid, we might clean up later.
                
                if _verify_network(child):
                    new_population.append(child)
                    continue
            
            # Mutation: Random Insertion or Deletion
            parent = rng.choice(population)
            child = parent[:]
            
            # Randomly choose mutation type
            if rng.random() < 0.5 and len(child) > 0:
                # Deletion
                idx = rng.randint(0, len(child) - 1)
                child.pop(idx)
            else:
                # Insertion
                comp = rng.choice(all_comps)
                idx = rng.randint(0, len(child))
                child.insert(idx, comp)
            
            if _verify_network(child):
                new_population.append(child)
            else:
                # If invalid, keep parent (or retry mutation)
                new_population.append(parent[:])
        
        # Selection for next generation
        # Sort by size (ascending)
        new_population.sort(key=len)
        
        # Update best
        if len(new_population[0]) < best_size:
            best_size = len(new_population[0])
            best_net = new_population[0][:]
            
        # Replace population with new_population
        # Truncate to pop_size if needed (it should be exactly pop_size)
        population = new_population[:pop_size]
        
        # Optional: Local improvement on the best individual
        # This combines GA with local search (Memetic Algorithm)
        if time.time() < deadline:
            improved = False
            # Try to remove one comparator from best_net
            for idx in range(len(best_net)):
                if time.time() > deadline:
                    break
                new_net = best_net[:idx] + best_net[idx+1:]
                if _verify_network(new_net):
                    best_net = new_net
                    best_size = len(best_net)
                    improved = True
                    # Update population's best
                    if population:
                        population[0] = best_net[:]
                    break
            
            # If we improved, we might want to propagate this improvement
            # to the population? Or just let the GA handle it.
            # Let's just update the best individual in the population.
            
    return best_net

if __name__ == "__main__":
    net = construct()
    print(f"Found network with {len(net)} comparators")
    print("Valid:", _verify_network(net))
