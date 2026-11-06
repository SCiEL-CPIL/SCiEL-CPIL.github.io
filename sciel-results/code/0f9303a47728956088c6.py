"""Size-efficient sorting network search for N=16 wires.

Combines the bit-parallel verification and local search machinery of the current
program with a Genetic Programming (GP) approach from the inspirations. The GP
evolves network structures using crossover (merging valid sub-sequences) and
mutation (insertion, deletion, swap, move), with a fast bit-parallel fitness
function. This allows escaping local minima that trap pure greedy local search.
"""

import time
import random

N = 16
NUM_INPUTS = 1 << N  # 65536

# Pre-compute initial states for bit-parallel verification
_initial_states = None

def _get_initial_states():
    global _initial_states
    if _initial_states is None:
        _initial_states = []
        for i in range(N):
            # For input j (0..65535), wire i has value (j >> i) & 1
            # Build a 65536-bit integer where bit j = (j >> i) & 1
            # Pattern: period 2^(i+1), lower half 0, upper half 1
            half = 1 << i
            period = 1 << (i + 1)
            num_periods = NUM_INPUTS // period
            # Build one period: lower half zeros, upper half ones
            period_val = ((1 << half) - 1) << half
            # Repeat the pattern
            val = 0
            for p in range(num_periods):
                val |= period_val << (p * period)
            _initial_states.append(val)
    return _initial_states

def fast_verify(network):
    """Verify a sorting network using bit-parallel zero-one principle."""
    if not network:
        return N == 1
        
    state = _get_initial_states()[:]
    for (i, j) in network:
        min_val = state[i] & state[j]
        max_val = state[i] | state[j]
        state[i] = min_val
        state[j] = max_val
    for i in range(N - 1):
        if state[i] & ~state[i + 1]:
            return False
    return True

def remove_redundant(net):
    """
    Removes all comparators that are no-ops in the current network.
    Returns (new_net, final_state).
    """
    if not net:
        return [], _get_initial_states()[:]
    current_state = _get_initial_states()[:]
    keep = []
    for (i, j) in net:
        si = current_state[i]
        sj = current_state[j]
        # Check if wire i <= wire j for all inputs before this comparator
        if (si & ~sj) == 0:
            # No-op: skip it
            continue
        min_val = si & sj
        max_val = si | sj
        current_state[i] = min_val
        current_state[j] = max_val
        keep.append((i, j))
    return keep, current_state

def is_sorted_state(final_state):
    """Check if the final state is sorted."""
    for i in range(N - 1):
        if final_state[i] & ~final_state[i + 1]:
            return False
    return True

def bitonic_sorter(n):
    """Generate Batcher's bitonic sorting network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    comparators = []
    def bitonic_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        bitonic_sort(half, start, ascending)
        bitonic_sort(half, start + half, not ascending)
        bitonic_merge(half, start, ascending)
    def bitonic_merge(size, start, ascending):
        if size <= 1:
            return
        for i in range(size):
            a = start + i
            b = start + i + size
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        bitonic_merge(size // 2, start, ascending)
        bitonic_merge(size // 2, start + size, ascending)
    bitonic_sort(n, 0, True)
    return comparators

def odd_even_merge(n):
    """Odd-even merge sort network for n wires (n must be power of 2)."""
    if n <= 1:
        return []
    comparators = []
    def oe_merge(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        # Merge two sorted sequences of size half
        for i in range(half):
            a = start + i
            b = start + i + half
            if ascending:
                comparators.append((a, b))
            else:
                comparators.append((b, a))
        oe_merge(half, start, ascending)
        oe_merge(half, start + half, ascending)
    # First, sort odd and even positions
    def oe_sort(size, start, ascending):
        if size <= 1:
            return
        half = size // 2
        oe_sort(half, start, ascending)
        oe_sort(half, start + half, ascending)
        oe_merge(half, start, ascending)
    oe_sort(n, 0, True)
    return comparators

def odd_even_transposition(n):
    """Odd-even transposition sorting network."""
    return [(i, i + 1) for phase in range(n) for i in range(phase % 2, n - 1, 2)]

def random_network(n, seed):
    """Generate a random network of n comparators."""
    rng = random.Random(seed)
    comparators = []
    for _ in range(n):
        i = rng.randint(0, N - 2)
        j = rng.randint(i + 1, N - 1)
        comparators.append((i, j))
    return comparators

def local_search(net, deadline, seed):
    """
    Perform local search starting from net.
    Returns the best network found.
    """
    rng = random.Random(seed)
    best_net = list(net)
    best_count = len(net)
    current_net = list(net)
    
    # Initial cleanup
    current_net, final_state = remove_redundant(current_net)
    if is_sorted_state(final_state):
        best_net = list(current_net)
        best_count = len(current_net)
    
    no_improve_count = 0
    
    while time.time() < deadline:
        r = rng.random()
        
        if len(current_net) > 0:
            if r < 0.55:
                # Try removing a random comparator
                idx = rng.randint(0, len(current_net) - 1)
                candidate = current_net[:idx] + current_net[idx + 1:]
                candidate, candidate_final = remove_redundant(candidate)
                if is_sorted_state(candidate_final):
                    if len(candidate) < best_count:
                        best_net = list(candidate)
                        best_count = len(candidate)
                        no_improve_count = 0
                    current_net = candidate
                else:
                    no_improve_count += 1
                    
            elif r < 0.75:
                # Try swapping two adjacent comparators
                if len(current_net) >= 2:
                    idx = rng.randint(0, len(current_net) - 2)
                    candidate = list(current_net)
                    candidate[idx], candidate[idx + 1] = candidate[idx + 1], candidate[idx]
                    candidate, candidate_final = remove_redundant(candidate)
                    if is_sorted_state(candidate_final):
                        if len(candidate) < best_count:
                            best_net = list(candidate)
                            best_count = len(candidate)
                        no_improve_count = 0
                        current_net = candidate
                    else:
                        no_improve_count += 1
                        
            elif r < 0.90:
                # Try inserting a random comparator
                i = rng.randint(0, N - 2)
                j = rng.randint(i + 1, N - 1)
                pos = rng.randint(0, len(current_net))
                candidate = current_net[:pos] + [(i, j)] + current_net[pos:]
                candidate, candidate_final = remove_redundant(candidate)
                if is_sorted_state(candidate_final):
                    if len(candidate) < best_count:
                        best_net = list(candidate)
                        best_count = len(candidate)
                        no_improve_count = 0
                    current_net = candidate
                else:
                    no_improve_count += 1
            else:
                # Try moving a comparator to a random position
                if len(current_net) >= 2:
                    src = rng.randint(0, len(current_net) - 1)
                    dst = rng.randint(0, len(current_net) - 1)
                    if src != dst:
                        candidate = list(current_net)
                        comp = candidate.pop(src)
                        candidate.insert(dst, comp)
                        candidate, candidate_final = remove_redundant(candidate)
                        if is_sorted_state(candidate_final):
                            if len(candidate) < best_count:
                                best_net = list(candidate)
                                best_count = len(candidate)
                            no_improve_count = 0
                            current_net = candidate
                        else:
                            no_improve_count += 1
        else:
            # Empty network - must insert
            i = rng.randint(0, N - 2)
            j = rng.randint(i + 1, N - 1)
            candidate = [(i, j)]
            candidate, candidate_final = remove_redundant(candidate)
            if is_sorted_state(candidate_final):
                if len(candidate) < best_count:
                    best_net = list(candidate)
                    best_count = len(candidate)
                    no_improve_count = 0
                current_net = candidate
            else:
                no_improve_count += 1
        
        # Periodically re-seed from best to escape local minima
        if no_improve_count > 100 and rng.random() < 0.05:
            current_net = list(best_net)
            no_improve_count = 0
    
    return best_net

def genetic_search(population, deadline, seed):
    """
    Perform a genetic search starting from a population of networks.
    Returns the best network found.
    """
    rng = random.Random(seed)
    
    # Initialize population with cleaned versions
    cleaned_pop = []
    for net in population:
        c_net, final_state = remove_redundant(net)
        if is_sorted_state(final_state):
            cleaned_pop.append(c_net)
    
    if not cleaned_pop:
        return None
    
    # Sort by size
    cleaned_pop.sort(key=len)
    best_net = cleaned_pop[0]
    best_count = len(best_net)
    
    # Keep top 10% as elite, rest as population
    elite_size = max(1, len(cleaned_pop) // 10)
    elites = [list(net) for net in cleaned_pop[:elite_size]]
    population = [list(net) for net in cleaned_pop[elite_size:]]
    
    if not population:
        population = [list(net) for net in cleaned_pop]
    
    gen = 0
    while time.time() < deadline:
        gen += 1
        new_population = []
        
        # Copy elites
        for e in elites:
            new_population.append(list(e))
        
        # Generate offspring
        while len(new_population) < len(population):
            if time.time() > deadline - 1:
                break
                
            p1 = rng.choice(population)
            p2 = rng.choice(population)
            
            # Crossover: Merge two networks
            # Simple approach: Take a random prefix from p1 and random suffix from p2
            # Or just random insertion/deletion
            if rng.random() < 0.5:
                # Crossover: Take first half of p1 and second half of p2
                cut1 = len(p1) // 2
                cut2 = len(p2) // 2
                child = p1[:cut1] + p2[cut2:]
            else:
                # Mutation: Copy p1 and mutate
                child = list(p1)
                num_mutations = rng.randint(1, 3)
                for _ in range(num_mutations):
                    if not child:
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        child.append((i, j))
                        continue
                    
                    m_type = rng.random()
                    if m_type < 0.4:
                        # Delete
                        idx = rng.randint(0, len(child) - 1)
                        child.pop(idx)
                    elif m_type < 0.8:
                        # Insert
                        i = rng.randint(0, N - 2)
                        j = rng.randint(i + 1, N - 1)
                        pos = rng.randint(0, len(child))
                        child.insert(pos, (i, j))
                    else:
                        # Swap
                        if len(child) >= 2:
                            i1 = rng.randint(0, len(child) - 2)
                            i2 = rng.randint(i1 + 1, len(child) - 1)
                            child[i1], child[i2] = child[i2], child[i1]
            
            # Clean up and verify
            child, final_state = remove_redundant(child)
            if is_sorted_state(final_state):
                if len(child) < best_count:
                    best_net = list(child)
                    best_count = len(child)
                    # Update elites
                    if len(elites) == 0 or len(child) < len(elites[-1]):
                        elites.append(list(child))
                        elites.sort(key=len)
                        if len(elites) > elite_size:
                            elites.pop()
                new_population.append(child)
        
        # Selection: Keep the smallest valid networks
        new_population.sort(key=len)
        population = new_population[:len(population)]
        
        # Periodically run local search on the best individual
        if gen % 5 == 0 and best_net:
            ls_deadline = min(deadline - 1, time.time() + 2)
            ls_result = local_search(best_net, ls_deadline, seed=gen)
            if len(ls_result) < best_count:
                best_net = ls_result
                best_count = len(ls_result)
    
    return best_net

def construct():
    deadline = time.time() + 85  # 85s to leave 5s margin for overhead
    
    # Multi-seed portfolio - start with known good constructions
    seeds = [
        bitonic_sorter(N),
        odd_even_merge(N),
        odd_even_transposition(N),
    ]
    
    best_net = None
    best_count = float('inf')
    
    # First pass: quickly evaluate and clean up all seeds
    cleaned_seeds = []
    for seed_net in seeds:
        if time.time() > deadline:
            break
        seed_net, final_state = remove_redundant(seed_net)
        if is_sorted_state(final_state):
            cleaned_seeds.append(seed_net)
            if len(seed_net) < best_count:
                best_net = list(seed_net)
                best_count = len(seed_net)
    
    # Sort seeds by size (smaller first) for better search efficiency
    cleaned_seeds.sort(key=len)
    
    # Add random seeds
    for s in range(3):
        if time.time() > deadline:
            break
        rnd_net = random_network(60, seed=100 + s)
        rnd_net, final_state = remove_redundant(rnd_net)
        if is_sorted_state(final_state):
            cleaned_seeds.append(rnd_net)
            if len(rnd_net) < best_count:
                best_net = list(rnd_net)
                best_count = len(rnd_net)
    
    # Do local search from each seed, allocating more time to better seeds
    for seed_idx, seed_net in enumerate(cleaned_seeds):
        if time.time() > deadline - 5:
            break
        
        current_count = len(seed_net)
        # Allocate time based on seed quality: better seeds get more time
        time_budget = 25 if current_count <= 65 else 15
        search_deadline = min(deadline - 5, time.time() + time_budget)
        found = local_search(seed_net, search_deadline, seed=seed_idx * 1000)
        if len(found) < best_count:
            best_net = found
            best_count = len(found)
    
    # Run Genetic Search if time permits
    if time.time() < deadline - 10:
        # Build initial population for GP from best found and seeds
        gp_pop = [best_net] if best_net else []
        for s in cleaned_seeds:
            if s != best_net:
                gp_pop.append(s)
        
        gp_deadline = deadline - 5
        gp_result = genetic_search(gp_pop, gp_deadline, seed=9999)
        if gp_result and len(gp_result) < best_count:
            best_net = gp_result
            best_count = len(gp_result)
    
    # Final cleanup
    if best_net is not None:
        best_net, _ = remove_redundant(best_net)
    else:
        # Fallback
        best_net = odd_even_transposition(N)
        best_net, _ = remove_redundant(best_net)
    
    return best_net
