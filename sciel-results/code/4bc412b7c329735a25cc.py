import time
import random
import math
from collections import deque

N = 12
FULL_MASK = (1 << N) - 1
DEADLINE_SECONDS = 82.0  # safety margin before 90s kill


def _build_subset_trans(trans):
    size = 1 << N
    next_subset = [0] * size
    for mask in range(1, size):
        lsb = mask & (-mask)
        i = lsb.bit_length() - 1
        rest = mask ^ lsb
        next_subset[mask] = next_subset[rest] | (1 << trans[i])
    return next_subset


def compute_threshold(trans_a, trans_b):
    """BFS on subset automaton; returns exact shortest reset word length, 0 if not synchronizing."""
    next_a = _build_subset_trans(trans_a)
    next_b = _build_subset_trans(trans_b)
    size = 1 << N
    dist = [-1] * size
    dist[FULL_MASK] = 0
    queue = deque([FULL_MASK])
    while queue:
        cur = queue.popleft()
        if cur & (cur - 1) == 0:
            return dist[cur]
        nxt_a = next_a[cur]
        if dist[nxt_a] == -1:
            dist[nxt_a] = dist[cur] + 1
            queue.append(nxt_a)
        nxt_b = next_b[cur]
        if dist[nxt_b] == -1:
            dist[nxt_b] = dist[cur] + 1
            queue.append(nxt_b)
    return 0


def _pinwheel_a(n):
    """Standard pinwheel: a[i] = (i+1) % n"""
    return [(i + 1) % n for i in range(n)]


def _pinwheel_b(n, k):
    """Pinwheel b: b[i] = i for i < k, b[k] = 0, b[i] = 0 for i > k"""
    b = [0] * n
    for i in range(k):
        b[i] = i
    return b


def _crossover(a1, b1, a2, b2, rng):
    """Uniform crossover: for each state, pick transition from parent 1 or 2."""
    a = [0] * N
    b = [0] * N
    for i in range(N):
        if rng.random() < 0.5:
            a[i] = a1[i]
            b[i] = b1[i]
        else:
            a[i] = a2[i]
            b[i] = b2[i]
    return a, b


def _mutation(a, b, rng, rate=0.15):
    """Random mutation: flip individual state transitions."""
    a = a[:]
    b = b[:]
    for i in range(N):
        if rng.random() < rate:
            a[i] = rng.randint(0, N - 1)
        if rng.random() < rate:
            b[i] = rng.randint(0, N - 1)
    return a, b


def _structural_mutation(a, b, rng):
    """Structure-aware mutation preserving cycle/compression properties."""
    a = a[:]
    b = b[:]
    mtype = rng.random()
    
    if mtype < 0.25:
        # Modify the cycle in 'a': pick one state and redirect
        state = rng.randint(0, N - 1)
        new_dest = rng.randint(0, N - 1)
        if new_dest != a[state]:
            a[state] = new_dest
    elif mtype < 0.50:
        # Modify compression in 'b': adjust which states map to themselves
        state = rng.randint(0, N - 1)
        if rng.random() < 0.5:
            b[state] = state  # make it fixed
        else:
            b[state] = rng.randint(0, min(3, N - 1))  # compress to low state
    elif mtype < 0.75:
        # Swap destinations of two states in one transition
        i, j = rng.sample(range(N), 2)
        if rng.random() < 0.5:
            a[i], a[j] = a[j], a[i]
        else:
            b[i], b[j] = b[j], b[i]
    else:
        # Random single edit
        if rng.random() < 0.5:
            state = rng.randint(0, N - 1)
            a[state] = rng.randint(0, N - 1)
        else:
            state = rng.randint(0, N - 1)
            b[state] = rng.randint(0, N - 1)
    
    return a, b


def _genetic_algorithm(deadline, rng):
    """
    Genetic algorithm with tournament selection, crossover, and structure-aware mutation.
    Maintains a population of candidate automata.
    """
    # Initialize population with diverse pinwheel variants and random starts
    population = []
    
    # Add pinwheel variants
    for k in range(3, N):
        a = _pinwheel_a(N)
        b = _pinwheel_b(N, k)
        population.append((a, b))
    
    # Add some random starting points
    for _ in range(5):
        a = [rng.randint(0, N - 1) for _ in range(N)]
        b = [rng.randint(0, N - 1) for _ in range(N)]
        population.append((a, b))
    
    # Evaluate initial population
    scored_pop = []
    for a, b in population:
        score = compute_threshold(a, b)
        if score > 0:
            scored_pop.append((score, a, b))
    
    if not scored_pop:
        return None, None, 0
    
    scored_pop.sort(key=lambda x: -x[0])
    best_score = scored_pop[0][0]
    best_a = scored_pop[0][1][:]
    best_b = scored_pop[0][2][:]
    
    pop_size = 20
    gen = 0
    
    while time.time() < deadline:
        gen += 1
        
        # Tournament selection
        def tournament_select():
            candidates = rng.sample(scored_pop, min(3, len(scored_pop)))
            return max(candidates, key=lambda x: x[0])
        
        # Create new generation
        new_pop = []
        
        # Elitism: keep top 2
        for i in range(min(2, len(scored_pop))):
            s, a, b = scored_pop[i]
            new_pop.append((s, a[:], b[:]))
        
        # Fill rest with offspring
        while len(new_pop) < pop_size and time.time() < deadline:
            # Select parents
            p1 = tournament_select()
            p2 = tournament_select()
            
            # Crossover
            child_a, child_b = _crossover(p1[1], p1[2], p2[1], p2[2], rng)
            
            # Mutation (mix of random and structural)
            if rng.random() < 0.5:
                child_a, child_b = _mutation(child_a, child_b, rng, rate=0.1)
            else:
                child_a, child_b = _structural_mutation(child_a, child_b, rng)
            
            # Evaluate
            score = compute_threshold(child_a, child_b)
            if score > 0:
                new_pop.append((score, child_a, child_b))
        
        # Update scored population
        scored_pop = new_pop
        scored_pop.sort(key=lambda x: -x[0])
        
        # Track best
        if scored_pop[0][0] > best_score:
            best_score = scored_pop[0][0]
            best_a = scored_pop[0][1][:]
            best_b = scored_pop[0][2][:]
        
        # Check for stagnation: if no improvement in last 5 generations, inject diversity
        if gen % 5 == 0 and scored_pop[0][0] == best_score:
            # Inject random individuals
            for _ in range(3):
                a = [rng.randint(0, N - 1) for _ in range(N)]
                b = [rng.randint(0, N - 1) for _ in range(N)]
                score = compute_threshold(a, b)
                if score > 0:
                    scored_pop.append((score, a, b))
            scored_pop.sort(key=lambda x: -x[0])
            scored_pop = scored_pop[:pop_size]
    
    return best_a, best_b, best_score


def _hill_climb(a, b, rng, deadline):
    """
    Local hill climbing from a starting point, accepting only improvements.
    Uses structural mutations for better exploration.
    """
    cur_a = a[:]
    cur_b = b[:]
    cur_score = compute_threshold(cur_a, cur_b)
    if cur_score == 0:
        return None, None, 0
    
    best_a = cur_a[:]
    best_b = cur_b[:]
    best_score = cur_score
    
    no_improve = 0
    max_no_improve = 50
    
    while time.time() < deadline and no_improve < max_no_improve:
        # Try structural mutation
        new_a, new_b = _structural_mutation(cur_a, cur_b, rng)
        new_score = compute_threshold(new_a, new_b)
        
        if new_score > cur_score:
            cur_a, cur_b = new_a, new_b
            cur_score = new_score
            no_improve = 0
            if cur_score > best_score:
                best_score = cur_score
                best_a = cur_a[:]
                best_b = cur_b[:]
        else:
            no_improve += 1
    
    return best_a, best_b, best_score


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_SECONDS
    rng = random.Random(42)
    
    best_holder = [0, None, None]  # [score, a, b]
    
    # Phase 1: Quick analytic scan of pinwheel family
    n = N
    best_family_score = 0
    best_family_a = None
    best_family_b = None
    
    for k in range(2, n):
        a = _pinwheel_a(n)
        b = _pinwheel_b(n, k)
        score = compute_threshold(a, b)
        if score > best_family_score:
            best_family_score = score
            best_family_a = a[:]
            best_family_b = b[:]
    
    if best_family_a is not None:
        best_holder[0] = best_family_score
        best_holder[1] = best_family_a[:]
        best_holder[2] = best_family_b[:]
    
    # Phase 2: Genetic algorithm (main search)
    if time.time() < deadline - 10.0:
        ga_deadline = deadline - 10.0
        ga_a, ga_b, ga_score = _genetic_algorithm(ga_deadline, rng)
        if ga_score > best_holder[0]:
            best_holder[0] = ga_score
            best_holder[1] = ga_a[:]
            best_holder[2] = ga_b[:]
    
    # Phase 3: Hill climbing from best found
    if best_holder[1] is not None and time.time() < deadline - 3.0:
        hc_deadline = deadline - 3.0
        hc_a, hc_b, hc_score = _hill_climb(best_holder[1], best_holder[2], rng, hc_deadline)
        if hc_score > best_holder[0]:
            best_holder[0] = hc_score
            best_holder[1] = hc_a[:]
            best_holder[2] = hc_b[:]
    
    # Fallback: guaranteed synchronizing automaton
    if best_holder[1] is None or best_holder[0] == 0:
        a = _pinwheel_a(N)
        b = [0] * N
        return [a, b]
    
    return [best_holder[1], best_holder[2]]


if __name__ == "__main__":
    a, b = construct()
    print("Best score:", compute_threshold(a, b))
    print("A:", a)
    print("B:", b)
