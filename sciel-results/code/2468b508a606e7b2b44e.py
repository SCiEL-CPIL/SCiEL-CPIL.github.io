"""Search for a 12-state binary synchronizing automaton with a large reset threshold."""
import time
import random

N = 12
DEADLINE = 55.0  # safety margin before the 60s kill

def construct():
    size = 1 << N
    full = size - 1
    a = [0] * size
    b = [0] * size
    
    # Precompute index of lowest set bit for speed? 
    # lbit.bit_length() - 1 is fast enough in Python 3.
    
    def build(fa, fb):
        # Reuse a and b arrays
        # a[0] and b[0] are always 0
        for s in range(1, size):
            lbit = s & -s
            idx = lbit.bit_length() - 1
            prev = s ^ lbit
            a[s] = a[prev] | (1 << fa[idx])
            b[s] = b[prev] | (1 << fb[idx])

    def threshold():
        # Fast BFS using list as queue
        if a[full] == full and b[full] == full:
            return None
        
        dist = [-1] * size
        dist[full] = 0
        queue = [full]
        head = 0
        while head < len(queue):
            s = queue[head]
            head += 1
            d = dist[s]
            
            n1 = a[s]
            if n1 == 1:
                return d + 1
            if dist[n1] == -1:
                dist[n1] = d + 1
                queue.append(n1)
                
            n2 = b[s]
            if n2 == 1:
                return d + 1
            if dist[n2] == -1:
                dist[n2] = d + 1
                queue.append(n2)
        return None

    rng = random.Random(20240517)
    
    # Initialize population with random "pinwheel-like" structures
    # One letter is a permutation (cycle), the other is random
    population = []
    best_score = 0
    best_fa = list(range(N))
    best_fb = [0] * N
    
    def evaluate_candidate(fa, fb):
        build(fa, fb)
        t = threshold()
        return t if t is not None else 0

    # Seed population
    for _ in range(50):
        fa = list(range(N))
        rng.shuffle(fa)
        fb = [rng.randrange(N) for _ in range(N)]
        score = evaluate_candidate(fa, fb)
        if score > best_score:
            best_score = score
            best_fa = fa[:]
            best_fb = fb[:]
        population.append((score, fa[:], fb[:]))
        
    # Keep top K
    K = 20
    population.sort(key=lambda x: x[0], reverse=True)
    population = population[:K]
    
    deadline = time.time() + DEADLINE
    
    # Local search / Genetic Algorithm
    while time.time() < deadline:
        # Pick a parent
        parent_score, p_fa, p_fb = population[rng.randrange(min(K, len(population)))]
        
        # Mutation: change one state's destination for one letter
        child_fa = p_fa[:]
        child_fb = p_fb[:]
        if rng.random() < 0.5:
            state = rng.randrange(N)
            child_fa[state] = rng.randrange(N)
        else:
            state = rng.randrange(N)
            child_fb[state] = rng.randrange(N)
            
        # Occasionally, do a larger mutation (swap two destinations)
        if rng.random() < 0.1:
            s1, s2 = rng.sample(range(N), 2)
            if rng.random() < 0.5:
                child_fa[s1], child_fa[s2] = child_fa[s2], child_fa[s1]
            else:
                child_fb[s1], child_fb[s2] = child_fb[s2], child_fb[s1]

        score = evaluate_candidate(child_fa, child_fb)
        
        if score > best_score:
            best_score = score
            best_fa = child_fa[:]
            best_fb = child_fb[:]
            
        # Update population
        population.append((score, child_fa, child_fb))
        population.sort(key=lambda x: x[0], reverse=True)
        population = population[:K]
        
        # If we have a very good score, we can stop early to save time?
        # But we want to maximize. Let's run until deadline.
        
    return best_fa, best_fb
