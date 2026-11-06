import time
import random
import numpy as np

def construct():
    n = 256
    # Deadline guard: 110s to be safe against the 120s/150s limits
    deadline = time.time() + 110
    
    # 1. Precompute all 4-term APs
    # a, a+d, a+2d, a+3d <= 256
    # Indices are 0-based: a-1, a+d-1, etc.
    aps = []
    for d in range(1, n // 3 + 1):
        for a in range(1, n - 3 * d + 1):
            p0 = a - 1
            p1 = a + d - 1
            p2 = a + 2 * d - 1
            p3 = a + 3 * d - 1
            aps.append((p0, p1, p2, p3))
    
    num_aps = len(aps)
    aps_arr = np.array(aps, dtype=np.int32)
    
    # Map position to list of (ap_index, offset_in_ap)
    pos_aps = [[] for _ in range(n)]
    for idx in range(num_aps):
        p0, p1, p2, p3 = aps[idx]
        pos_aps[p0].append((idx, 0))
        pos_aps[p1].append((idx, 1))
        pos_aps[p2].append((idx, 2))
        pos_aps[p3].append((idx, 3))

    def count_mono_numpy(color):
        """Fast count using NumPy."""
        c = np.array(color, dtype=np.int8)
        c0 = c[aps_arr[:, 0]]
        c1 = c[aps_arr[:, 1]]
        c2 = c[aps_arr[:, 2]]
        c3 = c[aps_arr[:, 3]]
        return int(np.sum((c0 == c1) & (c1 == c2) & (c2 == c3)))

    def init_state(color):
        """Initialize mono status and count for local search."""
        mono = [False] * num_aps
        count = 0
        for idx, (p0, p1, p2, p3) in enumerate(aps):
            if color[p0] == color[p1] == color[p2] == color[p3]:
                mono[idx] = True
                count += 1
        return mono, count

    def delta_flip(color, mono, p):
        """Calculate the change in mono count if we flip position p."""
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            
            if mono[ap_idx]:
                # Currently mono. Flipping p breaks it.
                delta -= 1
            else:
                # Currently non-mono. Check if flipping p makes it mono.
                if offset == 0:
                    if color[p1] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 1:
                    if color[p0] == new_val and color[p2] == new_val and color[p3] == new_val:
                        delta += 1
                elif offset == 2:
                    if color[p0] == new_val and color[p1] == new_val and color[p3] == new_val:
                        delta += 1
                else: # offset == 3
                    if color[p0] == new_val and color[p1] == new_val and color[p2] == new_val:
                        delta += 1
        return delta

    def apply_flip(color, mono, p):
        """Flip position p, update mono status. Returns delta."""
        delta = 0
        new_val = 1 - color[p]
        
        for ap_idx, offset in pos_aps[p]:
            p0, p1, p2, p3 = aps[ap_idx]
            was_mono = mono[ap_idx]
            
            if offset == 0:
                is_mono = (color[p1] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 1:
                is_mono = (color[p0] == new_val and color[p2] == new_val and color[p3] == new_val)
            elif offset == 2:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p3] == new_val)
            else:
                is_mono = (color[p0] == new_val and color[p1] == new_val and color[p2] == new_val)
                
            mono[ap_idx] = is_mono
            
            if is_mono and not was_mono:
                delta += 1
            elif not is_mono and was_mono:
                delta -= 1
                
        color[p] = new_val
        return delta

    def local_search(color, max_iter=1000):
        """
        Local search with greedy moves and occasional SA escapes.
        Returns best color and count found.
        """
        mono, count = init_state(color)
        best_count = count
        best_color = color[:]
        
        T = 0.5
        cooling = 0.95
        
        for _ in range(max_iter):
            if time.time() > deadline - 2:
                break
            
            # Find best greedy move
            best_delta = 0
            best_pos = -1
            
            for p in range(n):
                d = delta_flip(color, mono, p)
                if d < best_delta:
                    best_delta = d
                    best_pos = p
                    
            if best_pos != -1:
                delta = apply_flip(color, mono, best_pos)
                count += delta
                if count < best_count:
                    best_count = count
                    best_color = color[:]
            else:
                # No improving move. Try SA escape
                if T < 0.01:
                    break
                
                p = random.randint(0, n - 1)
                d = delta_flip(color, mono, p)
                
                # Accept if d < 0 or with probability exp(-d/T)
                if d < 0:
                    delta = apply_flip(color, mono, p)
                    count += delta
                    if count < best_count:
                        best_count = count
                        best_color = color[:]
                else:
                    if random.random() < np.exp(-d / T):
                        delta = apply_flip(color, mono, p)
                        count += delta
                        if count < best_count:
                            best_count = count
                            best_color = color[:]
                
                T *= cooling

        return best_color, best_count

    # --- Genetic Algorithm / Memetic Algorithm ---
    
    pop_size = 60
    population = []
    
    # Initializers
    def thue_morse(n):
        return [bin(i).count('1') % 2 for i in range(n)]
        
    def random_init(n):
        return [random.randint(0, 1) for _ in range(n)]
        
    def rademacher_init(n):
        return [(i // 2) % 2 for i in range(n)]

    random.seed(42)
    
    # Initialize population with diversity
    init_strategies = [
        thue_morse,
        random_init,
        rademacher_init,
        lambda: [1 - x for x in thue_morse(n)],
        lambda: [1 - x for x in rademacher(n)] if False else [1 - x for x in rademacher_init(n)],
    ]
    
    # Add some randomized structured inits
    for i in range(pop_size):
        if i < len(init_strategies):
            c = init_strategies[i](n)
        else:
            c = random_init(n)
        # Apply some random noise to structured ones to ensure diversity
        if i > 0 and i < len(init_strategies):
            for j in range(random.randint(0, 10)):
                c[random.randint(0, n-1)] = 1 - c[random.randint(0, n-1)]
        population.append(c)
    
    # Evaluate initial population
    scores = [count_mono_numpy(c) for c in population]
    best_score = min(scores)
    best_color = population[scores.index(best_score)][:]
    
    # GA Loop
    generation = 0
    while time.time() < deadline - 5:
        generation += 1
        
        # Selection: Tournament
        def select_parent():
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            if scores[idx1] < scores[idx2]:
                return idx1
            else:
                return idx2
        
        # Create offspring
        new_population = []
        # Elitism: Keep the top 10%
        sorted_indices = sorted(range(pop_size), key=lambda i: scores[i])
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            new_population.append(population[sorted_indices[i]][:])
        
        # Fill rest with GA children
        while len(new_population) < pop_size:
            if time.time() > deadline - 5:
                break
                
            p1_idx = select_parent()
            p2_idx = select_parent()
            
            # Crossover: Uniform
            child = [random.choice([population[p1_idx][j], population[p2_idx][j]]) for j in range(n)]
            
            # Mutation
            mut_rate = 0.05
            for j in range(n):
                if random.random() < mut_rate:
                    child[j] = 1 - child[j]
            
            # Local Search on child
            # We limit local search iterations for GA children to keep pace
            # but enough to reach a local optimum
            improved_child, child_score = local_search(child, max_iter=500)
            
            new_population.append(improved_child)
            if child_score < best_score:
                best_score = child_score
                best_color = improved_child[:]
        
        # Replace population
        population = new_population
        scores = [count_mono_numpy(c) for c in population]
        
        # Update global best from population just in case
        current_pop_best = min(scores)
        if current_pop_best < best_score:
            best_score = current_pop_best
            best_idx = scores.index(current_pop_best)
            best_color = population[best_idx][:]

    return best_color

