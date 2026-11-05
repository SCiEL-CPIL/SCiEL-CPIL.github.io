"""Construct a 12-state binary deterministic synchronizing automaton with a long reset word."""
import time
import random
from collections import deque

N = 12
_DEADLINE = 85.0  # safety margin below the 90s wall-clock kill
_start = time.time()


def _elapsed():
    return time.time() - _start


def _build_transition_table(a, b):
    """Precompute transition for all 2^N subsets under letters a and b.
    Returns (trans_a, trans_b) where trans_a[mask] = result of applying a to subset mask."""
    trans_a = [0] * (1 << N)
    trans_b = [0] * (1 << N)
    for mask in range(1 << N):
        na = 0
        nb = 0
        for i in range(N):
            if mask & (1 << i):
                na |= (1 << a[i])
                nb |= (1 << b[i])
        trans_a[mask] = na
        trans_b[mask] = nb
    return trans_a, trans_b


def _threshold_fast(a, b):
    """BFS on subset automaton using precomputed transition tables.
    Returns -1 if not synchronizing."""
    start_mask = (1 << N) - 1
    if start_mask & (start_mask - 1) == 0:
        return 0
    
    trans_a, trans_b = _build_transition_table(a, b)
    
    visited = [-1] * (1 << N)
    visited[start_mask] = 0
    q = deque([start_mask])
    
    while q:
        cur = q.popleft()
        d = visited[cur]
        # Check if single state: power of 2
        if cur & (cur - 1) == 0:
            return d
        
        # Next state under 'a'
        next_a = trans_a[cur]
        if visited[next_a] == -1:
            visited[next_a] = d + 1
            q.append(next_a)
            
        # Next state under 'b'
        next_b = trans_b[cur]
        if visited[next_b] == -1:
            visited[next_b] = d + 1
            q.append(next_b)
            
    return -1


def _threshold(a, b):
    """BFS on subset automaton to find shortest reset word length.
    Returns -1 if not synchronizing."""
    return _threshold_fast(a, b)


def construct():
    """Return two length-N lists of destination states in 0..N-1.

    Must define a complete binary deterministic automaton that synchronizes.
    Score is the exact shortest reset-word length, maximized.
    """
    best_thresh = -1
    best_a = None
    best_b = None
    rng = random.Random(12345)

    # A known-good fallback (cycle + sink) guarantees synchronization.
    fallback_a = [(i + 1) % N for i in range(N)]
    fallback_b = [0] * N
    fb_thr = _threshold(fallback_a, fallback_b)
    if fb_thr > best_thresh:
        best_thresh = fb_thr
        best_a = fallback_a[:]
        best_b = fallback_b[:]

    # ---- Structured search over a = cycle, b = perturbed monotone ----
    a_cycle = [(i + 1) % N for i in range(N)]
    
    # Construction A: b = [0,1,2,...,10,10]
    b_mono = list(range(N-1)) + [N-2]
    thr = _threshold(a_cycle, b_mono)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_mono[:]

    # Construction B: a = cycle, b = identity except b(0) = 1
    b_id = list(range(N))
    b_id[0] = 1
    thr = _threshold(a_cycle, b_id)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_id[:]

    # Construction C: a = cycle, b = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_shift = [0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_shift)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_shift[:]

    # Construction D: Try all "one defect" perturbations of the monotone b
    base_b = list(range(N-1)) + [N-2]
    for i in range(N):
        for new_val in range(N):
            if new_val == base_b[i]:
                continue
            if _elapsed() > _DEADLINE:
                break
            b_test = base_b[:]
            b_test[i] = new_val
            thr = _threshold(a_cycle, b_test)
            if thr > best_thresh:
                best_thresh = thr
                best_a = a_cycle[:]
                best_b = b_test[:]

    # Construction E: Random search with a = cycle (increased trials)
    for trial in range(1000):
        if _elapsed() > _DEADLINE:
            break
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_cycle, b_rand)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_rand[:]

    # Construction F: Random permutations for a, random b (increased trials)
    for trial in range(500):
        if _elapsed() > _DEADLINE:
            break
        a_perm = list(range(N))
        rng.shuffle(a_perm)
        b_rand = [rng.randint(0, N-1) for _ in range(N)]
        thr = _threshold(a_perm, b_rand)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_perm[:]
            best_b = b_rand[:]

    # Construction G: a = cycle in reverse direction
    a_rev = [(i - 1) % N for i in range(N)]
    b_mono2 = list(range(N-1)) + [N-2]
    thr = _threshold(a_rev, b_mono2)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_rev[:]
        best_b = b_mono2[:]

    # Construction H: Try a "bottleneck" construction
    for sink in range(N):
        if _elapsed() > _DEADLINE:
            break
        b_sink = [(i + 1) % N for i in range(N)]
        b_sink[sink] = sink
        thr = _threshold(a_cycle, b_sink)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_sink[:]

    # Construction I: Two-step bottleneck
    for i in range(N):
        if _elapsed() > _DEADLINE:
            break
        b_two = [(i + 1) % N for i in range(N)]
        b_two[i] = (i + 2) % N
        thr = _threshold(a_cycle, b_two)
        if thr > best_thresh:
            best_thresh = thr
            best_a = a_cycle[:]
            best_b = b_two[:]

    # Construction J: a = cycle, b = [0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    b_three = [0, 0, 0] + list(range(1, N-1))
    thr = _threshold(a_cycle, b_three)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_three[:]

    # Construction K: a = cycle, b = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b_dup = [0, 1, 1] + list(range(2, N-1))
    thr = _threshold(a_cycle, b_dup)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_dup[:]

    # Construction L: a = cycle, b = "fold" pattern [0,1,2,...,5,5,4,3,2,1,0]
    b_fold = [i for i in range(N//2)] + [N//2-1-i for i in range(N//2)]
    if len(b_fold) < N:
        b_fold.extend([0] * (N - len(b_fold)))
    thr = _threshold(a_cycle, b_fold)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_fold[:]

    # Construction M: a = cycle, b = "two peaks" [0,1,2,3,4,5,5,4,3,2,1,0]
    b_twopeaks = [min(i, 11-i) for i in range(N)]
    thr = _threshold(a_cycle, b_twopeaks)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_twopeaks[:]

    # Construction N: a = cycle, b = "staircase with gap" [0,1,2,3,4,4,5,6,7,8,9,10]
    b_gap = [0,1,2,3,4,4,5,6,7,8,9,10]
    thr = _threshold(a_cycle, b_gap)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_gap[:]

    # Construction O: a = cycle, b = "zigzag" [0,5,1,6,2,7,3,8,4,9,5,10]
    b_zigzag = [i if i % 2 == 0 else i + 5 for i in range(N)]
    b_zigzag = [x % N for x in b_zigzag]
    thr = _threshold(a_cycle, b_zigzag)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_zigzag[:]

    # Construction P: a = cycle, b = "reverse half" [0,1,2,3,4,5,10,9,8,7,6,5]
    b_revhalf = list(range(6)) + list(range(10, 4, -1))
    thr = _threshold(a_cycle, b_revhalf)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_revhalf[:]

    # Construction Q: a = cycle, b = "three clusters" [0,0,1,1,2,2,3,3,4,4,5,5]
    b_clusters = [i // 2 for i in range(N)]
    thr = _threshold(a_cycle, b_clusters)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_clusters[:]

    # Construction R: a = cycle, b = "four clusters" [0,0,0,1,1,1,2,2,2,3,3,3]
    b_fourclust = [i // 3 for i in range(N)]
    thr = _threshold(a_cycle, b_fourclust)
    if thr > best_thresh:
        best_thresh = thr
        best_a = a_cycle[:]
        best_b = b_fourclust[:]

    # ---- Genetic Algorithm for global search ----
    if best_a is not None and best_b is not None and _elapsed() < _DEADLINE - 10:
        pop_size = 60
        # Initialize population with best solution + random perturbations
        population = []
        population.append((best_a[:], best_b[:], best_thresh))
        
        for _ in range(pop_size - 1):
            a_gen = list(range(N))
            rng.shuffle(a_gen)
            b_gen = [rng.randint(0, N-1) for _ in range(N)]
            thr_gen = _threshold(a_gen, b_gen)
            if thr_gen > 0:
                population.append((a_gen, b_gen, thr_gen))
            else:
                population.append((a_gen, b_gen, 0))
        
        # Sort by fitness
        population.sort(key=lambda x: x[2], reverse=True)
        
        for gen in range(50):
            if _elapsed() > _DEADLINE:
                break
            
            new_population = []
            
            # Elitism: keep top 2
            new_population.append(population[0])
            if len(population) > 1:
                new_population.append(population[1])
            
            # Generate offspring
            while len(new_population) < pop_size:
                # Tournament selection
                def tournament():
                    cands = [population[rng.randrange(len(population))] for _ in range(3)]
                    return max(cands, key=lambda x: x[2])
                
                parent1 = tournament()
                parent2 = tournament()
                
                # Crossover: random mix of a and b from parents
                child_a = []
                child_b = []
                for i in range(N):
                    if rng.random() < 0.5:
                        child_a.append(parent1[0][i])
                        child_b.append(parent1[1][i])
                    else:
                        child_a.append(parent2[0][i])
                        child_b.append(parent2[1][i])
                
                # Mutation
                if rng.random() < 0.3:
                    idx = rng.randrange(N)
                    child_a[idx] = rng.randrange(N)
                if rng.random() < 0.3:
                    idx = rng.randrange(N)
                    child_b[idx] = rng.randrange(N)
                
                thr_child = _threshold(child_a, child_b)
                new_population.append((child_a, child_b, thr_child if thr_child > 0 else 0))
            
            population = new_population
            population.sort(key=lambda x: x[2], reverse=True)
            
            # Update best
            if population[0][2] > best_thresh:
                best_thresh = population[0][2]
                best_a = population[0][0][:]
                best_b = population[0][1][:]
    
    # ---- Local search: perturb best found automaton (increased iterations) ----
    if best_a is not None and best_b is not None:
        current_a = best_a[:]
        current_b = best_b[:]
        current_thr = best_thresh
        for _ in range(2000):
            if _elapsed() > _DEADLINE:
                break
            # Randomly perturb one transition
            new_a = current_a[:]
            new_b = current_b[:]
            if rng.random() < 0.5:
                idx = rng.randrange(N)
                new_a[idx] = rng.randrange(N)
            else:
                idx = rng.randrange(N)
                new_b[idx] = rng.randrange(N)
            thr = _threshold(new_a, new_b)
            if thr > current_thr:
                current_a = new_a
                current_b = new_b
                current_thr = thr
                if current_thr > best_thresh:
                    best_thresh = current_thr
                    best_a = current_a[:]
                    best_b = current_b[:]
            elif thr > 0 and rng.random() < 0.15:
                # Accept worse solutions occasionally (simulated annealing-like)
                current_a = new_a
                current_b = new_b
                current_thr = thr

    # ---- Second local search phase with different random seed ----
    if best_a is not None and best_b is not None and _elapsed() < _DEADLINE - 5:
        rng2 = random.Random(99999)
        current_a = best_a[:]
        current_b = best_b[:]
        current_thr = best_thresh
        for _ in range(1000):
            if _elapsed() > _DEADLINE:
                break
            new_a = current_a[:]
            new_b = current_b[:]
            if rng2.random() < 0.5:
                idx = rng2.randrange(N)
                new_a[idx] = rng2.randrange(N)
            else:
                idx = rng2.randrange(N)
                new_b[idx] = rng2.randrange(N)
            thr = _threshold(new_a, new_b)
            if thr > current_thr:
                current_a = new_a
                current_b = new_b
                current_thr = thr
                if current_thr > best_thresh:
                    best_thresh = current_thr
                    best_a = current_a[:]
                    best_b = current_b[:]
            elif thr > 0 and rng2.random() < 0.15:
                current_a = new_a
                current_b = new_b
                current_thr = thr

    # Fallback: ensure we have a valid synchronizing automaton
    if best_a is None or best_b is None or best_thresh <= 0:
        best_a = fallback_a
        best_b = fallback_b
        best_thresh = _threshold(best_a, best_b)

    return best_a, best_b
