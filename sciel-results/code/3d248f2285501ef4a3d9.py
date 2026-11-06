"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import numpy as np

def _score_set_fast(s):
    """Count distinct d > 0 such that a, a+d, a+2d all in s for some a."""
    s_set = set(s)
    diffs = set()
    for a in s:
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if a + d in s_set and a + 2 * d in s_set:
                diffs.add(d)
    return len(diffs)

def _score_and_get_diffs(s):
    """Return (score, set_of_diffs) for a set."""
    s_set = set(s)
    diffs = set()
    for a in s:
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if a + d in s_set and a + 2 * d in s_set:
                diffs.add(d)
    return len(diffs), diffs

def _try_add_element(s, candidate, s_set, diffs):
    """Try adding candidate to s, return new score and diffs if improved."""
    if candidate in s_set or not (1 <= candidate <= 512):
        return None, None
    # Check if adding this creates new diffs
    new_diffs = set(diffs)
    for a in s:
        max_d = (512 - a) // 2
        for d in range(1, max_d + 1):
            if a + d in s_set and a + 2 * d in s_set:
                continue  # already counted
            # Check if candidate helps form a progression
            # candidate could be a, a+d, or a+2d
            # This is complex; just recompute
            pass
    # Just recompute
    new_s = list(s) + [candidate]
    return _score_and_get_diffs(new_s)

def _local_search(s, max_iters=5000, time_limit=None, t0=None):
    """Local search with smart moves."""
    s = list(s)
    s_set = set(s)
    current_score, current_diffs = _score_and_get_diffs(s)
    best_score = current_score
    best_set = list(s)
    best_diffs = set(current_diffs)
    
    if time_limit is None:
        time_limit = 10.0
    if t0 is None:
        t0 = time.time()
    
    for it in range(max_iters):
        if time.time() - t0 > time_limit:
            break
        
        improved = False
        
        # Try removing an element and adding a better one
        for i in range(len(s)):
            old_val = s[i]
            # Try replacing with values that might create new diffs
            # Focus on values that could complete progressions
            candidates = set()
            
            # For each missing d, try to find a value that would create it
            for d in range(1, 256):
                if d in current_diffs:
                    continue
                # To create diff d, we need a, a+d, a+2d in set
                # If we have two of them, we can add the third
                for a in s:
                    if a + d in s_set and a + 2 * d in s_set:
                        candidates.add(a)  # a is already there
                    elif a in s_set and a + 2 * d in s_set:
                        candidates.add(a + d)
                    elif a + d in s_set and a + 2 * d in s_set:
                        candidates.add(a)
                    # Also try: if we have a and a+d, add a+2d
                    if a in s_set and a + d in s_set:
                        candidates.add(a + 2 * d)
                    # If we have a+d and a+2d, add a
                    if a + d in s_set and a + 2 * d in s_set:
                        candidates.add(a)
            
            # Also try random candidates
            for _ in range(3):
                candidates.add(random.randint(1, 512))
            
            for cand in candidates:
                if cand == old_val or cand in s_set:
                    continue
                if not (1 <= cand <= 512):
                    continue
                
                s[i] = cand
                s_set.remove(old_val)
                s_set.add(cand)
                new_score, new_diffs = _score_and_get_diffs(s)
                
                if new_score > current_score:
                    current_score = new_score
                    current_diffs = new_diffs
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(s)
                        best_diffs = set(current_diffs)
                    improved = True
                    break
                else:
                    s[i] = old_val
                    s_set.remove(cand)
                    s_set.add(old_val)
            
            if improved:
                break
        
        if not improved:
            # Try a random perturbation to escape local optimum
            if random.random() < 0.3:
                i = random.randint(0, len(s) - 1)
                old_val = s[i]
                new_val = random.randint(1, 512)
                if new_val != old_val and new_val not in s_set:
                    s[i] = new_val
                    s_set.remove(old_val)
                    s_set.add(new_val)
                    current_score, current_diffs = _score_and_get_diffs(s)
    
    return best_set, best_score

def _genetic_algorithm(pop_size=20, time_limit=10.0):
    """Genetic algorithm for finding good sets."""
    t0 = time.time()
    
    # Initialize population
    pop = []
    # Seed with various initializations
    inits = [
        list(range(1, 33)),
        [int(512 * (i + 1) / 33) for i in range(32)],
        list(range(1, 17)) + list(range(400, 416)),
        random.sample(range(1, 513), 32),
        [1 + i * 16 for i in range(32)],
        [2 ** i for i in range(9)] + [2 ** i * 3 for i in range(8)] + [2 ** i * 5 for i in range(7)] + [random.randint(1, 512) for _ in range(3)],
    ]
    
    for init in inits:
        init = list(set(x for x in init if 1 <= x <= 512))
        while len(init) < 32:
            v = random.randint(1, 512)
            if v not in init:
                init.append(v)
        pop.append(init[:32])
    
    while len(pop) < pop_size:
        pop.append(random.sample(range(1, 513), 32))
    
    # Evaluate initial population
    scored = [(_score_set_fast(s), s) for s in pop]
    scored.sort(key=lambda x: -x[0])
    best_score = scored[0][0]
    best_set = scored[0][1]
    
    gen = 0
    while time.time() - t0 < time_limit:
        gen += 1
        new_pop = []
        
        # Elitism: keep top 2
        for _, s in scored[:2]:
            new_pop.append(list(s))
        
        # Create offspring
        while len(new_pop) < pop_size:
            # Tournament selection
            def select():
                idxs = random.sample(range(len(scored)), min(3, len(scored)))
                best_idx = max(idxs, key=lambda i: scored[i][0])
                return scored[best_idx][1]
            
            parent1 = select()
            parent2 = select()
            
            # Crossover: uniform
            child = []
            for i in range(32):
                if random.random() < 0.5:
                    child.append(parent1[i])
                else:
                    child.append(parent2[i])
            
            # Ensure distinct
            child = list(set(child))
            while len(child) < 32:
                v = random.randint(1, 512)
                if v not in child:
                    child.append(v)
            child = child[:32]
            
            # Mutation
            for i in range(random.randint(1, 3)):
                idx = random.randint(0, len(child) - 1)
                child[idx] = random.randint(1, 512)
            
            # Ensure distinct again
            child = list(set(child))
            while len(child) < 32:
                v = random.randint(1, 512)
                if v not in child:
                    child.append(v)
            child = child[:32]
            
            new_pop.append(child)
        
        # Evaluate new population
        scored = [(_score_set_fast(s), s) for s in new_pop]
        scored.sort(key=lambda x: -x[0])
        
        if scored[0][0] > best_score:
            best_score = scored[0][0]
            best_set = scored[0][1]
        
        # Local search on best few
        for i in range(min(3, len(scored))):
            if time.time() - t0 > time_limit - 2:
                break
            ls_set, ls_score = _local_search(scored[i][1], max_iters=2000, time_limit=min(2.0, time_limit - (time.time() - t0)), t0=t0)
            if ls_score > best_score:
                best_score = ls_score
                best_set = ls_set
            scored[i] = (ls_score, ls_set)
        
        scored.sort(key=lambda x: -x[0])
    
    return best_set, best_score

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    t0 = time.time()
    
    best_set = list(range(1, 33))
    best_score = _score_set_fast(best_set)
    
    # Phase 1: Genetic algorithm
    ga_time = min(40.0, max(10.0, (deadline - time.time()) * 0.5))
    ga_set, ga_score = _genetic_algorithm(pop_size=25, time_limit=ga_time)
    if ga_score > best_score:
        best_score = ga_score
        best_set = ga_set
    
    # Phase 2: Local search with multiple restarts
    remaining = deadline - time.time()
    n_restarts = max(1, int(remaining / 8.0))
    
    for r in range(n_restarts):
        if time.time() > deadline - 5:
            break
        
        # Different initializations
        if r == 0:
            init = list(best_set)
        elif r == 1:
            init = list(range(1, 33))
        elif r == 2:
            init = [int(512 * (i + 1) / 33) for i in range(32)]
            init = list(set(init))
            while len(init) < 32:
                init.append(random.randint(1, 512))
            init = list(set(init))[:32]
        elif r == 3:
            init = list(range(1, 17)) + list(range(400, 416))
        elif r == 4:
            init = random.sample(range(1, 513), 32)
        elif r == 5:
            init = [1 + i * 16 for i in range(32)]
            init = [x for x in init if x <= 512]
            while len(init) < 32:
                init.append(random.randint(1, 512))
            init = list(set(init))[:32]
        elif r == 6:
            # Dense small + sparse large
            init = list(range(1, 21))
            init += [random.randint(250, 512) for _ in range(12)]
            init = list(set(init))[:32]
        elif r == 7:
            # Two arithmetic progressions
            init = [1 + i * 4 for i in range(16)] + [200 + i * 15 for i in range(16)]
            init = [x for x in init if x <= 512]
            while len(init) < 32:
                init.append(random.randint(1, 512))
            init = list(set(init))[:32]
        else:
            init = random.sample(range(1, 513), 32)
        
        init = list(set(init))
        while len(init) < 32:
            v = random.randint(1, 512)
            if v not in init:
                init.append(v)
        init = init[:32]
        
        ls_time = min(7.0, max(3.0, (deadline - time.time()) / (n_restarts - r)))
        ls_set, ls_score = _local_search(init, max_iters=5000, time_limit=ls_time, t0=time.time())
        
        if ls_score > best_score:
            best_score = ls_score
            best_set = ls_set
    
    return best_set
# EVOLVE-BLOCK-END
