"""Flow-shop permutation via fast heuristics, insertion local search, PSO, and simulated annealing."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import math


def _makespan(pt, pt_rows, perm):
    """Compute makespan for a given permutation."""
    n_jobs, n_machines = pt.shape
    if n_machines == 0 or n_jobs == 0:
        return 0.0
    
    # Use Python lists for speed
    prev = [0.0] * n_machines
    for j in perm:
        row = pt_rows[j]
        curr = [0.0] * n_machines
        curr[0] = prev[0] + row[0]
        for m in range(1, n_machines):
            val = prev[m] if prev[m] > curr[m-1] else curr[m-1]
            curr[m] = val + row[m]
        prev = curr
    return prev[-1]


def _neh_initial(pt):
    """NEH heuristic: sort by total processing time descending, insert each job at best position."""
    n_jobs, n_machines = pt.shape
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    totals = pt.sum(axis=1)
    order = np.argsort(-totals)
    
    perm = [int(order[0])]
    for i in range(1, n_jobs):
        job = int(order[i])
        best_pos = 0
        best_ms = float('inf')
        for pos in range(len(perm) + 1):
            test_perm = perm[:pos] + [job] + perm[pos:]
            ms = _makespan(pt, pt_rows, test_perm)
            if ms < best_ms:
                best_ms = ms
                best_pos = pos
        perm.insert(best_pos, job)
    return perm


def _pso_optimize(pt, pt_rows, initial_perms, deadline, n_particles=8, max_iter=5000):
    """
    Discrete Particle Swarm Optimization for flow-shop permutation.
    Each particle has a position (permutation), personal best, and velocity (Or-Opt move directives).
    """
    n_jobs = len(initial_perms[0])
    if n_jobs <= 1:
        return initial_perms[0], _makespan(pt, pt_rows, initial_perms[0])
    
    # Initialize particles
    particles = []
    pbests = []
    pbest_ms = []
    velocities = []  # Each velocity is a list of (job, from_pos, to_pos) tuples
    
    for i in range(n_particles):
        if i < len(initial_perms):
            pos = initial_perms[i][:]
        else:
            pos = list(initial_perms[i % len(initial_perms)])
            # Add some randomness
            for _ in range(min(3, n_jobs // 2)):
                a = random.randint(0, n_jobs - 2)
                b = random.randint(a + 1, n_jobs - 1)
                pos[a], pos[b] = pos[b], pos[a]
        
        ms = _makespan(pt, pt_rows, pos)
        particles.append(pos)
        pbests.append(pos[:])
        pbest_ms.append(ms)
        velocities.append([])  # Empty velocity initially
    
    # Global best
    gbest_idx = np.argmin(pbest_ms)
    gbest = pbests[gbest_idx][:]
    gbest_ms = pbest_ms[gbest_idx]
    
    w = 0.5  # Inertia
    c1 = 1.5  # Cognitive coefficient
    c2 = 1.5  # Social coefficient
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        for p in range(n_particles):
            if time.time() > deadline:
                break
            
            # Generate velocity: stochastic Or-Opt moves
            # Number of moves proportional to velocity magnitude
            n_moves = random.randint(1, max(1, n_jobs // 4))
            new_pos = particles[p][:]
            
            for _ in range(n_moves):
                # Choose between cognitive and social influence
                if random.random() < 0.5:
                    # Cognitive: move toward personal best
                    ref = pbests[p]
                else:
                    # Social: move toward global best
                    ref = gbest
                
                # Apply Or-Opt move: pick a job from current position, insert toward reference position
                # Find a job that is in a different relative position
                job = random.choice(new_pos)
                curr_idx = new_pos.index(job)
                ref_idx = ref.index(job)
                
                if curr_idx == ref_idx:
                    continue
                
                # Move job toward its reference position with some randomness
                target = ref_idx + random.randint(-1, 1)
                target = max(0, min(n_jobs - 1, target))
                
                if target == curr_idx:
                    continue
                
                # Remove job from current position
                new_pos.pop(curr_idx)
                # Insert at target position (adjust if target was before curr_idx)
                if target > curr_idx:
                    target -= 1
                new_pos.insert(target, job)
            
            # Evaluate new position
            new_ms = _makespan(pt, pt_rows, new_pos)
            
            # Update personal best
            if new_ms < pbest_ms[p]:
                pbest_ms[p] = new_ms
                pbests[p] = new_pos[:]
            
            # Update global best
            if new_ms < gbest_ms:
                gbest_ms = new_ms
                gbest = new_pos[:]
            
            # Update particle position
            particles[p] = new_pos
        
        # Gradually reduce exploration
        w = max(0.3, w * 0.999)
    
    return gbest, gbest_ms


def _palmer_initial(pt):
    """Palmer's heuristic: sort by priority key."""
    n_jobs, n_machines = pt.shape
    n = n_machines
    scores = np.zeros(n_jobs)
    for j in range(n_jobs):
        s = 0
        for m in range(n):
            if m < n // 2:
                s += (n - 1 - m) * pt[j, m]
            else:
                s -= m * pt[j, m]
        scores[j] = s
    return list(np.argsort(-scores))


def _lpt_initial(pt):
    """LPT: sort by total processing time descending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(-totals))


def _spt_initial(pt):
    """SPT: sort by total processing time ascending."""
    totals = pt.sum(axis=1)
    return list(np.argsort(totals))


def _random_initial(n_jobs, rng):
    """Random permutation."""
    return list(rng.permutation(n_jobs))


def _insertion_local_search(pt, pt_rows, perm, deadline, max_iter=100000):
    """
    Local search using insertion neighborhood with first-improvement.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    eval_count = 0
    
    while eval_count < max_iter:
        if time.time() > deadline:
            break
        
        improved = False
        
        for i in range(n_jobs):
            if time.time() > deadline or eval_count >= max_iter:
                break
            
            job = current_perm[i]
            temp_perm = current_perm[:i] + current_perm[i+1:]
            
            for j in range(len(temp_perm) + 1):
                if j == i:
                    continue
                
                new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                new_ms = _makespan(pt, pt_rows, new_perm)
                eval_count += 1
                
                if new_ms < current_ms:
                    current_ms = new_ms
                    current_perm = new_perm
                    improved = True
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = current_perm[:]
                    break
        
        if not improved:
            break
    
    return best_perm, best_ms


def _vnd_local_search(pt, pt_rows, perm, deadline, max_iter=50000):
    """
    Variable Neighborhood Descent: cycles through nested neighborhoods.
    N1: adjacent swap, N2: 2-opt reversal, N3: Or-Opt (move 1 job)
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    eval_count = 0
    
    def try_adjacent_swap():
        nonlocal current_perm, current_ms, best_perm, best_ms, eval_count
        for i in range(n_jobs - 1):
            if time.time() > deadline or eval_count >= max_iter:
                return False
            new_perm = current_perm[:]
            new_perm[i], new_perm[i+1] = new_perm[i+1], new_perm[i]
            new_ms = _makespan(pt, pt_rows, new_perm)
            eval_count += 1
            if new_ms < current_ms:
                current_ms = new_ms
                current_perm = new_perm
                if current_ms < best_ms:
                    best_ms = current_ms
                    best_perm = current_perm[:]
                return True
        return False
    
    def try_2opt():
        nonlocal current_perm, current_ms, best_perm, best_ms, eval_count
        for i in range(n_jobs - 2):
            if time.time() > deadline or eval_count >= max_iter:
                return False
            for j in range(i + 2, n_jobs):
                if time.time() > deadline or eval_count >= max_iter:
                    return False
                new_perm = current_perm[:i] + current_perm[i:j+1][::-1] + current_perm[j+1:]
                new_ms = _makespan(pt, pt_rows, new_perm)
                eval_count += 1
                if new_ms < current_ms:
                    current_ms = new_ms
                    current_perm = new_perm
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = current_perm[:]
                    return True
        return False
    
    def try_oropt():
        nonlocal current_perm, current_ms, best_perm, best_ms, eval_count
        for i in range(n_jobs):
            if time.time() > deadline or eval_count >= max_iter:
                return False
            job = current_perm[i]
            temp_perm = current_perm[:i] + current_perm[i+1:]
            for j in range(len(temp_perm) + 1):
                if j == i:
                    continue
                new_perm = temp_perm[:j] + [job] + temp_perm[j:]
                new_ms = _makespan(pt, pt_rows, new_perm)
                eval_count += 1
                if new_ms < current_ms:
                    current_ms = new_ms
                    current_perm = new_perm
                    if current_ms < best_ms:
                        best_ms = current_ms
                        best_perm = current_perm[:]
                    return True
        return False
    
    neighborhoods = [try_adjacent_swap, try_2opt, try_oropt]
    
    while eval_count < max_iter:
        if time.time() > deadline:
            break
        
        improved = False
        for neighborhood in neighborhoods:
            if time.time() > deadline or eval_count >= max_iter:
                break
            if neighborhood():
                improved = True
                break
        
        if not improved:
            break
    
    return best_perm, best_ms


def _simulated_annealing(pt, pt_rows, perm, deadline, max_iter=20000):
    """
    Simulated annealing with insertion neighborhood.
    """
    n_jobs = len(perm)
    if n_jobs <= 1:
        return perm[:], _makespan(pt, pt_rows, perm)
    
    current_perm = perm[:]
    current_ms = _makespan(pt, pt_rows, current_perm)
    best_perm = current_perm[:]
    best_ms = current_ms
    
    T = max(current_ms * 0.1, 1.0)
    T_min = max(current_ms * 0.001, 0.1)
    cooling = 0.995
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        # Generate neighbor: pick random job to move
        i = random.randint(0, n_jobs - 1)
        job = current_perm[i]
        temp_perm = current_perm[:i] + current_perm[i+1:]
        
        # Pick random insertion position
        j = random.randint(0, len(temp_perm))
        if j == i:
            continue  # No change
        
        new_perm = temp_perm[:j] + [job] + temp_perm[j:]
        new_ms = _makespan(pt, pt_rows, new_perm)
        
        delta = new_ms - current_ms
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_ms = new_ms
            current_perm = new_perm
            if current_ms < best_ms:
                best_ms = current_ms
                best_perm = current_perm[:]
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    return best_perm, best_ms


def order_jobs(processing_times):
    """
    Construct a job permutation minimizing makespan.
    """
    pt = np.asarray(processing_times)
    n_jobs, n_machines = pt.shape
    
    # Deadline guard: 145s to leave 5s margin before 150s kill
    start_time = time.time()
    deadline = start_time + 145
    
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
    if n_machines == 0:
        return list(range(n_jobs))
    
    rng = np.random.RandomState(42)
    
    # Pre-extract job rows as Python lists for faster access in makespan computation
    pt_rows = [list(pt[j]) for j in range(n_jobs)]
    
    # Generate diverse initial solutions
    candidates = []
    
    # 1. NEH
    candidates.append(_neh_initial(pt))
    
    # 2. Palmer
    candidates.append(_palmer_initial(pt))
    
    # 3. LPT
    candidates.append(_lpt_initial(pt))
    
    # 4. SPT
    candidates.append(_spt_initial(pt))
    
    # 5. Sort by first machine descending
    candidates.append(list(np.argsort(-pt[:, 0])))
    
    # 6. Sort by last machine descending
    candidates.append(list(np.argsort(-pt[:, -1])))
    
    # 7. Random permutations
    for _ in range(4):
        candidates.append(_random_initial(n_jobs, rng))
    
    best_perm = candidates[0]
    best_ms = _makespan(pt, pt_rows, best_perm)
    
    # Phase 1: Run VND local search on top candidates
    for cand in candidates[:6]:
        if time.time() > deadline - 40:  # Leave 40s for PSO + SA
            break
        
        perm, ms = _vnd_local_search(pt, pt_rows, cand, deadline - 40, max_iter=30000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Phase 2: Run PSO using top candidates as initial particles
    if time.time() < deadline - 15:
        # Use top 8 candidates as initial particles for PSO
        pso_candidates = candidates[:min(8, len(candidates))]
        # Sort by makespan to get best ones first
        scored = [(c, _makespan(pt, pt_rows, c)) for c in pso_candidates]
        scored.sort(key=lambda x: x[1])
        pso_starts = [s[0] for s in scored]
        
        perm, ms = _pso_optimize(pt, pt_rows, pso_starts, deadline - 15, 
                                  n_particles=min(8, n_jobs), max_iter=8000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Phase 3: Run SA from best solution found
    if time.time() < deadline - 5:
        perm, ms = _simulated_annealing(pt, pt_rows, best_perm, deadline - 5, max_iter=30000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    # Phase 4: If we still have time, run SA from a few other good starts
    if time.time() < deadline - 2:
        for cand in candidates[:3]:
            if time.time() > deadline - 2:
                break
            perm, ms = _simulated_annealing(pt, pt_rows, cand, deadline - 2, max_iter=8000)
            if ms < best_ms:
                best_ms = ms
                best_perm = perm
    
    # Final polish: quick insertion local search on best
    if time.time() < deadline - 0.5:
        perm, ms = _insertion_local_search(pt, pt_rows, best_perm, deadline - 0.5, max_iter=5000)
        if ms < best_ms:
            best_ms = ms
            best_perm = perm
    
    return best_perm

# EVOLVE-BLOCK-END
