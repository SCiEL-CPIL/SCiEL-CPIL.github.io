"""
Max K(2,2,2)-free triple system on 16 points.
Combines the structural idea of the Max-SAT count-based formulation (inspiration 1)
with the local search / multiple restart machinery of the current program.
"""
import time
import random
import itertools

def construct():
    deadline = time.time() + 115  # safety margin before 150s limit
    random.seed(42)
    
    N = 16
    all_triples = list(itertools.combinations(range(N), 3))
    n_triples = len(all_triples)  # 560
    
    # Map each triple to an index for fast lookup
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute all K(2,2,2) configurations using the efficient 6-point enumeration
    # For each set of 6 points, there are 15 ways to partition them into 3 pairs.
    # Each partition defines a K(2,2,2) configuration consisting of 8 triples.
    k222_configs = []
    seen_configs = set()
    
    def get_pairings(points):
        points = list(points)
        pairings = []
        def rec(remaining, current):
            if not remaining:
                pairings.append(current[:])
                return
            first = remaining[0]
            rest = remaining[1:]
            for i in range(len(rest)):
                second = rest[i]
                new_remaining = rest[:i] + rest[i+1:]
                rec(new_remaining, current + [(first, second)])
        rec(points, [])
        return pairings
    
    for six_points in itertools.combinations(range(N), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_to_idx[triple]
                        triples_in_config.append(idx)
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(n_triples)]
    for c_idx, config in enumerate(k222_configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
    
    # Convert to tuples for faster iteration
    triple_to_configs = [tuple(configs) for configs in triple_to_configs]
    
    def greedy_init():
        """Randomized greedy initialization with count-based safety checks."""
        sol = [False] * n_triples
        counts = [0] * num_configs
        candidates = list(range(n_triples))
        random.shuffle(candidates)
        
        for idx in candidates:
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
        return sol, counts, sum(sol)
    
    def try_add_candidates(sol, counts, candidates):
        """
        Tries to add triples from the candidates list.
        Returns the number of triples added and the updated candidates list.
        """
        added = 0
        new_candidates = []
        for idx in candidates:
            if sol[idx]:
                continue
            safe = True
            for c_idx in triple_to_configs[idx]:
                if counts[c_idx] == 7:
                    safe = False
                    break
            if safe:
                sol[idx] = True
                for c_idx in triple_to_configs[idx]:
                    counts[c_idx] += 1
                added += 1
            else:
                new_candidates.append(idx)
        return added, new_candidates
    
    def local_search(sol, counts, current_count, max_iters=5000):
        """
        WalkSAT-style local search: kick a tight triple, then try to re-add candidates.
        """
        iters = 0
        while iters < max_iters and time.time() < deadline - 5:
            iters += 1
            
            # Try to add from candidates
            candidates = [i for i in range(n_triples) if not sol[i]]
            random.shuffle(candidates)
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                continue
            
            # Stuck: perform a kick
            # Find tight clauses (count == 7) and pick a random active triple from them
            tight_triples = []
            for c_idx, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in k222_configs[c_idx]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)
            
            if tight_triples:
                kick_idx = random.choice(tight_triples)
            else:
                # Pick random active triple
                active = [i for i in range(n_triples) if sol[i]]
                if not active:
                    break
                kick_idx = random.choice(active)
            
            # Remove kick_idx
            sol[kick_idx] = False
            for c_idx in triple_to_configs[kick_idx]:
                counts[c_idx] -= 1
            current_count -= 1
            
            # Try to add from candidates including kick_idx
            candidates.append(kick_idx)
            random.shuffle(candidates)
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
        
        return sol, counts, current_count
    
    def two_kick_local_search(sol, counts, current_count, max_iters=3000):
        """
        2-kick local search: remove two tight triples, then try to re-add.
        """
        iters = 0
        while iters < max_iters and time.time() < deadline - 5:
            iters += 1
            
            # Try to add from candidates
            candidates = [i for i in range(n_triples) if not sol[i]]
            random.shuffle(candidates)
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                continue
            
            # Find tight clauses
            tight_triples = []
            for c_idx, cnt in enumerate(counts):
                if cnt == 7:
                    for t_idx in k222_configs[c_idx]:
                        if sol[t_idx]:
                            tight_triples.append(t_idx)
            
            if len(tight_triples) < 2:
                # Not enough tight triples, do single kick
                active = [i for i in range(n_triples) if sol[i]]
                if len(active) < 2:
                    break
                kick1, kick2 = random.sample(active, 2)
            else:
                kick1, kick2 = random.sample(tight_triples, min(2, len(tight_triples)))
                if len(tight_triples) < 2:
                    break
            
            # Remove both
            for kick_idx in [kick1, kick2]:
                sol[kick_idx] = False
                for c_idx in triple_to_configs[kick_idx]:
                    counts[c_idx] -= 1
                current_count -= 1
            
            # Try to add from candidates
            candidates.extend([kick1, kick2])
            random.shuffle(candidates)
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
        
        return sol, counts, current_count
    
    best_sol = None
    best_count = -1
    
    num_restarts = 0
    while time.time() < deadline - 10:
        num_restarts += 1
        
        # Greedy initialization
        sol, counts, current_count = greedy_init()
        
        # Local search
        sol, counts, current_count = local_search(sol, counts, current_count, max_iters=3000)
        
        # 2-kick local search for deeper escape
        if time.time() < deadline - 15:
            sol, counts, current_count = two_kick_local_search(sol, counts, current_count, max_iters=2000)
        
        # Final greedy pass
        candidates = [i for i in range(n_triples) if not sol[i]]
        random.shuffle(candidates)
        added, _ = try_add_candidates(sol, counts, candidates)
        current_count += added
        
        if current_count > best_count:
            best_count = current_count
            best_sol = sol[:]
        
        # Check if we're running low on time
        if time.time() > deadline - 20:
            break
    
    # Final polish on the best solution
    if best_sol is not None and time.time() < deadline - 3:
        sol = best_sol[:]
        # Recompute counts
        counts = [0] * num_configs
        for i in range(n_triples):
            if sol[i]:
                for c_idx in triple_to_configs[i]:
                    counts[c_idx] += 1
        current_count = sum(sol)
        
        sol, counts, current_count = local_search(sol, counts, current_count, max_iters=3000)
        if time.time() < deadline - 5:
            sol, counts, current_count = two_kick_local_search(sol, counts, current_count, max_iters=1500)
        
        # Final greedy
        candidates = [i for i in range(n_triples) if not sol[i]]
        random.shuffle(candidates)
        added, _ = try_add_candidates(sol, counts, candidates)
        current_count += added
        
        if current_count > best_count:
            best_count = current_count
            best_sol = sol
    
    if best_sol is None:
        best_sol, _, _ = greedy_init()
    
    result = [list(all_triples[i]) for i in range(n_triples) if best_sol[i]]
    return result
