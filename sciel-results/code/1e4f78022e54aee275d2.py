"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
from itertools import combinations

def construct():
    """
    Construct a triple system on 16 points without the complete 3-partite K(2,2,2).
    Uses a Max-SAT style local search with efficient candidate management.
    """
    n = 16
    num_triples = 560  # C(16,3)
    
    # Precompute all possible triples
    all_triples = []
    triple_index = {}
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                triple = (i, j, k)
                triple_index[triple] = len(all_triples)
                all_triples.append(triple)
    
    # Precompute all K(2,2,2) configurations
    k222_configs = []
    
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
    
    seen_configs = set()
    for six_points in combinations(range(n), 6):
        pairings = get_pairings(six_points)
        for pairing in pairings:
            p1, p2, p3 = pairing
            triples_in_config = []
            for a in p1:
                for b in p2:
                    for c in p3:
                        triple = tuple(sorted((a, b, c)))
                        idx = triple_index[triple]
                        triples_in_config.append(idx)
            config_key = tuple(sorted(triples_in_config))
            if config_key not in seen_configs:
                seen_configs.add(config_key)
                k222_configs.append(triples_in_config)
    
    num_configs = len(k222_configs)
    
    # Precompute for each triple, which configs it belongs to
    triple_to_configs = [[] for _ in range(num_triples)]
    for c_idx, config in enumerate(k222_configs):
        for t_idx in config:
            triple_to_configs[t_idx].append(c_idx)
            
    random.seed(42)
    start_time = time.time()
    time_limit = 115
    
    best_solution = None
    best_count = -1
    
    def greedy_init():
        sol = [False] * num_triples
        counts = [0] * num_configs
        candidates = list(range(num_triples))
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
        Returns the number of triples added and the updated candidates list (those not added).
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

    for restart in range(300):
        if time.time() - start_time > time_limit:
            break
            
        sol, counts, current_count = greedy_init()
        
        # Initialize candidates with all inactive triples
        candidates = [i for i in range(num_triples) if not sol[i]]
        random.shuffle(candidates)
        
        if current_count > best_count:
            best_count = current_count
            best_solution = sol[:]
            
        stagnation_counter = 0
        max_stagnation = 30
        
        while time.time() - start_time < time_limit:
            # Try to add from candidates
            added, candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                stagnation_counter = 0
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
                continue
            
            # Stuck: candidates are all unsafe.
            # Perform a kick
            # Find tight clauses (count 7)
            tight_clause_indices = [c_idx for c_idx, cnt in enumerate(counts) if cnt == 7]
            
            if not tight_clause_indices:
                break

            # Strategy: Pick a tight clause. Among active triples in it, pick one that is in the most tight clauses.
            target_clause = random.choice(tight_clause_indices)
            
            # Get active triples in this target clause
            active_in_target = [t_idx for t_idx in k222_configs[target_clause] if sol[t_idx]]
            
            if not active_in_target:
                continue
                
            # Score each active triple by how many tight clauses it is in
            best_kick = None
            max_tight_count = -1
            for t_idx in active_in_target:
                t_tight_count = 0
                for c_idx in triple_to_configs[t_idx]:
                    if counts[c_idx] == 7:
                        t_tight_count += 1
                if t_tight_count > max_tight_count:
                    max_tight_count = t_tight_count
                    best_kick = t_idx
                    if random.random() < 0.2:
                        break
            if best_kick is None:
                best_kick = random.choice(active_in_target)
            
            kick_idx = best_kick
            
            # Save state
            sol_backup = sol[:]
            counts_backup = counts[:]
            candidates_backup = candidates[:]
            count_backup = current_count

            # Remove kick_idx
            sol[kick_idx] = False
            for c_idx in triple_to_configs[kick_idx]:
                counts[c_idx] -= 1
            current_count -= 1
            
            # Add kick_idx to candidates
            candidates.append(kick_idx)
            
            # Try to add from candidates
            added, new_candidates = try_add_candidates(sol, counts, candidates)
            current_count += added
            
            if added > 0:
                # Accept (Gain or Sideways)
                candidates = new_candidates
                if added > 1:
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                if current_count > best_count:
                    best_count = current_count
                    best_solution = sol[:]
            else:
                # Reject (Loss)
                sol = sol_backup
                counts = counts_backup
                candidates = candidates_backup
                current_count = count_backup
                stagnation_counter += 1
                
                if stagnation_counter > max_stagnation:
                    break
    
    if best_solution is None:
        best_solution, _, _ = greedy_init()
        best_count = sum(best_solution)

    result = [list(all_triples[i]) for i in range(num_triples) if best_solution[i]]
    return result
# EVOLVE-BLOCK-END
