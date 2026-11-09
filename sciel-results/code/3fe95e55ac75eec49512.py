"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

P = {'n': 12, 'm': 40}

def _calc_score_bitmask(family):
    """Calculate score for a family of bitmasks."""
    m = len(family)
    count = 0
    for i in range(m):
        si = family[i]
        for j in range(i + 1, m):
            sj = family[j]
            if (si & sj) == si and si != sj:
                count += 1
            elif (si & sj) == sj and si != sj:
                count += 1
    return count

def _generate_maximal_chain(n, seed_bits=None):
    """Generate a maximal chain by adding bits in a given order."""
    if seed_bits is None:
        seed_bits = list(range(n))
    random.shuffle(seed_bits)
    chain = [0]
    mask = 0
    for b in seed_bits:
        mask |= (1 << b)
        chain.append(mask)
    return chain

def _greedy_chain_init(n, m):
    """Initialize with 3 disjoint maximal chains (39 sets) + 1 extra set."""
    family = []
    used = set()
    
    # Generate 3 disjoint maximal chains
    max_attempts = 500
    chains_found = 0
    for attempt in range(max_attempts):
        if chains_found >= 3:
            break
        chain = _generate_maximal_chain(n)
        if not any(x in used for x in chain):
            family.extend(chain)
            used.update(chain)
            chains_found += 1
    
    # If we couldn't get 3 disjoint chains, fill in with other sets
    if len(family) < m:
        for mask in range(1 << n):
            if len(family) >= m:
                break
            if mask not in used:
                family.append(mask)
                used.add(mask)
    
    # If still short (shouldn't happen), add random
    while len(family) < m:
        mask = random.randint(0, (1 << n) - 1)
        if mask not in used:
            family.append(mask)
            used.add(mask)
    
    return family[:m]

def _two_opt_local_search(family, n, m, deadline):
    """Vectorized 2-opt local search with simulated annealing escape."""
    max_mask = 1 << n
    current_family = list(family)
    current_fs = set(current_family)
    current_score = _calc_score_bitmask(current_family)
    
    best_family = list(current_family)
    best_score = current_score
    
    all_masks = np.arange(max_mask, dtype=np.int32)
    
    # Precompute comparison matrix for efficiency
    # For each pair (i,j), store whether they're comparable
    no_improvement_count = 0
    temp = 5.0
    cooling = 0.9995
    min_temp = 0.1
    
    while time.time() < deadline:
        best_delta = 0
        best_idx = -1
        best_new_set = -1
        
        # Try each position with vectorized evaluation
        for idx in range(m):
            if time.time() >= deadline:
                break
            old_set = current_family[idx]
            
            # Build others array (exclude idx)
            others_list = []
            for j in range(m):
                if j != idx:
                    others_list.append(current_family[j])
            others = np.array(others_list, dtype=np.int32)
            
            # Vectorized: compute new comparabilities for all 4096 candidates
            cand = all_masks[:, np.newaxis]  # (4096, 1)
            oth = others[np.newaxis, :]       # (1, 39)
            
            # Strict subset checks
            cand_subset_oth = (cand & oth) == cand
            oth_subset_cand = (oth & cand) == oth
            eq_mask = (cand == oth)
            
            strict_cand_subset = cand_subset_oth & ~eq_mask
            strict_oth_subset = oth_subset_cand & ~eq_mask
            
            new_comps = (strict_cand_subset.sum(axis=1) + strict_oth_subset.sum(axis=1)).astype(np.int32)
            
            # Old comparabilities
            old_comps = 0
            for other in others_list:
                if (old_set & other) == old_set and old_set != other:
                    old_comps += 1
                elif (old_set & other) == other and old_set != old_set:
                    old_comps += 1
            
            deltas = new_comps - old_comps
            
            # Mask invalid candidates
            valid_mask = np.ones(max_mask, dtype=bool)
            for s in current_fs:
                valid_mask[s] = False
            valid_mask[old_set] = False
            
            deltas[~valid_mask] = -999999
            
            max_idx = np.argmax(deltas)
            max_delta = int(deltas[max_idx])
            
            if max_delta > best_delta:
                best_delta = max_delta
                best_idx = idx
                best_new_set = int(all_masks[max_idx])
        
        # Decide whether to accept
        accept = False
        if best_delta > 0:
            accept = True
        elif best_delta == 0:
            accept = random.random() < 0.3
        else:
            # Simulated annealing: accept worse moves with probability
            if temp > min_temp:
                accept = random.random() < math.exp(best_delta / temp)
        
        if accept and best_idx >= 0 and best_new_set >= 0:
            old_set = current_family[best_idx]
            current_fs.remove(old_set)
            current_fs.add(best_new_set)
            current_family[best_idx] = best_new_set
            current_score += best_delta
            no_improvement_count = 0
            
            if current_score > best_score:
                best_score = current_score
                best_family = list(current_family)
        else:
            no_improvement_count += 1
            if no_improvement_count >= 5:
                # Random perturbation
                for _ in range(10):
                    if time.time() >= deadline:
                        break
                    idx = random.randint(0, m - 1)
                    new_set = random.randint(0, max_mask - 1)
                    if new_set in current_fs:
                        continue
                    old_set = current_family[idx]
                    
                    delta = 0
                    for j in range(m):
                        if j == idx:
                            continue
                        other = current_family[j]
                        if (old_set & other) == old_set and old_set != other:
                            delta -= 1
                        elif (old_set & other) == other and old_set != other:
                            delta -= 1
                        if (new_set & other) == new_set and new_set != other:
                            delta += 1
                        elif (new_set & other) == other and new_set != old_set:
                            delta += 1
                    
                    if delta >= 0 or random.random() < 0.5:
                        current_fs.remove(old_set)
                        current_fs.add(new_set)
                        current_family[idx] = new_set
                        current_score += delta
                        if current_score > best_score:
                            best_score = current_score
                            best_family = list(current_family)
                        break
                no_improvement_count = 0
        
        # Cool down
        if temp > min_temp:
            temp *= cooling
    
    return best_family, best_score

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin for 90s wall-clock
    n = P['n']
    m = P['m']
    
    # Multiple restarts with different seeds
    best_overall_family = None
    best_overall_score = -1
    
    num_restarts = 3
    time_per_restart = (deadline - start_time) / num_restarts
    
    for restart in range(num_restarts):
        restart_deadline = start_time + (restart + 1) * time_per_restart
        if restart_deadline > deadline:
            restart_deadline = deadline
        
        # Initialize
        current_family = _greedy_chain_init(n, m)
        
        # Local search
        best_family, best_score = _two_opt_local_search(current_family, n, m, restart_deadline)
        
        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall_family = best_family
        
        if time.time() >= deadline:
            break
    
    # Final polish: one more pass on the best family
    if time.time() < deadline - 5 and best_overall_family is not None:
        final_family, final_score = _two_opt_local_search(best_overall_family, n, m, deadline)
        if final_score > best_overall_score:
            best_overall_score = final_score
            best_overall_family = final_family
    
    # Convert back to lists of ints
    result = []
    for s in best_overall_family:
        lst = []
        for b in range(n):
            if s & (1 << b):
                lst.append(b)
        result.append(lst)
    
    return result
# EVOLVE-BLOCK-END
