import time
import random
import itertools
import math

def construct():
    start_time = time.time()
    # Evaluator kills at 150s. Leave a 5s buffer.
    deadline = start_time + 145.0
    
    N = 16
    vertices = range(N)
    all_triples = list(itertools.combinations(vertices, 3))
    num_triples = len(all_triples)  # 560
    triple_to_idx = {t: i for i, t in enumerate(all_triples)}
    
    # Precompute K(2,2,2) configurations
    # A K(2,2,2) is determined by 3 disjoint pairs.
    # The 8 transversals are the combinations of picking one vertex from each pair.
    # We need to map each triple to the list of K(2,2,2) IDs it belongs to.
    
    k222_configs = []  # List of lists of triple indices
    triple_to_k222 = [[] for _ in range(num_triples)]
    
    # Generate all sets of 6 vertices
    # For each set of 6 vertices, there are 15 ways to partition into 3 pairs.
    for six_vertices in itertools.combinations(vertices, 6):
        a, b, c, d, e, f = six_vertices
        # The 15 partitions of 6 elements into 3 pairs
        pairs_list = [
            ((a,b),(c,d),(e,f)),
            ((a,b),(c,e),(d,f)),
            ((a,b),(c,f),(d,e)),
            ((a,c),(b,d),(e,f)),
            ((a,c),(b,e),(d,f)),
            ((a,c),(b,f),(d,e)),
            ((a,d),(b,c),(e,f)),
            ((a,d),(b,e),(c,f)),
            ((a,d),(b,f),(c,e)),
            ((a,e),(b,c),(d,f)),
            ((a,e),(b,d),(c,f)),
            ((a,e),(b,f),(c,d)),
            ((a,f),(b,c),(d,e)),
            ((a,f),(b,d),(c,e)),
            ((a,f),(b,e),(c,d)),
        ]
        
        for p1, p2, p3 in pairs_list:
            config_triples = []
            for x in p1:
                for y in p2:
                    for z in p3:
                        t = tuple(sorted((x, y, z)))
                        idx = triple_to_idx[t]
                        config_triples.append(idx)
            
            config_id = len(k222_configs)
            k222_configs.append(config_triples)
            
            for t_idx in config_triples:
                triple_to_k222[t_idx].append(config_id)
                
    num_k222 = len(k222_configs)
    
    def update_counts(ti, sign, k222_counts, num_violations):
        """
        sign = 1 for add, -1 for remove
        returns new num_violations
        """
        old_viol = num_violations
        for cid in triple_to_k222[ti]:
            old_count = k222_counts[cid]
            new_count = old_count + sign
            k222_counts[cid] = new_count
            
            if sign == 1:
                if old_count == 7:
                    old_viol += 1
            else:
                if old_count == 8:
                    old_viol -= 1
        return old_viol

    def greedy_init(rng, strategy='random'):
        present = [False] * num_triples
        k222_counts = [0] * num_k222
        num_viol = 0
        count = 0
        
        if strategy == 'degree_desc':
            degrees = [len(triple_to_k222[i]) for i in range(num_triples)]
            order = list(range(num_triples))
            order.sort(key=lambda x: degrees[x], reverse=True)
            rng.shuffle(order)
        elif strategy == 'degree_asc':
            degrees = [len(triple_to_k222[i]) for i in range(num_triples)]
            order = list(range(num_triples))
            order.sort(key=lambda x: degrees[x])
            rng.shuffle(order)
        else:
            order = list(range(num_triples))
            rng.shuffle(order)
        
        for ti in order:
            ok = True
            for cid in triple_to_k222[ti]:
                if k222_counts[cid] >= 7:
                    ok = False
                    break
            if ok:
                present[ti] = True
                count += 1
                num_viol = update_counts(ti, 1, k222_counts, num_viol)
                
        return present, k222_counts, count, num_viol

    def simulated_annealing(present, k222_counts, count, num_viol, rng, time_limit):
        # We want to maximize count, subject to num_viol == 0.
        # Score = count if num_viol == 0, else count - 1000 * num_viol
        # Actually, simpler: Score = count - 1000 * num_viol.
        # Since we start valid, we want to keep it valid while increasing count.
        
        current_score = count * 1.0 - num_viol * 1000.0
        best_score = current_score
        best_present = present[:]
        best_count = count
        best_viol = num_viol
        
        # Temperature schedule
        T = 10.0
        cooling_rate = 0.999
        min_T = 0.05
        
        steps = 0
        stuck_counter = 0
        max_stuck = 1000
        
        while time.time() < time_limit:
            if time.time() > deadline:
                break
                
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to Add a triple
                # Optimization: Sample from absent triples? 
                # Just sample random, check if present.
                ti = rng.randint(0, num_triples - 1)
                if present[ti]:
                    continue
                
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 7:
                        delta_viol += 1
                
                delta_score = 1.0 - delta_viol * 1000.0
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti] = True
                    count += 1
                    num_viol = update_counts(ti, 1, k222_counts, num_viol)
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
                        
            elif move_type < 0.8:
                # Try to Remove a triple
                ti = rng.randint(0, num_triples - 1)
                if not present[ti]:
                    continue
                
                delta_viol = 0
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] == 8:
                        delta_viol -= 1
                
                delta_score = -1.0 - delta_viol * 1000.0
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[ti] = False
                    count -= 1
                    num_viol = update_counts(ti, -1, k222_counts, num_viol)
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
            else:
                # Swap: Remove one, Add one
                # This is more powerful as it can escape local optima where 
                # single moves don't change the score (e.g. removing a 'bad' triple 
                # and adding a 'good' one might net 0 score change in count but 
                # change the structure).
                
                # Pick ti1 to remove, ti2 to add
                ti1 = rng.randint(0, num_triples - 1)
                ti2 = rng.randint(0, num_triples - 1)
                
                if ti1 == ti2:
                    continue
                if present[ti1] == present[ti2]:
                    continue
                
                # Ensure ti1 is present and ti2 is absent
                if present[ti1]:
                    remove_ti = ti1
                    add_ti = ti2
                else:
                    remove_ti = ti2
                    add_ti = ti1
                    
                # Calculate delta for removing remove_ti
                delta_viol_remove = 0
                for cid in triple_to_k222[remove_ti]:
                    if k222_counts[cid] == 8:
                        delta_viol_remove -= 1
                
                # Calculate delta for adding add_ti
                # Note: k222_counts are current. 
                # If add_ti and remove_ti share a config, the count changes.
                # Let's compute effective count for add_ti after removal.
                delta_viol_add = 0
                for cid in triple_to_k222[add_ti]:
                    current_c = k222_counts[cid]
                    # If remove_ti is in this config, count drops by 1
                    if cid in triple_to_k222[remove_ti]:
                        current_c -= 1
                    if current_c == 7:
                        delta_viol_add += 1
                        
                delta_viol = delta_viol_remove + delta_viol_add
                # Count stays same (remove 1, add 1)
                delta_score = 0.0 - delta_viol * 1000.0
                
                if delta_score > 0 or rng.random() < math.exp(delta_score / T):
                    present[remove_ti] = False
                    present[add_ti] = True
                    num_viol = update_counts(remove_ti, -1, k222_counts, num_viol)
                    num_viol = update_counts(add_ti, 1, k222_counts, num_viol)
                    current_score += delta_score
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_present = present[:]
                        best_count = count
                        best_viol = num_viol
                        stuck_counter = 0
                    else:
                        stuck_counter += 1
            
            T *= cooling_rate
            if T < min_T:
                T = min_T
                if stuck_counter > max_stuck:
                    # Reheat
                    T = 5.0
                    stuck_counter = 0
            else:
                if stuck_counter > max_stuck:
                    # Small reheat
                    T = max(T * 1.5, 1.0)
                    stuck_counter = 0

        return best_present, best_count, best_viol

    def polish(present, k222_counts, count, num_viol, rng, time_limit):
        """Greedy polish: remove violations, then add valid triples."""
        while time.time() < time_limit:
            if time.time() > deadline:
                break
            
            # 1. Remove violations
            violated = [i for i in range(num_k222) if k222_counts[i] >= 8]
            if violated:
                cid = rng.choice(violated)
                config_triples = k222_configs[cid]
                candidates = [ti for ti in config_triples if present[ti]]
                if candidates:
                    # Remove the one with highest "impact" (most conflicts) or random?
                    # Random is fine, or remove the one that appears in most other violated configs.
                    ti = rng.choice(candidates)
                    present[ti] = False
                    count -= 1
                    num_viol = update_counts(ti, -1, k222_counts, num_viol)
                    continue
            
            # 2. Add valid triples
            added = False
            # Try to find a triple that can be added
            # Optimization: Iterate through all triples? 560 is small.
            # Or random sampling? Random might miss the only valid one.
            # Let's do a full scan if we are stuck, otherwise random.
            # For polish, let's do a full scan to be sure.
            for ti in range(num_triples):
                if present[ti]:
                    continue
                ok = True
                for cid in triple_to_k222[ti]:
                    if k222_counts[cid] >= 7:
                        ok = False
                        break
                if ok:
                    present[ti] = True
                    count += 1
                    num_viol = update_counts(ti, 1, k222_counts, num_viol)
                    added = True
                    break
            
            if not added:
                break
        
        return present, k222_counts, count, num_viol

    best_global_count = 0
    best_global_result = []
    
    # Strategies for restarts
    strategies = ['random', 'degree_desc', 'degree_asc']
    
    # We will run multiple rounds of SA with perturbations.
    # Time budget allocation:
    # 1. Initial Greedy + SA
    # 2. Perturbation + SA (VNS style)
    
    total_time_budget = deadline - time.time()
    
    # Divide time into blocks
    num_blocks = 5
    block_time = total_time_budget / num_blocks
    
    for b in range(num_blocks):
        if time.time() > deadline:
            break
        
        block_deadline = time.time() + block_time
        if b < num_blocks - 1:
            block_deadline = min(block_deadline, deadline - 5.0)
        
        rng = random.Random(1000 + b * 13)
        
        # 1. Generate a starting solution
        if b == 0:
            # First block: try different greedy inits
            present, k222_counts, count, num_viol = greedy_init(rng, 'random')
        else:
            # Subsequent blocks: Perturb the best solution found so far
            if best_global_count > 0 and best_global_result:
                # Reconstruct present from best_global_result
                present = [False] * num_triples
                for t in best_global_result:
                    idx = triple_to_idx[tuple(t)]
                    present[idx] = True
                
                k222_counts = [0] * num_k222
                count = 0
                num_viol = 0
                for i in range(num_triples):
                    if present[i]:
                        count += 1
                        num_viol = update_counts(i, 1, k222_counts, num_viol)
                
                # Perturb: Remove k random triples
                k_remove = rng.randint(3, 8)
                present_indices = [i for i in range(num_triples) if present[i]]
                rng.shuffle(present_indices)
                for _ in range(min(k_remove, len(present_indices))):
                    ti = present_indices[_]
                    present[ti] = False
                    count -= 1
                    num_viol = update_counts(ti, -1, k222_counts, num_viol)
            else:
                present, k222_counts, count, num_viol = greedy_init(rng, 'random')
        
        # 2. Run SA
        best_present, best_count, best_viol = simulated_annealing(
            present, k222_counts, count, num_viol, rng, block_deadline
        )
        
        # 3. Polish
        if best_viol == 0:
            # Rebuild counts for polish
            k222_counts_polish = [0] * num_k222
            count_polish = 0
            num_viol_polish = 0
            for i in range(num_triples):
                if best_present[i]:
                    count_polish += 1
                    num_viol_polish = update_counts(i, 1, k222_counts_polish, num_viol_polish)
            
            polish_deadline = min(time.time() + 2.0, deadline - 2.0)
            polished_present, polished_counts, polished_count, polished_viol = polish(
                best_present, k222_counts_polish, count_polish, num_viol_polish, rng, polish_deadline
            )
            
            if polished_count > best_global_count:
                best_global_count = polished_count
                best_global_result = [list(all_triples[i]) for i in range(num_triples) if polished_present[i]]
        
        if best_count > best_global_count and best_viol == 0:
            best_global_count = best_count
            best_global_result = [list(all_triples[i]) for i in range(num_triples) if best_present[i]]

    if not best_global_result:
        # Fallback
        present, k222_counts, count, num_viol = greedy_init(random.Random(0), 'random')
        best_global_result = [list(all_triples[i]) for i in range(num_triples) if present[i]]
        
    return best_global_result

