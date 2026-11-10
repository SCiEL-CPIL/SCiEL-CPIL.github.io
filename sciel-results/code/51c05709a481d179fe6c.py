import itertools
import random
import time
import math
from collections import defaultdict

def construct(v, k, t):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 150s limit
    
    # 1. Precomputation
    points = list(range(v))
    all_blocks = list(itertools.combinations(points, k))
    n_blocks = len(all_blocks)
    
    all_triples = list(itertools.combinations(points, t))
    n_triples = len(all_triples)
    triple_index = {trip: i for i, trip in enumerate(all_triples)}
    
    # Map each block to the list of triple indices it covers
    block_triples = []
    for block in all_blocks:
        covered = []
        for trip in itertools.combinations(block, t):
            covered.append(triple_index[trip])
        block_triples.append(covered)
        
    # Precompute bitmask for each block for fast intersection and counting
    block_masks = [0] * n_blocks
    for i, trips in enumerate(block_triples):
        m = 0
        for t_idx in trips:
            m |= (1 << t_idx)
        block_masks[i] = m

    # 2. Greedy Initialization
    def greedy_init(rng):
        uncovered_mask = (1 << n_triples) - 1
        selected = []
        
        while uncovered_mask:
            best_gain = -1
            best_block = -1
            
            max_possible = uncovered_mask.bit_count()
            
            for i in range(n_blocks):
                inter = block_masks[i] & uncovered_mask
                if inter:
                    gain = inter.bit_count()
                    if gain > best_gain:
                        best_gain = gain
                        best_block = i
                        if best_gain == max_possible:
                            break
            
            if best_block == -1:
                break
                
            selected.append(best_block)
            uncovered_mask &= ~block_masks[best_block]
            
        return selected

    # Run a few greedy restarts to find a good initial solution
    best_initial = None
    best_initial_len = float('inf')
    
    for _ in range(5):
        if time.time() > deadline - 30:
            break
        rng = random.Random()
        sel = greedy_init(rng)
        if len(sel) < best_initial_len:
            best_initial_len = len(sel)
            best_initial = sel
            
    if best_initial is None:
        best_initial = greedy_init(random.Random())
        
    # 3. Greedy Reduction: Try to remove redundant blocks
    def greedy_reduce(block_set):
        if not block_set:
            return set()
        
        current_set = set(block_set)
        cover_count = [0] * n_triples
        for idx in current_set:
            for t_idx in block_triples[idx]:
                cover_count[t_idx] += 1
        
        changed = True
        while changed:
            changed = False
            for idx in list(current_set):
                redundant = True
                for t_idx in block_triples[idx]:
                    if cover_count[t_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    for t_idx in block_triples[idx]:
                        cover_count[t_idx] -= 1
                    current_set.remove(idx)
                    changed = True
        
        return current_set

    best_initial = greedy_reduce(set(best_initial))
    best_initial_len = len(best_initial)
    
    # 4. Simulated Annealing / Local Search
    current_set = set(best_initial)
    
    cover_count = [0] * n_triples
    for idx in current_set:
        for t_idx in block_triples[idx]:
            cover_count[t_idx] += 1
            
    uncovered_count = sum(1 for c in cover_count if c == 0)
    
    def calc_cost():
        if uncovered_count == 0:
            return len(current_set)
        else:
            return len(current_set) + 1000 * uncovered_count

    current_cost = calc_cost()
    best_set = set(current_set)
    best_cost = current_cost
    
    T = 5.0
    cooling_rate = 0.9995
    min_T = 0.1
    
    all_indices = list(range(n_blocks))
    
    while time.time() < deadline - 5:
        if T < min_T:
            T = 5.0
            
        move_type = random.random()
        delta_cost = 0
        old_block = -1
        new_block = -1
        valid_move = True
        
        if move_type < 0.3:
            if uncovered_count == 0:
                new_block = random.choice(all_indices)
                if new_block in current_set:
                    valid_move = False
                else:
                    for t_idx in block_triples[new_block]:
                        if cover_count[t_idx] == 0:
                            uncovered_count -= 1
                            delta_cost -= 1000
                        cover_count[t_idx] += 1
                    delta_cost += 1
                    current_set.add(new_block)
            else:
                best_gain = -1
                best_block = -1
                for _ in range(20):
                    candidate = random.choice(all_indices)
                    if candidate in current_set:
                        continue
                    gain = 0
                    for t_idx in block_triples[candidate]:
                        if cover_count[t_idx] == 0:
                            gain += 1
                    if gain > best_gain:
                        best_gain = gain
                        best_block = candidate
                if best_block == -1 or best_gain == 0:
                    valid_move = False
                else:
                    new_block = best_block
                    for t_idx in block_triples[new_block]:
                        if cover_count[t_idx] == 0:
                            uncovered_count -= 1
                            delta_cost -= 1000
                        cover_count[t_idx] += 1
                    delta_cost += 1
                    current_set.add(new_block)
                
        elif move_type < 0.6:
            if not current_set:
                valid_move = False
            else:
                best_candidate = -1
                for _ in range(10):
                    candidate = random.choice(list(current_set))
                    safe = True
                    for t_idx in block_triples[candidate]:
                        if cover_count[t_idx] <= 1:
                            safe = False
                            break
                    if safe:
                        best_candidate = candidate
                        break
                if best_candidate == -1:
                    valid_move = False
                else:
                    old_block = best_candidate
                    for t_idx in block_triples[old_block]:
                        cover_count[t_idx] -= 1
                        if cover_count[t_idx] == 0:
                            uncovered_count += 1
                            delta_cost += 1000
                    delta_cost -= 1
                    current_set.remove(old_block)
                    
        else:
            if not current_set:
                valid_move = False
            else:
                old_block = random.choice(list(current_set))
                
                would_uncover = []
                for t_idx in block_triples[old_block]:
                    if cover_count[t_idx] <= 1:
                        would_uncover.append(t_idx)
                
                if not would_uncover:
                    best_gain = -1
                    best_block = -1
                    for _ in range(30):
                        candidate = random.choice(all_indices)
                        if candidate in current_set:
                            continue
                        gain = 0
                        for t_idx in block_triples[candidate]:
                            if cover_count[t_idx] > 0:
                                if cover_count[t_idx] > 1 or t_idx not in block_triples[old_block]:
                                    gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = candidate
                    if best_block == -1:
                        valid_move = False
                    else:
                        new_block = best_block
                        for t_idx in block_triples[old_block]:
                            cover_count[t_idx] -= 1
                            if cover_count[t_idx] == 0:
                                uncovered_count += 1
                                delta_cost += 1000
                        for t_idx in block_triples[new_block]:
                            if cover_count[t_idx] == 0:
                                uncovered_count -= 1
                                delta_cost -= 1000
                            cover_count[t_idx] += 1
                        current_set.remove(old_block)
                        current_set.add(new_block)
                else:
                    best_gain = -1
                    best_block = -1
                    for _ in range(30):
                        candidate = random.choice(all_indices)
                        if candidate in current_set:
                            continue
                        gain = 0
                        for t_idx in block_triples[candidate]:
                            if cover_count[t_idx] == 0:
                                gain += 1
                        if gain > best_gain:
                            best_gain = gain
                            best_block = candidate
                    if best_block == -1 or best_gain == 0:
                        valid_move = False
                    else:
                        new_block = best_block
                        for t_idx in block_triples[old_block]:
                            cover_count[t_idx] -= 1
                            if cover_count[t_idx] == 0:
                                uncovered_count += 1
                                delta_cost += 1000
                        for t_idx in block_triples[new_block]:
                            if cover_count[t_idx] == 0:
                                uncovered_count -= 1
                                delta_cost -= 1000
                            cover_count[t_idx] += 1
                        current_set.remove(old_block)
                        current_set.add(new_block)

        if not valid_move:
            continue
            
        new_cost = current_cost + delta_cost
        
        if delta_cost < 0 or random.random() < math.exp(-delta_cost / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_set = set(current_set)
        else:
            if move_type < 0.3:
                current_set.remove(new_block)
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1
            elif move_type < 0.6:
                current_set.add(old_block)
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
            else:
                current_set.remove(new_block)
                current_set.add(old_block)
                for t_idx in block_triples[old_block]:
                    cover_count[t_idx] += 1
                    if cover_count[t_idx] == 0:
                        uncovered_count -= 1
                for t_idx in block_triples[new_block]:
                    cover_count[t_idx] -= 1
                    if cover_count[t_idx] == 0:
                        uncovered_count += 1

        T *= cooling_rate

    # 5. Final Polish: Apply greedy reduction to best_set
    best_set = greedy_reduce(best_set)
    
    # Verify best_set
    final_cover = [0] * n_triples
    for idx in best_set:
        for t_idx in block_triples[idx]:
            final_cover[t_idx] += 1
            
    if any(c == 0 for c in final_cover):
        current_cover = [0] * n_triples
        for idx in current_set:
            for t_idx in block_triples[idx]:
                current_cover[t_idx] += 1
        if not any(c == 0 for c in current_cover):
            best_set = current_set
        else:
            best_set = set(best_initial)

    final_blocks = [list(all_blocks[i]) for i in best_set]
    return final_blocks
