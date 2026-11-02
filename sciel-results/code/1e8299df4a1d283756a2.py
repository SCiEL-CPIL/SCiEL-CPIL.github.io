import time
import numpy as np
import itertools
import random

def _greedy_cover(balls, cover_count, current_uncovered, deadline, margin=5.0):
    """Add codewords greedily to cover all uncovered positions."""
    selected = []
    while np.any(current_uncovered) and time.time() < deadline - margin:
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        if max_count == 0:
            break
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
    return selected, current_uncovered

def _prune_code(selected, balls, cover_count, deadline, margin=5.0):
    """Remove redundant codewords (those whose removal keeps all words covered)."""
    changed = True
    while changed and time.time() < deadline - margin:
        changed = False
        new_selected = []
        for c in selected:
            if np.min(cover_count[balls[c]]) > 1:
                cover_count[balls[c]] -= 1
                changed = True
            else:
                new_selected.append(c)
        selected = new_selected
    return selected

def _local_search_sa(selected, balls, cover_count, deadline, margin=5.0, rng=None):
    """
    Simulated annealing local search with swap and delete moves.
    Uses larger candidate samples for better swap evaluation.
    """
    if rng is None:
        rng = random.Random(42)
    
    total = len(cover_count)
    best_selected = list(selected)
    best_cover = cover_count.copy()
    best_size = len(best_selected)
    
    cur_selected = list(best_selected)
    cur_cover = best_cover.copy()
    cur_set = set(cur_selected)
    
    # Temperature schedule
    t_start = 3.0
    t_end = 0.01
    sa_deadline = deadline - margin
    start_time = time.time()
    
    max_iters = 8000
    for it in range(max_iters):
        if time.time() > sa_deadline:
            break
        
        # Cooling schedule
        elapsed = time.time() - start_time
        total_budget = max(1.0, sa_deadline - start_time)
        progress = min(1.0, elapsed / total_budget)
        temperature = t_start * (t_end / t_start) ** progress
        
        cur_size = len(cur_selected)
        
        # Try a random move: delete or swap
        move_type = rng.random()
        
        if move_type < 0.3 and cur_size > 1:
            # Delete move: try removing a codeword
            rm_idx = rng.randint(0, cur_size - 1)
            rm_c = cur_selected[rm_idx]
            
            min_cov = np.min(cur_cover[balls[rm_c]])
            if min_cov > 1:
                # Safe deletion
                cur_cover[balls[rm_c]] -= 1
                cur_selected.pop(rm_idx)
                cur_set.discard(rm_c)
                
                if len(cur_selected) < best_size:
                    best_size = len(cur_selected)
                    best_selected = list(cur_selected)
                    best_cover = cur_cover.copy()
            else:
                uncovered_after = int(np.sum((cur_cover[balls[rm_c]] == 1)))
                delta_cost = uncovered_after
                if rng.random() < np.exp(-delta_cost / max(temperature, 1e-10)):
                    cur_cover[balls[rm_c]] -= 1
                    cur_selected.pop(rm_idx)
                    cur_set.discard(rm_c)
                    # Greedy repair
                    unc = (cur_cover == 0)
                    repair_count = 0
                    while np.any(unc) and repair_count < 5 and time.time() < sa_deadline:
                        counts = np.sum(unc[balls], axis=1)
                        mx = np.max(counts)
                        if mx == 0:
                            break
                        cands = np.where(counts == mx)[0]
                        add_c = cands[0]
                        cur_cover[balls[add_c]] += 1
                        cur_selected.append(add_c)
                        cur_set.add(add_c)
                        unc = (cur_cover == 0)
                        repair_count += 1
        else:
            # Swap move
            if cur_size < 2:
                continue
            
            rm_idx = rng.randint(0, cur_size - 1)
            rm_c = cur_selected[rm_idx]
            
            cur_cover[balls[rm_c]] -= 1
            cur_selected.pop(rm_idx)
            cur_set.discard(rm_c)
            
            unc_mask = (cur_cover == 0)
            sample_n = min(500, total)
            candidates = rng.sample(range(total), sample_n)
            
            best_add = -1
            best_gain = 0
            for cand in candidates:
                if cand in cur_set:
                    continue
                gain = int(np.sum(unc_mask[balls[cand]]))
                if gain > best_gain:
                    best_gain = gain
                    best_add = cand
            
            if best_add >= 0 and best_gain > 0:
                cur_cover[balls[best_add]] += 1
                cur_selected.append(best_add)
                cur_set.add(best_add)
                
                if len(cur_selected) < best_size:
                    best_size = len(cur_selected)
                    best_selected = list(cur_selected)
                    best_cover = cur_cover.copy()
            else:
                cur_cover[balls[rm_c]] += 1
                cur_selected.insert(rm_idx, rm_c)
                cur_set.add(rm_c)
    
    # Final pruning
    changed = True
    while changed and time.time() < sa_deadline:
        changed = False
        new_sel = []
        for c in cur_selected:
            if np.min(cur_cover[balls[c]]) > 1:
                cur_cover[balls[c]] -= 1
                changed = True
            else:
                new_sel.append(c)
        cur_selected = new_sel
    
    if len(cur_selected) < best_size:
        best_size = len(cur_selected)
        best_selected = list(cur_selected)
        best_cover = cur_cover.copy()
    
    return best_selected, best_cover

def _ga_improve(selected, balls, cover_count, deadline, margin=5.0):
    """
    Genetic improvement with SA local search refinement.
    """
    total = len(cover_count)
    best_selected = list(selected)
    best_cover = cover_count.copy()
    best_size = len(best_selected)
    
    rng = random.Random(123)
    
    def compute_cover(code):
        cov = np.zeros(total, dtype=np.int32)
        for c in code:
            cov[balls[c]] += 1
        return cov
    
    # Phase 1: SA local search
    if time.time() < deadline - margin - 10.0:
        best_selected, best_cover = _local_search_sa(
            best_selected, balls, best_cover, deadline, margin=margin + 10.0, rng=rng
        )
    
    # Phase 2: GA with local search on offspring
    population_size = 12
    max_generations = 20
    
    population = [list(best_selected)]
    
    for _ in range(population_size - 1):
        pop = list(best_selected)
        k = rng.randint(2, min(6, max(1, len(pop) // 20)))
        if len(pop) > k:
            to_remove = rng.sample(pop, k)
            to_remove_set = set(to_remove)
            pop = [c for c in pop if c not in to_remove_set]
            temp_cover = compute_cover(pop)
            uncovered = (temp_cover == 0)
            pop, _ = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin + 15)
            pop = _prune_code(pop, balls, temp_cover, deadline, margin=margin + 15)
        population.append(pop)
    
    for gen in range(max_generations):
        if time.time() > deadline - margin:
            break
        
        scored = []
        for code in population:
            cover = compute_cover(code)
            unc = int(np.sum(cover == 0))
            f = len(code) + unc * 100
            scored.append((f, unc, code, cover))
        
        scored.sort(key=lambda x: (x[0], x[1]))
        
        bf, bu, bc, bcv = scored[0]
        if bf < best_size:
            best_size = bf
            best_selected = list(bc)
            best_cover = bcv.copy()
        
        def select():
            idxs = rng.sample(range(len(scored)), min(3, len(scored)))
            best_i = min(idxs, key=lambda i: scored[i][0])
            return scored[best_i][2]
        
        new_pop = []
        for i in range(min(2, len(scored))):
            new_pop.append(list(scored[i][2]))
        
        while len(new_pop) < population_size:
            if time.time() > deadline - margin:
                break
            p1 = select()
            p2 = select()
            
            set1 = set(p1)
            set2 = set(p2)
            union = set1 | set2
            child = []
            for c in union:
                in1 = c in set1
                in2 = c in set2
                if in1 and in2:
                    if rng.random() < 0.7:
                        child.append(c)
                elif in1:
                    if rng.random() < 0.6:
                        child.append(c)
                else:
                    if rng.random() < 0.4:
                        child.append(c)
            
            if child and rng.random() < 0.5:
                rm = rng.choice(child)
                child.remove(rm)
            if rng.random() < 0.5:
                add = rng.randint(0, total - 1)
                if add not in child:
                    child.append(add)
            
            temp_cover = compute_cover(child)
            uncovered = (temp_cover == 0)
            child, _ = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin + 20)
            if not (temp_cover == 0).any():
                child = _prune_code(child, balls, temp_cover, deadline, margin=margin + 20)
            
            new_pop.append(child)
        
        population = new_pop
    
    return best_selected, best_cover

def _cur_uncovered_check(cover):
    return int(np.sum(cover == 0))

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1_optimized(deadline, start_time)
    
    # Fallbacks for other cases
    if q == 3 and n >= 4:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n == 4 and radius == 1:
            return tetracode
        
    # Generic fallback: return all words (valid format, worst case)
    return [list(c) for c in itertools.product(range(q), repeat=n)]


def _construct_ternary_8_1_optimized(deadline, start_time):
    n = 8
    q = 3
    total = q ** n  # 6561
    ball_size = 1 + 2 * n  # 17
    
    # Weights for positions (big-endian)
    weights = np.array([3 ** (n - 1 - i) for i in range(n)], dtype=np.int32)
    
    # Precompute balls for all words
    balls = np.zeros((total, ball_size), dtype=np.int32)
    for w in range(total):
        balls[w, 0] = w
        idx = 1
        for pos in range(n):
            weight = weights[pos]
            digit = (w // weight) % 3
            for new_digit in range(q):
                if new_digit == digit:
                    continue
                new_w = w + (new_digit - digit) * weight
                balls[w, idx] = new_w
                idx += 1
    
    # Precompute a mapping from word index to its position in the balls array
    # For quick lookup: given a word, which codewords cover it?
    # We'll maintain cover_count array dynamically.
    
    # 1. Greedy Construction with tie-breaking by word index for reproducibility
    cover_count = np.zeros(total, dtype=np.int32)
    selected = []
    current_uncovered = np.ones(total, dtype=bool)
    
    # Main greedy loop
    while np.any(current_uncovered):
        if time.time() > deadline - 45.0:
            break
            
        # Count uncovered neighbors for each word
        counts = np.sum(current_uncovered[balls], axis=1)
        max_count = np.max(counts)
        
        if max_count == 0:
            break
        
        # Pick candidate with max count; break ties by smallest index
        candidates = np.where(counts == max_count)[0]
        best_idx = candidates[0]
        
        selected.append(best_idx)
        cover_count[balls[best_idx]] += 1
        current_uncovered = (cover_count == 0)
        
    # Force completion if needed
    if np.any(current_uncovered) and time.time() < deadline - 10.0:
        while np.any(current_uncovered):
            uncovered_indices = np.where(current_uncovered)[0]
            if len(uncovered_indices) == 0:
                break
            best_idx = uncovered_indices[0]
            selected.append(best_idx)
            cover_count[balls[best_idx]] += 1
            current_uncovered = (cover_count == 0)
            if time.time() > deadline - 10.0:
                break
    
    # 2. Aggressive Pruning (Multiple passes in different orders)
    prune_deadline = deadline - 25.0
    for pass_num in range(6):
        if time.time() > prune_deadline:
            break
        changed = True
        while changed and time.time() < prune_deadline:
            changed = False
            new_selected = []
            # Alternate forward/reverse/shuffled order
            if pass_num % 3 == 0:
                order = selected
            elif pass_num % 3 == 1:
                order = list(reversed(selected))
            else:
                order = list(selected)
                random.Random(pass_num * 7 + 1).shuffle(order)
            
            for c in order:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
    
    # Multi-start: run greedy with random tie-breaking as second start
    if time.time() < deadline - 35.0:
        rng2 = random.Random(7)
        cover_count2 = np.zeros(total, dtype=np.int32)
        selected2 = []
        current_uncovered2 = np.ones(total, dtype=bool)
        
        while np.any(current_uncovered2) and time.time() < deadline - 35.0:
            counts = np.sum(current_uncovered2[balls], axis=1)
            max_count = np.max(counts)
            if max_count == 0:
                break
            candidates = np.where(counts == max_count)[0]
            best_idx = rng2.choice(candidates.tolist())
            selected2.append(best_idx)
            cover_count2[balls[best_idx]] += 1
            current_uncovered2 = (cover_count2 == 0)
        
        if np.any(current_uncovered2) and time.time() < deadline - 10.0:
            while np.any(current_uncovered2) and time.time() < deadline - 10.0:
                uncovered_indices = np.where(current_uncovered2)[0]
                if len(uncovered_indices) == 0:
                    break
                best_idx = uncovered_indices[0]
                selected2.append(best_idx)
                cover_count2[balls[best_idx]] += 1
                current_uncovered2 = (cover_count2 == 0)
        
        changed = True
        while changed and time.time() < deadline - 25.0:
            changed = False
            new_sel = []
            for c in selected2:
                if np.min(cover_count2[balls[c]]) > 1:
                    cover_count2[balls[c]] -= 1
                    changed = True
                else:
                    new_sel.append(c)
            selected2 = new_sel
        
        if len(selected2) < len(selected):
            selected = selected2
            cover_count = cover_count2
    
    # 3. SA + GA Improvement
    if time.time() < deadline - 15.0:
        selected, cover_count = _ga_improve(selected, balls, cover_count, deadline, margin=15.0)
    
    # 4. Final aggressive pruning
    if time.time() < deadline - 5.0:
        changed = True
        while changed and time.time() < deadline - 5.0:
            changed = False
            new_selected = []
            for c in selected:
                if np.min(cover_count[balls[c]]) > 1:
                    cover_count[balls[c]] -= 1
                    changed = True
                else:
                    new_selected.append(c)
            selected = new_selected
    
    # 5. Final Verification and Conversion
    final_cover = np.zeros(total, dtype=np.int32)
    for c in selected:
        final_cover[balls[c]] += 1
    
    if np.any(final_cover == 0):
        missing = np.where(final_cover == 0)[0]
        for m in missing:
            selected.append(m)
            final_cover[balls[m]] += 1

    # Convert to digit lists
    result = []
    for w in selected:
        digits = []
        temp = w
        for _ in range(n):
            digits.append(temp % 3)
            temp //= 3
        digits.reverse()
        result.append(digits)
    
    return result
