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

def _ga_improve(selected, balls, cover_count, deadline, margin=5.0):
    """
    Improved local search + genetic hybrid to reduce code size.
    Uses swap-based moves and incremental cover tracking.
    """
    total = len(cover_count)
    best_selected = list(selected)
    best_cover = cover_count.copy()
    best_size = len(best_selected)
    best_uncovered = int(np.sum(best_cover == 0))
    
    rng = random.Random(42)
    
    def compute_cover(code):
        cov = np.zeros(total, dtype=np.int32)
        for c in code:
            cov[balls[c]] += 1
        return cov
    
    # Phase 1: Aggressive local search with swap moves
    # Try to replace each codeword with one that covers more uncovered/under-covered words
    cur_selected = list(best_selected)
    cur_cover = best_cover.copy()
    cur_size = len(cur_selected)
    
    # Build a set for O(1) membership
    cur_set = set(cur_selected)
    
    improved = True
    local_iter = 0
    while improved and time.time() < deadline - margin - 5.0:
        improved = False
        local_iter += 1
        
        # Try removing each codeword if all its neighbors are still covered
        for i in range(len(cur_selected) - 1, -1, -1):
            if time.time() > deadline - margin - 5.0:
                break
            c = cur_selected[i]
            if np.min(cur_cover[balls[c]]) > 1:
                cur_cover[balls[c]] -= 1
                cur_selected.pop(i)
                cur_set.discard(c)
                cur_size -= 1
                improved = True
        
        # Try swap: remove one, add another that helps
        if not improved and cur_size > 1:
            # Find words with high redundancy
            # For each codeword, compute how many unique words it covers
            # Try swapping low-utility words with high-utility words
            # Simplified: try random swaps
            max_swap_attempts = 50
            for _ in range(max_swap_attempts):
                if time.time() > deadline - margin - 5.0:
                    break
                if not cur_selected:
                    break
                rm_idx = rng.randint(0, len(cur_selected) - 1)
                rm_c = cur_selected[rm_idx]
                
                # Temporarily remove
                cur_cover[balls[rm_c]] -= 1
                cur_selected.pop(rm_idx)
                cur_set.discard(rm_c)
                cur_size -= 1
                
                # Find uncovered words
                uncovered = (cur_cover == 0)
                if np.any(uncovered):
                    # Try to add a word that covers many uncovered
                    unc_indices = np.where(uncovered)[0]
                    if len(unc_indices) > 0:
                        # Score all words by how many uncovered they cover
                        # But this is expensive, so sample
                        sample_size = min(200, total)
                        candidates = rng.sample(range(total), sample_size)
                        best_add = -1
                        best_gain = 0
                        for cand in candidates:
                            if cand in cur_set:
                                continue
                            gain = int(np.sum(uncovered[balls[cand]]))
                            if gain > best_gain:
                                best_gain = gain
                                best_add = cand
                        
                        if best_add >= 0 and best_gain >= 2:
                            cur_cover[balls[best_add]] += 1
                            cur_selected.append(best_add)
                            cur_set.add(best_add)
                            cur_size += 1
                            improved = True
                        else:
                            # Revert
                            cur_cover[balls[rm_c]] += 1
                            cur_selected.insert(rm_idx, rm_c)
                            cur_set.add(rm_c)
                            cur_size += 1
                    else:
                        # Revert
                        cur_cover[balls[rm_c]] += 1
                        cur_selected.insert(rm_idx, rm_c)
                        cur_set.add(rm_c)
                        cur_size += 1
                else:
                    # No uncovered, keep removal
                    improved = True
        
        # Update best
        if _cur_uncovered_check(cur_cover) == 0 and cur_size < best_size:
            best_size = cur_size
            best_selected = list(cur_selected)
            best_cover = cur_cover.copy()
            best_uncovered = 0
    
    # Phase 2: Genetic improvement on the best found
    population_size = 15
    max_generations = 30
    
    # Initialize population
    population = [list(best_selected)]
    pop_covers = [best_cover.copy()]
    
    for _ in range(population_size - 1):
        pop = list(best_selected)
        k = rng.randint(2, min(8, max(1, len(pop) // 15)))
        if len(pop) > k:
            to_remove = rng.sample(pop, k)
            to_remove_set = set(to_remove)
            pop = [c for c in pop if c not in to_remove_set]
            temp_cover = compute_cover(pop)
            uncovered = (temp_cover == 0)
            _, uncovered = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin + 5)
            pop = _prune_code(pop, balls, temp_cover, deadline, margin=margin + 5)
        population.append(pop)
    
    for gen in range(max_generations):
        if time.time() > deadline - margin:
            break
        
        # Evaluate fitness
        scored_pop = []
        for code in population:
            cover = compute_cover(code)
            unc = int(np.sum(cover == 0))
            if unc > 0:
                f = len(code) + unc * 50
            else:
                f = len(code)
            scored_pop.append((f, unc, code, cover))
        
        scored_pop.sort(key=lambda x: (x[0], x[1]))
        
        # Update best
        best_f, best_unc, best_code, best_cov = scored_pop[0]
        if best_f < best_size or (best_f == best_size and best_unc < best_uncovered):
            best_size = best_f
            best_uncovered = best_unc
            best_selected = list(best_code)
            best_cover = best_cov.copy()
        
        # Selection
        def select():
            idxs = rng.sample(range(len(scored_pop)), min(3, len(scored_pop)))
            best_idx = min(idxs, key=lambda i: scored_pop[i][0])
            return scored_pop[best_idx][2]
        
        # Generate new population
        new_population = []
        # Elitism
        for i in range(min(3, len(scored_pop))):
            new_population.append(list(scored_pop[i][2]))
        
        while len(new_population) < population_size:
            if time.time() > deadline - margin:
                break
            p1 = select()
            p2 = select()
            
            # Uniform crossover
            set1 = set(p1)
            set2 = set(p2)
            union = set1 | set2
            child = [c for c in union if rng.random() < 0.6]
            
            # Mutation
            if child and rng.random() < 0.7:
                rm = rng.choice(child)
                child.remove(rm)
            if rng.random() < 0.7:
                if rng.random() < 0.5 and union:
                    add = rng.choice(list(union))
                else:
                    add = rng.randint(0, total - 1)
                if add not in child:
                    child.append(add)
            
            # Repair and Prune
            temp_cover = compute_cover(child)
            uncovered = (temp_cover == 0)
            _, uncovered = _greedy_cover(balls, temp_cover, uncovered, deadline, margin=margin + 10)
            if not uncovered.any():
                child = _prune_code(child, balls, temp_cover, deadline, margin=margin + 10)
            
            new_population.append(child)
        
        population = new_population
    
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
    
    # 3. Genetic Algorithm Improvement
    if time.time() < deadline - 12.0:
        selected, cover_count = _ga_improve(selected, balls, cover_count, deadline, margin=12.0)
    
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
