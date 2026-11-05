import time
import random
import itertools
import numpy as np

def construct(q, n, radius):
    # Handle the specific known case with a strong algorithm
    if q == 3 and n == 8 and radius == 1:
        return _construct_ternary_8_1()
    
    # Fallback for other cases: simple greedy covering
    return _fallback_construct(q, n, radius)


def _fallback_construct(q, n, radius):
    # Simple greedy set cover for general case
    all_words = list(itertools.product(range(q), repeat=n))
    total = len(all_words)
    
    if radius != 1:
        return [list(w) for w in all_words]
    
    covered = [False] * total
    codewords = []
    
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[w] = i
    
    covers = {}
    for i, w in enumerate(all_words):
        w_list = list(w)
        nbrs = [i]
        for pos in range(n):
            orig = w_list[pos]
            for val in range(q):
                if val != orig:
                    w_list[pos] = val
                    nbrs.append(word_to_idx[tuple(w_list)])
            w_list[pos] = orig
        covers[i] = set(nbrs)
        
    while not all(covered):
        best_idx = -1
        best_count = -1
        for i in range(total):
            if all(covered[j] for j in covers[i]):
                continue
            cnt = sum(1 for j in covers[i] if not covered[j])
            if cnt > best_count:
                best_count = cnt
                best_idx = i
        if best_idx == -1:
            break
        codewords.append(list(all_words[best_idx]))
        for j in covers[best_idx]:
            covered[j] = True
            
    return codewords


def _construct_ternary_8_1():
    q = 3
    n = 8
    radius = 1
    total = q ** n  # 6561
    
    # Precompute neighbors for all 6561 words using numpy for speed
    all_words = np.array(list(itertools.product(range(q), repeat=n)), dtype=np.int8)
    
    # Build a mapping from word tuple to index
    word_to_idx = {}
    for i, w in enumerate(all_words):
        word_to_idx[tuple(w)] = i
    
    neighbors = [None] * total
    for idx in range(total):
        w = all_words[idx]
        nbrs = [idx]
        for pos in range(n):
            orig = w[pos]
            for val in range(q):
                if val != orig:
                    w_new = w.copy()
                    w_new[pos] = val
                    nbrs.append(word_to_idx[tuple(w_new)])
        neighbors[idx] = nbrs
    
    # Start with greedy construction to get a good starting point
    cover_count = [0] * total
    uncovered = [True] * total
    
    code_list = []
    
    start_time = time.time()
    deadline = start_time + 145  # Leave 5s margin
    
    # Greedy phase: pick word with max uncovered coverage
    # Use numpy for faster counting
    while time.time() < deadline - 100:
        best_idx = -1
        best_count = -1
        for idx in range(total):
            cnt = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
            if cnt > best_count:
                best_count = cnt
                best_idx = idx
        if best_idx == -1 or best_count == 0:
            break
        code_list.append(best_idx)
        for nbr in neighbors[best_idx]:
            uncovered[nbr] = False
            cover_count[nbr] += 1
    
    # Check if we've covered everything
    if not any(uncovered):
        pass
    else:
        # Continue greedy until covered
        while any(uncovered) and time.time() < deadline - 80:
            best_idx = -1
            best_count = -1
            for idx in range(total):
                cnt = sum(1 for nbr in neighbors[idx] if uncovered[nbr])
                if cnt > best_count:
                    best_count = cnt
                    best_idx = idx
            if best_idx == -1 or best_count == 0:
                break
            code_list.append(best_idx)
            for nbr in neighbors[best_idx]:
                uncovered[nbr] = False
                cover_count[nbr] += 1
    
    # Now we have a valid code. Do local search to minimize size.
    code_set = set(code_list)
    
    # Recompute cover_count properly
    cover_count = [0] * total
    for c in code_list:
        for nbr in neighbors[c]:
            cover_count[nbr] += 1
    
    best_size = len(code_list)
    best_code = list(code_list)
    
    # Local search with removal and swap moves
    no_improvement_count = 0
    max_no_improvement = 100
    
    while time.time() < deadline - 10:
        # Phase 1: Try to remove words
        removed_any = False
        random.shuffle(code_list)
        
        for i in range(len(code_list) - 1, -1, -1):
            if time.time() > deadline - 15:
                break
            c = code_list[i]
            removable = True
            for nbr in neighbors[c]:
                if cover_count[nbr] <= 1:
                    removable = False
                    break
            if removable:
                code_list.pop(i)
                code_set.discard(c)
                for nbr in neighbors[c]:
                    cover_count[nbr] -= 1
                removed_any = True
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
        
        if removed_any:
            no_improvement_count = 0
            continue
        
        # Phase 2: Try swaps to escape local minimum
        # Pick a random word, remove it, find a replacement that covers uncovered words
        if len(code_list) > 450:
            idx_c = random.randint(0, len(code_list) - 1)
            c = code_list[idx_c]
            
            # Temporarily remove
            for nbr in neighbors[c]:
                cover_count[nbr] -= 1
            
            # Find uncovered words
            uncovered_words = [nbr for nbr in neighbors[c] if cover_count[nbr] == 0]
            
            if not uncovered_words:
                # Shouldn't happen, but if it does, keep the removal
                code_list.pop(idx_c)
                code_set.discard(c)
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = list(code_list)
                no_improvement_count = 0
                continue
            
            # Find a replacement word that covers all uncovered words
            # Optimization: The replacement must be a neighbor of every uncovered word.
            # We only need to check neighbors of the first uncovered word (max 17 candidates).
            found_swap = False
            if uncovered_words:
                first_uncovered = uncovered_words[0]
                # Candidates must be in neighbors[first_uncovered]
                for idx_cp in neighbors[first_uncovered]:
                    if idx_cp == c:
                        continue
                    if idx_cp in code_set:
                        continue
                    # Check if idx_cp covers all other uncovered words
                    all_covered = True
                    cp_nbrs_set = set(neighbors[idx_cp])
                    for u in uncovered_words:
                        if u not in cp_nbrs_set:
                            all_covered = False
                            break
                    if all_covered:
                        # Perform swap
                        code_list[idx_c] = idx_cp
                        code_set.discard(c)
                        code_set.add(idx_cp)
                        for nbr in neighbors[idx_cp]:
                            cover_count[nbr] += 1
                        found_swap = True
                        no_improvement_count = 0
                        break
            
            if not found_swap:
                # Revert
                for nbr in neighbors[c]:
                    cover_count[nbr] += 1
                no_improvement_count += 1
        
        if no_improvement_count > max_no_improvement:
            # Try a more aggressive restart: remove a random subset and rebuild
            if time.time() < deadline - 40:
                # Remove 10% of codewords randomly
                num_remove = max(1, len(code_list) // 10)
                to_remove = random.sample(code_list, num_remove)
                for c in to_remove:
                    if c in code_set:
                        code_set.discard(c)
                        for nbr in neighbors[c]:
                            cover_count[nbr] -= 1
                        code_list.remove(c)
                
                # Rebuild: greedily add words to cover uncovered
                while any(cover_count[x] == 0 for x in range(total)) and time.time() < deadline - 20:
                    best_idx = -1
                    best_count = -1
                    for idx in range(total):
                        if idx in code_set:
                            continue
                        cnt = sum(1 for nbr in neighbors[idx] if cover_count[nbr] == 0)
                        if cnt > best_count:
                            best_count = cnt
                            best_idx = idx
                    if best_idx == -1 or best_count == 0:
                        break
                    code_list.append(best_idx)
                    code_set.add(best_idx)
                    for nbr in neighbors[best_idx]:
                        cover_count[nbr] += 1
                
                no_improvement_count = 0
    
    # Return the best code found
    if len(best_code) < len(code_list):
        return [list(all_words[c]) for c in best_code]
    return [list(all_words[c]) for c in code_list]
