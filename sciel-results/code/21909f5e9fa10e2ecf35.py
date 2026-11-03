"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses randomized local search with multiple restarts, simulated annealing, and 2-opt moves.
    """
    if n == 0:
        return [[]]
    
    total_words = q ** n
    
    def int_to_word(val, n, q):
        word = []
        for _ in range(n):
            word.append(val % q)
            val //= q
        return word
    
    def word_to_int(word, q):
        val = 0
        for i in range(len(word) - 1, -1, -1):
            val = val * q + word[i]
        return val
    
    # Precompute neighbors as lists for faster iteration
    neighbors = []
    for w in range(total_words):
        word = int_to_word(w, n, q)
        neigh = {w}
        if radius >= 1:
            for pos in range(n):
                for val in range(q):
                    if val == word[pos]:
                        continue
                    new_word = word[:]
                    new_word[pos] = val
                    neigh.add(word_to_int(new_word, q))
        if radius >= 2:
            for pos1 in range(n):
                for pos2 in range(pos1 + 1, n):
                    for val1 in range(q):
                        if val1 == word[pos1]:
                            continue
                        for val2 in range(q):
                            if val2 == word[pos2]:
                                continue
                            new_word = word[:]
                            new_word[pos1] = val1
                            new_word[pos2] = val2
                            neigh.add(word_to_int(new_word, q))
        neighbors.append(list(neigh))
    
    # Precompute neighbor sets for O(1) lookup
    neighbor_sets = [set(neigh) for neigh in neighbors]
    
    def greedy_init():
        """Fast greedy using priority queue approach."""
        covered = np.zeros(total_words, dtype=bool)
        codewords = []
        uncovered_count = total_words
        
        # Track how many uncovered words each candidate covers
        uncovered_cover_count = np.zeros(total_words, dtype=np.int32)
        for w in range(total_words):
            for u in neighbors[w]:
                uncovered_cover_count[w] += 1
        
        while uncovered_count > 0:
            # Find word with max new coverage
            best_word = -1
            best_cov = -1
            
            # Sample a subset for speed if too many candidates
            if total_words > 1000:
                candidates = random.sample(range(total_words), min(200, total_words))
            else:
                candidates = range(total_words)
            
            for w in candidates:
                if w in codewords:
                    continue
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_cov:
                    best_cov = new_cov
                    best_word = w
                if new_cov == uncovered_count:
                    break
            
            if best_word == -1:
                break
            
            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered_count -= 1
        
        return codewords
    
    def optimize(codewords, time_limit):
        """Optimize using 1-swap, 2-swap, and simulated annealing."""
        coverage_count = np.zeros(total_words, dtype=np.int32)
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def try_remove_redundant():
            removed_any = False
            changed = True
            while changed:
                changed = False
                indices = list(range(len(codewords)))
                random.shuffle(indices)
                for i in indices:
                    cw = codewords[i]
                    can_remove = True
                    for u in neighbors[cw]:
                        if coverage_count[u] == 1:
                            can_remove = False
                            break
                    if can_remove:
                        for u in neighbors[cw]:
                            coverage_count[u] -= 1
                        codewords.pop(i)
                        cw_set.discard(cw)
                        changed = True
                        removed_any = True
                        break
            return removed_any
        
        try_remove_redundant()
        
        start_time = time.time()
        temperature = 5.0
        cooling = 0.995
        best_size = len(codewords)
        no_improve_counter = 0
        
        while time.time() - start_time < time_limit:
            if try_remove_redundant():
                continue
                
            if len(codewords) == 0:
                break
            
            move_type = random.random()
            
            if move_type < 0.6:
                # 1-swap: remove one, add one
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                
                would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
                
                if len(would_uncover) == 0:
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    codewords.pop(idx)
                    cw_set.discard(old_cw)
                    no_improve_counter = 0
                    continue
                
                # Find candidates that cover all would_uncover
                first_uncovered = would_uncover[0]
                potential_candidates = [c for c in neighbors[first_uncovered] if c not in cw_set and c != old_cw]
                random.shuffle(potential_candidates)
                
                best_new_cw = -1
                best_score = -1
                
                for new_cw in potential_candidates[:50]:
                    valid = all(u in neighbor_sets[new_cw] for u in would_uncover)
                    if not valid:
                        continue
                    
                    # Score: net change in total coverage redundancy
                    score = 0
                    for u in neighbors[new_cw]:
                        if u not in neighbor_sets[old_cw]:
                            score += 1
                        else:
                            score -= 1
                    
                    if score > best_score:
                        best_score = score
                        best_new_cw = new_cw
                
                if best_new_cw != -1:
                    delta = -best_score  # negative delta means improvement
                    # Simulated annealing: accept if better or with probability
                    if delta < 0 or random.random() < np.exp(-delta / max(temperature, 0.01)):
                        for u in neighbors[old_cw]:
                            coverage_count[u] -= 1
                        for u in neighbors[best_new_cw]:
                            coverage_count[u] += 1
                        codewords[idx] = best_new_cw
                        cw_set.discard(old_cw)
                        cw_set.add(best_new_cw)
                        temperature *= cooling
                        if len(codewords) < best_size:
                            best_size = len(codewords)
                            no_improve_counter = 0
                        else:
                            no_improve_counter += 1
                else:
                    no_improve_counter += 1
                    
            elif move_type < 0.85:
                # 2-swap: remove two, add two
                if len(codewords) < 2:
                    no_improve_counter += 1
                    continue
                
                idx1, idx2 = random.sample(range(len(codewords)), 2)
                old1, old2 = codewords[idx1], codewords[idx2]
                
                # Find words uniquely covered by old1 or old2
                would_uncover = set()
                for u in neighbors[old1]:
                    if coverage_count[u] == 1:
                        would_uncover.add(u)
                for u in neighbors[old2]:
                    if coverage_count[u] == 1 and u not in neighbors[old1]:
                        would_uncover.add(u)
                
                if not would_uncover:
                    for u in neighbors[old1]:
                        coverage_count[u] -= 1
                    for u in neighbors[old2]:
                        coverage_count[u] -= 1
                    codewords.pop(idx2)
                    codewords.pop(idx1)
                    cw_set.discard(old1)
                    cw_set.discard(old2)
                    no_improve_counter = 0
                    continue
                
                # Find two new codewords that cover all would_uncover
                first_uncovered = next(iter(would_uncover))
                candidates1 = [c for c in neighbors[first_uncovered] if c not in cw_set and c not in (old1, old2)]
                random.shuffle(candidates1)
                
                best_pair = None
                best_score = -1
                
                for new1 in candidates1[:20]:
                    for u in would_uncover:
                        if u not in neighbor_sets[new1]:
                            break
                    else:
                        # new1 covers all, try to find new2
                        remaining_uncover = set(would_uncover) - set(neighbors[new1])
                        if not remaining_uncover:
                            best_pair = (new1, -1)
                            best_score = 1000
                            break
                        else:
                            second_uncovered = next(iter(remaining_uncover))
                            candidates2 = [c for c in neighbors[second_uncovered] if c not in cw_set and c not in (old1, old2, new1)]
                            random.shuffle(candidates2)
                            for new2 in candidates2[:10]:
                                if all(u in neighbor_sets[new2] for u in remaining_uncover):
                                    score = 0
                                    for u in neighbors[new1] + neighbors[new2]:
                                        if u not in neighbor_sets[old1] and u not in neighbor_sets[old2]:
                                            score += 1
                                    if score > best_score:
                                        best_score = score
                                        best_pair = (new1, new2)
                            if best_pair and best_pair[1] != -1:
                                break
                
                if best_pair:
                    new1, new2 = best_pair
                    if new2 == -1:
                        # Only need one new codeword
                        for u in neighbors[old1]:
                            coverage_count[u] -= 1
                        for u in neighbors[old2]:
                            coverage_count[u] -= 1
                        for u in neighbors[new1]:
                            coverage_count[u] += 1
                        codewords[idx1] = new1
                        codewords.pop(idx2)
                        cw_set.discard(old1)
                        cw_set.discard(old2)
                        cw_set.add(new1)
                    else:
                        for u in neighbors[old1]:
                            coverage_count[u] -= 1
                        for u in neighbors[old2]:
                            coverage_count[u] -= 1
                        for u in neighbors[new1]:
                            coverage_count[u] += 1
                        for u in neighbors[new2]:
                            coverage_count[u] += 1
                        codewords[idx1] = new1
                        codewords[idx2] = new2
                        cw_set.discard(old1)
                        cw_set.discard(old2)
                        cw_set.add(new1)
                        cw_set.add(new2)
                    
                    temperature *= cooling
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1
                    
            else:
                # Random restart: replace a random codeword with a random one
                idx = random.randint(0, len(codewords) - 1)
                old_cw = codewords[idx]
                new_cw = random.randint(0, total_words - 1)
                
                if new_cw in cw_set or new_cw == old_cw:
                    no_improve_counter += 1
                    continue
                
                # Check if this maintains coverage
                would_uncover = [u for u in neighbors[old_cw] if coverage_count[u] == 1]
                if all(u in neighbor_sets[new_cw] for u in would_uncover):
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[new_cw]:
                        coverage_count[u] += 1
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                    temperature *= cooling
                    if len(codewords) < best_size:
                        best_size = len(codewords)
                        no_improve_counter = 0
                    else:
                        no_improve_counter += 1
                else:
                    no_improve_counter += 1
            
            # Reset temperature if stuck
            if no_improve_counter > 100:
                temperature = max(temperature * 1.5, 5.0)
                no_improve_counter = 0
        
        try_remove_redundant()
        return codewords
    
    # Multiple restarts
    start_time = time.time()
    time_limit = 115
    best_codewords = []
    
    while time.time() - start_time < time_limit:
        iter_time = min(15, time_limit - (time.time() - start_time))
        if iter_time <= 0:
            break
        
        cw = greedy_init()
        cw = optimize(cw, iter_time - 2)
        
        if len(cw) < len(best_codewords) or not best_codewords:
            best_codewords = cw
        
        if time.time() - start_time > time_limit - 5:
            break
    
    result = [int_to_word(cw, n, q) for cw in best_codewords]
    return result
# EVOLVE-BLOCK-END
