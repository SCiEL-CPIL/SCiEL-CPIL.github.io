# EVOLVE-BLOCK-START
import random
import numpy as np
from collections import defaultdict, deque
import time

def construct(q, n, radius):
    """
    Construct a covering code of length n over alphabet size q with covering radius radius.
    Uses a sampled greedy initialization followed by a local search with 1-swaps
    and random perturbations (adding a codeword) to escape local minima.
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

    # Precompute neighbors
    neighbors = [None] * total_words
    
    # Fast path for radius=1
    if radius == 1:
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            for pos in range(n):
                cur = word[pos]
                for val in range(q):
                    if val == cur:
                        continue
                    word[pos] = val
                    neigh.append(word_to_int(word, q))
                word[pos] = cur
            neighbors[w] = set(neigh)
    else:
        # General case (slower, but handles radius > 1)
        for w in range(total_words):
            neigh = [w]
            word = int_to_word(w, n, q)
            if radius >= 1:
                for pos in range(n):
                    cur = word[pos]
                    for val in range(q):
                        if val == cur:
                            continue
                        word[pos] = val
                        neigh.append(word_to_int(word, q))
                    word[pos] = cur
            if radius >= 2:
                # Re-fetch word as it was modified
                word = int_to_word(w, n, q)
                for pos1 in range(n):
                    c1 = word[pos1]
                    for val1 in range(q):
                        if val1 == c1:
                            continue
                        word[pos1] = val1
                        for pos2 in range(pos1 + 1, n):
                            c2 = word[pos2]
                            for val2 in range(q):
                                if val2 == c2:
                                    continue
                                word[pos2] = val2
                                neigh.append(word_to_int(word, q))
                            word[pos2] = c2
                    word[pos1] = c1
            neighbors[w] = set(neigh)

    def greedy_init():
        """
        Sampled Greedy: Repeatedly pick a word that covers the most uncovered words.
        To speed up, sample a subset of uncovered words and consider their neighbors as candidates.
        """
        covered = [False] * total_words
        codewords = []
        uncovered = set(range(total_words))
        
        while uncovered:
            best_word = -1
            best_new_cov = -1
            
            # Sample uncovered words to find candidates
            # If uncovered is large, sample a few; otherwise use all
            sample_size = min(200, len(uncovered))
            sample_uncovered = random.sample(list(uncovered), sample_size)
            
            # Candidates are neighbors of the sampled uncovered words
            candidates = set()
            for u in sample_uncovered:
                candidates.update(neighbors[u])
            
            # Evaluate candidates
            for w in candidates:
                new_cov = 0
                for u in neighbors[w]:
                    if not covered[u]:
                        new_cov += 1
                if new_cov > best_new_cov:
                    best_new_cov = new_cov
                    best_word = w
                    if new_cov == len(uncovered):
                        break
            
            if best_word == -1:
                # Fallback: pick a neighbor of any uncovered word
                u = next(iter(uncovered))
                best_word = next(iter(neighbors[u]))
                best_new_cov = 1 # At least covers itself/neighbor

            codewords.append(best_word)
            for u in neighbors[best_word]:
                if not covered[u]:
                    covered[u] = True
                    uncovered.discard(u)
                    
        return codewords

    def optimize(codewords, time_limit):
        """
        Local search with 1-swaps and random kicks.
        """
        coverage_count = [0] * total_words
        for cw in codewords:
            for u in neighbors[cw]:
                coverage_count[u] += 1
        
        cw_set = set(codewords)
        
        def remove_redundant():
            """Remove codewords that don't uniquely cover any word."""
            removed_any = False
            changed = True
            while changed:
                changed = False
                # Shuffle indices to avoid bias
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

        # Initial cleanup
        remove_redundant()
        
        start_time = time.time()
        last_improvement_time = start_time
        no_improvement_count = 0
        kick_threshold = 50 # Number of failed swaps before a kick
        
        while time.time() - start_time < time_limit:
            # 1. Try to remove redundant codewords
            if remove_redundant():
                no_improvement_count = 0
                last_improvement_time = time.time()
                continue
            
            if not codewords:
                break
            
            # 2. Try 1-Swap
            # Find codewords with smallest unique coverage
            candidates_to_swap = []
            for i, cw in enumerate(codewords):
                unique_words = []
                for u in neighbors[cw]:
                    if coverage_count[u] == 1:
                        unique_words.append(u)
                if unique_words:
                    candidates_to_swap.append((len(unique_words), i, unique_words))
            
            if not candidates_to_swap:
                # All codewords are redundant? Should have been caught by remove_redundant.
                # Or we are in a state where no single swap works, but maybe a kick helps.
                pass
            else:
                # Sort by size of unique coverage
                candidates_to_swap.sort(key=lambda x: x[0])
                
                swap_done = False
                # Try top few candidates
                for _, idx, unique_words in candidates_to_swap[:5]:
                    old_cw = codewords[idx]
                    
                    # Find new_cw covering all unique_words
                    # Intersection of neighbors of all unique_words
                    possible_new_cws = set(neighbors[unique_words[0]])
                    for u in unique_words[1:]:
                        if not possible_new_cws:
                            break
                        possible_new_cws &= set(neighbors[u])
                    
                    if not possible_new_cws:
                        continue
                    
                    # Must not be an existing codeword
                    valid_new_cws = [cw for cw in possible_new_cws if cw not in cw_set]
                    
                    if not valid_new_cws:
                        continue
                    
                    # Pick one randomly
                    new_cw = random.choice(valid_new_cws)
                    
                    # Perform swap
                    for u in neighbors[old_cw]:
                        coverage_count[u] -= 1
                    for u in neighbors[new_cw]:
                        coverage_count[u] += 1
                    
                    codewords[idx] = new_cw
                    cw_set.discard(old_cw)
                    cw_set.add(new_cw)
                    
                    swap_done = True
                    no_improvement_count = 0
                    last_improvement_time = time.time()
                    break
                
                if swap_done:
                    continue
            
            # 3. If no swap was made, increment no_improvement_count
            no_improvement_count += 1
            
            # 4. Kick: Add a random codeword to escape local minimum
            if no_improvement_count >= kick_threshold:
                # Pick a random word not in cw_set
                # To make it faster, we can pick from a precomputed list or just try random ints
                # Since total_words is 6561, and cw_set is small (~400-500), most random picks are valid
                new_cw = random.randint(0, total_words - 1)
                if new_cw in cw_set:
                    # Retry a few times
                    for _ in range(10):
                        new_cw = random.randint(0, total_words - 1)
                        if new_cw not in cw_set:
                            break
                    else:
                        # If still in set, skip kick (should be rare)
                        no_improvement_count = 0
                        continue
                
                # Add the codeword
                codewords.append(new_cw)
                cw_set.add(new_cw)
                for u in neighbors[new_cw]:
                    coverage_count[u] += 1
                
                # Reset counter, let the removal phase clean up
                no_improvement_count = 0
                last_improvement_time = time.time()
                
                # Optionally, run remove_redundant immediately to stabilize
                remove_redundant()

        # Final cleanup
        remove_redundant()
        
        return codewords

    # Run the algorithm
    initial_cw = greedy_init()
    # Use 110s for optimization to be safe
    final_cw = optimize(initial_cw, 110)
    
    result = [int_to_word(cw, n, q) for cw in final_cw]
    return result
# EVOLVE-BLOCK-END
