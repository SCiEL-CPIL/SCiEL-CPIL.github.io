"""Football pool problem: ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

_DEADLINE = 110.0  # seconds, safety margin before 120s kill

def _word_to_int(word, q, n):
    """Convert a word (list of ints) to a single integer index."""
    idx = 0
    for v in word:
        idx = idx * q + v
    return idx

def _int_to_word(idx, q, n):
    """Convert integer index back to word (list of ints)."""
    word = [0] * n
    for i in range(n - 1, -1, -1):
        word[i] = idx % q
        idx //= q
    return word

def _build_balls(q, n, radius):
    """Precompute for each word the list of words within Hamming distance radius."""
    total = q ** n
    balls = []
    for w in range(total):
        word = _int_to_word(w, q, n)
        neighbors = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for v in range(q):
                    if v == orig:
                        continue
                    word[pos] = v
                    neighbors.append(_word_to_int(word, q, n))
                    word[pos] = orig
        balls.append(neighbors)
    return balls

def construct(q, n, radius):
    start_time = time.time()
    total = q ** n
    rng = random.Random(42)
    
    # Precompute balls
    balls = _build_balls(q, n, radius)
    
    # Initial code: tetracode sum (81 words)
    # For q=3, n=6, tetracode is 9 words of length 4.
    # Direct sum: 9 * 3^(6-4) = 9 * 9 = 81 words.
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(q), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    best_code = set(_word_to_int(c, q, n) for c in initial_code)
    best_size = len(best_code)
    
    def compute_coverage(code_set):
        cov = np.zeros(total, dtype=np.int32)
        for ci in code_set:
            for nb in balls[ci]:
                cov[nb] += 1
        return cov

    # We aim to find a code of size 73 or smaller.
    # Strategy:
    # 1. Greedy removal based on "slack" (how much extra coverage a word provides).
    # 2. Local search: try to swap a word with another to enable further removals.
    
    for attempt in range(30):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        # Start from the best found so far, or the initial code for the first attempt
        if attempt == 0:
            code_set = set(_word_to_int(c, q, n) for c in initial_code)
        else:
            code_set = set(best_code)
            
        coverage = compute_coverage(code_set)
        
        # Phase 1: Greedy Removal
        # Repeatedly remove the codeword with the highest "slack" that is still redundant.
        # Slack = sum(coverage[nb] - 1 for nb in ball(ci))
        # A word is redundant if min(coverage[nb] for nb in ball(ci)) >= 2.
        
        for _ in range(100):
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            best_ci = -1
            best_slack = -1
            found_redundant = False
            
            # We iterate over a shuffled list to break ties randomly
            candidates = list(code_set)
            rng.shuffle(candidates)
            
            for ci in candidates:
                min_cov = 0
                slack = 0
                redundant = True
                for nb in balls[ci]:
                    c_val = coverage[nb]
                    if c_val < min_cov or min_cov == 0:
                        min_cov = c_val if min_cov == 0 else min(min_cov, c_val)
                    if c_val < 2:
                        redundant = False
                        break
                    slack += c_val - 1
                
                if redundant:
                    found_redundant = True
                    if slack > best_slack:
                        best_slack = slack
                        best_ci = ci
            
            if not found_redundant:
                break
            
            # Remove the best candidate
            for nb in balls[best_ci]:
                coverage[nb] -= 1
            code_set.discard(best_ci)
            
            # If we hit the target size, we can break early from this attempt's removal phase
            if len(code_set) <= 73:
                break

        # Phase 2: Local Search (Swaps)
        # Try to replace a codeword with another to potentially enable more removals.
        # This helps escape local minima where no single word is redundant, but a swap makes one redundant.
        
        if time.time() - start_time < _DEADLINE - 10 and len(code_set) > 73:
            for _ in range(50):
                if time.time() - start_time > _DEADLINE - 5:
                    break
                
                # Pick a random codeword to try removing/swapping
                ci = rng.choice(list(code_set))
                
                # Try adding a random word not in the code, then see if we can remove ci or another
                # Actually, a simpler heuristic: try to replace ci with a neighbor that covers more "critical" words.
                # But for now, let's just try random swaps and see if they enable removals.
                
                # Find a candidate to swap with: a word not in the code
                # We can pick a random word from the space, or a neighbor of ci.
                # Neighbors of ci are more likely to be useful.
                neighbors_of_ci = balls[ci]
                candidates_to_add = [nb for nb in neighbors_of_ci if nb not in code_set]
                
                if not candidates_to_add:
                    continue
                
                new_ci = rng.choice(candidates_to_add)
                
                # Perform swap: remove ci, add new_ci
                for nb in balls[ci]:
                    coverage[nb] -= 1
                code_set.discard(ci)
                
                for nb in balls[new_ci]:
                    coverage[nb] += 1
                code_set.add(new_ci)
                
                # Try to remove any redundant words now
                removed_any = False
                for ci2 in list(code_set):
                    min_cov = min(coverage[nb] for nb in balls[ci2])
                    if min_cov >= 2:
                        for nb in balls[ci2]:
                            coverage[nb] -= 1
                        code_set.discard(ci2)
                        removed_any = True
                        break # Remove one at a time to keep it simple and fast
                
                # If we didn't improve (remove anything), revert the swap
                if not removed_any:
                    for nb in balls[new_ci]:
                        coverage[nb] -= 1
                    code_set.discard(new_ci)
                    
                    for nb in balls[ci]:
                        coverage[nb] += 1
                    code_set.add(ci)
        
        # Update best
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
            
            # If we found 73 or less, we can stop early as this is the known optimal/best.
            if best_size <= 73:
                break

    # Final: convert back to list of words
    result = [_int_to_word(ci, q, n) for ci in sorted(best_code)]
    return result
# EVOLVE-BLOCK-END
