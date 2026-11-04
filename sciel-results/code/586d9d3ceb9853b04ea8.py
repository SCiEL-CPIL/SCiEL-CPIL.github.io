"""Football pool problem: ternary covering code K_3(6,1)."""
# EVOLVE-BLOCK-START
import time
import itertools
import numpy as np
import random

_DEADLINE = 145.0  # seconds, safety margin before 150s kill


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
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=n - 4):
            initial_code.append(h + list(tail))
    
    best_code = set(_word_to_int(c, q, n) for c in initial_code)
    best_size = len(best_code)
    
    def compute_coverage(code_set):
        cov = np.zeros(total, dtype=np.int32)
        for ci in code_set:
            for nb in balls[ci]:
                cov[nb] += 1
        return cov
    
    # Try multiple random removal orders
    for attempt in range(20):
        if time.time() - start_time > _DEADLINE - 5:
            break
        
        code_set = set(best_code) if attempt > 0 else set(_word_to_int(c, q, n) for c in initial_code)
        coverage = compute_coverage(code_set)
        
        # Greedy removal: repeatedly remove a codeword that is "most redundant"
        # A codeword is redundant if all words in its ball have coverage >= 2
        # Among redundant codewords, prefer the one whose removal causes least damage
        # (i.e., the one with the most "slack" in its ball)
        
        for _ in range(100):  # max iterations per attempt
            if time.time() - start_time > _DEADLINE - 2:
                break
            
            # Find all removable codewords
            removable = []
            for ci in code_set:
                min_cov = min(coverage[nb] for nb in balls[ci])
                if min_cov >= 2:
                    # Slack = how much extra coverage this codeword provides
                    slack = sum(coverage[nb] - 1 for nb in balls[ci])
                    removable.append((slack, ci))
            
            if not removable:
                break
            
            # Pick the one with minimum slack (least damage)
            removable.sort()
            _, ci_to_remove = removable[0]
            
            # Remove it
            for nb in balls[ci_to_remove]:
                coverage[nb] -= 1
            code_set.discard(ci_to_remove)
        
        # Try swaps: for each codeword, try replacing with a neighbor
        # that might enable further removals
        if time.time() - start_time < _DEADLINE - 5:
            for _ in range(50):
                if time.time() - start_time > _DEADLINE - 2:
                    break
                if len(code_set) >= best_size:
                    break
                
                # Pick a random codeword to try swapping
                ci = rng.choice(list(code_set))
                # Try a random neighbor not in code_set
                neighbors_of_ci = balls[ci]
                candidates = [nb for nb in neighbors_of_ci if nb not in code_set]
                if not candidates:
                    continue
                new_ci = rng.choice(candidates)
                
                # Try the swap
                code_set.discard(ci)
                for nb in balls[ci]:
                    coverage[nb] -= 1
                code_set.add(new_ci)
                for nb in balls[new_ci]:
                    coverage[nb] += 1
                
                # Check if this enables any removals
                improved = False
                for ci2 in list(code_set):
                    min_cov = min(coverage[nb] for nb in balls[ci2])
                    if min_cov >= 2:
                        for nb in balls[ci2]:
                            coverage[nb] -= 1
                        code_set.discard(ci2)
                        improved = True
                        break
                
                if not improved:
                    # Revert
                    code_set.discard(new_ci)
                    for nb in balls[new_ci]:
                        coverage[nb] -= 1
                    code_set.add(ci)
                    for nb in balls[ci]:
                        coverage[nb] += 1
        
        # Update best
        if len(code_set) < best_size:
            best_size = len(code_set)
            best_code = set(code_set)
    
    # Final: convert back to list of words
    result = [_int_to_word(ci, q, n) for ci in sorted(best_code)]
    return result
# EVOLVE-BLOCK-END
