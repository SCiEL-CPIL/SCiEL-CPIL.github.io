import numpy as np
import itertools
import time
import random
from collections import defaultdict

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill
    
    # Handle the specific task parameters
    if q != 3 or n != 6 or radius != 1:
        return _generic_construct(q, n, radius)
    
    total_words = q ** n  # 729
    
    # Precompute all words
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute coverage: covers[c] = list of word indices within Hamming distance 1 of c
    covers = [None] * total_words
    for i, w in enumerate(all_words):
        cov = [i]
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                cov.append(word_to_idx[tuple(new_w)])
        covers[i] = cov
    
    # Precompute covered_by: for each word w, which codewords cover it
    covered_by = [set() for _ in range(total_words)]
    for c in range(total_words):
        for w in covers[c]:
            covered_by[w].add(c)
    
    # Start with tetracode direct sum: 81 codewords
    tetracode = [[a, b, (a + b) % q, (a + 2 * b) % q] for a in range(q) for b in range(q)]
    tails = list(itertools.product(range(q), repeat=n - 4))
    initial_indices = []
    for h in tetracode:
        for tail in tails:
            cw = tuple(h + list(tail))
            initial_indices.append(word_to_idx[cw])
    
    active = set(initial_indices)
    
    # Initialize coverage counts
    cover_count = np.zeros(total_words, dtype=np.int32)
    for cw in active:
        cover_count[covers[cw]] += 1
    
    best_size = len(active)
    best_active = set(active)
    
    def is_removable(cw):
        """Check if codeword cw can be removed without losing coverage."""
        for idx in covers[cw]:
            if cover_count[idx] == 1:
                return False
        return True
    
    def greedy_remove():
        """Remove all redundant codewords."""
        nonlocal active, cover_count
        changed = True
        while changed:
            changed = False
            to_check = list(active)
            random.shuffle(to_check)
            for cw in to_check:
                if cw in active and is_removable(cw):
                    active.remove(cw)
                    cover_count[covers[cw]] -= 1
                    changed = True
    
    # Initial greedy removal
    greedy_remove()
    if len(active) < best_size:
        best_size = len(active)
        best_active = set(active)
    
    # Simulated annealing local search
    steps = 0
    max_steps = 500000
    temperature = 1.0
    cooling_rate = 0.99995
    
    while time.time() < deadline and steps < max_steps:
        steps += 1
        
        # Periodic greedy removal
        if steps % 10 == 0:
            greedy_remove()
            if len(active) < best_size:
                best_size = len(active)
                best_active = set(active)
                if best_size <= 73:
                    break
        
        if len(active) < 2:
            break
        
        active_list = list(active)
        
        # --- Move type 1: Swap (replace one codeword with another) ---
        u = random.choice(active_list)
        
        # Find words uniquely covered by u
        unique_words = []
        for w in covers[u]:
            if cover_count[w] == 1:
                unique_words.append(w)
        
        if unique_words:
            # Candidate v must cover all unique_words
            candidate_set = covered_by[unique_words[0]].copy()
            for w in unique_words[1:]:
                candidate_set &= covered_by[w]
            candidate_set.discard(u)
            candidate_set -= active  # Don't swap with existing codewords
        else:
            # u is fully redundant
            candidate_set = set(range(total_words)) - active - {u}
        
        if candidate_set:
            # Pick a candidate, preferring ones that cover more currently under-covered words
            sample_size = min(20, len(candidate_set))
            candidates = random.sample(list(candidate_set), sample_size)
            best_v = None
            best_score = -1
            for v in candidates:
                # Score: number of words in covers[v] that have low coverage
                score = 0
                for w in covers[v]:
                    if cover_count[w] <= 2:
                        score += 1
                if score > best_score:
                    best_score = score
                    best_v = v
            
            if best_v is not None:
                # Verify the swap is valid
                valid = True
                for w in covers[u]:
                    if cover_count[w] == 1 and w not in covers[best_v]:
                        valid = False
                        break
                
                if valid:
                    # Calculate delta for simulated annealing
                    # New size would be same, but we want to minimize
                    delta = 0  # Size doesn't change with swap
                    
                    # Accept if it improves or with probability based on temperature
                    if delta < 0 or random.random() < np.exp(-delta / max(temperature, 1e-10)):
                        # Perform swap
                        active.remove(u)
                        cover_count[covers[u]] -= 1
                        active.add(best_v)
                        cover_count[covers[best_v]] += 1
                        
                        # Try to remove redundant codewords after swap
                        greedy_remove()
                        
                        if len(active) < best_size:
                            best_size = len(active)
                            best_active = set(active)
                            if best_size <= 73:
                                break
        
        # --- Move type 2: 2-for-1 (replace two codewords with one) ---
        if steps % 30 == 0 and len(active) > 2:
            # Pick a random pair
            c1, c2 = random.sample(active_list, 2)
            union_cov = set(covers[c1]) | set(covers[c2])
            
            # Find c3 that covers all of union_cov
            sample_word = random.choice(list(union_cov))
            candidate_c3s = covered_by[sample_word]
            found_c3 = -1
            for c3 in candidate_c3s:
                if c3 == c1 or c3 == c2:
                    continue
                if union_cov.issubset(covers[c3]):
                    found_c3 = c3
                    break
            
            if found_c3 != -1:
                # Perform 2-for-1: remove c1, c2, add c3
                active.remove(c1)
                active.remove(c2)
                cover_count[covers[c1]] -= 1
                cover_count[covers[c2]] -= 1
                active.add(found_c3)
                cover_count[covers[found_c3]] += 1
                
                greedy_remove()
                
                if len(active) < best_size:
                    best_size = len(active)
                    best_active = set(active)
                    if best_size <= 73:
                        break
        
        # --- Move type 3: Random perturbation (add a random codeword, then remove redundants) ---
        if steps % 50 == 0 and len(active) < total_words:
            v = random.randint(0, total_words - 1)
            if v not in active:
                active.add(v)
                cover_count[covers[v]] += 1
                greedy_remove()
                if len(active) < best_size:
                    best_size = len(active)
                    best_active = set(active)
        
        # Cool down
        temperature *= cooling_rate
    
    # Convert best_active back to list of lists
    final_code = []
    for idx in best_active:
        final_code.append(list(all_words[idx]))
    
    return final_code


def _generic_construct(q, n, radius):
    """Fallback for non-(3,6,1) parameters."""
    start_time = time.time()
    deadline = start_time + 145.0
    
    space_size = q ** n
    all_words = [tuple(w) for w in itertools.product(range(q), repeat=n)]
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute balls
    balls = []
    for i, w in enumerate(all_words):
        ball = set([i])
        for pos in range(n):
            for val in range(q):
                if val == w[pos]:
                    continue
                new_w = list(w)
                new_w[pos] = val
                ball.add(word_to_idx[tuple(new_w)])
        balls.append(ball)
    
    # Greedy set cover
    active = set()
    covered = set()
    remaining = set(range(space_size))
    
    while remaining and time.time() < deadline:
        best_c = -1
        best_count = -1
        for c in range(space_size):
            if c in active:
                continue
            count = len(balls[c] & remaining)
            if count > best_count:
                best_count = count
                best_c = c
        if best_c == -1 or best_count == 0:
            break
        active.add(best_c)
        covered |= balls[best_c]
        remaining -= balls[best_c]
    
    # Greedy removal
    cover_count = np.zeros(space_size, dtype=np.int32)
    for cw in active:
        cover_count[list(balls[cw])] += 1
    
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for cw in list(active):
            removable = True
            for idx in balls[cw]:
                if cover_count[idx] == 1:
                    removable = False
                    break
            if removable:
                active.remove(cw)
                cover_count[list(balls[cw])] -= 1
                changed = True
    
    return [list(all_words[idx]) for idx in active]
