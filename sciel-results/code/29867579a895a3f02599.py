import itertools
import random
import time
import sys

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin for 120s limit
    
    # Specific case for the task: q=3, n=6, radius=1
    if q == 3 and n == 6 and radius == 1:
        return _solve_q3_n6_r1(deadline)
    
    # Fallback for other parameters (simple product construction)
    if n <= 0:
        return [[]]
    
    # Generic greedy set cover for other cases
    all_words = list(itertools.product(range(q), repeat=n))
    space_size = q ** n
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    neighbors = [[] for _ in range(space_size)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
    
    # Greedy set cover
    covered = [False] * space_size
    code = []
    remaining = set(range(space_size))
    
    while remaining and time.time() < deadline - 5.0:
        # Find uncovered word with most potential
        best_idx = -1
        best_count = -1
        for idx in list(remaining):
            if not covered[idx]:
                count = sum(1 for w in neighbors[idx] if not covered[w])
                if count > best_count:
                    best_count = count
                    best_idx = idx
        if best_idx == -1:
            break
        code.append(all_words[best_idx])
        for w in neighbors[best_idx]:
            covered[w] = True
        remaining.discard(best_idx)
    
    return code

def _solve_q3_n6_r1(deadline):
    n = 6
    q = 3
    space_size = 729
    
    # Generate all words
    all_words = list(itertools.product(range(q), repeat=n))
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors (ball of radius 1) for each word
    neighbors = [[] for _ in range(space_size)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        for j in range(n):
            for val in range(q):
                if val == w[j]:
                    continue
                new_w = list(w)
                new_w[j] = val
                idx = word_to_idx[tuple(new_w)]
                neighbors[i].append(idx)
    
    # Initial code: Tetracode product (81 words)
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = []
    for h in tetracode:
        for tail in itertools.product(range(3), repeat=2):
            initial_code.append(h + list(tail))
    initial_indices = [word_to_idx[tuple(c)] for c in initial_code]
    
    best_code = initial_code
    best_size = len(best_code)
    
    # Branch and bound with symmetry breaking
    # Key insight: we can fix the all-zero word as a codeword (symmetry)
    # Then use greedy removal with multiple restarts
    
    rng = random.Random(42)
    zero_idx = word_to_idx[(0, 0, 0, 0, 0, 0)]
    
    # First, do greedy removal from the initial 81-word code
    # Then try to improve with local search
    
    # Greedy removal function
    def greedy_remove(indices, rng_seed=42):
        local_rng = random.Random(rng_seed)
        current = list(indices)
        local_rng.shuffle(current)
        
        cover_count = [0] * space_size
        for idx in current:
            for w_idx in neighbors[idx]:
                cover_count[w_idx] += 1
        
        active = set(current)
        
        changed = True
        while changed:
            changed = False
            for idx in list(active):
                redundant = True
                for w_idx in neighbors[idx]:
                    if cover_count[w_idx] <= 1:
                        redundant = False
                        break
                if redundant:
                    active.remove(idx)
                    for w_idx in neighbors[idx]:
                        cover_count[w_idx] -= 1
                    changed = True
        
        return list(active)
    
    # Try multiple random orderings
    for trial in range(50):
        if time.time() > deadline - 2.0:
            break
        
        current_indices = greedy_remove(initial_indices, rng_seed=trial * 7 + 13)
        
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_code = [all_words[i] for i in current_indices]
    
    # Local search: try to swap codewords to reduce size
    # Or try to add a codeword that allows removing two others
    for trial in range(30):
        if time.time() > deadline - 2.0:
            break
        
        # Start from best code so far
        current = list(best_code)
        current_indices = [word_to_idx[tuple(c)] for c in current]
        
        # Try removing one codeword and adding another
        improved = True
        while improved:
            improved = False
            for i in range(len(current_indices)):
                # Try removing current_indices[i]
                test_indices = [j for j in range(len(current_indices)) if j != i]
                
                # Check if still valid
                cover_count = [0] * space_size
                for idx in test_indices:
                    for w_idx in neighbors[idx]:
                        cover_count[w_idx] += 1
                
                if all(cover_count[w] >= 1 for w in range(space_size)):
                    # Valid! Remove it
                    current_indices.pop(i)
                    improved = True
                    break
        
        if len(current_indices) < best_size:
            best_size = len(current_indices)
            best_code = [all_words[i] for i in current_indices]
    
    return best_code
