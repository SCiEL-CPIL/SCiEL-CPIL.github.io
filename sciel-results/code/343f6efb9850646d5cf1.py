import numpy as np
import itertools
import time
import random

def construct(q, n, radius):
    # Deadline guard: 120s wall-clock limit, keep 20s safety margin
    start_time = time.time()
    deadline = start_time + 100  # 100s for computation, 20s buffer
    
    # Parameters
    N = q ** n
    
    # Helper functions
    def word_to_idx(word):
        idx = 0
        for i in range(n):
            idx = idx * q + word[i]
        return idx
    
    def idx_to_word(idx):
        word = [0] * n
        for i in range(n-1, -1, -1):
            word[i] = idx % q
            idx //= q
        return word
    
    # Precompute all words and their neighbors
    all_words = [idx_to_word(i) for i in range(N)]
    
    # Precompute neighbors for each word (words within Hamming distance <= radius)
    # For efficiency, store as list of lists
    neighbors = [[] for _ in range(N)]
    
    for c_idx in range(N):
        c_word = all_words[c_idx]
        neighbors[c_idx] = set([c_idx])
        
        # Generate all words within distance radius
        # For radius 1: self + one-position changes
        if radius == 1:
            for i in range(n):
                orig = c_word[i]
                for new_val in range(q):
                    if new_val != orig:
                        new_word = c_word.copy()
                        new_word[i] = new_val
                        neighbors[c_idx].add(word_to_idx(new_word))
        else:
            # General radius: use combinations
            for dist in range(1, radius + 1):
                for positions in itertools.combinations(range(n), dist):
                    for vals in itertools.product(range(q), repeat=dist):
                        if all(vals[j] == c_word[positions[j]] for j in range(dist)):
                            continue
                        new_word = c_word.copy()
                        for j, pos in enumerate(positions):
                            new_word[pos] = vals[j]
                        neighbors[c_idx].add(word_to_idx(new_word))
    
    # Convert to lists for faster iteration
    neighbors_list = [list(neighbors[i]) for i in range(N)]
    
    # Precompute ball_sets: frozenset of neighbors for O(1) subset checks
    ball_sets = [frozenset(neighbors_list[i]) for i in range(N)]
    
    # Greedy covering algorithm
    def greedy_cover():
        """Greedy set cover: repeatedly pick the codeword that covers the most uncovered words."""
        uncovered = set(range(N))
        code = []
        code_set = set()
        
        while uncovered:
            if time.time() > deadline:
                break
            
            best_code = None
            best_count = -1
            
            # For each candidate, count how many uncovered words it covers
            for c_idx in range(N):
                if c_idx in code_set:
                    continue
                
                count = 0
                for w_idx in neighbors_list[c_idx]:
                    if w_idx in uncovered:
                        count += 1
                
                if count > best_count:
                    best_count = count
                    best_code = c_idx
            
            if best_code is None:
                break
            
            code.append(best_code)
            code_set.add(best_code)
            # Remove covered words
            for w_idx in neighbors_list[best_code]:
                uncovered.discard(w_idx)
        
        return code
    
    # Local search: try to improve by removing redundant codewords or swapping
    def local_search(code):
        """Improve the code by trying to remove codewords or swap them for better ones."""
        code_set = set(code)
        uncovered = set(range(N))
        
        # Compute which words are covered
        for c_idx in code:
            for w_idx in neighbors_list[c_idx]:
                uncovered.discard(w_idx)
        
        # If not fully covered, add missing coverage first
        while uncovered:
            if time.time() > deadline:
                break
            best_code = None
            best_count = -1
            for c_idx in range(N):
                if c_idx in code_set:
                    continue
                count = sum(1 for w_idx in neighbors_list[c_idx] if w_idx in uncovered)
                if count > best_count:
                    best_count = count
                    best_code = c_idx
            if best_code is None:
                break
            code.append(best_code)
            code_set.add(best_code)
            for w_idx in neighbors_list[best_code]:
                uncovered.discard(w_idx)
        
        # Now try to remove redundant codewords
        improved = True
        while improved and time.time() < deadline:
            improved = False
            
            # Try removing each codeword
            for i in range(len(code)):
                c_idx = code[i]
                
                # Check if removing c_idx would leave any word uncovered
                # Count how many codewords cover each word
                cover_count = {}
                for c in code:
                    for w in neighbors_list[c]:
                        cover_count[w] = cover_count.get(w, 0) + 1
                
                # Check if c_idx is the only cover for any word
                is_redundant = True
                for w in neighbors_list[c_idx]:
                    if cover_count.get(w, 0) == 1:
                        is_redundant = False
                        break
                
                if is_redundant:
                    # Remove c_idx
                    code.pop(i)
                    code_set.discard(c_idx)
                    improved = True
                    break
        
        return code
    
    # Simulated annealing with pair swaps
    def simulated_annealing(code, max_time=60):
        """Simulated annealing to escape local minima via pair swaps."""
        sa_start = time.time()
        sa_deadline = min(sa_start + max_time, deadline)
        
        code = list(code)
        code_set = set(code)
        
        # Precompute coverage counts
        cover_count = {}
        for c in code:
            for w in neighbors_list[c]:
                cover_count[w] = cover_count.get(w, 0) + 1
        
        best_code = list(code)
        best_len = len(code)
        
        # Temperature schedule
        T = 10.0
        cooling_rate = 0.995
        min_T = 0.1
        
        while time.time() < sa_deadline and T > min_T:
            # Randomly pick two codewords to swap
            if len(code) < 2:
                break
            
            i, j = random.sample(range(len(code)), 2)
            c_i, c_j = code[i], code[j]
            
            # Try swapping: remove both, add two new random codewords
            new_candidates = []
            for _ in range(5):
                new_c = random.randint(0, N-1)
                if new_c not in code_set and new_c not in new_candidates:
                    new_candidates.append(new_c)
            
            if len(new_candidates) < 2:
                T *= cooling_rate
                continue
            
            new_c1, new_c2 = random.sample(new_candidates, 2)
            
            # Check if this swap maintains coverage
            # Words only covered by c_i or c_j must be covered by new_c1 or new_c2
            only_i = ball_sets[c_i] - ball_sets[c_j]
            only_j = ball_sets[c_j] - ball_sets[c_i]
            
            new_cover = ball_sets[new_c1] | ball_sets[new_c2]
            
            # Check coverage of words that were only covered by c_i
            uncovered_i = only_i - new_cover
            # Check coverage of words that were only covered by c_j
            uncovered_j = only_j - new_cover
            
            if len(uncovered_i) == 0 and len(uncovered_j) == 0:
                # Valid swap: accept it
                code[i] = new_c1
                code[j] = new_c2
                code_set.discard(c_i)
                code_set.discard(c_j)
                code_set.add(new_c1)
                code_set.add(new_c2)
                
                # Update cover_count
                for w in neighbors_list[c_i]:
                    cover_count[w] -= 1
                for w in neighbors_list[c_j]:
                    cover_count[w] -= 1
                for w in neighbors_list[new_c1]:
                    cover_count[w] = cover_count.get(w, 0) + 1
                for w in neighbors_list[new_c2]:
                    cover_count[w] = cover_count.get(w, 0) + 1
                
                # Try to remove redundant codewords
                improved = True
                while improved and time.time() < sa_deadline:
                    improved = False
                    for k in range(len(code)):
                        c_k = code[k]
                        is_redundant = True
                        for w in neighbors_list[c_k]:
                            if cover_count.get(w, 0) == 1:
                                is_redundant = False
                                break
                        if is_redundant:
                            code.pop(k)
                            code_set.discard(c_k)
                            for w in neighbors_list[c_k]:
                                cover_count[w] -= 1
                            improved = True
                            break
                
                if len(code) < best_len:
                    best_code = list(code)
                    best_len = len(code)
            
            T *= cooling_rate
        
        return best_code
    
    # Run greedy cover
    code = greedy_cover()
    
    # Run local search
    if time.time() < deadline:
        code = local_search(code)
    
    # Run simulated annealing if time permits
    if time.time() < deadline - 20:
        code = simulated_annealing(code, max_time=min(40, deadline - time.time() - 10))
    
    # Final local search: try swapping codewords
    if time.time() < deadline:
        # Try to replace each codeword with a better one
        for i in range(len(code)):
            if time.time() > deadline:
                break
            
            current = code[i]
            # Try replacing with a random codeword
            for _ in range(10):
                new_code = random.randint(0, N-1)
                if new_code in code:
                    continue
                
                # Test if replacing current with new_code maintains coverage
                test_code = code.copy()
                test_code[i] = new_code
                
                # Verify coverage
                covered = set()
                for c in test_code:
                    covered.update(neighbors_list[c])
                
                if len(covered) == N:
                    code = test_code
                    break
    
    # Verify the final code is valid
    covered = set()
    for c in code:
        covered.update(neighbors_list[c])
    
    if len(covered) < N:
        # Fallback: add missing coverage
        while len(covered) < N and time.time() < deadline:
            best_code = None
            best_count = -1
            for c_idx in range(N):
                if c_idx in code:
                    continue
                count = sum(1 for w in neighbors_list[c_idx] if w not in covered)
                if count > best_count:
                    best_count = count
                    best_code = c_idx
            if best_code is None:
                break
            code.append(best_code)
            covered.update(neighbors_list[best_code])
    
    # Tetracode fallback if something went wrong
    if not code:
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert indices to words
    result = [all_words[i] for i in code]
    return result
