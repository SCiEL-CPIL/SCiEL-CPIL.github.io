import itertools
import time
import random

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s margin for safety
    
    # Precompute neighbors for all words in the space
    # Space size is q^n. For q=3, n=8, size=6561.
    # Each word has 1 + n*(q-1) neighbors. For q=3, n=8: 1 + 16 = 17 neighbors.
    space_size = q ** n
    neighbors = [[] for _ in range(space_size)]
    
    def word_to_idx(w):
        idx = 0
        for d in range(n):
            idx = idx * q + w[d]
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for d in range(n - 1, -1, -1):
            w[d] = idx % q
            idx //= q
        return w

    # Build neighbor list
    for idx in range(space_size):
        w = idx_to_word(idx)
        neighbors[idx].append(idx) # self
        for i in range(n):
            for d in range(1, q):
                new_w = list(w)
                new_w[i] = (w[i] + d) % q
                neighbors[idx].append(word_to_idx(new_w))

    # Generate the initial 729-word code
    tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
    initial_code = [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
    
    # Convert to indices
    code_indices = set(word_to_idx(w) for w in initial_code)
    code_list = list(code_indices)
    
    # Build count array
    count = [0] * space_size
    for idx in code_list:
        for nb in neighbors[idx]:
            count[nb] += 1
            
    # Prune function: removes redundant codewords
    def prune(code_list, code_set, count):
        improved = True
        while improved and time.time() < deadline:
            improved = False
            # Shuffle to avoid bias
            random.shuffle(code_list)
            for idx in code_list:
                if time.time() > deadline:
                    break
                # Check if idx is redundant: all its neighbors are covered by >1 codeword
                # Actually, if removing idx leaves all neighbors with count >= 1, it's redundant.
                # Current count includes idx. So we need count[nb] - 1 >= 1 => count[nb] >= 2 for all nb in neighbors[idx]
                redundant = True
                for nb in neighbors[idx]:
                    if count[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    # Remove idx
                    code_set.remove(idx)
                    code_list.remove(idx)
                    for nb in neighbors[idx]:
                        count[nb] -= 1
                    improved = True
                    break
        return code_list, code_set, count

    # Initial pruning
    code_list, code_set, count = prune(code_list, code_set, count)
    best_size = len(code_list)
    best_code = code_list.copy()
    
    # Local Search with Swaps
    # We try to swap a codeword c with a non-codeword w.
    # If the swap maintains validity and allows removal of other codewords, we accept.
    
    # Precompute random candidates? No, just pick random idx from space.
    # To speed up, we can focus on neighbors of current codewords.
    
    for _ in range(1000): # Max iterations, guarded by time
        if time.time() > deadline - 5:
            break
        
        # Pick a random codeword to try swapping
        if not code_list:
            break
        c = random.choice(code_list)
        
        # Pick a candidate replacement w
        # Bias towards neighbors of c to maintain locality? Or random?
        # Random from space is 6561 options. Neighbors are 17.
        # Let's try random from space first, as it explores more.
        # But checking validity is O(17). 6561 * 17 is ~100k ops per step.
        # 1000 steps * 100k = 100M ops. Might be too slow in Python.
        # Let's restrict candidates to neighbors of c and maybe distance 2.
        # Or just try a few random candidates per c.
        
        candidates = []
        # Add neighbors of c
        candidates.extend(neighbors[c])
        # Add some random words
        for _ in range(5):
            candidates.append(random.randint(0, space_size - 1))
            
        random.shuffle(candidates)
        
        for w in candidates:
            if w in code_set:
                continue
                
            # Simulate swap: remove c, add w
            # Check if removing c breaks coverage
            # count[nb] becomes count[nb] - 1 + (1 if nb in neighbors[w] else 0)
            # We need new_count[nb] >= 1 for all nb.
            
            # Optimization: 
            # 1. Check if any neighbor of c has count 1. If so, w MUST cover it.
            # 2. Check if any neighbor of w (not in c) has count 0? No, count is global.
            
            # Let's just compute the change.
            # This is O(17 + 17).
            
            valid_swap = True
            # We need to check if any word becomes uncovered.
            # A word x is uncovered if count[x] - (1 if x in N(c) else 0) + (1 if x in N(w) else 0) == 0
            
            # Since we only change counts for N(c) and N(w), we only need to check those.
            affected = set(neighbors[c])
            affected.update(neighbors[w])
            
            for x in affected:
                old_count = count[x]
                delta = 0
                if x in neighbors[c]:
                    delta -= 1
                if x in neighbors[w]:
                    delta += 1
                new_count = old_count + delta
                if new_count < 1:
                    valid_swap = False
                    break
            
            if not valid_swap:
                continue
                
            # Swap is valid. Now check if it allows removal of other codewords.
            # Apply swap temporarily to count
            for x in affected:
                delta = 0
                if x in neighbors[c]:
                    delta -= 1
                if x in neighbors[w]:
                    delta += 1
                count[x] += delta
                
            # Try to prune
            # We can't just call the full prune because it modifies lists.
            # We check if any codeword (other than w) is redundant.
            # A codeword y is redundant if all its neighbors have count >= 2 (since y contributes 1).
            # Wait, count now includes w and excludes c.
            # So for a codeword y (y != w), if all nb in neighbors[y] have count[nb] >= 2, then y is redundant.
            
            removable = []
            for y in code_list:
                if y == c:
                    continue # c is being removed
                if y == w:
                    continue # w is being added
                redundant = True
                for nb in neighbors[y]:
                    if count[nb] == 1:
                        redundant = False
                        break
                if redundant:
                    removable.append(y)
                    
            # If we can remove at least one codeword, the size decreases.
            # If removable is empty, size stays same.
            # If we accept a same-size swap, we might get stuck.
            # Let's only accept if size decreases OR if it increases "slack".
            # Slack: number of words with count > 1.
            
            if removable:
                # Accept swap and removals
                code_set.remove(c)
                code_list.remove(c)
                code_set.add(w)
                code_list.append(w)
                
                for y in removable:
                    code_set.remove(y)
                    code_list.remove(y)
                    for nb in neighbors[y]:
                        count[nb] -= 1
                        
                if len(code_list) < best_size:
                    best_size = len(code_list)
                    best_code = code_list.copy()
                    
            else:
                # No immediate reduction. 
                # Should we accept? 
                # If we accept, we might escape a local minimum.
                # Let's accept with a small probability or if it improves slack.
                # Calculate slack change?
                # Old slack: sum(1 for x in affected if count[x] >= 2) ... complicated.
                # Let's just revert for now to keep it simple and fast.
                # Or, accept if random() < 0.1 (Simulated Annealing style)
                if random.random() < 0.05:
                    code_set.remove(c)
                    code_list.remove(c)
                    code_set.add(w)
                    code_list.append(w)
                else:
                    # Revert
                    for x in affected:
                        delta = 0
                        if x in neighbors[c]:
                            delta += 1
                        if x in neighbors[w]:
                            delta -= 1
                        count[x] += delta

    return [idx_to_word(idx) for idx in best_code]
