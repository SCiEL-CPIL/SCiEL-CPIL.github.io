import numpy as np
import time
from itertools import combinations
import random

def construct():
    N = 17
    MIN_DIST = 6
    start_time = time.time()
    time_limit = 58  # Leave margin for 60s limit
    
    # Precompute masks for distances 0 to 5
    masks_by_dist = [[] for _ in range(MIN_DIST)]
    for d in range(MIN_DIST):
        for combo in combinations(range(N), d):
            mask = 0
            for i in combo:
                mask |= (1 << i)
            masks_by_dist[d].append(mask)
    
    all_masks = []
    for d in range(MIN_DIST):
        all_masks.extend(masks_by_dist[d])
        
    # blocked_count[i] is the number of codewords within distance < MIN_DIST of i
    blocked_count = np.zeros(1 << N, dtype=np.int32)
    
    code = []
    
    def add_word(w):
        code.append(w)
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    def remove_word(w):
        code.remove(w)
        for mask in all_masks:
            blocked_count[w ^ mask] -= 1

    def greedy_construction(order):
        nonlocal blocked_count, code
        # Reset
        blocked_count[:] = 0
        code = []
        
        for w in order:
            if blocked_count[w] == 0:
                add_word(w)
                
    def find_mis_in_set(S):
        # S is a numpy array of candidate vertices
        # We want the largest subset of S such that pairwise distance >= MIN_DIST
        # This is an Independent Set in the conflict graph induced by S
        # Conflict edge if distance < MIN_DIST
        
        if len(S) == 0:
            return []
        
        # Greedy MIS heuristic
        # Sort by degree in the induced subgraph (ascending)
        # To speed up, if S is small, we can be more exact.
        # For now, greedy is safe and fast.
        
        # Calculate degrees within S
        # This is O(|S|^2)
        # If |S| is large, this might be slow.
        # But S is usually small in this context.
        
        s_list = S.tolist()
        n_s = len(s_list)
        
        if n_s == 1:
            return s_list
            
        # Build adjacency list or matrix for S
        # adj[i][j] = True if d(s_i, s_j) < MIN_DIST
        # We can use bit_count
        
        # To optimize, we can just use a simple greedy:
        # Pick vertex with lowest degree, add to MIS, remove it and neighbors.
        
        # Let's implement a simple branch and bound for small n_s, else greedy.
        # Given time constraints, let's stick to a robust greedy with random restarts if needed?
        # Or just standard greedy.
        
        # Standard Greedy:
        # 1. Compute degrees.
        # 2. Pick min degree.
        # 3. Add to MIS.
        # 4. Remove from S and all its neighbors.
        
        # To avoid O(N^2) overhead if S is huge, we check size.
        # In practice, S is the set of words compatible with code\{c}.
        # In a code of size ~256, the density is 256/131072 ~ 0.002.
        # The number of words blocked by c is ~9400.
        # Most of these are also blocked by others.
        # So S is likely small.
        
        remaining = set(s_list)
        mis = []
        
        while remaining:
            # Calculate degree of each vertex in remaining
            # Degree = number of neighbors in remaining
            # This is O(|remaining|^2)
            # If |remaining| is large, this is slow.
            # Let's cap it or use a faster heuristic.
            
            # Heuristic: Pick random vertex? No, min degree is better.
            # Let's optimize degree calculation.
            # We can just pick a random vertex if |remaining| > 100?
            # Let's try to be smart.
            
            if len(remaining) <= 50:
                # Exact-ish or careful greedy
                degs = {}
                for v in remaining:
                    deg = 0
                    for u in remaining:
                        if v != u and np.bit_count(v ^ u) < MIN_DIST:
                            deg += 1
                    degs[v] = deg
                
                # Pick min degree
                min_v = min(degs, key=degs.get)
                mis.append(min_v)
                
                # Remove min_v and its neighbors
                to_remove = {min_v}
                for u in remaining:
                    if u != min_v and np.bit_count(min_v ^ u) < MIN_DIST:
                        to_remove.add(u)
                
                remaining -= to_remove
            else:
                # Fast greedy: pick random or first
                # Let's pick the one with lowest "potential"
                # Just pick random to avoid bias?
                # Or pick the one that conflicts with the fewest others?
                # Let's just pick a random vertex from remaining to be fast
                # This might yield a smaller MIS, but it's fast.
                # Actually, let's try to pick one with low degree approximately.
                # Sample a few and pick the best?
                
                # Let's just pick a random one for speed in large sets
                # The quality of MIS might drop, but we are looking for ANY improvement > 1.
                # If we find a pair, that's an improvement.
                
                # Let's try to find a pair first!
                # If we can find 2, we are good.
                # If we find 3, even better.
                
                # Let's just do a simple greedy:
                # Pick random v
                v = random.choice(list(remaining))
                mis.append(v)
                
                to_remove = {v}
                for u in remaining:
                    if u != v and np.bit_count(v ^ u) < MIN_DIST:
                        to_remove.add(u)
                
                remaining -= to_remove
                
        return mis

    best_code = []
    best_size = 0
    
    # Phase 1: Multiple Greedy Starts
    # Try deterministic first
    order = list(range(1 << N))
    greedy_construction(order)
    if len(code) > best_size:
        best_size = len(code)
        best_code = list(code)
        
    # Try a few random orders
    random.seed(42)
    num_restarts = 5
    for _ in range(num_restarts):
        if time.time() - start_time > time_limit - 10:
            break
        random.shuffle(order)
        greedy_construction(order)
        if len(code) > best_size:
            best_size = len(code)
            best_code = list(code)
            
    # Use the best code found so far for local search
    # Re-initialize state with best_code
    blocked_count[:] = 0
    code = list(best_code)
    for w in code:
        for mask in all_masks:
            blocked_count[w ^ mask] += 1
            
    # Phase 2: Local Search
    # Replace 1 word with MIS of newly free words
    
    no_improvement_count = 0
    max_no_improvement = 20
    
    while time.time() - start_time < time_limit:
        improved = False
        indices = list(range(len(code)))
        random.shuffle(indices)
        
        for idx in indices:
            if time.time() - start_time > time_limit - 2:
                break
            
            c = code[idx]
            
            # Temporarily remove c
            for mask in all_masks:
                blocked_count[c ^ mask] -= 1
            
            # Find all free words
            free_words = np.where(blocked_count == 0)[0]
            
            # We only care about words that were blocked by c?
            # Actually, any free word is a candidate.
            # But words that were free before removing c would have been added in greedy.
            # So free_words should be small.
            
            if len(free_words) > 1:
                # Find MIS in free_words
                # We want to replace c (1 word) with as many as possible from free_words
                # If MIS size > 1, we improve.
                
                # Optimization: If len(free_words) is small, find exact MIS?
                # Let's use the greedy MIS function.
                mis_words = find_mis_in_set(free_words)
                
                if len(mis_words) > 1:
                    # Perform swap
                    code.remove(c)
                    
                    for w in mis_words:
                        code.append(w)
                        for mask in all_masks:
                            blocked_count[w ^ mask] += 1
                    
                    improved = True
                    if len(code) > best_size:
                        best_size = len(code)
                        best_code = list(code)
                    
                    # Break to restart with new state
                    break
                else:
                    # No improvement, restore c
                    for mask in all_masks:
                        blocked_count[c ^ mask] += 1
            else:
                # Not enough candidates, restore c
                for mask in all_masks:
                    blocked_count[c ^ mask] += 1
            
        if not improved:
            no_improvement_count += 1
            if no_improvement_count > max_no_improvement:
                # Try to perturb: Swap 1 for 1
                # Pick a random codeword c and a random free word w
                # such that d(c, w) >= 6? No, w must be free.
                # If w is free, it is compatible with all others.
                # So we can swap c with w.
                # This doesn't change size, but might unlock future improvements.
                
                c_idx = random.randint(0, len(code) - 1)
                c = code[c_idx]
                
                # Find a free word w
                # Sample random words
                swapped = False
                for _ in range(50):
                    w = random.randint(0, (1 << N) - 1)
                    if blocked_count[w] == 0:
                        # Check if w is distinct from c (it is, since c is in code, blocked_count[c] > 0)
                        # Swap
                        code.remove(c)
                        for mask in all_masks:
                            blocked_count[c ^ mask] -= 1
                            
                        code.append(w)
                        for mask in all_masks:
                            blocked_count[w ^ mask] += 1
                            
                        swapped = True
                        break
                        
                if not swapped:
                    break
                else:
                    no_improvement_count = 0

    return best_code
