import itertools
import random
import time
import math

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0
    
    # Generate all words in {0, 1, ..., q-1}^n
    all_words = list(itertools.product(range(q), repeat=n))
    num_words = len(all_words)
    word_to_idx = {w: i for i, w in enumerate(all_words)}
    
    # Precompute neighbors for each word (Hamming distance <= radius)
    # For radius 1, neighbors are self + 1-step changes
    neighbors = [[] for _ in range(num_words)]
    for i, w in enumerate(all_words):
        neighbors[i].append(i)
        w_list = list(w)
        for pos in range(n):
            orig_val = w_list[pos]
            for new_val in range(q):
                if new_val == orig_val:
                    continue
                w_list[pos] = new_val
                new_w = tuple(w_list)
                neighbors[i].append(word_to_idx[new_w])
            w_list[pos] = orig_val

    # State variables
    coverage = [0] * num_words
    code = set()
    uncovered_count = num_words
    
    def add_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            if coverage[nb_idx] == 0:
                uncovered_count -= 1
            coverage[nb_idx] += 1
        code.add(w)
        
    def remove_word(w):
        nonlocal uncovered_count
        idx = word_to_idx[w]
        for nb_idx in neighbors[idx]:
            coverage[nb_idx] -= 1
            if coverage[nb_idx] == 0:
                uncovered_count += 1
        code.remove(w)
        
    def is_dominating():
        return uncovered_count == 0

    def try_remove(w):
        """Try to remove w if it doesn't break domination. Returns True if removed."""
        if w not in code:
            return False
        idx = word_to_idx[w]
        # Check if all neighbors are covered by at least 2 codewords (so removing 1 leaves >= 1)
        for nb_idx in neighbors[idx]:
            if coverage[nb_idx] <= 1:
                return False
        remove_word(w)
        return True

    # 1. Greedy Initialization
    # Start with empty set, add words that cover most uncovered vertices
    while not is_dominating():
        best_w = None
        best_gain = -1
        # To speed up, we can sample or use a heap, but 729 is small.
        # We just pick the one with max coverage of currently uncovered nodes.
        # Note: We need to calculate gain for each candidate.
        # Optimization: Keep track of uncovered nodes.
        
        # For simplicity and speed in Python, we iterate all words not in code.
        # This is O(N^2) per step in worst case, but N=729, steps~100. 729*729*100 ~ 50M ops. Might be slow in Python.
        # Let's use a heuristic: Pick a random uncovered node, find its neighbors, pick the one with highest degree in code?
        # Or just pick the neighbor of a random uncovered node that covers most uncovered nodes.
        
        # Let's try: Pick a random uncovered node u.
        # Candidates are neighbors of u.
        # Pick the candidate v that covers the most uncovered nodes.
        
        if uncovered_count > 0:
            # Find an uncovered node
            # We need a list of uncovered nodes to pick from.
            # Rebuilding list is O(N).
            # Since N=729, it's fine.
            uncovered_nodes = [i for i in range(num_words) if coverage[i] == 0]
            u = random.choice(uncovered_nodes)
            
            candidates = neighbors[word_to_idx[all_words[u]]]
            # Actually u is index.
            candidates = neighbors[u]
            
            best_cand = None
            max_covers = -1
            for v_idx in candidates:
                if all_words[v_idx] in code:
                    continue
                # Count how many uncovered nodes this v_idx covers
                covers = 0
                for nb in neighbors[v_idx]:
                    if coverage[nb] == 0:
                        covers += 1
                if covers > max_covers:
                    max_covers = covers
                    best_cand = all_words[v_idx]
            
            if best_cand is not None:
                add_word(best_cand)
            else:
                # Fallback: add any neighbor
                if candidates:
                    v_idx = random.choice(candidates)
                    add_word(all_words[v_idx])
                else:
                    # Should not happen
                    add_word(all_words[0])

    best_code = set(code)
    best_size = len(code)
    
    # 2. Local Search
    no_improvement_count = 0
    
    while time.time() < deadline:
        # Prune phase
        improved = False
        while True:
            # Find a removable word
            # Iterate through a copy of code to avoid modification during iteration
            removable = False
            for w in list(code):
                if try_remove(w):
                    improved = True
                    removable = True
                    break # Restart scan after removal
            if not removable:
                break
        
        if improved:
            if len(code) < best_size:
                best_code = set(code)
                best_size = len(code)
            no_improvement_count = 0
        else:
            no_improvement_count += 1
            
        # If stuck, try swaps
        if no_improvement_count > 10:
            # Try a swap
            if code:
                w_old = random.choice(list(code))
                # Pick a random word not in code
                # To avoid O(N) scan for random choice, we can just pick random index and check
                w_new = None
                for _ in range(100): # Try 100 times to find a valid swap candidate
                    idx_new = random.randint(0, num_words - 1)
                    candidate = all_words[idx_new]
                    if candidate not in code:
                        w_new = candidate
                        break
                
                if w_new is not None:
                    # Check if swap is valid
                    idx_old = word_to_idx[w_old]
                    idx_new = word_to_idx[w_new]
                    
                    # Simulate swap
                    # Remove old
                    old_cov_changes = {}
                    for nb in neighbors[idx_old]:
                        old_cov_changes[nb] = old_cov_changes.get(nb, 0) - 1
                    
                    # Add new
                    new_cov_changes = {}
                    for nb in neighbors[idx_new]:
                        new_cov_changes[nb] = new_cov_changes.get(nb, 0) + 1
                        
                    # Check validity: for all nb, coverage[nb] + change[nb] >= 1
                    valid = True
                    for nb, chg in old_cov_changes.items():
                        if coverage[nb] + chg < 1:
                            valid = False
                            break
                    if valid:
                        for nb, chg in new_cov_changes.items():
                            if coverage[nb] + chg < 1:
                                valid = False
                                break
                            
                    if valid:
                        # Apply swap
                        remove_word(w_old)
                        add_word(w_new)
                        no_improvement_count = 0 # Reset counter, we made a move
                        
        # If stuck for a long time, do a kick
        if no_improvement_count > 50:
            # Remove a random word (breaking domination)
            if code:
                w_old = random.choice(list(code))
                remove_word(w_old)
                
                # Greedily fix
                while not is_dominating():
                    if time.time() > deadline - 5:
                        break
                    uncovered_nodes = [i for i in range(num_words) if coverage[i] == 0]
                    if not uncovered_nodes:
                        break
                    u = random.choice(uncovered_nodes)
                    candidates = neighbors[u]
                    best_cand = None
                    max_covers = -1
                    for v_idx in candidates:
                        if all_words[v_idx] in code:
                            continue
                        covers = 0
                        for nb in neighbors[v_idx]:
                            if coverage[nb] == 0:
                                covers += 1
                        if covers > max_covers:
                            max_covers = covers
                            best_cand = all_words[v_idx]
                    if best_cand is not None:
                        add_word(best_cand)
                    else:
                        if candidates:
                            v_idx = random.choice(candidates)
                            add_word(all_words[v_idx])
            
            # Prune again
            while True:
                removable = False
                for w in list(code):
                    if try_remove(w):
                        removable = True
                        break
                if not removable:
                    break
            
            if len(code) < best_size:
                best_code = set(code)
                best_size = len(code)
                
            no_improvement_count = 0

    # Ensure best_code is dominating (it should be, as we only keep valid states or prune valid states)
    # But just in case, if best_code is empty or invalid, fallback to initial greedy?
    # The greedy initial code is dominating.
    # The local search maintains domination (prune/swap) or fixes it (kick).
    
    return [list(w) for w in best_code]
