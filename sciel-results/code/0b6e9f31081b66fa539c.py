import time
import random
from collections import Counter, defaultdict

def construct(n):
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    if n <= 1:
        return "1" if n == 1 else ""
    
    # 1. Generate standard superpermutation via recursive construction
    word = [1]
    for m in range(2, n + 1):
        perms, seen = [], set()
        w_size = m - 1
        for i in range(len(word) - w_size + 1):
            window = tuple(word[i:i + w_size])
            if len(set(window)) == w_size and window not in seen:
                seen.add(window)
                perms.append(window)
        
        new = []
        for p in perms:
            block = list(p) + [m] + list(p)
            max_overlap = min(len(new), len(block))
            overlap = 0
            for o in range(max_overlap, -1, -1):
                if new[len(new) - o:] == block[:o]:
                    overlap = o
                    break
            new += block[overlap:]
        word = new
    
    best_seq = list(word)
    best_len = len(best_seq)
    
    # Helper: compute window counts
    def get_counts(seq, n):
        counts = Counter()
        for i in range(len(seq) - n + 1):
            counts[tuple(seq[i:i+n])] += 1
        return counts
    
    # Helper: check if index i is deletable (all windows containing i have count > 1)
    def is_deletable(seq, i, n, counts):
        start_min = max(0, i - n + 1)
        start_max = min(len(seq) - n, i)
        for start in range(start_min, start_max + 1):
            if counts[tuple(seq[start:start+n])] <= 1:
                return False
        return True
    
    # Helper: compute deletion cost (lower = safer/more redundant)
    def deletion_cost(seq, i, n, counts):
        start_min = max(0, i - n + 1)
        start_max = min(len(seq) - n, i)
        min_count = float('inf')
        for start in range(start_min, start_max + 1):
            c = counts[tuple(seq[start:start+n])]
            if c < min_count:
                min_count = c
        return min_count if min_count != float('inf') else 0
    
    # Greedy deletion with cost-based priority
    def greedy_delete_cost(seq, n, deadline, rng, strategy='min_cost'):
        seq = list(seq)
        L = len(seq)
        counts = get_counts(seq, n)
        
        # Precompute costs for all positions
        costs = [deletion_cost(seq, i, n, counts) for i in range(L)]
        deletable = [i for i in range(L) if costs[i] > 1]
        
        while deletable and time.time() < deadline:
            if strategy == 'min_cost':
                # Pick the position with minimum cost (most redundant)
                min_c = min(costs[i] for i in deletable)
                candidates = [i for i in deletable if costs[i] == min_c]
                idx = rng.choice(candidates)
            else:
                idx = rng.choice(deletable)
            
            # Perform deletion
            start_min = max(0, idx - n + 1)
            start_max = min(L - n, idx)
            for start in range(start_min, start_max + 1):
                w = tuple(seq[start:start+n])
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
            
            seq.pop(idx)
            L -= 1
            
            # Update counts for new windows
            new_start_min = max(0, idx - n + 1)
            new_start_max = min(L - n, idx)
            for start in range(new_start_min, new_start_max + 1):
                w = tuple(seq[start:start+n])
                counts[w] += 1
            
            # Rebuild deletable and costs
            costs = [deletion_cost(seq, i, n, counts) for i in range(L)]
            deletable = [i for i in range(L) if costs[i] > 1]
            
        return seq
    
    # Local search polish: try small perturbations
    def local_polish(seq, n, deadline, rng):
        seq = list(seq)
        L = len(seq)
        counts = get_counts(seq, n)
        best = list(seq)
        best_len = L
        
        # Try random deletions and reinsertions
        for _ in range(200):
            if time.time() > deadline:
                break
            
            # Try deleting a random deletable position
            deletable = [i for i in range(len(seq)) if is_deletable(seq, i, n, counts)]
            if not deletable:
                break
            
            idx = rng.choice(deletable)
            old_val = seq[idx]
            
            # Delete
            start_min = max(0, idx - n + 1)
            start_max = min(len(seq) - n, idx)
            for start in range(start_min, start_max + 1):
                w = tuple(seq[start:start+n])
                counts[w] -= 1
                if counts[w] == 0:
                    del counts[w]
            
            seq.pop(idx)
            L -= 1
            
            # Update new windows
            new_start_min = max(0, idx - n + 1)
            new_start_max = min(L - n, idx)
            for start in range(new_start_min, new_start_max + 1):
                w = tuple(seq[start:start+n])
                counts[w] += 1
            
            # Try reinserting at a random nearby position
            insert_pos = max(0, min(L, idx + rng.randint(-5, 5)))
            seq.insert(insert_pos, old_val)
            L += 1
            
            # Recompute counts
            counts = get_counts(seq, n)
            
            if L < best_len:
                best = list(seq)
                best_len = L
        
        return best
    
    # Run multiple strategies
    strategies = ['min_cost', 'random']
    tries = 0
    while time.time() < deadline - 5.0:
        strategy = strategies[tries % len(strategies)]
        rng = random.Random(tries * 1000 + 42)
        candidate = greedy_delete_cost(word, n, deadline - 2.0, rng, strategy)
        
        if len(candidate) < best_len:
            best_len = len(candidate)
            best_seq = candidate
            # Polish
            rng2 = random.Random(tries * 2000 + 13)
            polished = local_polish(best_seq, n, deadline - 2.0, rng2)
            if len(polished) < best_len:
                best_len = len(polished)
                best_seq = polished
            
            if n == 7 and best_len <= 5905:
                break
        
        tries += 1
        if tries > 80:
            break
    
    return "".join(map(str, best_seq))
