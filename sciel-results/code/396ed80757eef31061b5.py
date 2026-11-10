import time
import random

def construct():
    deadline = time.time() + 85
    n = 256
    
    # Precompute conflicts
    conflicts = [set() for _ in range(n + 1)]
    for a in range(1, n + 1):
        for b in range(a + 1, n + 1):
            if (a * b) % (a + b) == 0:
                conflicts[a].add(b)
                conflicts[b].add(a)
    
    def is_valid_set(s):
        """Check if set s has no conflicting pairs."""
        for a in s:
            if conflicts[a] & s:
                return False
        return True
    
    def greedy_build(start_set, order):
        """Greedily add elements from order to start_set."""
        s = set(start_set)
        for e in order:
            if e not in s and not (conflicts[e] & s):
                s.add(e)
        return s
    
    def local_search(s, max_iters=50):
        """Iterative local search: try to add elements, remove blockers."""
        s = set(s)
        best_size = len(s)
        for _ in range(max_iters):
            if time.time() > deadline - 3:
                break
            improved = False
            # Try adding each non-member
            for b in range(1, n + 1):
                if b in s:
                    continue
                conflicts_with_b = conflicts[b] & s
                if not conflicts_with_b:
                    s.add(b)
                    best_size = len(s)
                    improved = True
                    break
                # Try removing one conflicting element to add b
                for a in list(conflicts_with_b):
                    new_set = s - {a} | {b}
                    if not (conflicts[b] & (new_set - {b})):
                        s = new_set
                        if len(s) > best_size:
                            best_size = len(s)
                        improved = True
                        break
                if improved:
                    break
            # Try removing one element and adding two non-conflicting ones
            if not improved:
                for a in list(s):
                    remaining = s - {a}
                    candidates = [b for b in range(1, n + 1) if b not in remaining and not (conflicts[b] & remaining)]
                    if len(candidates) >= 1:
                        # Add one candidate
                        for b in candidates:
                            new_set = remaining | {b}
                            if len(new_set) > best_size:
                                s = new_set
                                best_size = len(s)
                                improved = True
                                break
                    if improved:
                        break
            if not improved:
                break
        return s, best_size
    
    best = []
    best_size = 0
    
    # Phase 1: Multiple greedy starts
    all_elements = list(range(1, n + 1))
    random.seed(42)
    
    # Start 1: all odds
    odds = set(range(1, n + 1, 2))
    evens = list(range(2, n + 1, 2))
    s = greedy_build(odds, evens)
    s, sz = local_search(s)
    if sz > best_size:
        best = list(s)
        best_size = sz
    
    # Phase 2: Random orderings from odds
    for trial in range(40):
        if time.time() > deadline - 5:
            break
        evens_shuffled = list(range(2, n + 1, 2))
        random.shuffle(evens_shuffled)
        s = greedy_build(odds, evens_shuffled)
        s, sz = local_search(s, max_iters=30)
        if sz > best_size:
            best = list(s)
            best_size = sz
    
    # Phase 3: Random subsets as seeds
    for trial in range(30):
        if time.time() > deadline - 5:
            break
        # Random seed: pick random subset of odds + some evens
        seed_size = random.randint(50, 100)
        random_seed = random.sample(all_elements, seed_size)
        # Filter to valid set
        valid_seed = set()
        for e in random_seed:
            if not (conflicts[e] & valid_seed):
                valid_seed.add(e)
        
        remaining = [e for e in all_elements if e not in valid_seed]
        random.shuffle(remaining)
        s = greedy_build(valid_seed, remaining)
        s, sz = local_search(s, max_iters=30)
        if sz > best_size:
            best = list(s)
            best_size = sz
    
    # Phase 4: More aggressive local search on best
    if time.time() < deadline - 5:
        s_set = set(best)
        for iteration in range(100):
            if time.time() > deadline - 3:
                break
            improved = False
            for b in range(1, n + 1):
                if b in s_set:
                    continue
                conflicts_with_b = conflicts[b] & s_set
                if not conflicts_with_b:
                    s_set.add(b)
                    best_size = len(s_set)
                    improved = True
                    break
                for a in list(conflicts_with_b):
                    new_set = s_set - {a} | {b}
                    if not (conflicts[b] & (new_set - {b})):
                        s_set = new_set
                        if len(s_set) > best_size:
                            best_size = len(s_set)
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        best = list(s_set)
    
    return best
