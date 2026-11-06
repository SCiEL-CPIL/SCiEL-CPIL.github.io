"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def construct():
    deadline = time.time() + 140.0

    def instance(size, seed, limit=10000):
        rng = Random(seed)
        values = set()
        while len(values) < size:
            x = rng.randint(-limit, limit)
            if x:
                values.add(x)
        return sorted(values)

    def is_sum_free(S):
        s = set(S)
        for a in S:
            for b in S:
                if a + b in s:
                    return False
        return True

    def can_add(x, cur_set):
        """Check if x can be added to cur_set while maintaining sum-free."""
        if x in cur_set:
            return False
        if x + x in cur_set:
            return False
        # a + x in cur_set for some a
        for a in cur_set:
            if a + x in cur_set:
                return False
        # a + b == x for some a, b in cur_set
        for a in cur_set:
            for b in cur_set:
                if a + b == x:
                    return False
        return True

    def randomized_greedy(S, rng, max_iter=200000):
        """Randomized greedy: start empty, repeatedly add a random feasible element."""
        current = set()
        candidates = list(S)
        rng.shuffle(candidates)
        for _ in range(max_iter):
            if time.time() > deadline:
                break
            added = False
            for x in candidates:
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    added = True
                    break
            if not added:
                break
        return list(current)

    def greedy_from_seed(S, seed_set, rng):
        """Start from seed_set, greedily add elements from S."""
        current = set(seed_set)
        for _ in range(500000):
            if time.time() > deadline:
                break
            added = False
            for x in S:
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    added = True
                    break
            if not added:
                break
        return list(current)

    def one_swap_local_search(S, initial, rng, max_iters=20000):
        """1-swap local search: remove one, try to add another."""
        current = set(initial)
        best = set(initial)
        best_size = len(best)
        
        for _ in range(max_iters):
            if time.time() > deadline:
                break
            improved = False
            # Try removing a random element from current
            if not current:
                break
            removed = rng.choice(list(current))
            current.discard(removed)
            
            # Try to add a different element
            for x in S:
                if x in current:
                    continue
                if can_add(x, current):
                    current.add(x)
                    if len(current) > best_size:
                        best = set(current)
                        best_size = len(current)
                        improved = True
                    break
            if not improved:
                current.add(removed)
        
        return list(best)

    def solve_one(S, seed):
        """Multi-start ensemble for one instance."""
        rng = Random(seed + 1000)
        best = []
        best_size = 0
        
        # Budget: try multiple diverse starts
        n_starts = 8
        for i in range(n_starts):
            if time.time() > deadline - 2:
                break
            
            # Diverse initial seeds
            if i == 0:
                # Odds
                init = [x for x in S if x % 2]
            elif i == 1:
                # Evens
                init = [x for x in S if x % 2 == 0]
            elif i == 2:
                # Positives
                init = [x for x in S if x > 0]
            elif i == 3:
                # Negatives
                init = [x for x in S if x < 0]
            elif i == 4:
                # Large magnitude
                init = sorted(S, key=abs, reverse=True)[:len(S)//2]
            elif i == 5:
                # Small magnitude
                init = sorted(S, key=abs)[:len(S)//2]
            elif i == 6:
                # Random half
                init = rng.sample(S, len(S)//2)
            else:
                # Random quarter
                init = rng.sample(S, max(1, len(S)//4))
            
            # Greedy from seed
            result = greedy_from_seed(S, init, rng)
            
            # Local search
            result = one_swap_local_search(S, result, rng)
            
            if len(result) > best_size:
                best_size = len(result)
                best = result
        
        # Also try pure randomized greedy
        if time.time() < deadline - 2:
            result = randomized_greedy(S, rng)
            result = one_swap_local_search(S, result, rng)
            if len(result) > best_size:
                best = result
        
        return best

    results = []
    for size, seed in ((60, 801), (120, 802), (240, 803)):
        S = instance(size, seed)
        results.append(solve_one(S, seed))
    
    return results
# EVOLVE-BLOCK-END
