import time
import random

def construct():
    # LCM of 1..22
    L = 232792560
    weights = [0] * 23
    for i in range(1, 23):
        weights[i] = L // i

    start_time = time.time()
    deadline = start_time + 85  # 85s margin (evaluator kills at 90s)

    # Large prime for modular filtering
    P = 1000000007

    # Precompute modular weights
    mod_weights = [0] * 23
    for i in range(1, 23):
        mod_weights[i] = weights[i] % P

    def is_dissociated_mod(current_set):
        """Check if all subset sums modulo P are distinct (necessary condition)."""
        sums = {0}
        for elem in current_set:
            w = mod_weights[elem]
            new_sums = set()
            for s in sums:
                new_sums.add((s + w) % P)
            if len(new_sums) != len(sums) * 2:
                return False
            sums = new_sums
        return True

    def is_dissociated_exact(current_set):
        """Check if all subset sums (exact integers) are distinct."""
        sums = {0}
        for elem in current_set:
            w = weights[elem]
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            if len(new_sums) != len(sums) * 2:
                return False
            sums = new_sums
        return True

    def greedy_construct(order):
        """Greedy construction: add elements in given order if no collision."""
        current_set = []
        sums = {0}
        mod_sums = {0}
        
        for elem in order:
            w = weights[elem]
            mw = mod_weights[elem]
            
            # Fast modular check
            new_mod_sums = set()
            for s in mod_sums:
                new_mod_sums.add((s + mw) % P)
            if len(new_mod_sums) != len(mod_sums) * 2:
                continue
            
            # Exact check
            new_sums = set()
            for s in sums:
                new_sums.add(s + w)
            if len(new_sums) != len(sums) * 2:
                continue
            
            # Add element
            current_set.append(elem)
            sums = new_sums
            mod_sums = new_mod_sums
        
        return current_set

    best_set = [1, 2, 3, 5, 7, 11, 13, 17, 19]  # fallback size 9
    best_size = len(best_set)

    # Strategy 1: Try deterministic orderings
    deterministic_orders = [
        sorted(range(1, 23), key=lambda x: -weights[x]),  # descending weight
        sorted(range(1, 23), key=lambda x: weights[x]),    # ascending weight
        sorted(range(1, 23), key=lambda x: x),             # natural order
        sorted(range(1, 23), key=lambda x: -x),            # reverse natural
        list(range(1, 23)),  # 1,2,3,...,22
        list(range(22, 0, -1)),  # 22,21,...,1
    ]

    for order in deterministic_orders:
        if time.time() > deadline:
            break
        result = greedy_construct(order)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)

    # Strategy 2: Randomized restarts with different orderings
    random.seed(42)
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        # Create a random ordering with some structure
        elems = list(range(1, 23))
        
        # Mix strategies: sometimes sort by weight, sometimes random
        strategy = random.random()
        if strategy < 0.3:
            elems.sort(key=lambda x: -weights[x])
            # Shuffle last half
            mid = len(elems) // 2
            elems[mid:] = random.sample(elems[mid:], len(elems) - mid)
        elif strategy < 0.6:
            elems.sort(key=lambda x: weights[x])
            mid = len(elems) // 2
            elems[:mid] = random.sample(elems[:mid], mid)
        else:
            random.shuffle(elems)
        
        result = greedy_construct(elems)
        if len(result) > best_size:
            best_set = result
            best_size = len(result)
            
            # If we found size 12+, try to verify and potentially improve
            if best_size >= 12:
                break

        # If we've tried enough and haven't improved, increase randomness
        if attempts % 100 == 0 and best_size < 12:
            # Try a more aggressive random approach
            for _ in range(10):
                if time.time() > deadline:
                    break
                elems = list(range(1, 23))
                random.shuffle(elems)
                result = greedy_construct(elems)
                if len(result) > best_size:
                    best_set = result
                    best_size = len(result)
                    if best_size >= 12:
                        break
            if best_size >= 12:
                break

    # Final verification
    if is_dissociated_exact(best_set):
        return best_set
    else:
        # Fallback
        return [1, 2, 3, 5, 7, 11, 13, 17, 19]
