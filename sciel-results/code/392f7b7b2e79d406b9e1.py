import math
import random
import time
from itertools import combinations

def construct():
    """
    Return a nonempty subset of [1..22] such that ALL reciprocal subset sums,
    including the empty sum, are distinct. Maximize cardinality.
    
    A set {a_1, ..., a_n} is "dissociated" for reciprocal subset sums iff
    the only integer combination sum c_i * (1/a_i) = 0 with c_i in {-1,0,1}
    is the trivial one. Equivalently, all 2^n subset sums of {1/a_i} are distinct.
    
    We use exact rational arithmetic via LCM to avoid floating-point errors.
    """
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit

    # LCM of 1..22
    def lcm(a, b):
        return abs(a * b) // math.gcd(a, b)

    L = 1
    for i in range(1, 23):
        L = lcm(L, i)

    # A[i] = L / (i+1) for i in 0..21, so 1/(i+1) = A[i] / L
    A = [L // (i + 1) for i in range(22)]

    def check_dissociated(indices):
        """
        Check if the set of indices (0-based, representing elements 1..22)
        has all distinct reciprocal subset sums.
        Returns True if dissociated, False otherwise.
        """
        k = len(indices)
        if k == 0:
            return True

        vals = [A[i] for i in indices]

        # Generate all subset sums and check for duplicates
        seen = {0}
        current_sums = [0]

        for i in range(k):
            v = vals[i]
            new_sums = []
            for s in current_sums:
                ns = s + v
                if ns in seen:
                    return False
                seen.add(ns)
                new_sums.append(ns)
            current_sums.extend(new_sums)

        return True

    def find_collision(indices):
        """
        Find a collision in the subset sums of the given indices.
        Returns (element_to_remove, collision_info) or (None, None) if no collision.
        The element_to_remove is chosen to maximize the number of collisions it's involved in.
        """
        k = len(indices)
        if k == 0:
            return None, None

        vals = [A[i] for i in indices]

        # Generate all subset sums with their masks
        # Use iterative approach
        sum_to_masks = {0: [0]}
        all_sums = [(0, 0)]

        for i in range(k):
            v = vals[i]
            new_entries = []
            for s, m in all_sums:
                ns = s + v
                nm = m | (1 << i)
                if ns in sum_to_masks:
                    # Add this mask to the existing list
                    sum_to_masks[ns].append(nm)
                else:
                    sum_to_masks[ns] = [nm]
                new_entries.append((ns, nm))
            all_sums.extend(new_entries)

        # Find all collisions and count how many times each element is involved
        collision_count = [0] * k
        has_collision = False

        for s, masks in sum_to_masks.items():
            if len(masks) > 1:
                has_collision = True
                # For each pair of masks, find the symmetric difference
                for i in range(len(masks)):
                    for j in range(i + 1, len(masks)):
                        diff = masks[i] ^ masks[j]
                        for bit in range(k):
                            if (diff >> bit) & 1:
                                collision_count[bit] += 1

        if not has_collision:
            return None, None

        # Find the element with the most collisions
        max_count = max(collision_count)
        candidates = [i for i in range(k) if collision_count[i] == max_count]
        
        # Among candidates, prefer larger indices (larger denominators, smaller reciprocals)
        # as they're less likely to cause future collisions
        best_local = max(candidates)
        
        return indices[best_local], (collision_count, has_collision)

    # Greedy approach: start with all 22, remove elements causing collisions
    # Try multiple strategies and pick the best (largest cardinality)

    best_set = [0]  # At minimum, a single element is always valid

    def greedy_remove_v2():
        """
        Start with all 22 elements, repeatedly find a collision and remove
        the element most involved in collisions.
        """
        if time.time() > deadline:
            return None

        S = list(range(22))

        while True:
            if time.time() > deadline:
                return S if S else None

            k = len(S)
            if k == 0:
                return None

            # Check if S is dissociated
            if check_dissociated(S):
                return S

            # Find a collision and the best element to remove
            elem_to_remove, _ = find_collision(S)
            
            if elem_to_remove is None:
                # Shouldn't happen, but return current set
                return S

            if elem_to_remove in S:
                S.remove(elem_to_remove)
            else:
                # Fallback: remove the largest index
                S.remove(max(S))

        return S if S else None

    # Try the improved greedy removal
    result = greedy_remove_v2()
    if result and len(result) > len(best_set):
        best_set = result

    # Try some random restarts with the improved method
    for _ in range(30):
        if time.time() > deadline - 5:
            break
        
        # Randomize the order of elements before greedy removal
        random_order = list(range(22))
        random.shuffle(random_order)
        
        # Use a modified greedy that respects the random order for tie-breaking
        S = random_order[:]
        
        while True:
            if time.time() > deadline - 5:
                break
            
            k = len(S)
            if k == 0:
                break

            if check_dissociated(S):
                break

            elem_to_remove, _ = find_collision(S)
            
            if elem_to_remove is None:
                break

            if elem_to_remove in S:
                S.remove(elem_to_remove)
            else:
                S.remove(max(S))

        if S and len(S) > len(best_set):
            best_set = S

    # Try a different approach: build up from scratch using greedy addition
    # Start empty, try to add each element if it maintains dissociation
    if time.time() < deadline - 10:
        def greedy_build(order):
            S = []
            for idx in order:
                if time.time() > deadline - 5:
                    break
                new_S = S + [idx]
                if check_dissociated(new_S):
                    S = new_S
            return S if S else None

        # Try different orderings
        # 1. Descending (larger denominators first - smaller reciprocals)
        result = greedy_build(list(range(21, -1, -1)))
        if result and len(result) > len(best_set):
            best_set = result

        # 2. Ascending
        result = greedy_build(list(range(22)))
        if result and len(result) > len(best_set):
            best_set = result

        # 3. Some random orderings
        for _ in range(15):
            if time.time() > deadline - 5:
                break
            order = list(range(22))
            random.shuffle(order)
            result = greedy_build(order)
            if result and len(result) > len(best_set):
                best_set = result

    # Final safety: ensure best_set is nonempty
    if not best_set:
        best_set = [0]

    # Return 1-based indices
    return [i + 1 for i in sorted(best_set)]

# EVOLVE-BLOCK-END
