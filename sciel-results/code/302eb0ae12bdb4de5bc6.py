"""
Extended Sidon set construction using beam search + constraint propagation + local search.
Goal: Maximize size of set A subset [1, 284] containing {1, 2, 5, 11}
such that all pairwise sums a+b (a<=b) are distinct.

Strategy:
1. Fast greedy initialization to get a good baseline.
2. Beam search for broad exploration with constraint propagation.
3. Local search (swaps) for deep exploitation to escape local maxima.
4. Time-bounded: return best found if search doesn't complete.
"""
import time
import sys
import random
from typing import Set, List, Tuple

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill

    prescribed = [1, 2, 5, 11]
    prescribed_set = set(prescribed)
    max_val = 284
    n = max_val

    def get_sums(s: Set[int]) -> Set[int]:
        sums = set()
        sl = sorted(s)
        for i in range(len(sl)):
            for j in range(i, len(sl)):
                sums.add(sl[i] + sl[j])
        return sums

    def can_add(x: int, s: Set[int], used_sums: Set[int]) -> bool:
        new_sums = {x + y for y in s} | {2 * x}
        if new_sums & used_sums:
            return False
        return True

    def add_to_sums(x: int, used_sums: Set[int], s: Set[int]) -> Set[int]:
        new_sums = set(used_sums)
        for y in s:
            if y != x:
                new_sums.add(x + y)
        new_sums.add(2 * x)
        return new_sums

    # --- Phase 1: Greedy initialization ---
    def greedy_init() -> List[int]:
        current_set = set(prescribed)
        used_sums = get_sums(current_set)
        for x in range(1, n + 1):
            if x in current_set:
                continue
            if can_add(x, current_set, used_sums):
                current_set.add(x)
                used_sums = add_to_sums(x, used_sums, current_set)
        return sorted(current_set)

    best_set = greedy_init()
    best_size = len(best_set)

    # --- Phase 2: Beam Search ---
    # Each beam state: (current_set, used_sums, next_candidate_idx)
    # We maintain a beam of the most promising partial solutions.

    all_candidates = [x for x in range(1, n + 1) if x not in prescribed_set]
    beam_width = 200
    
    def beam_search(target_size: int) -> bool:
        nonlocal best_set, best_size
        
        # Initial beam: start with prescribed elements
        initial_set = set(prescribed)
        initial_sums = get_sums(initial_set)
        
        beam = [(initial_set, initial_sums, 0)]
        
        for depth in range(target_size - len(prescribed)):
            if time.time() > deadline:
                return False
            
            next_beam = []
            seen_states = set()
            
            for curr_set, curr_sums, start_idx in beam:
                if time.time() > deadline:
                    return False
                
                # Try adding each candidate from start_idx onwards
                for i in range(start_idx, len(all_candidates)):
                    x = all_candidates[i]
                    if not can_add(x, curr_set, curr_sums):
                        continue
                    
                    new_set = curr_set | {x}
                    new_sums = add_to_sums(x, curr_sums, curr_set)
                    new_idx = i + 1
                    
                    # State key for deduplication (use frozenset of elements + min_idx)
                    state_key = (frozenset(new_set), new_idx)
                    if state_key in seen_states:
                        continue
                    seen_states.add(state_key)
                    
                    next_beam.append((new_set, new_sums, new_idx))
                    
                    if len(next_beam) >= beam_width * 3:
                        break
                
                if time.time() > deadline:
                    return False
            
            if not next_beam:
                return False
            
            # Score and prune beam
            # Score: prefer states with more "flexibility" - fewer conflicts
            # Simple heuristic: prefer smaller max element (more room to grow)
            scored = []
            for state in next_beam:
                s = state[0]
                max_elem = max(s) if s else 0
                # Lower max element is better (more room)
                scored.append((max_elem, state))
            
            scored.sort(key=lambda x: x[0])
            beam = [s for _, s in scored[:beam_width]]
            
            # Check if any state reached target size
            for curr_set, curr_sums, _ in beam:
                if len(curr_set) >= target_size:
                    best_set = sorted(curr_set)
                    best_size = target_size
                    return True
        
        return False

    # Try increasing target sizes
    for target in range(best_size + 1, min(best_size + 4, 21)):
        if time.time() > deadline - 30:
            break
        if beam_search(target):
            pass  # best_set already updated
        else:
            break  # Can't find larger, stop

    # --- Helper: greedy extend a set to local maximum ---
    def greedy_extend(s: Set[int], used_sums: Set[int]) -> Tuple[Set[int], Set[int]]:
        """Add as many valid elements as possible in increasing order."""
        changed = True
        while changed:
            changed = False
            for x in range(1, n + 1):
                if x in s:
                    continue
                if can_add(x, s, used_sums):
                    s.add(x)
                    used_sums = add_to_sums(x, used_sums, s)
                    changed = True
        return s, used_sums

    # --- Phase 3: Local search refinement ---
    # Use swaps to escape local maxima
    if time.time() < deadline - 10:
        rng = random.Random(42)
        current_set = set(best_set)
        current_sums = get_sums(current_set)
        # First, greedily extend to make sure we're at a local max
        current_set, current_sums = greedy_extend(current_set, current_sums)
        if len(current_set) > best_size:
            best_set = sorted(current_set)
            best_size = len(best_set)

        for iteration in range(100000):
            if time.time() > deadline - 5:
                break

            # Try a swap: remove one non-prescribed element, add one new one
            removable = [x for x in current_set if x not in prescribed_set]
            if not removable:
                break
            x_remove = rng.choice(removable)
            candidates = [x for x in range(1, n + 1) if x not in current_set]
            if not candidates:
                break
            x_add = rng.choice(candidates)

            new_set = set(current_set)
            new_set.remove(x_remove)

            # Check if x_add can be added to the reduced set
            new_sums_temp = get_sums(new_set)
            if can_add(x_add, new_set, new_sums_temp):
                new_set.add(x_add)
                new_sums = get_sums(new_set)

                # Try to extend: add any valid element
                extended = False
                for y in range(1, n + 1):
                    if y in new_set:
                        continue
                    if can_add(y, new_set, new_sums):
                        new_set.add(y)
                        new_sums = get_sums(new_set)
                        if len(new_set) > best_size:
                            best_set = sorted(new_set)
                            best_size = len(best_set)
                        extended = True
                        break

                # Accept swap if it led to extension or with some probability
                if extended or rng.random() < 0.3:
                    current_set = new_set
                    current_sums = new_sums

    # --- Phase 4: Randomized greedy with perturbation ---
    # Per the cell findings, randomized greedy with perturbation can escape
    # deterministic local optima. Run several randomized greedy passes.
    if time.time() < deadline - 5:
        rng2 = random.Random(12345)
        for trial in range(50):
            if time.time() > deadline - 3:
                break
            # Start with prescribed, then add candidates in a randomized order
            # but with a bias toward smaller values (to keep room for growth)
            current_set = set(prescribed)
            used_sums = get_sums(current_set)
            candidates = [x for x in range(1, n + 1) if x not in prescribed_set]
            # Shuffle but weight toward smaller values
            # Use a simple trick: sort by (random_key, x) where random_key is uniform
            shuffled = [(rng2.random(), x) for x in candidates]
            shuffled.sort(key=lambda t: t[0] * 0.7 + t[1] / n * 0.3)
            for _, x in shuffled:
                if can_add(x, current_set, used_sums):
                    current_set.add(x)
                    used_sums = add_to_sums(x, used_sums, current_set)
            # Greedily extend
            current_set, used_sums = greedy_extend(current_set, used_sums)
            if len(current_set) > best_size:
                best_set = sorted(current_set)
                best_size = len(best_set)

    # Final verification
    final_set = set(best_set)
    sums = get_sums(final_set)
    expected = len(final_set) * (len(final_set) + 1) // 2
    if len(sums) != expected:
        best_set = list(prescribed)

    return best_set

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    print(f"Size: {len(res)}, Time: {end-start:.2f}s")
    # Verify
    sums = set()
    valid = True
    r = sorted(res)
    for i in range(len(r)):
        for j in range(i, len(r)):
            s = r[i] + r[j]
            if s in sums:
                valid = False
            sums.add(s)
    if not all(x in res for x in [1, 2, 5, 11]):
        valid = False
    if not all(1 <= x <= 284 for x in res):
        valid = False
    if len(set(res)) != len(res):
        valid = False
    print(f"Valid: {valid}")
