"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations_with_replacement

def construct():
    start_time = time.time()
    deadline = start_time + 85  # 85s safety margin (90s limit)
    
    n = 256
    max_size = 64
    
    def is_sidon(s):
        """Check if s is a Sidon set: all pair sums (including a+a) are unique."""
        if not s:
            return True
        sums = set()
        for i in range(len(s)):
            for j in range(i, len(s)):
                sm = s[i] + s[j]
                if sm in sums:
                    return False
                sums.add(sm)
        return True
    
    def get_diffs(s):
        """Get the set of nonzero absolute differences from set s."""
        diffs = set()
        for i in range(len(s)):
            for j in range(i + 1, len(s)):
                d = abs(s[j] - s[i])
                if d > 0:
                    diffs.add(d)
        return diffs
    
    def try_add_greedy(sets, candidate_pool, order='asc'):
        """
        Greedily add elements from candidate_pool to sets[0] and sets[1].
        Returns the resulting sets.
        """
        result = [list(sets[0]), list(sets[1])]
        diff_sets = [get_diffs(result[0]), get_diffs(result[1])]
        
        # Sort candidates by the given order
        if order == 'asc':
            candidates = list(candidate_pool)
        elif order == 'desc':
            candidates = list(reversed(candidate_pool))
        else:
            candidates = list(candidate_pool)
            random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > deadline:
                break
            if len(result[0]) >= max_size and len(result[1]) >= max_size:
                break
            
            # Try adding to side 0
            if len(result[0]) < max_size:
                new_s0 = result[0] + [x]
                if is_sidon(new_s0):
                    new_diffs0 = get_diffs(new_s0)
                    if not (new_diffs0 & diff_sets[1]):
                        result[0] = new_s0
                        diff_sets[0] = new_diffs0
                        continue
            
            # Try adding to side 1
            if len(result[1]) < max_size:
                new_s1 = result[1] + [x]
                if is_sidon(new_s1):
                    new_diffs1 = get_diffs(new_s1)
                    if not (new_diffs1 & diff_sets[0]):
                        result[1] = new_s1
                        diff_sets[1] = new_diffs1
                        continue
        
        return result
    
    best_result = None
    best_score = 0
    
    # Strategy 1: Greedy ascending
    for attempt in range(3):
        if time.time() > deadline:
            break
        sets = [[], []]
        result = try_add_greedy(sets, range(1, n + 1), 'asc')
        score = len(result[0]) + len(result[1])
        if score > best_score:
            best_score = score
            best_result = result
    
    # Strategy 2: Greedy descending
    for attempt in range(3):
        if time.time() > deadline:
            break
        sets = [[], []]
        result = try_add_greedy(sets, range(1, n + 1), 'desc')
        score = len(result[0]) + len(result[1])
        if score > best_score:
            best_score = score
            best_result = result
    
    # Strategy 3: Random shuffles with multiple restarts
    num_restarts = 50
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        sets = [[], []]
        result = try_add_greedy(sets, range(1, n + 1), 'random')
        score = len(result[0]) + len(result[1])
        if score > best_score:
            best_score = score
            best_result = result
    
    # Strategy 4: Start with different initial seeds
    for seed_offset in [0, 1, 2, 3]:
        if time.time() > deadline:
            break
        for attempt in range(5):
            if time.time() > deadline:
                break
            # Start with a few random elements
            init = [random.randint(1, n) for _ in range(2)]
            init = list(set(init))
            sets = [list(init[:1]), list(init[1:2])]
            # Verify initial validity
            if not is_sidon(sets[0]) or not is_sidon(sets[1]):
                continue
            d0 = get_diffs(sets[0])
            d1 = get_diffs(sets[1])
            if d0 & d1:
                continue
            result = try_add_greedy(sets, range(1, n + 1), 'random')
            score = len(result[0]) + len(result[1])
            if score > best_score:
                best_score = score
                best_result = result
    
    # Final polish: try to add more elements using a different ordering
    if best_result and time.time() < deadline - 5:
        # Try adding remaining elements in ascending order to both sets
        for x in range(1, n + 1):
            if time.time() > deadline - 2:
                break
            if len(best_result[0]) >= max_size and len(best_result[1]) >= max_size:
                break
            
            # Try side 0
            if len(best_result[0]) < max_size:
                new_s0 = best_result[0] + [x]
                if is_sidon(new_s0):
                    new_diffs0 = get_diffs(new_s0)
                    d1 = get_diffs(best_result[1])
                    if not (new_diffs0 & d1):
                        best_result[0] = new_s0
                        continue
            
            # Try side 1
            if len(best_result[1]) < max_size:
                new_s1 = best_result[1] + [x]
                if is_sidon(new_s1):
                    new_diffs1 = get_diffs(new_s1)
                    d0 = get_diffs(best_result[0])
                    if not (new_diffs1 & d0):
                        best_result[1] = new_s1
                        continue
    
    if best_result is None:
        best_result = [[1], [2]]
    
    return best_result
# EVOLVE-BLOCK-END
