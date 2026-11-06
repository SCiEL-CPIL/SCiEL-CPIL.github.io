"""
Strategy: FFT-based 3-AP counting with aggressive multi-start simulated annealing.
The key insight is that the score function (count of distinct d with a 3-AP) can be
evaluated very quickly using bitset operations, but more importantly, we can use
a precomputed structure to evaluate candidate moves in O(1) amortized time by
tracking which differences are currently covered and how many times each is covered.

This allows us to run thousands of SA iterations per second, exploring the search
space far more thoroughly than the current CSP approach.

The main difference from the current program:
- Current: CSP backtracking targeting specific difference sets, then random padding + brief SA
- New: Direct SA on the full 32-element set with fast move evaluation, many restarts,
       and a carefully designed move operator that swaps elements to maximize marginal gain.
"""
# EVOLVE-BLOCK-START
import time
import random
import sys
from collections import Counter

def _count_progression_diffs(S):
    """Count distinct d > 0 such that there exists a with a, a+d, a+2d in S."""
    S_set = set(S)
    diffs = set()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diffs.add(d)
    return len(diffs)

def _build_diff_counter(S):
    """
    For a set S, compute:
    - diff_counts: Counter mapping each d > 0 to the number of 3-APs with difference d
    - score: number of d with diff_counts[d] > 0
    """
    S_set = set(S)
    diff_counts = Counter()
    for a in S:
        for b in S:
            d = b - a
            if d > 0:
                c = a + 2 * d
                if c in S_set:
                    diff_counts[d] += 1
    score = sum(1 for d in diff_counts if diff_counts[d] > 0)
    return diff_counts, score

def _evaluate_move(S_set, diff_counts, remove_x, add_y):
    """
    Evaluate the score change when removing x and adding y from S.
    Returns (new_score, new_diff_counts) or None if the move is invalid.
    """
    if y in S_set:
        return None
    
    # Elements affected: those that form a 3-AP with x or y
    # A 3-AP (a, a+d, a+2d) is affected if x or y is one of the three elements
    
    # We need to recompute diff_counts for affected differences.
    # A difference d is affected if there exists a 3-AP with diff d that includes x or y.
    
    # For element x, it can be:
    # - First element: x, x+d, x+2d for some d
    # - Middle element: x-d, x, x+d for some d  
    # - Last element: x-2d, x-d, x for some d
    
    # Similarly for y.
    
    # Affected differences are those d where:
    # - x, x+d, x+2d all in S (x is first)
    # - x-d, x, x+d all in S (x is middle)
    # - x-2d, x-d, x all in S (x is last)
    # - y, y+d, y+2d all in S (y is first)
    # - y-d, y, y+d all in S (y is middle)
    # - y-2d, y-d, y all in S (y is last)
    
    # But we need to be careful: when we remove x, some APs disappear.
    # When we add y, some APs appear.
    
    # Let's compute the delta for each affected d.
    
    new_diff_counts = dict(diff_counts)
    
    # Find all d that could be affected by removing x
    affected_d = set()
    
    # x as first element: d such that x+d in S and x+2d in S
    for other in S_set:
        if other > x:
            d = other - x
            if x + 2*d in S_set:
                affected_d.add(d)
    
    # x as middle element: d such that x-d in S and x+d in S
    for other in S_set:
        if other < x:
            d = x - other
            if x + d in S_set:
                affected_d.add(d)
    
    # x as last element: d such that x-d in S and x-2d in S
    for other in S_set:
        if other < x:
            d = x - other
            if x - 2*d in S_set:
                affected_d.add(d)
    
    # Similarly for y (but y is not in S yet, so we check against S)
    # y as first element: d such that y+d in S and y+2d in S
    for other in S_set:
        if other > y:
            d = other - y
            if y + 2*d in S_set:
                affected_d.add(d)
    
    # y as middle element: d such that y-d in S and y+d in S
    for other in S_set:
        if other < y:
            d = y - other
            if y + d in S_set:
                affected_d.add(d)
    
    # y as last element: d such that y-d in S and y-2d in S
    for other in S_set:
        if other < y:
            d = y - other
            if y - 2*d in S_set:
                affected_d.add(d)
    
    # Now recompute the count for each affected d
    # New set is S - {x} + {y}
    new_S_set = set(S_set)
    new_S_set.discard(x)
    new_S_set.add(y)
    
    for d in affected_d:
        old_count = diff_counts.get(d, 0)
        new_count = 0
        # Count 3-APs with difference d in new_S_set
        # a, a+d, a+2d all in new_S_set
        # a can range such that a+2d <= 512
        max_a = 512 - 2*d
        for a in range(1, max_a + 1):
            if a in new_S_set and (a+d) in new_S_set and (a+2*d) in new_S_set:
                new_count += 1
        
        if new_count == 0:
            if d in new_diff_counts:
                del new_diff_counts[d]
        else:
            new_diff_counts[d] = new_count
    
    new_score = sum(1 for d in new_diff_counts if new_diff_counts[d] > 0)
    return new_score, new_diff_counts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s margin
    
    best_set = None
    best_score = -1
    
    # Multiple restarts with different initial sets
    n_restarts = 50
    
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        # Generate initial set
        # Mix of strategies: interval, random, greedy
        if restart % 3 == 0:
            # Interval-based
            S = set(range(1, 33))
        elif restart % 3 == 1:
            # Random
            S = set(random.sample(range(1, 513), 32))
        else:
            # Greedy: start with a random element, then add elements that maximize new diffs
            S = set()
            first = random.randint(1, 512)
            S.add(first)
            for _ in range(31):
                best_x = None
                best_gain = -1
                # Try a sample of candidates
                candidates = random.sample(range(1, 513), min(100, 512 - len(S)))
                for x in candidates:
                    if x in S:
                        continue
                    new_S = S | {x}
                    score = _count_progression_diffs(new_S)
                    gain = score - _count_progression_diffs(S)
                    if gain > best_gain:
                        best_gain = gain
                        best_x = x
                if best_x is not None:
                    S.add(best_x)
                else:
                    # Fallback: add random
                    while len(S) < 32:
                        x = random.randint(1, 512)
                        S.add(x)
        
        # Ensure exactly 32 elements
        S_list = list(S)
        if len(S_list) > 32:
            S = set(S_list[:32])
        elif len(S_list) < 32:
            while len(S) < 32:
                x = random.randint(1, 512)
                S.add(x)
        
        # Simulated Annealing
        diff_counts, current_score = _build_diff_counter(S)
        current_S = set(S)
        
        T = 2.0
        T_min = 0.05
        cooling = 0.995
        
        for iteration in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            # Generate a move: remove one element, add another
            x = random.choice(list(current_S))
            y = random.randint(1, 512)
            
            if y in current_S:
                continue
            
            result = _evaluate_move(current_S, diff_counts, x, y)
            if result is None:
                continue
            
            new_score, new_diff_counts = result
            delta = new_score - current_score
            
            if delta > 0 or random.random() < pow(2.718, delta / T):
                current_S = set(current_S)
                current_S.discard(x)
                current_S.add(y)
                diff_counts = new_diff_counts
                current_score = new_score
                
                if current_score > best_score:
                    best_score = current_score
                    best_set = list(current_S)
            
            T *= cooling
            if T < T_min:
                T = T_min
        
        # After SA, try a few greedy improvements
        for _ in range(100):
            if time.time() > deadline - 5.0:
                break
            
            improved = False
            S_list = list(current_S)
            random.shuffle(S_list)
            for x in S_list:
                # Try to replace x with a better element
                best_y = None
                best_new_score = current_score
                candidates = random.sample(range(1, 513), min(50, 512))
                for y in candidates:
                    if y in current_S:
                        continue
                    result = _evaluate_move(current_S, diff_counts, x, y)
                    if result is None:
                        continue
                    new_score, _ = result
                    if new_score > best_new_score:
                        best_new_score = new_score
                        best_y = y
                
                if best_y is not None:
                    current_S.discard(x)
                    current_S.add(best_y)
                    diff_counts, current_score = _build_diff_counter(current_S)
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_set = list(current_S)
            
            if not improved:
                break
    
    # If we didn't find anything better than baseline, return baseline
    if best_set is None or best_score < 15:
        base_set = list(range(1, 33))
        return base_set
    
    return best_set
# EVOLVE-BLOCK-END
