"""Improved B2[2] set construction with smarter ordering and local search."""
# EVOLVE-BLOCK-START
import time
import random
import math
from collections import defaultdict

def construct():
    start_time = time.time()
    deadline = start_time + 135  # 15s safety margin
    
    N = 1000
    best_result = []
    best_size = 0
    
    def _build_counts(chosen):
        """Build sum counts for a chosen set."""
        counts = defaultdict(int)
        n = len(chosen)
        for i in range(n):
            for j in range(i, n):
                s = chosen[i] + chosen[j]
                counts[s] += 1
        return counts
    
    def _can_add(x, chosen, counts):
        """Check if x can be added without violating B2[2]."""
        s_xx = 2 * x
        if counts.get(s_xx, 0) >= 2:
            return False
        for y in chosen:
            s_xy = x + y
            if counts.get(s_xy, 0) >= 2:
                return False
        return True
    
    def _add_element(x, chosen, counts):
        """Add x to chosen and update counts."""
        chosen.append(x)
        counts[2 * x] = counts.get(2 * x, 0) + 1
        for y in chosen[:-1]:
            s_xy = x + y
            counts[s_xy] = counts.get(s_xy, 0) + 1
    
    def _greedy_pass(order, time_limit):
        """Run greedy pass with given order."""
        chosen = []
        counts = defaultdict(int)
        for x in order:
            if time.time() > time_limit:
                break
            if _can_add(x, chosen, counts):
                _add_element(x, chosen, counts)
        return chosen
    
    def _local_improve(chosen, time_limit):
        """Try to add more elements via local search."""
        if not chosen:
            return chosen
        counts = _build_counts(chosen)
        used = set(chosen)
        
        # Sort candidates by how "easy" they are to add (fewer conflicts)
        candidates = []
        for x in range(1, N + 1):
            if x in used:
                continue
            # Count how many sums would be at capacity
            conflicts = 0
            s_xx = 2 * x
            if counts.get(s_xx, 0) >= 2:
                conflicts += 1
            for y in chosen:
                s_xy = x + y
                if counts.get(s_xy, 0) >= 2:
                    conflicts += 1
            candidates.append((conflicts, x))
        
        # Try adding in order of fewest conflicts
        candidates.sort()
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            for conflicts, x in candidates:
                if x in used:
                    continue
                if _can_add(x, chosen, counts):
                    _add_element(x, chosen, counts)
                    used.add(x)
                    improved = True
                    # Re-sort candidates after each addition
                    candidates = []
                    for y in range(1, N + 1):
                        if y in used:
                            continue
                        conf = 0
                        s_yy = 2 * y
                        if counts.get(s_yy, 0) >= 2:
                            conf += 1
                        for z in chosen:
                            s_yz = y + z
                            if counts.get(s_yz, 0) >= 2:
                                conf += 1
                        candidates.append((conf, y))
                    candidates.sort()
                    break
        
        return chosen
    
    def _score_result(result):
        """Validate and return size of result."""
        if not result:
            return 0
        # Check all elements are in range and distinct
        if len(set(result)) != len(result):
            return 0
        if any(x < 1 or x > N for x in result):
            return 0
        # Check B2[2] property
        counts = _build_counts(result)
        for s, c in counts.items():
            if c > 2:
                return 0
        return len(result)
    
    # Strategy 1: Increasing order
    if time.time() < deadline - 5:
        order1 = list(range(1, N + 1))
        res1 = _greedy_pass(order1, deadline - 5)
        res1 = _local_improve(res1, deadline - 3)
        score1 = _score_result(res1)
        if score1 > best_size:
            best_size = score1
            best_result = res1
    
    # Strategy 2: Decreasing order
    if time.time() < deadline - 5:
        order2 = list(range(N, 0, -1))
        res2 = _greedy_pass(order2, deadline - 5)
        res2 = _local_improve(res2, deadline - 3)
        score2 = _score_result(res2)
        if score2 > best_size:
            best_size = score2
            best_result = res2
    
    # Strategy 3: Random restarts with different biases
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 10:
            break
        
        # Different random strategies
        order_r = list(range(1, N + 1))
        
        # Mix of strategies
        if i % 3 == 0:
            random.shuffle(order_r)
        elif i % 3 == 1:
            # Bias towards smaller numbers
            order_r = sorted(order_r, key=lambda x: random.random() * x)
        else:
            # Bias towards larger numbers  
            order_r = sorted(order_r, key=lambda x: random.random() * (N + 1 - x))
        
        res_r = _greedy_pass(order_r, deadline - 10)
        res_r = _local_improve(res_r, deadline - 3)
        score_r = _score_result(res_r)
        if score_r > best_size:
            best_size = score_r
            best_result = res_r
    
    # Final local improvement on best result
    if best_result and time.time() < deadline - 5:
        improved = _local_improve(best_result, deadline - 2)
        final_score = _score_result(improved)
        if final_score > best_size:
            best_size = final_score
            best_result = improved
    
    return best_result
# EVOLVE-BLOCK-END
