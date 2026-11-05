import time
import random
import numpy as np
from collections import defaultdict, deque

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    N = 128
    min_sum = 128
    max_sum = 256
    target_sums = list(range(min_sum, max_sum + 1))
    num_targets = len(target_sums)  # 129 targets
    
    def get_counts_fast(A):
        """Returns array of counts for sums in [0, 256] using convolution"""
        indicator = np.zeros(2 * N + 1, dtype=np.int64)
        for x in A:
            indicator[x] = 1
        # Convolution gives ordered pair counts
        conv = np.convolve(indicator, indicator, mode='full')
        # For unordered pairs with equal summands allowed:
        # count(s) = (conv[s] + indicator[s//2] if s even) / 2
        counts = np.zeros(max_sum + 1, dtype=np.int64)
        for s in range(min_sum, max_sum + 1):
            if s % 2 == 0 and s // 2 <= N:
                counts[s] = (conv[s] + 1) // 2
            else:
                counts[s] = conv[s] // 2
        return counts

    def get_cost(A):
        counts = get_counts_fast(A)
        target_counts = counts[min_sum:max_sum+1]
        if np.any(target_counts == 0):
            return 1000 + int(np.sum(target_counts == 0))
        return int(np.max(target_counts))

    def max_flow_feasible(capacity, A_set):
        """
        Check if there exists a subset B of [0,128] with |B| <= capacity
        such that every sum in [128,256] can be represented as x+y with x,y in B.
        
        We use a bipartite matching formulation:
        - Left nodes: target sums s in [128, 256]
        - Right nodes: elements x in [0, 128]
        - Edge (s, x) exists if s-x is also in [0, 128]
        
        But this doesn't directly model the constraint that we need to CHOOSE a set B.
        
        Alternative: Binary search on peak multiplicity t.
        For a given t, we ask: does there exist a set A such that every sum in [128,256]
        has at most t representations?
        
        This is hard to formulate as flow directly. Instead, let's use a different approach:
        Start with a feasible set (coverage), then use local search to minimize peak.
        """
        pass

    def find_initial_cover():
        """Find a set A that covers all sums in [128,256]"""
        A = set()
        # Greedy: add elements that cover the most uncovered sums
        uncovered = set(target_sums)
        
        while uncovered and time.time() < deadline - 10:
            best_x = None
            best_gain = -1
            
            for x in range(0, N + 1):
                if x in A:
                    continue
                # How many uncovered sums would x cover?
                # x + y for y in A, where x+y in uncovered
                gain = 0
                for y in A:
                    s = x + y
                    if s in uncovered:
                        gain += 1
                # Also x + x
                s = x + x
                if s in uncovered:
                    gain += 1
                
                if gain > best_gain:
                    best_gain = gain
                    best_x = x
            
            if best_x is None or best_gain == 0:
                # Try random
                candidates = [x for x in range(0, N + 1) if x not in A]
                if not candidates:
                    break
                best_x = random.choice(candidates)
            
            A.add(best_x)
            # Update uncovered
            for y in list(A):
                for z in list(A):
                    if y >= z:
                        s = y + z
                        if s in uncovered:
                            uncovered.discard(s)
        
        return A, uncovered

    # Phase 1: Get initial coverage
    random.seed(42)
    A, uncovered = find_initial_cover()
    
    # If still uncovered, add random elements
    while uncovered and time.time() < deadline - 5:
        x = random.randint(0, N)
        if x not in A:
            A.add(x)
            for y in A:
                s = x + y
                if s in uncovered:
                    uncovered.discard(s)
            s = 2 * x
            if s in uncovered:
                uncovered.discard(s)
    
    # If we can't cover everything, return what we have
    if uncovered:
        return list(A)
    
    best_cost = get_cost(A)
    best_A = set(A)
    current_cost = best_cost
    
    # Phase 2: Local search to minimize peak multiplicity
    # Use swap moves and occasional add/remove
    while time.time() < deadline - 2:
        # Try a swap
        if len(A) > 0:
            x_in = random.choice(list(A))
            # Pick a random element outside A
            x_out = random.randint(0, N)
            while x_out in A:
                x_out = random.randint(0, N)
            
            A.discard(x_in)
            A.add(x_out)
            
            new_cost = get_cost(A)
            
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
                # Accept
            else:
                # Revert
                A.discard(x_out)
                A.add(x_in)
        
        # Occasionally try to remove an element if it doesn't hurt
        if random.random() < 0.05 and len(A) > 16:
            x = random.choice(list(A))
            A.discard(x)
            new_cost = get_cost(A)
            if new_cost <= current_cost:
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_A = set(A)
            else:
                A.add(x)
        
        # Occasionally try to add an element
        if random.random() < 0.05:
            x = random.randint(0, N)
            if x not in A:
                A.add(x)
                new_cost = get_cost(A)
                if new_cost <= current_cost:
                    current_cost = new_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_A = set(A)
                else:
                    A.discard(x)
        
        if best_cost <= 2:
            break
    
    return list(best_A)
