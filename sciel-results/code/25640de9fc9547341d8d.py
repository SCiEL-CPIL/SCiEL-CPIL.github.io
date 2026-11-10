import time
import sys
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # 5s buffer for 60s limit
    
    # Initial best known valid set
    best_set = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    best_size = len(best_set)
    
    def compute_sums_by_cardinality(elements):
        """Compute sums_by_k[k] = set of all subset sums of cardinality k."""
        n = len(elements)
        sums_by_k = [set() for _ in range(n + 1)]
        sums_by_k[0].add(0)
        for v in elements:
            # Process in reverse to avoid using v multiple times
            for k in range(n, 0, -1):
                for s in sums_by_k[k-1]:
                    sums_by_k[k].add(s + v)
        return sums_by_k

    def check_valid_fast(sums_by_k):
        """Check if all cardinality layers are pairwise disjoint."""
        max_k = len(sums_by_k) - 1
        for i in range(max_k + 1):
            for j in range(i + 1, max_k + 1):
                if sums_by_k[i] and sums_by_k[j] and not sums_by_k[i].isdisjoint(sums_by_k[j]):
                    return False
        return True

    def find_collisions(sums_by_k):
        """Return list of (i, j, sum) where sums_by_k[i] and sums_by_k[j] collide."""
        collisions = []
        max_k = len(sums_by_k) - 1
        for i in range(max_k + 1):
            for j in range(i + 1, max_k + 1):
                if sums_by_k[i] and sums_by_k[j]:
                    common = sums_by_k[i] & sums_by_k[j]
                    for s in common:
                        collisions.append((i, j, s))
        return collisions

    def greedy_build():
        """Greedy construction: add elements that don't cause collisions."""
        current = []
        sums_by_k = [set([0])]
        
        for v in range(256, 0, -1):
            # Try adding v
            new_sums_by_k = [set() for _ in range(len(current) + 2)]
            for k in range(len(current) + 1):
                new_sums_by_k[k].update(sums_by_k[k])
            for k in range(1, len(current) + 2):
                for s in sums_by_k[k-1]:
                    new_sums_by_k[k].add(s + v)
            
            if check_valid_fast(new_sums_by_k):
                current.append(v)
                sums_by_k = new_sums_by_k
        
        return current

    def local_search(start_set, max_time):
        """Local search: try swapping elements to improve the set."""
        current = list(start_set)
        sums_by_k = compute_sums_by_cardinality(current)
        best_local = list(current)
        best_local_size = len(current)
        
        while time.time() < max_time:
            improved = False
            # Try adding a new element
            for v in range(256, 0, -1):
                if v in current:
                    continue
                new_set = current + [v]
                new_sums = compute_sums_by_cardinality(new_set)
                if check_valid_fast(new_sums):
                    current = new_set
                    sums_by_k = new_sums
                    if len(current) > best_local_size:
                        best_local = list(current)
                        best_local_size = len(current)
                    improved = True
                    break
            
            if not improved:
                # Try swapping: remove one, add another
                for i in range(len(current)):
                    for v in range(256, 0, -1):
                        if v in current:
                            continue
                        new_set = current[:i] + current[i+1:] + [v]
                        new_sums = compute_sums_by_cardinality(new_set)
                        if check_valid_fast(new_sums):
                            # Check if this swap actually helps (allows future additions)
                            # For now, accept if valid
                            current = new_set
                            sums_by_k = new_sums
                            improved = True
                            break
                    if improved:
                        break
            
            if not improved:
                break
        
        return best_local

    # Phase 1: Greedy construction
    greedy_result = greedy_build()
    if len(greedy_result) > best_size:
        best_set = list(greedy_result)
        best_size = len(best_set)
    
    # Phase 2: Local search from greedy result
    if time.time() < deadline - 10:
        ls_result = local_search(greedy_result, deadline - 10)
        if len(ls_result) > best_size:
            best_set = list(ls_result)
            best_size = len(best_set)
    
    # Phase 3: Randomized restarts if time remains
    if time.time() < deadline - 5:
        for restart in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            random_start = random.sample(range(1, 257), min(10, 256))
            random_start.sort(reverse=True)
            
            # Try to extend randomly
            current = list(random_start)
            sums_by_k = compute_sums_by_cardinality(current)
            
            if not check_valid_fast(sums_by_k):
                continue
            
            # Greedy extend
            for v in range(256, 0, -1):
                if v in current:
                    continue
                new_set = current + [v]
                new_sums = compute_sums_by_cardinality(new_set)
                if check_valid_fast(new_sums):
                    current = new_set
                    sums_by_k = new_sums
            
            if len(current) > best_size:
                best_set = list(current)
                best_size = len(best_set)
    
    # Ensure types are correct
    return [int(x) for x in best_set]
