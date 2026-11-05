import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin
    
    n = 256
    
    # Initialize with the known valid upper tail
    # For a in [171, 256], 3a > 512, so no multiples of a in [2(a+1), 512]
    # Thus [171, 256] is valid.
    S = set(range(171, 257))
    
    def get_conflicts(x, S):
        """Return list of pairs (b,c) in S such that x divides b+c and x < b <= c."""
        conflicts = []
        S_list = sorted(S)
        # We need b, c in S such that b > x, c >= b, and (b+c) % x == 0
        # To speed up, we can iterate b and calculate required c
        for b in S_list:
            if b <= x:
                continue
            # We need c >= b and c in S such that (b+c) % x == 0
            # (b+c) % x == 0 => c % x == (-b) % x
            rem = (-b) % x
            # Find c in S with c >= b and c % x == rem
            # Since S is a set, we can just check candidates
            # But S is large.
            # Better: iterate c in S?
            # Given n=256, O(n^2) is fine.
            for c in S_list:
                if c < b:
                    continue
                if (b + c) % x == 0:
                    conflicts.append((b, c))
        return conflicts

    def is_valid(S):
        S_list = sorted(S)
        m = len(S_list)
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        return False
        return True

    def count_violations(S):
        """Count number of violating triples."""
        S_list = sorted(S)
        m = len(S_list)
        count = 0
        for i in range(m):
            a = S_list[i]
            for j in range(i + 1, m):
                b = S_list[j]
                for k in range(j, m):
                    c = S_list[k]
                    if (b + c) % a == 0:
                        count += 1
        return count

    # Local Search
    best_S = S
    best_size = len(S)
    
    rng = random.Random(42)
    
    # Try to improve by swapping or adding
    # Since adding a low element requires removing many high ones,
    # we focus on swaps that might allow better packing, or just verifying optimality.
    
    # A more effective search: Random Restart with Greedy + Local Search
    # But the Upper Tail is very strong.
    
    # Let's try to see if we can add any element by removing a minimal set.
    # We'll do a "Repair" pass.
    
    for x in range(170, 0, -1):
        if time.time() > deadline - 10:
            break
            
        if x in S:
            continue
            
        conflicts = get_conflicts(x, S)
        if not conflicts:
            S.add(x)
            best_S = S
            best_size = len(S)
            continue
        
        # If there are conflicts, can we remove a small set of elements to resolve them?
        # The conflicts are pairs (b,c). We need to remove at least one element from each pair.
        # This is a Vertex Cover problem on the conflict graph.
        # For small number of conflicts, we can try all subsets of elements to remove.
        
        # Collect unique elements involved in conflicts
        involved = set()
        for b, c in conflicts:
            involved.add(b)
            involved.add(c)
            
        if len(involved) > 8:
            # Too many to brute force subsets
            continue
            
        # Try removing subsets of 'involved' of size 1, 2, 3...
        # If removing k elements allows adding x, and 1 - k > 0 (impossible)
        # But maybe removing k elements allows adding x AND x-1?
        # For now, just check if size increases. It won't.
        # So we skip this unless we have a multi-add strategy.
        
        # Let's try a simple swap: Remove one 'involved' element, add x.
        # Size stays same. But maybe it opens up for others?
        # This is a heuristic.
        
        for elem in involved:
            new_S = S - {elem} | {x}
            # Quick check
            if len(new_S) != len(S):
                continue
            # Validate new_S
            if is_valid(new_S):
                # If valid, we have a set of same size.
                # Does it allow more additions?
                # Let's accept it and see.
                S = new_S
                break
                
    # Final polish: try to add any remaining elements
    for x in range(1, n + 1):
        if x in S:
            continue
        new_S = S | {x}
        if is_valid(new_S):
            S = new_S
            
    if len(S) > best_size:
        best_S = S
        best_size = len(S)
        
    return list(best_S)
