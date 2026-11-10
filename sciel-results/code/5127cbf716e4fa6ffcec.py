import time
import random
import numpy as np

N = 512

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # 90s wall-clock limit, 5s safety margin

    # Precompute all 4-term arithmetic progressions
    # For each element x, store the list of other 3 elements that form a 4-AP with x
    aps = [[] for _ in range(N + 1)]
    
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            aps[a].append((b, c, e))
            aps[b].append((a, c, e))
            aps[c].append((a, b, e))
            aps[e].append((a, b, c))

    # For efficiency, precompute conflict contributions
    # conflict_contrib[x] = list of (partner1, partner2, partner3) tuples
    # When x is added to S, for each (p1, p2, p3) in conflict_contrib[x],
    # if p1, p2, p3 are all in S, we get a conflict.
    # But we track conflict counts differently:
    # conflict[y] = number of 4-APs containing y where the other 3 are in S
    
    # Actually, let's use a different approach with bitsets for speed
    
    # Use numpy bitsets for fast operations
    # S as a bitset
    def get_conflicts(S_bits):
        """
        Given S as a numpy boolean array of length N+1,
        return conflict array where conflict[x] = number of 4-APs {x, a, b, c} 
        with a, b, c all in S.
        """
        conflict = np.zeros(N + 1, dtype=np.int32)
        for x in range(1, N + 1):
            if S_bits[x]:
                continue  # Only count for x not in S... wait, we need conflict for all x
            # Actually, we need conflict[x] for x NOT in S to determine if we can add x
            # Let's compute it differently
        
        # Better: for each x not in S, count how many 4-APs containing x have other 3 in S
        # This is expensive. Let's use a different strategy.
        
        # Alternative: maintain conflict incrementally
        pass
    
    # Let's use a set-based approach but with careful optimization
    # Key insight: use a "conflict count" array that we update incrementally
    
    best_size = 0
    best_set = set()
    
    def greedy_init(random_order=True):
        """Greedy initialization with optional randomization."""
        S = set()
        conflict = [0] * (N + 1)
        
        candidates = list(range(1, N + 1))
        if random_order:
            random.shuffle(candidates)
        else:
            # Use a structured order based on modular arithmetic
            # Elements with higher "4-AP density" might be better to skip
            # For now, just use natural order or reverse
            candidates = list(range(1, N + 1))
        
        for x in candidates:
            if conflict[x] == 0:
                S.add(x)
                # Update conflict counts for all elements that share a 4-AP with x
                for a, b, c in aps[x]:
                    # If a and b are in S, then conflict[c] increases (for c not in S)
                    # But we need to check all three combinations
                    if a in S and b in S:
                        if c not in S:
                            conflict[c] += 1
                    if a in S and c in S:
                        if b not in S:
                            conflict[b] += 1
                    if b in S and c in S:
                        if a not in S:
                            conflict[a] += 1
        
        return S, conflict
    
    def add_element(S, conflict, x):
        """Add element x to S and update conflict counts."""
        S.add(x)
        for a, b, c in aps[x]:
            if a in S and b in S:
                if c not in S:
                    conflict[c] += 1
            if a in S and c in S:
                if b not in S:
                    conflict[b] += 1
            if b in S and c in S:
                if a not in S:
                    conflict[a] += 1
    
    def remove_element(S, conflict, x):
        """Remove element x from S and update conflict counts."""
        S.discard(x)
        for a, b, c in aps[x]:
            if a in S and b in S:
                if c not in S:
                    conflict[c] -= 1
            if a in S and c in S:
                if b not in S:
                    conflict[b] -= 1
            if b in S and c in S:
                if a not in S:
                    conflict[a] -= 1
    
    def local_search(S, conflict, max_iterations=5000):
        """Local search: try to add elements, remove when stuck."""
        S = S.copy()
        conflict = conflict.copy()
        current_size = len(S)
        best_local_size = current_size
        best_local_set = S.copy()
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Find all elements that can be added
            valids = [x for x in range(1, N + 1) if x not in S and conflict[x] == 0]
            
            if valids:
                # Add a random valid element
                x = random.choice(valids)
                add_element(S, conflict, x)
                current_size += 1
                
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local_set = S.copy()
            else:
                # Stuck: try to remove an element to unblock
                if not S:
                    break
                
                # Try removing elements that are involved in many conflicts
                # For simplicity, remove a random element
                x = random.choice(list(S))
                remove_element(S, conflict, x)
                current_size -= 1
        
        return best_local_set, best_local_size
    
    def two_exchange_search(S, conflict, max_iterations=2000):
        """Try 2-exchanges: remove one element, add two others."""
        S = S.copy()
        conflict = conflict.copy()
        current_size = len(S)
        best_local_size = current_size
        best_local_set = S.copy()
        
        for _ in range(max_iterations):
            if time.time() > deadline:
                break
            
            # Find elements that can be added
            valids = [x for x in range(1, N + 1) if x not in S and conflict[x] == 0]
            
            if len(valids) >= 2:
                # Try to add two elements
                x1, x2 = random.sample(valids, 2)
                # Check if adding both is valid
                # Temporarily add x1
                S.add(x1)
                # Update conflict for x1
                for a, b, c in aps[x1]:
                    if a in S and b in S:
                        if c not in S:
                            conflict[c] += 1
                    if a in S and c in S:
                        if b not in S:
                            conflict[b] += 1
                    if b in S and c in S:
                        if a not in S:
                            conflict[a] += 1
                
                # Check if x2 is still valid
                if conflict[x2] == 0:
                    S.add(x2)
                    # Update conflict for x2
                    for a, b, c in aps[x2]:
                        if a in S and b in S:
                            if c not in S:
                                conflict[c] += 1
                        if a in S and c in S:
                            if b not in S:
                                conflict[b] += 1
                        if b in S and c in S:
                            if a not in S:
                                conflict[a] += 1
                    
                    current_size += 2
                    if current_size > best_local_size:
                        best_local_size = current_size
                        best_local_set = S.copy()
                else:
                    # Revert x1
                    S.discard(x1)
                    for a, b, c in aps[x1]:
                        if a in S and b in S:
                            if c not in S:
                                conflict[c] -= 1
                        if a in S and c in S:
                            if b not in S:
                                conflict[b] -= 1
                        if b in S and c in S:
                            if a not in S:
                                conflict[a] -= 1
            elif valids:
                # Add one valid element
                x = valids[0]
                S.add(x)
                for a, b, c in aps[x]:
                    if a in S and b in S:
                        if c not in S:
                            conflict[c] += 1
                    if a in S and c in S:
                        if b not in S:
                            conflict[b] += 1
                    if b in S and c in S:
                        if a not in S:
                            conflict[a] += 1
                current_size += 1
                if current_size > best_local_size:
                    best_local_size = current_size
                    best_local_set = S.copy()
            else:
                # Remove a random element to unblock
                if not S:
                    break
                x = random.choice(list(S))
                S.discard(x)
                for a, b, c in aps[x]:
                    if a in S and b in S:
                        if c not in S:
                            conflict[c] -= 1
                    if a in S and c in S:
                        if b not in S:
                            conflict[b] -= 1
                    if b in S and c in S:
                        if a not in S:
                            conflict[a] -= 1
                current_size -= 1
        
        return best_local_set, best_local_size
    
    # Run multiple rounds of initialization and local search
    num_rounds = 0
    while time.time() < deadline:
        num_rounds += 1
        
        # Different initialization strategies
        if num_rounds % 3 == 1:
            # Random greedy
            S, conflict = greedy_init(random_order=True)
        elif num_rounds % 3 == 2:
            # Natural order greedy
            S, conflict = greedy_init(random_order=False)
        else:
            # Skip some elements first, then fill
            S, conflict = greedy_init(random_order=True)
            # Remove some elements to create space
            if len(S) > 10:
                for _ in range(min(5, len(S) // 2)):
                    if not S:
                        break
                    x = random.choice(list(S))
                    remove_element(S, conflict, x)
        
        current_size = len(S)
        if current_size > best_size:
            best_size = current_size
            best_set = S.copy()
        
        # Local search phase 1
        S, local_size = local_search(S, conflict, max_iterations=2000)
        if local_size > best_size:
            best_size = local_size
            best_set = S.copy()
        
        # Rebuild conflict for the improved set
        if best_size > current_size:
            # Recompute conflict from scratch for best_set
            conflict = [0] * (N + 1)
            S_rebuilt = set(best_set)
            for x in range(1, N + 1):
                if x in S_rebuilt:
                    for a, b, c in aps[x]:
                        if a in S_rebuilt and b in S_rebuilt:
                            if c not in S_rebuilt:
                                conflict[c] += 1
                        if a in S_rebuilt and c in S_rebuilt:
                            if b not in S_rebuilt:
                                conflict[b] += 1
                        if b in S_rebuilt and c in S_rebuilt:
                            if a not in S_rebuilt:
                                conflict[a] += 1
            S = S_rebuilt
        
        # Local search phase 2: two-exchange
        S, local_size2 = two_exchange_search(S, conflict, max_iterations=1000)
        if local_size2 > best_size:
            best_size = local_size2
            best_set = S.copy()
    
    # Final verification
    # Ensure best_set is 4-AP-free
    final_set = set(best_set)
    # Verify no 4-APs
    for a in range(1, N + 1):
        max_d = (N - a) // 3
        for d in range(1, max_d + 1):
            b = a + d
            c = a + 2 * d
            e = a + 3 * d
            if a in final_set and b in final_set and c in final_set and e in final_set:
                # This should not happen, but if it does, remove one element
                final_set.discard(e)
                break
    
    return sorted(final_set)
